"""RQ1 at scale: when a coding agent states a fact about the repository (a file path, a symbol,
a function signature), had it already SEEN that fact in the run, and is the fact TRUE at the
task's base commit? If it is wrong, does the agent notice and change course?

Extends provenance_pilot.py (file paths only, 3 SWE-bench Verified models) to symbols and
signatures, and to the DeepSWE sample (39 models, 5 languages).

Fact types (each is counted once per trajectory, at its first mention):
  path       a repo file path, e.g. django/contrib/auth/__init__.py            (all languages)
  module     a dotted Python module, e.g. django.utils.encoding                  (Python)
  symbol     a name inside a module, e.g. django.utils.encoding.force_str, from a dotted
             mention, an import in code the agent writes, or an alias of such an import
  code_name  a function/class/method name called in code the agent writes, e.g. .get_field(
             (only names that no standard library and no other studied repo of the same
             language defines, so library names like .append( or np.array( drop out)
  sig_def    a signature stated in the agent's text: "def f(a, b)", "func F(a int)", ...
  sig_call   a call-shaped statement in the agent's text whose arguments are all parameter
             names of f, e.g. "get_permission_codename(action, opts)": checks the ORDER
  kwarg      a keyword argument the agent passes to a repo function in its code, e.g. f(x, strict=True)

Named before seen (NBS): the first mention is not preceded by the fact appearing in the task
text or in an earlier tool output (the agent's own earlier words do not count as seeing).
  path:    pilot rule: full path, or its file name, or its dotted module name seen -> seen
  dotted:  the dotted string (or its path form) seen -> seen; its last name seen -> seen_name
  names:   the name appeared as a whole word before -> seen
  sig:     the definition line, or the same name with the same parameter list, appeared before
So NBS is a lower bound: anything that might have been read counts as seen.

Truth at the base commit (see rq1_repo_index.py): real / stale (only in older versions) /
future (only in newer versions) / never; signatures: match / order_wrong / names_differ;
keyword arguments: valid / invalid.

Self-correction: for a wrong NBS fact, the first later tool output that contradicts it (an error
line naming it: No such file, ImportError, AttributeError, NameError, TypeError, undefined:,
Cannot find name, ... or an empty search for it), and whether the agent stops using it afterwards.

Usage (from the project folder; every step can be rerun, caches live in data/repos_deepswe/_rq1_cache):
  .venv/bin/python -I scripts/rq1_fact_provenance.py index swebv      # fetch + parse repo files
  .venv/bin/python -I scripts/rq1_fact_provenance.py index deepswe
  .venv/bin/python -I scripts/rq1_fact_provenance.py vocab            # library names to ignore
  .venv/bin/python -I scripts/rq1_fact_provenance.py run swebv   [--limit N] [--workers N]
  .venv/bin/python -I scripts/rq1_fact_provenance.py run deepswe [--limit N] [--workers N]
  .venv/bin/python -I scripts/rq1_fact_provenance.py summarize
Outputs: outputs/rq1/facts_swebv.jsonl, facts_deepswe.jsonl, facts_summary.json, fig_*.csv.
DeepSWE rows carry ids, facts and step numbers only (no trajectory text: no redistribution license).
"""
import builtins
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I does not add the script folder; our helper modules live here
from rq1_repo_index import Repo, split_params, split_top  # noqa: E402
from summarize_deepswe import iter_steps  # noqa: E402

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs', 'rq1')
CACHE = os.path.join(DATA, 'repos_deepswe', '_rq1_cache')
SWEBV_PARQUET = os.path.join(DATA, 'meta', 'swebv', 'data', 'test-00000-of-00001.parquet')
SWEBV_RUNS = {  # model label -> run folder in data/raw
    'deepseek-v3.2': '20251201_mini-v1.17.1_deepseek-v3.2-reasoner',
    'kimi-k2.5': '20260217_mini-v2.0.0_kimi-k2-5-high',
    'minimax-m2.5': '20260217_mini-v2.0.0_minimax-2-5-high',
}

# Open vs closed weights, by vendor family (NOT verified per release: several of these models
# came out after the cutoff of the model writing this script). "open" = the family has published
# weights for earlier versions (DeepSeek, Kimi/Moonshot, MiniMax, GLM/Zhipu, MiMo/Xiaomi).
# Qwen "Plus"/"Max" are hosted-only products, so "closed".
OPEN_FAMILIES = ('deepseek', 'kimi', 'minimax', 'glm', 'mimo')


def weights_of(model):
    return 'open' if model.lower().startswith(OPEN_FAMILIES) else 'closed'


# ----------------------------------------------------------------------------------------------
# 1. Reading trajectories into steps
#    step = {'step', 'reasoning', 'visible', 'commands', 'obs', 'rc', 'fake'}
# ----------------------------------------------------------------------------------------------

def text_of(content):
    if isinstance(content, list):
        return '\n'.join(c.get('text', '') for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ''


RC_RE = re.compile(r'<returncode>(-?\d+)</returncode>')


def read_minisweagent(path):
    """SWE-bench Verified runs (mini-swe-agent v1 text actions or v2 tool calls).
    Same logic as provenance_pilot.steps_of, plus return codes and fenced code in the text."""
    with open(path) as f:
        traj = json.load(f)
    msgs = traj['messages']
    task = '\n'.join(text_of(m.get('content')) for m in msgs[:2])  # system prompt + task
    steps = []
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        resp = (m.get('extra') or {}).get('response') or {}
        msg = ((resp.get('choices') or [{}])[0].get('message') or {}) if isinstance(resp, dict) else {}
        reasoning = m.get('reasoning') or msg.get('reasoning_content') or msg.get('reasoning') or ''
        if not reasoning and isinstance(msg.get('thinking_blocks'), list):
            reasoning = '\n'.join(b.get('thinking', '') for b in msg['thinking_blocks'] if isinstance(b, dict))
        content = text_of(m.get('content'))
        commands = re.findall(r'```(?:bash|sh)?\n(.*?)```', content, re.S)  # v1: action in a bash block
        for tc in (m.get('tool_calls') or msg.get('tool_calls') or []):     # v2: tool call
            try:
                commands.append(json.loads(tc['function']['arguments']).get('command', ''))
            except Exception:
                pass
        visible = re.sub(r'```(?:bash|sh)?\n.*?```', ' ', content, flags=re.S)
        obs = []
        for n in msgs[i + 1:]:
            if n.get('role') == 'assistant':
                break
            obs.append(text_of(n.get('content')))
        obs = '\n'.join(obs)
        rc = RC_RE.search(obs)
        steps.append({'step': len(steps) + 1, 'reasoning': str(reasoning), 'visible': visible,
                      'commands': commands, 'obs': obs, 'rc': int(rc.group(1)) if rc else None,
                      # the model sometimes types a fake tool result into its own message
                      'fake': '<returncode>' in content or '<output>' in content})
    patch = (traj.get('info') or {}).get('submission') or ''
    return task, steps, patch


def decode_obs(raw):
    """DeepSWE tool outputs are JSON strings {"returncode", "output" | "output_head"/"output_tail", ...}."""
    try:
        d = json.loads(raw)
    except Exception:
        return raw, None
    if not isinstance(d, dict):
        return raw, None
    parts = [d.get(k) for k in ('output', 'output_head', 'output_tail', 'exception_info') if isinstance(d.get(k), str)]
    rc = d.get('returncode')
    return '\n'.join(parts), rc if isinstance(rc, int) else None


def read_atif(folder):
    """DeepSWE trials (Harbor ATIF trajectories, converted from mini-swe-agent)."""
    with open(os.path.join(folder, 'trajectory.json')) as f:
        traj = json.load(f)
    task = '\n'.join(s.get('message') or '' for s in traj.get('steps', []) if s.get('source') in ('system', 'user'))
    steps = []
    for s in iter_steps(traj):
        outs = [decode_obs(o) for o in s['outputs']]
        rcs = [rc for _, rc in outs if rc is not None]
        reasoning = s['reasoning']
        if reasoning and reasoning == s['message']:
            reasoning = ''  # some harness versions copy the visible message into the reasoning field
        steps.append({'step': len(steps) + 1, 'reasoning': reasoning, 'visible': s['message'],
                      'commands': [c for c in s['commands'] if c], 'obs': '\n'.join(o for o, _ in outs),
                      'rc': max(rcs) if rcs else None, 'fake': False})
    patch_file = os.path.join(folder, 'model.patch')
    patch = open(patch_file, errors='replace').read() if os.path.exists(patch_file) else ''
    tokens = ((traj.get('final_metrics') or {}).get('extra') or {}).get('total_reasoning_tokens')
    return task, steps, patch, tokens


# ----------------------------------------------------------------------------------------------
# 2. Finding fact mentions
# ----------------------------------------------------------------------------------------------

# file paths: the pilot's pattern, with more file types and the /app/ work folder of DeepSWE
EXT = (r'(?:py|pyx|pxd|pyi|c|h|cc|cpp|hpp|cfg|ini|toml|txt|rst|md|mdx|json|jsonc|yaml|yml|html|js|jsx|mjs|cjs|'
       r'ts|tsx|mts|cts|go|mod|sum|rs|css|scss|less|vue|svelte|po|sh|in|lock|proto|sql|graphql|gql|xml|j2|tmpl|snap)')
PATH_RE = re.compile(r'(?<![\w.\-/:@~$])((?:\.{1,2}/|/testbed/|/app/)?(?:[A-Za-z_@][\w.\-@]*/)+[A-Za-z_][\w.\-]*\.'
                     + EXT + r')(?![\w/])')
PY_WRITE_RE = re.compile(r"""open\(\s*['"]([^'"]+)['"]\s*,\s*['"][wax]|Path\(\s*['"]([^'"]+)['"]\s*\)\.write_""")
WRITE_RE = re.compile(r'(?:>\s*|tee\s+(?:-a\s+)?|touch\s+|cp\s+\S+\s+|mv\s+\S+\s+|mkdir\s+-p\s+)(\S+)')
# third-party, cache and build-output folders: files there are not repo files
PATH_SKIP = {'node_modules', '.git', '__pycache__', 'site-packages', 'dist-packages', '.venv', 'venv',
             '.pytest_cache', '.mypy_cache', '.tox', '.cargo', 'go', '.npm', '.pnpm-store',
             'build', 'dist', 'target', 'coverage', '.next', '.turbo', '.cache', '_build', 'htmlcov', '.eggs'}
PLAN_RE = re.compile(r"\b(creat\w*|add|adding|added|new|writ\w*|implement\w*|introduc\w*|generat\w*|"
                     r"will (?:make|put|define)|we'll|i'll|let's make|should (?:be )?(?:added|created|defined))\b", re.I)
HEDGE = re.compile(r"\b(probably|likely|might|may be|maybe|perhaps|i think|i believe|should be|presumably|"
                   r"if i recall|i recall|i remember|from memory|typically|usually|i'm not sure|not sure|"
                   r"let me (?:check|find|look|verify|search|see|confirm|locate))\b", re.I)

DOTTED_RE = re.compile(r'(?<![\w./\-:$@])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?!\w)')
PY_FROM_RE = re.compile(r'(?:^|(?<=[;\'"]))[ \t]*from[ \t]+([A-Za-z_][\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n#;\'"]*)', re.M)
PY_IMPORT_RE = re.compile(r'(?:^|(?<=[;\'"]))[ \t]*import[ \t]+([\w.]+(?:[ \t]+as[ \t]+\w+)?(?:[ \t]*,[ \t]*[\w.]+(?:[ \t]+as[ \t]+\w+)?)*)', re.M)
TS_IMPORT_RE = re.compile(r'\bimport[ \t]+(?:type[ \t]+)?([^;\'"]*?)[ \t]+from[ \t]+[\'"]([^\'"]+)[\'"]')
REQUIRE_RE = re.compile(r'\bconst[ \t]+(\{[^}]*\}|[A-Za-z_$][\w$]*)[ \t]*=[ \t]*require\([\'"]([^\'"]+)[\'"]\)')
GO_IMPORT_RE = re.compile(r'^[ \t]*(?:import[ \t]+)?(?:([A-Za-z_]\w*|\.)[ \t]+)?"([\w.\-/]+)"', re.M)
RS_USE_RE = re.compile(r'\buse[ \t]+((?:crate|super|self|[a-z_]\w*)::[\w:{}, \t*]+);')

HEREDOC_RE = re.compile(r"<<-?[ \t]*['\"]?([A-Za-z_]\w*)['\"]?[^\n]*\n(.*?)\n[ \t]*\1[ \t]*(?:\n|$)", re.S)
PY_C_RE = re.compile(r"""\bpython[\d.]*[ \t]+(?:-[A-Za-z]+[ \t]+)*-c[ \t]+(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""", re.S)
NODE_E_RE = re.compile(r"""\b(?:node|deno|bun|tsx|ts-node)[ \t]+(?:-[A-Za-z-]+[ \t]+)*-(?:e|p|-eval)[ \t]+(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""", re.S)
FENCE_RE = re.compile(r'```([\w+-]*)\n(.*?)```', re.S)
CALL_RE = re.compile(r'(?<![\w$\\])([A-Za-z_$][\w$]*)!?\(')   # name( ; see is_member() for x.name(
QUAL_CALL_RE = re.compile(r'(?<![\w$.\\])([A-Za-z_$][\w$]*)(\.|::)([A-Za-z_$][\w$]*)\(')

# names the agent defines itself in code it writes (not repo facts)
AGENT_DEF_RES = [
    re.compile(r'\b(?:def|class|function\*?|func|fn|struct|enum|trait|interface|type|const|let|var|static)[ \t]+'
               r'(?:\([^)]*\)[ \t]*)?([A-Za-z_$][\w$]*)'),
    re.compile(r'^[ \t]*([A-Za-z_]\w*)[ \t]*(?::[^=\n]*)?=(?!=)', re.M),         # x = ... / x: T = ...
    re.compile(r'^[ \t]*([A-Za-z_]\w*)[ \t]*:=', re.M),                          # Go x := ...
    re.compile(r'^[ \t]*(?:var[ \t]+)?[A-Za-z_]\w*(?:[ \t]*,[ \t]*([A-Za-z_]\w*))+[ \t]*:?=(?!=)', re.M),
    re.compile(r'\bfor[ \t]+([A-Za-z_]\w*)[ \t]+in\b'),
    re.compile(r'\bas[ \t]+([A-Za-z_]\w*)'),
    # TS/JS class methods (foo(...) {) and property functions (foo: (...) => / foo = function)
    re.compile(r'^[ \t]*(?:(?:public|private|protected|static|async|get|set|override|readonly)[ \t]+)*'
               r'#?([A-Za-z_$][\w$]*)[ \t]*(?:<[^>\n]*>)?\((?:[^)\n]*\)[ \t]*(?::[^{\n;]*)?\{|[^)\n;]*$)', re.M),
    re.compile(r'^[ \t]*([A-Za-z_$][\w$]*)[ \t]*[:=][ \t]*(?:async[ \t]*)?(?:function|\([^)\n]*\)[ \t]*=>)', re.M),
]

PY_KEYWORDS = {'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
               'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not',
               'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield', 'print', 'exec', 'match', 'case'}
OTHER_KEYWORDS = {
    'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'typeof', 'new', 'await', 'super', 'import',
    'export', 'else', 'do', 'try', 'with', 'yield', 'delete', 'void', 'case', 'throw', 'instanceof', 'in', 'of',
    'func', 'go', 'defer', 'select', 'range', 'map', 'chan', 'struct', 'interface', 'type', 'package', 'var', 'const',
    'fn', 'let', 'match', 'loop', 'impl', 'where', 'move', 'unsafe', 'pub', 'mod', 'use', 'crate', 'self', 'Self',
    'async', 'static', 'extern', 'ref', 'mut', 'as', 'dyn', 'this', 'constructor', 'require', 'assert', 'sizeof',
    # JS/TS globals and test helpers
    'Object', 'Array', 'String', 'Number', 'Boolean', 'Symbol', 'BigInt', 'Math', 'JSON', 'Date', 'RegExp', 'Error',
    'TypeError', 'RangeError', 'SyntaxError', 'Promise', 'Map', 'Set', 'WeakMap', 'WeakSet', 'Proxy', 'Reflect',
    'console', 'process', 'module', 'exports', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'setImmediate', 'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURIComponent', 'decodeURIComponent',
    'encodeURI', 'decodeURI', 'structuredClone', 'fetch', 'describe', 'it', 'test', 'expect', 'beforeEach',
    'afterEach', 'beforeAll', 'afterAll', 'vi', 'jest', 'Buffer', 'URL', 'URLSearchParams', 'TextEncoder',
    'TextDecoder', 'queueMicrotask', 'globalThis', 'window', 'document', 'Uint8Array', 'Int32Array', 'Float64Array',
    'ArrayBuffer', 'DataView', 'AbortController', 'Response', 'Request', 'Headers', 'FormData', 'Blob', 'atob', 'btoa',
    'Function', 'Intl', 'Iterator', 'AggregateError', 'alert', 'eval', 'unescape', 'escape', 'suite', 'bench',
    # Go builtins and conversions
    'append', 'cap', 'close', 'complex', 'copy', 'imag', 'len', 'make', 'panic', 'print', 'println', 'real',
    'recover', 'min', 'max', 'clear', 'int', 'int8', 'int16', 'int32', 'int64', 'uint', 'uint8', 'uint16', 'uint32',
    'uint64', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'string', 'byte', 'rune', 'bool', 'error',
    'any', 'nil', 'true', 'false', 'iota',
    # Rust prelude
    'Some', 'None', 'Ok', 'Err', 'Box', 'Vec', 'Option', 'Result', 'Rc', 'Arc', 'RefCell', 'Cell', 'HashMap',
    'HashSet', 'BTreeMap', 'BTreeSet', 'VecDeque', 'Default', 'From', 'Into', 'Iterator', 'drop', 'format',
    'vec', 'write', 'writeln', 'unreachable', 'todo', 'unimplemented', 'matches', 'dbg', 'eprintln', 'include_str'}
PY_BUILTINS = set(dir(builtins))  # Python builtins: never Go/TS/Rust repo names
# common aliases of non-repo Python modules: x.name( with these receivers is a library call
PY_EXT_ALIASES = {'np', 'pd', 'os', 'sys', 're', 'json', 'math', 'subprocess', 'shutil', 'pytest', 'unittest', 'mock',
                  'random', 'time', 'datetime', 'collections', 'itertools', 'functools', 'pathlib', 'typing', 'io',
                  'textwrap', 'inspect', 'ast', 'tempfile', 'glob', 'warnings', 'logging', 'copy', 'pickle', 'string',
                  'struct', 'decimal', 'operator', 'traceback', 'threading', 'asyncio', 'socket', 'csv', 'hashlib',
                  'base64', 'uuid', 'platform', 'contextlib', 'dataclasses', 'enum', 'types', 'importlib', 'doctest',
                  'argparse', 'shlex', 'difflib', 'pprint', 'heapq', 'bisect', 'codecs', 'tracemalloc', 'gc', 'weakref'}
GO_STD_PKGS = {'fmt', 'strings', 'os', 'io', 'errors', 'bytes', 'strconv', 'time', 'sort', 'sync', 'context', 'reflect',
               'regexp', 'math', 'path', 'filepath', 'testing', 'json', 'http', 'bufio', 'unicode', 'utf8', 'atomic',
               'log', 'flag', 'exec', 'runtime', 'slices', 'maps', 'cmp', 'rand', 'hex', 'base64', 'sha256', 'md5',
               'url', 'net', 'ioutil', 'filepathx', 'template', 'debug', 'signal', 'syscall', 'unsafe', 'big',
               'binary', 'csv', 'xml', 'gzip', 'zip', 'tar', 'heap', 'list', 'ring', 'fs', 'embed', 'iter',
               'httptest', 'assert', 'require', 'yaml', 'slog', 'errgroup', 'cobra', 'pflag', 'color', 'semver'}
ERROR_RE = re.compile(
    r"No such file or directory|cannot access|can't open|can't read|cannot open|does not exist|not found|"
    r"[Cc]annot find|[Cc]ould not find|No module named|ModuleNotFoundError|ImportError|cannot import name|"
    r"AttributeError|has no attribute|NameError|is not defined|TypeError|unexpected keyword argument|"
    r"positional argument|got multiple values|undefined|undeclared|has no exported member|is not exported|"
    r"unresolved import|no method named|no function or associated item|not a function|does not provide an export|"
    r"error\[E\d+\]|no field or method|unknown field|did you mean|cannot use|not declared|no such|is not a module")
SEARCH_CMD_RE = re.compile(r'\b(?:grep|rg|ag|ack|git[ \t]+grep|find|ls|fd)\b')


def is_member(code, m):
    """True if the call matched at m is written x.name( or X::name( (a method or a qualified name),
    also when a chained call continues on the next line (x.\n    Name()."""
    before = code[max(0, m.start() - 200):m.start()].rstrip()
    return before.endswith('.') or before.endswith('::')


PY_STRING_RE = re.compile(r'(?s)[rRbBuUfF]{0,2}("""|\'\'\')(.*?)\1|[rRbBuUfF]{0,2}(["\'])((?:\\.|(?!\3)[^\n])*)\3')


def python_strings(code):
    """The text inside the string literals of a Python script (an edit script's new code lives there)."""
    return '\n'.join((m.group(2) or m.group(4) or '') for m in PY_STRING_RE.finditer(code))


DOC_EXT = {'.md', '.mdx', '.rst', '.txt', '.json', '.yaml', '.yml', '.html', '.css', '.toml', '.csv', '.snap'}


def strip_noise(code):
    """Remove fenced examples, comments and string literals from Go/TS/JS/Rust code, so example
    snippets in docs comments and SQL or messages inside strings are not read as calls."""
    code = re.sub(r'```.*?```', ' ', code, flags=re.S)
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.S)
    code = re.sub(r'(?<![:\w])//[^\n]*', ' ', code)
    code = re.sub(r'`(?:\\.|[^`\\])*`', '""', code, flags=re.S)
    code = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\\n])*'", "''", code)
    return code


def norm_path(p):
    p = p.strip('`\'"(),;:')
    for pre in ('/testbed/', '/app/', './', 'testbed/'):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


SED_SUB_RE = re.compile(r"""\bsed\b[^'"\n]*['"]\d*,?\d*s([/|#,])((?:\\.|(?!\1).)*)\1((?:\\.|(?!\1).)*)\1""")
SED_ADD_RE = re.compile(r"""\bsed\b[^'"\n]*['"]\d*[aic]\\?[ \t]*((?:[^'"\\]|\\.)*)['"]""")
ECHO_RE = re.compile(r"""\b(?:echo|printf)[ \t]+(?:-[a-z]+[ \t]+)*(['"])(.*?)\1[ \t]*>>?[ \t]*([^\s;&|]+)""", re.S)


def code_segments(command):
    """Code the agent writes inside a shell command: heredoc bodies, python -c / node -e strings,
    sed replacement/insert text, echo/printf into a file. Returns [(target_file_or_interpreter, code)]."""
    segs = []
    for m in SED_SUB_RE.finditer(command):
        segs.append(('sed', m.group(3).replace('\\n', '\n')))
    for m in SED_ADD_RE.finditer(command):
        segs.append(('sed', m.group(1).replace('\\n', '\n')))
    for m in ECHO_RE.finditer(command):
        segs.append((m.group(3), m.group(2).replace('\\n', '\n')))
    for m in HEREDOC_RE.finditer(command):
        before = command[max(0, m.start() - 200):m.start()]
        target = re.findall(r'(?:>|tee(?:[ \t]+-a)?)[ \t]*([^\s<>|;&]+)[ \t]*$', before.split('\n')[-1])
        interp = re.findall(r'\b(python[\d.]*|node|go|cargo|tsx|ts-node|deno|bun)\b', before.split('\n')[-1])
        segs.append((target[-1] if target else (interp[-1] if interp else ''), m.group(2)))
    for m in PY_C_RE.finditer(command):
        segs.append(('python', (m.group(1) or m.group(2) or '').replace('\\"', '"')))
    for m in NODE_E_RE.finditer(command):
        segs.append(('node', (m.group(1) or m.group(2) or '').replace('\\"', '"')))
    return segs


def agent_defined(code):
    names = set()
    for rx in AGENT_DEF_RES:
        names.update(rx.findall(code))
    for m in re.finditer(r'^[ \t]*(?:var[ \t]+)?((?:[A-Za-z_]\w*[ \t]*,[ \t]*)+[A-Za-z_]\w*)[ \t]*:?=(?!=)', code, re.M):
        names.update(n.strip() for n in m.group(1).split(','))  # a, b = ... / x, ok := ...
    # parameters of its own functions (Python def, Go func, TS function, Rust fn, arrow functions)
    for m in re.finditer(r'\b(?:def|func|function|fn)\b[^(\n]*\(([^)]*)\)|\(([^()]*)\)[ \t]*(?::[^=\n]*)?=>', code):
        for part in split_top(m.group(1) or m.group(2) or ''):
            w = re.match(r'[*.&\s]*(?:mut\s+)?([A-Za-z_$][\w$]*)', part)
            if w:
                names.add(w.group(1))
    return names


def py_imports(code):
    """[(module, name_or_None, alias)] for absolute imports in Python code."""
    out = []
    for mod, names in PY_FROM_RE.findall(code):
        for n in names.strip('()').replace('\n', ' ').split(','):
            n = n.strip()
            if not n or n == '*':
                continue
            parts = n.split()
            if re.fullmatch(r'\w+', parts[0]):
                out.append((mod, parts[0], parts[2] if len(parts) == 3 and parts[1] == 'as' else parts[0]))
    for group in PY_IMPORT_RE.findall(code):
        for n in group.split(','):
            parts = n.split()
            if not parts:
                continue
            mod = parts[0]
            alias = parts[2] if len(parts) == 3 and parts[1] == 'as' else mod.split('.')[0]
            out.append((mod, None, alias))
    return out


def find_calls_with_kwargs(code):
    """Yield (callee, [keyword names]) for plain calls like f(a, b=1). Method calls x.f(b=1) are
    skipped: x may be another library's object. Capitalized callees (classes) are skipped too."""
    for m in re.finditer(r'(?<![\w$\\.])([a-z_]\w*)\(', code):
        if code[max(0, m.start() - 4):m.start()].strip() in ('def', 'class'):
            continue
        depth, i = 1, m.end()
        while i < len(code) and depth and i - m.end() < 2000:
            depth += {'(': 1, ')': -1}.get(code[i], 0)
            i += 1
        kws = [re.match(r'([A-Za-z_]\w*)[ \t]*=(?!=)', a).group(1) for a in split_top(code[m.end():i - 1])
               if re.match(r'([A-Za-z_]\w*)[ \t]*=(?!=)', a)]
        if kws:
            yield m.group(1), kws


SIG_TEXT_RES = {
    'py': re.compile(r'\bdef[ \t]+([A-Za-z_]\w*)[ \t]*\(([^)]{0,300})\)'),
    'go': re.compile(r'\bfunc[ \t]+(?:\([^)]*\)[ \t]*)?([A-Za-z_]\w*)[ \t]*\(([^)]{0,300})\)'),
    'ts': re.compile(r'\bfunction\*?[ \t]+([A-Za-z_$][\w$]*)[ \t]*(?:<[^>\n]*>)?\(([^)]{0,300})\)'),
    'rs': re.compile(r'\bfn[ \t]+([A-Za-z_]\w*)[ \t]*(?:<[^>{\n]*>)?\(([^)]{0,300})\)'),
}
SIG_TEXT_RES['js'] = SIG_TEXT_RES['ts']
CALL_SHAPE_RE = re.compile(r'(?<![\w$.])((?:[A-Za-z_]\w*\.)*[A-Za-z_$][\w$]*)\(([A-Za-z_$][\w$]*(?:[ \t]*,[ \t]*[A-Za-z_$][\w$]*)+)\)')


def claimed_params(raw, lang):
    """(positional names, keyword-only names) of a signature the agent wrote; self/cls dropped."""
    from rq1_repo_index import go_param_names, params_from_names
    names = go_param_names(raw) if lang == 'go' else split_params(raw)
    p = params_from_names([n for n in names if n not in ('self', 'cls', 'this')])
    return p['pos'], p['kwonly']


def def_params(sig):
    """Positional parameter names of a definition, without self/cls (receivers are not listed in Go)."""
    pos = [p for p in sig['pos'] if p not in ('self', 'cls')]
    return pos


# ----------------------------------------------------------------------------------------------
# 3. Checking a fact against the repo
# ----------------------------------------------------------------------------------------------

class RepoView:
    """A Repo plus the commits indexed for it, with the truth checks used below."""

    def __init__(self, repo, snapshots, lang, vocab, go_module=None):
        self.repo, self.snaps, self.lang, self.vocab, self.go_module = repo, snapshots, lang, vocab, go_module
        self.times = {c: repo.commit_time(c) for c in snapshots}
        self._py_roots = {}
        self._go_dirs = {}
        self._by_name = {}
        self._top_dirs = None
        self._memo = {}

    # -- paths (pilot logic) --
    def resolve_path(self, mention, commit):
        """Exact path, or a partial path that is the tail of a real path (pilot rule)."""
        paths = self.repo.paths(commit)
        if mention in paths:
            return mention
        if commit not in self._by_name:
            idx = defaultdict(list)
            for p in paths:
                idx[p.rsplit('/', 1)[-1]].append(p)
            self._by_name[commit] = idx
        hits = [p for p in self._by_name[commit].get(mention.rsplit('/', 1)[-1], []) if p.endswith('/' + mention)]
        return hits[0] if hits else None

    def path_truth(self, p, commit):
        """('real'|'stale'|'future'|'never'|'outside', years from base to first add)."""
        if self.resolve_path(p, commit):
            return 'real', None
        adds = self.repo.path_adds()
        if self._top_dirs is None:
            self._top_dirs = {q.split('/', 1)[0] for q in adds} | {q.split('/', 1)[0] for q in self.repo.paths(commit)}
            self._all_dirs = {d for q in adds for d in q.split('/')[:-1]}
        if p.split('/', 1)[0] not in self._all_dirs:
            return 'outside', None  # e.g. subdir/page.html, testapp/models.py: no such folder ever existed
        hit = p if p in adds else None
        if not hit:
            name = p.rsplit('/', 1)[-1]
            cands = [q for q in self.repo._adds_by_name.get(name, []) if q.endswith('/' + p)]
            hit = cands[0] if cands else None
        if not hit:
            return 'never', None
        t0 = self.repo.commit_time(commit)
        first = adds[hit][0]
        return ('stale' if first < t0 else 'future'), round((first - t0) / 31557600, 2)

    # -- symbols defined anywhere --
    def name_truth(self, name, commit):
        """real if some source file at the base commit defines the name (or, in Python, imports it
        at module level); else stale/future/never by the snapshots that define it."""
        key = ('name', name, commit)
        if key not in self._memo:
            if self.repo.defined(name, commit) or self.repo.imported(name, commit):
                self._memo[key] = 'real'
            else:
                self._memo[key] = self._other_versions(lambda c: self.repo.defined(name, c), commit)
        return self._memo[key]

    def _other_versions(self, test, commit):
        t0 = self.times.get(commit) or self.repo.commit_time(commit)
        older = any(test(c) for c in self.snaps if c != commit and (self.times.get(c) or 0) < t0)
        newer = any(test(c) for c in self.snaps if c != commit and (self.times.get(c) or 0) > t0)
        return 'stale' if older else ('future' if newer else 'never')

    # -- Python modules --
    def py_roots(self, commit):
        """Top-level names a dotted Python path can start with: packages, src/ packages, and
        top-level folders holding .py files (e.g. Django's tests/)."""
        if commit not in self._py_roots:
            roots = set()
            for p in self.repo.source_blobs(commit):
                parts = p.split('/')
                if parts[0] in ('src', 'lib') and len(parts) == 3 and parts[2] == '__init__.py':
                    roots.add(parts[1])
                elif len(parts) == 2 and parts[1] == '__init__.py':
                    roots.add(parts[0])
                elif parts[0] in ('tests', 'test', 'testing') and p.endswith('.py'):
                    roots.add(parts[0])  # e.g. Django's tests/ folder, used as tests.settings
            self._py_roots[commit] = roots
        return self._py_roots[commit]

    def resolve_dotted(self, dotted, commit):
        """'ok_module' | 'ok_symbol' | 'missing' | 'unknown' (file contents not available).
        Only the module path and the first name inside it are checked (django.db.models.Model in
        django.db.models.Model._meta): deeper attributes are often set at run time."""
        parts = dotted.split('.')
        for k in range(len(parts), 0, -1):
            hit = self.repo.py_module_file('.'.join(parts[:k]), commit)
            if hit:
                break
        else:
            return 'missing'
        if k == len(parts):
            return 'ok_module'
        names = self.repo.py_module_names(hit[0], commit)
        if names is None:
            return 'unknown'
        if parts[k] not in names:
            return 'unknown' if '<dynamic>' in names else 'missing'
        return 'ok_symbol'

    def judged_prefix(self, dotted, commit):
        """The part of a dotted name that resolve_dotted checks: module path + first name."""
        parts = dotted.split('.')
        for k in range(len(parts), 0, -1):
            if self.repo.py_module_file('.'.join(parts[:k]), commit):
                return '.'.join(parts[:k + 1])
        return dotted

    def dotted_truth(self, dotted, commit):
        key = ('dotted', dotted, commit)
        if key not in self._memo:
            r = self.resolve_dotted(dotted, commit)
            if r.startswith('ok'):
                self._memo[key] = ('real', r[3:])
            elif r == 'unknown':
                self._memo[key] = ('unknown', None)
            else:
                # label it a module if it is one in some other version, else a symbol
                kind = 'module' if any(self.resolve_dotted(dotted, c) == 'ok_module' for c in self.snaps) else 'symbol'
                self._memo[key] = (self._other_versions(
                    lambda c: self.resolve_dotted(dotted, c).startswith('ok'), commit), kind)
        return self._memo[key]

    # -- signatures --
    def sig_truth(self, name, claimed, commit, mode, kwonly=()):
        """mode 'def': the claimed positional names must be the first positional parameters of some
        definition (in order), and claimed keyword-only names must be parameters of it.
        mode 'call': claimed names are all parameter names; only the order is judged.
        Returns 'match' | 'order_wrong' | 'names_differ' | None (not judgeable)."""
        sigs = self.repo.signatures(name, commit)
        if not sigs:
            return None
        lists = [def_params(s) for s in sigs]
        if mode == 'call':
            pools = [l for l in lists if set(claimed) <= set(l)]
            if not pools:
                return None  # arguments are not parameter names: not a signature statement
            for l in pools:
                if [p for p in l if p in claimed] == claimed:
                    return 'match'
            return 'order_wrong'
        for s, l in zip(sigs, lists):
            # claimed positional names appear in this order (optional ones may be left out)
            it = iter(l + s['kwonly'])
            if all(c in it for c in claimed) and set(kwonly) <= set(s['kwonly']) | set(l):
                return 'match'
        for s, l in zip(sigs, lists):
            if len(claimed) >= 2 and set(claimed) <= set(l + s['kwonly']):
                return 'order_wrong'  # right names, wrong order
        return 'names_differ'

    def kwarg_truth(self, name, kw, commit):
        """'valid' | 'invalid' | None (no definition, or it takes **kwargs)."""
        sigs = self.repo.signatures(name, commit)
        if not sigs or any(s['varkw'] for s in sigs):
            return None
        return 'valid' if any(kw in s['pos'] or kw in s['kwonly'] for s in sigs) else 'invalid'

    def kwarg_other_versions(self, name, kw, commit):
        return self._other_versions(lambda c: self.kwarg_truth(name, kw, c) == 'valid', commit)

    # -- Go packages --
    def go_dirs(self, commit):
        if commit not in self._go_dirs:
            self._go_dirs[commit] = {p.rsplit('/', 1)[0].rsplit('/', 1)[-1] for p in self.repo.source_blobs(commit)
                                     if p.endswith('.go') and '/' in p}
        return self._go_dirs[commit]


# ----------------------------------------------------------------------------------------------
# 4. One trajectory
# ----------------------------------------------------------------------------------------------

IDENT_RE = re.compile(r'[A-Za-z_$][\w$]*')


def sentence_around(text, needle, width=200):
    i = text.find(needle)
    if i < 0:
        return ''
    a = max(text.rfind('.', 0, i), text.rfind('\n', 0, i), i - width)
    b = min([x for x in (text.find('.\n', i), text.find('. ', i + len(needle)), text.find('\n\n', i), i + width)
             if x >= 0])
    return text[a + 1:b + 1].strip()


def analyze(meta, task, steps, patch, view):
    """Return (fact rows, trajectory summary) for one trajectory."""
    commit, lang = meta['base_commit'], view.lang
    repo = view.repo
    py_roots = view.py_roots(commit) if lang == 'py' else set()
    vocab = view.vocab

    # everything the agent writes into files, and every name it defines in its own code
    written, own_names, own_names_cmd, segs_by_step = set(), set(), set(), {}
    for st in steps:
        segs = []
        for cmd in st['commands']:
            for w in WRITE_RE.findall(cmd):
                written.add(norm_path(w))
            for target, code in code_segments(cmd):
                segs.append(('command', target, code))
                if target and ('/' in target or '.' in target):
                    written.add(norm_path(target))
                for a, b in PY_WRITE_RE.findall(code):
                    written.add(norm_path(a or b))
        # (fenced code inside the agent's reasoning/text is a draft or plan, not code it runs: skipped)
        segs_by_step[st['step']] = segs
        for where, _, code in segs:
            names = agent_defined(code)
            own_names |= names
            if where == 'command':
                own_names_cmd |= names

    # names the final patch defines (new functions/methods/constants the agent adds to the repo)
    added = '\n'.join(l[1:] for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
    patch_defs = agent_defined(added)
    own_names |= patch_defs
    own_names_cmd |= patch_defs

    seen_text = task
    seen_words = set(IDENT_RE.findall(task))
    first = {}  # fact key -> row

    def add(ftype, fact, st, field, seen, extra=None, ctx_text='', needle=None):
        key = (ftype, fact)
        if key in first:
            return
        row = {'fact_type': ftype, 'fact': fact, 'step': st['step'], 'field': field, 'seen': seen,
               'nbs': seen == 'unseen', 'in_fake_output': bool(st['fake'] and field == 'visible')}
        if extra:
            row.update(extra)
        if ctx_text and field in ('reasoning', 'visible'):
            nd = needle or fact.split('(')[0].rsplit('.', 1)[-1]
            row['_ctx'] = sentence_around(ctx_text, nd)
            i = ctx_text.find(nd)
            # the agent talking about something it will create is a plan, not a claim about the repo
            row['planned'] = bool(i >= 0 and PLAN_RE.search(ctx_text[max(0, i - 100):i + len(nd) + 60]))
        first[key] = row

    def seen_dotted(d):
        if d in seen_text or d.replace('.', '/') in seen_text:
            return 'seen'
        return 'seen_name' if d.rsplit('.', 1)[-1] in seen_words else 'unseen'

    def def_seen(name):
        pats = {'py': [f'def {name}('], 'go': [f'func {name}(', f') {name}('], 'rs': [f'fn {name}('],
                'ts': [f'function {name}(', f' {name}('], 'js': [f'function {name}(', f' {name}(']}[lang]
        return any(p in seen_text for p in pats)

    for st in steps:
        fields = [('reasoning', st['reasoning']), ('visible', st['visible'])] + \
                 [('command', c) for c in st['commands']]
        for field, text in fields:
            if not text:
                continue
            # --- file paths ---
            for pm in PATH_RE.finditer(text):
                raw = pm.group(1)
                if raw.startswith(('./', '../')) and re.search(r'(?:\bfrom|\bimport|require\(|import\()\s*[\'"]$',
                                                               text[max(0, pm.start() - 30):pm.start()]):
                    continue  # './x.js' in an import statement is relative to that file, not the repo root
                p = norm_path(raw)
                if p.startswith(('/', '../')) or '/../' in p or '/' not in p or set(p.split('/')[:-1]) & PATH_SKIP \
                        or '.egg-info/' in p:
                    continue  # outside the repo, or a third-party / generated file
                name = p.rsplit('/', 1)[-1]
                mod = p.rsplit('.', 1)[0].replace('/', '.').split('.')
                dotted = '.'.join(mod[-2:]) if len(mod) >= 2 else None
                stem = '/'.join(p.rsplit('.', 1)[0].split('/')[-2:])  # utils/regex for src/utils/regex.ts
                seen = ('seen' if p in seen_text else 'seen_name' if name in seen_text
                        else 'derived' if (dotted and dotted in seen_text) or ('/' in stem and stem in seen_text)
                        else 'unseen')
                add('path', p, st, field, seen, ctx_text=text, needle=raw)
            # --- dotted Python names ---
            if lang == 'py':
                for d in DOTTED_RE.findall(text):
                    if d.endswith('.py'):
                        d = d[:-3]
                    if '.' in d and d.split('.')[0] in py_roots:
                        add('dotted', d, st, field, seen_dotted(d), {'how': 'dotted'}, ctx_text=text)
            # --- signatures stated in the agent's own text ---
            if field in ('reasoning', 'visible'):
                for m in SIG_TEXT_RES[lang].finditer(text):
                    name, raw = m.group(1), m.group(2)
                    pos, kwonly = claimed_params(raw, lang)
                    before = text[max(0, m.start() - 400):m.start()]
                    if name.startswith('test') or re.search(r'<<-?\s*[\'"]?EOF|cat\s+>', before):
                        continue  # the agent drafting its own test or a file it will write, not a claim
                    # skipped: library-like names (the standard library or other repos define them) and
                    # functions the agent itself (re)defines in code it runs (a plan, not a claim)
                    if (pos or kwonly) and name not in PY_KEYWORDS and name not in vocab and name not in view.cross \
                            and name not in own_names_cmd:  # (own_names_cmd includes names the final patch defines)
                        fact = f'{name}({", ".join(pos + (["*"] + kwonly if kwonly else []))})'
                        claim_seen = re.search(re.escape(name) + r'\(\s*' + r'[^)]*?'.join(map(re.escape, pos + kwonly)),
                                               seen_text)
                        add('sig_def', fact, st, field, 'seen' if def_seen(name) or claim_seen else 'unseen',
                            {'func': name, 'params': pos, 'kwonly': kwonly}, ctx_text=text)
                for callee, args in CALL_SHAPE_RE.findall(text):
                    name = callee.rsplit('.', 1)[-1]
                    claimed = [a.strip() for a in args.split(',')]
                    if name in PY_KEYWORDS or name in OTHER_KEYWORDS or name in vocab or name in view.cross \
                            or len(set(claimed)) != len(claimed):
                        continue
                    fact = f'{name}({", ".join(claimed)})'
                    claim_seen = re.search(re.escape(name) + r'\(\s*' + r'\s*,\s*'.join(map(re.escape, claimed)),
                                           seen_text)
                    add('sig_call', fact, st, field, 'seen' if def_seen(name) or claim_seen else 'unseen',
                        {'func': name, 'params': claimed}, ctx_text=text)
        # --- code the agent writes (commands, fenced code in its text) ---
        for field, target, code in segs_by_step[st['step']]:
            extract_code_facts(code, target, field, st, lang, py_roots, view, commit, own_names, own_names_cmd,
                               vocab, seen_words, seen_text, add, def_seen)
        # tool output becomes visible from the next step on
        seen_text += '\n' + st['obs']
        seen_words.update(IDENT_RE.findall(st['obs']))

    rows, kept = [], set()
    for (ftype, fact), row in sorted(first.items(), key=lambda kv: kv[1]['step']):
        judge_fact(row, view, commit, written, lang, own_names)
        if row.get('truth') is None or (row['fact_type'], row['fact']) in kept:
            continue  # unjudgeable, or the same judged fact as an earlier mention (dotted prefixes)
        kept.add((row['fact_type'], row['fact']))
        rows.append(row)
    self_correction(rows, steps, patch)
    for row in rows:
        ctx = row.pop('_ctx', '')
        row['hedged'] = bool(HEDGE.search(ctx)) if ctx else None
        if meta['keep_context'] and row['nbs']:
            row['context'] = ctx[:200]
        # per-trajectory facts (task, repo, resolved, ...) live in the trajectory table; join on 'trial'
        row.update({'model': meta['model'], 'trial': meta['trial']})
    summary = {k: meta[k] for k in ('bench', 'source', 'model', 'trial', 'task', 'repo', 'lang', 'resolved',
                                    'reasoning_visible', 'weights')}
    summary['n_steps'] = len(steps)
    return rows, summary


def extract_code_facts(code, target, field, st, lang, py_roots, view, commit, own_names, own_names_cmd, vocab,
                       seen_words, seen_text, add, def_seen):
    """Symbols used in a piece of code the agent wrote."""
    repo = view.repo
    seg_lang = lang
    ext = os.path.splitext(target)[1] if target else ''
    if target.startswith('python') or ext in ('.py', '.pyi') or (lang == 'py' and not ext):
        seg_lang = 'py'
    if lang == 'py' and seg_lang == 'py':
        aliases = {}
        ext_names = set(PY_EXT_ALIASES)  # names bound to other libraries (np, pd, docutils' nodes, ...)
        for mod, name, alias in py_imports(code):
            if mod.split('.')[0] not in py_roots:
                ext_names.add(alias)
                continue
            dotted = f'{mod}.{name}' if name else mod
            add('dotted', dotted, st, field, 'seen' if dotted in seen_text or
                (name or mod.rsplit('.', 1)[-1]) in seen_words else 'unseen',
                {'how': 'import', 'in_code': True, 'claimed_module': name is None})
            aliases[alias] = dotted if name else (mod if alias != mod.split('.')[0] else mod.split('.')[0])
        for alias, dotted in aliases.items():  # x.attr where x is an alias of a repo module
            for attr in set(re.findall(r'(?<![\w.])' + re.escape(alias) + r'\.([A-Za-z_]\w*)', code)):
                full = f'{dotted}.{attr}'
                if alias == dotted:
                    continue  # plain 'import django' then django.x.y: the dotted pass covers it
                add('dotted', full, st, field, 'seen' if full in seen_text or attr in seen_words else 'unseen',
                    {'how': 'alias', 'in_code': True})
        for d in DOTTED_RE.findall(code):
            if '.' in d and d.split('.')[0] in py_roots and d.split('.')[0] not in ext_names:
                add('dotted', d, st, field, 'seen' if d in seen_text or d.rsplit('.', 1)[-1] in seen_words
                    else 'unseen', {'how': 'dotted', 'in_code': True})
        # calls: name( and .name(
        for m in CALL_RE.finditer(code):
            dot, name = is_member(code, m), m.group(1)
            prev = code[max(0, m.start() - 40):m.start()]
            if re.search(r'\b(?:def|class)[ \t]*$', prev) or name in aliases or name in ext_names:
                continue
            recv = re.search(r'([A-Za-z_][\w.]*)\.$', prev) if dot else None
            if recv and recv.group(1).split('.')[0] in ext_names:
                continue  # np.something(, nodes.paragraph(, importlib.util.x( ... : another library's name
            if name in PY_KEYWORDS or name.startswith('__') or len(name) < 3 or name in vocab:
                continue
            if name in own_names and not repo.defined(name, commit):
                continue
            how = 'call' if not dot else ('self_method' if recv and recv.group(1).split('.')[0] in ('self', 'cls')
                                          else 'method')
            add('code_name', name, st, field, 'seen' if name in seen_words else 'unseen', {'how': how, 'in_code': True})
        # keyword arguments passed to repo functions
        for callee, kws in find_calls_with_kwargs(code):
            if callee in vocab or callee in view.cross or callee in PY_KEYWORDS or callee in own_names_cmd or \
                    callee in ext_names or (callee in own_names and not repo.defined(callee, commit)):
                continue  # library functions, and functions the agent itself (re)defines
            for kw in kws:
                fact = f'{callee}({kw}=)'
                add('kwarg', fact, st, field, 'seen' if kw in seen_words else 'unseen',
                    {'func': callee, 'kw': kw, 'in_code': True})
        return
    if lang == 'py':
        return  # non-Python code inside a Python repo task: skip
    # ---- Go / TypeScript / JavaScript / Rust code ----
    if ext in DOC_EXT:
        return  # documentation, config or data the agent writes: example code there is not a claim
    if seg_lang == 'py' or target.startswith('python') or ext in ('.py', '.pyi'):
        code = python_strings(code)  # an edit script: the repo code it writes is inside its strings
    code = strip_noise(code)
    repo_ns = set()   # local names that refer to repo modules/packages in this code
    external = set()  # names imported from other packages (not repo facts)
    if lang in ('ts', 'js'):
        for clause, spec in TS_IMPORT_RE.findall(code) + REQUIRE_RE.findall(code):
            if not (spec.startswith('.') or view.repo_pkg(spec)):
                external.update(re.findall(r'[A-Za-z_$][\w$]*', clause))
                continue
            for part in re.split(r'[{},]', clause):
                part = part.strip()
                if not part:
                    continue
                if part.startswith('* as '):
                    repo_ns.add(part[5:].strip())
                    continue
                bits = part.replace('type ', '').split(' as ')
                name = bits[0].strip()
                repo_ns.add(bits[-1].strip())
                if re.fullmatch(r'[A-Za-z_$][\w$]*', name) and not (name in own_names and not repo.defined(name, commit)):
                    add('code_name', name, st, field, 'seen' if name in seen_words else 'unseen',
                        {'how': 'import', 'in_code': True})
    elif lang == 'go':
        for alias, path in GO_IMPORT_RE.findall(code):
            if view.go_module and (path == view.go_module or path.startswith(view.go_module + '/')):
                repo_ns.add(alias or path.rsplit('/', 1)[-1])
    elif lang == 'rs':
        for use in RS_USE_RE.findall(code):
            for name in re.findall(r'[A-Za-z_]\w*', use.split('::', 1)[1]):
                if name not in ('self', 'super', 'crate') and name not in OTHER_KEYWORDS and \
                        not (name in own_names and not repo.defined(name, commit)):
                    add('code_name', name, st, field, 'seen' if name in seen_words else 'unseen',
                        {'how': 'use', 'in_code': True})
    # qualified calls into the repo: ns.Name( / Type::name(
    for recv, sep, name in QUAL_CALL_RE.findall(code):
        repo_recv = recv in repo_ns or (lang == 'go' and recv in view.go_dirs(commit) and recv not in GO_STD_PKGS) or \
            (lang in ('ts', 'js', 'rs') and recv[:1].isupper() and repo.defined(recv, commit) and recv not in OTHER_KEYWORDS)
        if repo_recv and name not in OTHER_KEYWORDS and not (name in own_names and not repo.defined(name, commit)):
            add('code_name', name, st, field, 'seen' if name in seen_words else 'unseen',
                {'how': 'qualified', 'in_code': True})
    # unqualified calls: name(
    for m in CALL_RE.finditer(code):
        dot, name = is_member(code, m), m.group(1)
        if dot or name in OTHER_KEYWORDS or len(name) < 3 or name in vocab or name in external or name in PY_BUILTINS:
            continue
        if '!' in m.group(0):
            continue  # Rust macro
        prev = code[max(0, m.start() - 12):m.start()]
        if re.search(r'\b(?:function\*?|func|fn|def|class|new)[ \t]*$', prev) or re.search(r'\)[ \t]*$', prev):
            continue
        if name in own_names and not repo.defined(name, commit):
            continue
        add('code_name', name, st, field, 'seen' if name in seen_words else 'unseen', {'how': 'call', 'in_code': True})


def judge_fact(row, view, commit, written, lang, own_names=frozenset()):
    """Set row['truth'] (and 'wrong') for one fact; truth None drops the row."""
    ft, fact = row['fact_type'], row['fact']
    row['truth'] = None
    if ft == 'path':
        truth, years = view.path_truth(fact, commit)
        if truth != 'real' and (fact in written or any(w.endswith('/' + fact) or w == fact for w in written)):
            return  # the agent made this file itself
        row['truth'], row['years'] = truth, years
    elif ft == 'dotted':
        parts = fact.split('.')
        dunder = [i for i, x in enumerate(parts) if i and x.startswith('__') and x.endswith('__')]
        if dunder:
            fact = '.'.join(parts[:dunder[0]])  # x.__file__, X.__dict__: runtime attributes, not checked
            if '.' not in fact:
                return
        fact = view.judged_prefix(fact, commit)
        row['fact'] = fact
        truth, kind = view.dotted_truth(fact, commit)
        if truth != 'real' and fact.rsplit('.', 1)[-1] in own_names:
            return  # a name the agent itself adds to the repo (imported before/after writing it)
        if row.pop('claimed_module', False):
            kind = 'module'  # 'import a.b.c' claims a module
        row['dotted_kind'] = kind if truth != 'never' else 'unresolved'
        if truth == 'unknown':
            return
        row['truth'] = truth
    elif ft == 'code_name':
        row['truth'] = view.name_truth(fact, commit)
        if row['truth'] == 'never' and (fact in view.cross or row.get('how') == 'method'):
            # not in this repo, but other projects define it, or it is a method called on an object
            # of unknown type (x.name(): x may be another library's object): not judged
            row['truth'] = 'external'
    elif ft in ('sig_def', 'sig_call'):
        name, claimed, kwonly = row.pop('func'), row.pop('params'), row.pop('kwonly', [])
        if not view.repo.defined(name, commit):
            return  # the name itself is checked as a code_name/symbol fact
        mode = 'def' if ft == 'sig_def' else 'call'
        res = view.sig_truth(name, claimed, commit, mode, kwonly)
        if res is None:
            return
        row['truth'] = res
        if res != 'match':
            row['other_version_match'] = view._other_versions(
                lambda c: view.sig_truth(name, claimed, c, mode, kwonly) == 'match', commit)
    elif ft == 'kwarg':
        name, kw = row.pop('func'), row.pop('kw')
        if not view.repo.defined(name, commit):
            return
        res = view.kwarg_truth(name, kw, commit)
        if res is None:
            return
        row['truth'] = res
        if res == 'invalid':
            row['other_version_match'] = view.kwarg_other_versions(name, kw, commit)
    if row['truth'] is not None and row.get('planned'):
        row['truth_if_claim'] = row['truth']
        row['truth'] = 'planned'  # judged neither right nor wrong (both real and non-real plans are set aside)
    if row['truth'] is not None:
        row['wrong'] = (None if row['truth'] in ('external', 'planned', 'outside')
                        else row['truth'] not in ('real', 'match', 'valid'))


def fact_key_terms(row):
    """(word to look for in tool output, full string) for self-correction checks."""
    ft, fact = row['fact_type'], row['fact']
    if ft == 'path':
        return fact.rsplit('/', 1)[-1], fact
    if ft in ('module', 'symbol'):
        return fact.rsplit('.', 1)[-1], fact
    if ft == 'kwarg':
        return fact.split('(', 1)[1].rstrip('=)'), fact.split('(', 1)[0]
    if ft in ('sig_def', 'sig_call'):
        return fact.split('(', 1)[0], fact.split('(', 1)[0]
    return fact, fact


def self_correction(rows, steps, patch):
    """For each wrong NBS fact: first contradicting tool output, and whether the agent stops using it."""
    added = '\n'.join(l[1:] for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
    for row in rows:
        if not (row['nbs'] and row['wrong']):
            continue
        word, full = fact_key_terms(row)
        word_re = re.compile(r'(?<![\w$])' + re.escape(word) + r'(?![\w$])')
        contradicted = None
        is_path = row['fact_type'] == 'path'
        for st in steps[row['step'] - 1:]:
            hit = any(ERROR_RE.search(line) for line in st['obs'].split('\n')
                      if (word in line if is_path else word in line and word_re.search(line)))
            if not hit and any(SEARCH_CMD_RE.search(c) and word in c for c in st['commands']):
                body = re.sub(r'</?(returncode|output)>|^\s*-?\d+\s*$', '', st['obs'], flags=re.M).strip()
                hit = (not body) or (st['rc'] == 1 and len(body) < 80)  # the search found nothing
            if hit:
                contradicted = st['step']
                break
        row['contradicted_step'] = contradicted
        row['steps_to_contradiction'] = (contradicted - row['step']) if contradicted else None
        later_cmds = [c for st in steps[(contradicted or row['step']):] for c in st['commands']]
        reused = any((full in c) if row['fact_type'] == 'path' else bool(word_re.search(c)) for c in later_cmds)
        row['changed_course'] = bool(contradicted) and not reused
        row['in_final_patch'] = bool(word_re.search(added)) if row['fact_type'] != 'path' else (full in patch)


# ----------------------------------------------------------------------------------------------
# 5. Datasets: which trajectories, which repo, which commits
# ----------------------------------------------------------------------------------------------

def swebv_jobs(limit=None):
    import pyarrow.parquet as pq
    t = pq.read_table(SWEBV_PARQUET, columns=['instance_id', 'repo', 'base_commit']).to_pandas()
    tasks = {r.instance_id: (r.repo, r.base_commit) for r in t.itertuples()}
    jobs = []
    for model, run in SWEBV_RUNS.items():
        rd = os.path.join(DATA, 'raw', run)
        for iid in sorted(os.listdir(os.path.join(rd, 'trajs'))):
            f = os.path.join(rd, 'trajs', iid, iid + '.traj.json')
            if iid not in tasks or not os.path.exists(f):
                continue
            rep = os.path.join(rd, 'logs', iid, 'report.json')
            resolved = json.load(open(rep)).get('resolved') if os.path.exists(rep) else None
            repo, base = tasks[iid]
            jobs.append({'bench': 'swebv', 'source': 'swebv', 'model': model, 'trial': f'{model}/{iid}', 'task': iid,
                         'repo': repo, 'repo_dir': os.path.join(DATA, 'repos', repo.replace('/', '_')),
                         'base_commit': base, 'resolved': resolved, 'path': f, 'weights': weights_of(model),
                         'keep_context': True})
    if limit:
        jobs = jobs[::max(1, len(jobs) // limit)][:limit]
    return jobs


def deepswe_jobs(limit=None):
    base = {}
    with open(os.path.join(DATA, 'deepswe', 'derived', 'base_commit_dates.csv')) as f:
        for r in csv.DictReader(f):
            base[r['task_id']] = (r['repository'], r['full_sha'] or r['base_commit'])
    with open(os.path.join(DATA, 'deepswe', 'sample_manifest.csv')) as f:
        manifest = list(csv.DictReader(f))
    with open(os.path.join(DATA, 'deepswe', 'meta', 'v1.1', 'tasks.json')) as f:
        task_lang = {t['id']: t['language'] for t in json.load(f)['rows']}
    # SWE-bench Pro tasks have no language field here: use the language of the files that the
    # sample's patches change (majority over all trials of the task)
    pro_lang = defaultdict(Counter)
    from rq1_repo_index import source_lang
    for m in manifest:
        if m['source'] != 'deep-swe':
            pf = os.path.join(DATA, 'deepswe', 'trials', m['version'], m['trial_name'], 'model.patch')
            if os.path.exists(pf):
                for path in re.findall(r'^\+\+\+ b/(\S+)', open(pf, errors='replace').read(), re.M):
                    if source_lang(path):
                        pro_lang[m['task_name']][source_lang(path)] += 1
    jobs = []
    for m in manifest:
        if m['source'] == 'deep-swe':
            repo, sha = base[m['task_name']]
            lang = {'python': 'py', 'go': 'go', 'rust': 'rs', 'typescript': 'ts', 'javascript': 'js'}.get(
                task_lang.get(m['task_name'], ''), None)
        else:
            hit = re.match(r'instance_(.+?)__(.+?)-([0-9a-f]{40})', m['task_name'])
            repo, sha = f'{hit.group(1)}/{hit.group(2)}', hit.group(3)
            c = pro_lang.get(m['task_name'])
            lang = c.most_common(1)[0][0] if c else None
        folder = os.path.join(DATA, 'deepswe', 'trials', m['version'], m['trial_name'])
        if not os.path.exists(os.path.join(folder, 'trajectory.json')):
            continue
        jobs.append({'bench': 'deepswe', 'source': m['source'], 'model': m['model'],
                     'effort': m['reasoning_effort'], 'version': m['version'], 'trial': m['trial_name'],
                     'task': m['task_name'], 'repo': repo,
                     'repo_dir': os.path.join(DATA, 'repos_deepswe', repo.replace('/', '_')),
                     'base_commit': sha, 'resolved': m['passed'] == 'True', 'path': folder,
                     'weights': weights_of(m['model']), 'keep_context': False, 'lang_hint': lang})
    if limit:
        jobs = jobs[::max(1, len(jobs) // limit)][:limit]
    return jobs


def repo_lang(repo, commit):
    """Main language of a repo at a commit, by counting source files."""
    from rq1_repo_index import source_lang
    c = Counter(source_lang(p) for p in repo.paths(commit))
    c.pop(None, None)
    if not c:
        return 'py'
    # JavaScript and TypeScript share one parser
    ts = c.get('ts', 0)
    return 'ts' if ts and ts >= c.get('js', 0) * 0.3 and c.most_common(1)[0][0] in ('ts', 'js') else c.most_common(1)[0][0]


def read_go_module(repo, commit):
    """Module path from go.mod at the commit ('' if unavailable)."""
    files = repo.tree(commit)
    if 'go.mod' not in files:
        return ''
    oid = files['go.mod'][0]
    text = repo.read_blobs([oid]).get(oid)
    if text is None and repo.allow_fetch:
        import subprocess
        subprocess.run(['git', '-C', repo.path, '-c', 'fetch.negotiationAlgorithm=noop', '-c', 'gc.auto=0', 'fetch',
                        'origin', '--no-tags', '--no-write-fetch-head', '--filter=blob:none', '--stdin'],
                       input=oid + '\n', capture_output=True, text=True)
        text = repo.read_blobs([oid]).get(oid)
    m = re.search(r'^module[ \t]+(\S+)', text or '', re.M)
    return m.group(1) if m else ''


def repo_groups(jobs):
    groups = defaultdict(list)
    for j in jobs:
        groups[j['repo']].append(j)
    return groups


def index_repos(bench):
    """Fetch the source files we need (one batch per repo) and parse them; write a summary per repo."""
    jobs = swebv_jobs() if bench == 'swebv' else deepswe_jobs()
    years = (2, 5) if bench == 'swebv' else (1, 2, 4)
    os.makedirs(CACHE, exist_ok=True)
    log = []
    for repo_name, js in sorted(repo_groups(jobs).items()):
        t0 = time.time()
        rd = js[0]['repo_dir']
        if not os.path.isdir(rd):
            log.append({'repo': repo_name, 'status': 'no clone'})
            print(json.dumps(log[-1]), flush=True)
            continue
        old = rd.rstrip('/') + '.adds.json'
        repo = Repo(rd, CACHE, allow_fetch=True, old_adds_cache=old if os.path.exists(old) else None)
        bases = sorted({j['base_commit'] for j in js if repo.has_commit(j['base_commit'])})
        snaps = repo.snapshots(bases, years)
        status = repo.fetch_missing(snaps)
        for c in snaps:
            repo.index_commit(c)
        repo.path_adds()
        hints = Counter(j.get('lang_hint') for j in js if j.get('lang_hint'))
        lang = hints.most_common(1)[0][0] if hints else (repo_lang(repo, bases[0]) if bases else None)
        info = {'repo': repo_name, 'bench': bench, 'lang': lang, 'bases': bases, 'snapshots': snaps,
                'snapshot_times': [repo.commit_time(c) for c in snaps],
                'coverage': {c: round(repo.coverage(c) or 0, 4) for c in snaps}, 'fetch': status,
                'missing_bases': sorted({j['base_commit'] for j in js} - set(bases)),
                'go_module': read_go_module(repo, bases[0]) if lang == 'go' and bases else '',
                'seconds': round(time.time() - t0)}
        with open(os.path.join(CACHE, repo.name + f'.{bench}.json'), 'w') as f:
            json.dump(info, f)
        log.append({k: info[k] for k in ('repo', 'lang', 'fetch', 'seconds')} |
                   {'n_bases': len(bases), 'n_snapshots': len(snaps), 'lang_hints': dict(hints),
                    'min_base_coverage': min(info['coverage'][c] for c in bases) if bases else None})
        print(json.dumps(log[-1]), flush=True)
    with open(os.path.join(CACHE, f'index_{bench}.log.json'), 'w') as f:
        json.dump(log, f, indent=1)


# ----------------------------------------------------------------------------------------------
# 6. Library-name vocabulary (names that are not repo-specific)
# ----------------------------------------------------------------------------------------------

def build_vocab():
    """Python: builtins, builtin types, standard library, numpy, pandas, pytest.
    Every language: names defined (at the newest commit) by the OTHER studied repos of that
    language, computed per repo at run time from the parse caches."""
    import builtins
    import importlib
    import warnings
    names = set(dir(builtins))
    for t in (str, bytes, list, dict, set, frozenset, tuple, int, float, complex, object, type, BaseException):
        names.update(dir(t))
    skip = {'antigravity', 'this', 'idlelib', 'tkinter', 'turtle', 'turtledemo', 'pydoc_data', 'ensurepip', 'venv',
            'msilib', 'winreg', 'winsound', 'lib2to3', 'test', 'xxsubtype', '__phello__', 'pty', 'tty', 'curses'}
    mods = [m for m in sorted(sys.stdlib_module_names) if not m.startswith('_') and m not in skip]
    mods += ['numpy', 'numpy.linalg', 'numpy.random', 'numpy.testing', 'numpy.ma', 'pandas', 'pandas.api.types',
             'pandas.testing', 'pytest', 'unittest.mock', 'os.path', 'collections.abc', 'importlib.metadata',
             'xml.etree.ElementTree', 'urllib.parse', 'concurrent.futures', 'email.message', 'logging.handlers',
             'importlib.util', 'importlib.machinery', 'importlib.resources', 'urllib.request', 'urllib.error',
             'http.client', 'http.server', 'json.decoder', 'multiprocessing.pool', 'xml.dom.minidom', 'email.utils',
             'email.mime.text', 'unittest.case', 'asyncio.subprocess', 'ctypes.util', 'wsgiref.simple_server',
             'logging.config', 'os', 'sys', 'typing', 'dataclasses', 'pathlib']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for m in mods:
            try:
                mod = importlib.import_module(m)
            except BaseException:
                continue
            for n in dir(mod):
                names.add(n)
                obj = getattr(mod, n, None)
                if isinstance(obj, type):
                    names.update(dir(obj))
    names = sorted(n for n in names if re.fullmatch(r'[A-Za-z_]\w*', n))
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, 'py_vocab.json'), 'w') as f:
        json.dump(names, f)
    print(len(names), 'Python library names ->', os.path.join(CACHE, 'py_vocab.json'))


def head_names(info):
    """Names defined at the newest indexed commit of a repo (for the cross-repo vocabulary)."""
    path = os.path.join(CACHE, os.path.basename(info['_dir']) + '.parsed.jsonl')
    repo = Repo(info['_dir'], CACHE, allow_fetch=False)
    head = info['snapshots'][-1]
    blobs = set(repo.source_blobs(head).values())
    names = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d['oid'] in blobs:
                    names.update(d['defs'])
    return names


def load_repo_infos():
    infos = {}
    for f in sorted(os.listdir(CACHE)):
        if f.endswith(('.swebv.json', '.deepswe.json')):
            with open(os.path.join(CACHE, f)) as fh:
                info = json.load(fh)
            info['_dir'] = os.path.join(DATA, 'repos' if info['bench'] == 'swebv' else 'repos_deepswe',
                                        f.rsplit('.', 2)[0])
            infos[(info['bench'], info['repo'])] = info
    return infos


def cross_repo_vocab():
    """Write {repo: {'lang', 'names'}}: names defined at the newest commit of every studied repo.
    At run time, a repo's "cross vocabulary" is the union over the OTHER repos of its language."""
    out = {}
    for key, info in load_repo_infos().items():
        if info.get('snapshots') and info.get('lang'):
            out[info['repo']] = {'lang': info['lang'] if info['lang'] != 'js' else 'ts',
                                 'names': sorted(head_names(info))}
    with open(os.path.join(CACHE, 'cross_vocab.json'), 'w') as f:
        json.dump(out, f)
    print(len(out), 'repos ->', os.path.join(CACHE, 'cross_vocab.json'))


# ----------------------------------------------------------------------------------------------
# 7. Running
# ----------------------------------------------------------------------------------------------

_VOCAB = {}


def run_group(args):
    """Worker: all trajectories of one repo."""
    bench, repo_name, jobs = args
    infos_path = os.path.join(CACHE, os.path.basename(jobs[0]['repo_dir']) + f'.{bench}.json')
    if not os.path.exists(infos_path):
        return [], [{'trial': j['trial'], 'skipped': 'repo not indexed'} for j in jobs]
    with open(infos_path) as f:
        info = json.load(f)
    old = jobs[0]['repo_dir'].rstrip('/') + '.adds.json'
    repo = Repo(jobs[0]['repo_dir'], CACHE, allow_fetch=False, old_adds_cache=old if os.path.exists(old) else None,
                readonly=True)
    repo.index_commit(info['snapshots'][0])  # loads the parse cache
    repo.path_adds()
    lang = info['lang'] if info['lang'] != 'js' else 'ts'
    with open(os.path.join(CACHE, 'py_vocab.json')) as f:
        vocab = set(json.load(f)) if lang == 'py' else set()   # library names: never repo facts
    cross = set()  # names the other studied repos of this language define
    with open(os.path.join(CACHE, 'cross_vocab.json')) as f:
        for other, d in json.load(f).items():
            if other != repo_name and d['lang'] == lang:
                cross.update(d['names'])
    view = RepoView(repo, info['snapshots'], lang, vocab, info.get('go_module') or None)
    view.cross = cross
    view.repo_pkg = make_repo_pkg_test(repo, info['bases'])
    rows, summaries = [], []
    for j in jobs:
        if j['base_commit'] not in info['bases']:
            summaries.append({'trial': j['trial'], 'skipped': 'base commit missing'})
            continue
        try:
            if bench == 'swebv':
                task, steps, patch = read_minisweagent(j['path'])
                tokens = None
            else:
                task, steps, patch, tokens = read_atif(j['path'])
        except Exception as err:
            summaries.append({'trial': j['trial'], 'skipped': f'unreadable: {err}'[:200]})
            continue
        has_reasoning = any(s['reasoning'].strip() for s in steps)
        j = dict(j, lang=lang, reasoning_visible=has_reasoning)
        r, s = analyze(j, task, steps, patch, view)
        s['reasoning_chars'] = sum(len(st['reasoning']) for st in steps)
        s['reasoning_tokens'] = tokens
        s['effort'] = j.get('effort')
        s['base_coverage'] = info['coverage'].get(j['base_commit'])
        rows.extend(r)
        summaries.append(s)
    return rows, summaries


def make_repo_pkg_test(repo, bases):
    """For TS/JS: is an import specifier one of the repo's own package names (from package.json files)?"""
    names = set()
    if bases:
        files = repo.tree(bases[0])
        pj = [oid for p, (oid, _) in files.items() if p.endswith('package.json') and 'node_modules' not in p
              and p.count('/') <= 3]
        for text in repo.read_blobs(pj[:300]).values():
            m = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            if m:
                names.add(m.group(1))

    def test(spec):
        return any(spec == n or spec.startswith(n + '/') for n in names)
    return test


def run(bench, limit=None, workers=4):
    jobs = swebv_jobs(limit) if bench == 'swebv' else deepswe_jobs(limit)
    groups = repo_groups(jobs)
    # split big repos into chunks so workers stay busy
    tasks = []
    for repo_name, js in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        for i in range(0, len(js), 120):
            tasks.append((bench, repo_name, js[i:i + 120]))
    os.makedirs(OUT, exist_ok=True)
    out_rows = os.path.join(OUT, f'facts_{bench}.jsonl')
    out_traj = os.path.join(CACHE, f'trajectories_{bench}.jsonl')
    t0 = time.time()
    n_rows = 0
    with open(out_rows, 'w') as fr, open(out_traj, 'w') as ft, Pool(workers) as pool:
        for k, (rows, summaries) in enumerate(pool.imap_unordered(run_group, tasks)):
            for r in rows:
                fr.write(json.dumps(r) + '\n')
            for s in summaries:
                ft.write(json.dumps(s) + '\n')
            n_rows += len(rows)
            print(f'  {k + 1}/{len(tasks)} chunks, {n_rows} fact rows, {time.time() - t0:.0f}s', flush=True)
    print('wrote', out_rows, 'and', out_traj)


def main():
    cmd = sys.argv[1]
    opts = dict(zip(sys.argv[3::2], sys.argv[4::2])) if len(sys.argv) > 3 else {}
    if cmd == 'index':
        index_repos(sys.argv[2])
    elif cmd == 'vocab':
        build_vocab()
        cross_repo_vocab()
    elif cmd == 'run':
        run(sys.argv[2], int(opts['--limit']) if '--limit' in opts else None, int(opts.get('--workers', 4)))
    elif cmd == 'summarize':
        import rq1_summarize
        rq1_summarize.main(OUT, CACHE)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
