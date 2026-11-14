#!/usr/bin/env python3
"""Memorization probes with small local models (experiment E5 in notes/PLAN.md).

Question: when a coding agent relies on a repo fact it never read (argument order, a default value, a keyword
name, the module a name is imported from), is that fact the kind of thing language models carry in their
weights? The agents' own models (DeepSeek-V3.2, Kimi K2.5, MiniMax M2.5) are far too big to run here, so we
probe small open models of one family (Qwen2.5 0.5B to 7B, general and coder versions, plus a few extras)
on the same facts. These are NOT the agents studied; they show how such memory grows with size and code data.

Two measures per fact:
 (a) likelihood. Show the model the 40 real lines before the definition (or import line) at the task's base
     commit, then compute log P(line | those lines) for three versions of the line:
       original  the real line
       trapped   the fact changed (two arguments swapped, a default flipped, a keyword renamed, an import moved)
       placebo   an edit that keeps every fact a caller relies on (add **kwargs to the signature, or import one
                 more real name from the same module)
     memory strength = logp(original) - logp(trapped)  (> 0 means the model prefers the real fact)
 (b) closed-book recall. Ask "In Django 3.0 (2019), what is the exact signature of X?" (or "the import line
     for X") with no code shown, and score parameter order, names and defaults (or the module).

Facts probed (groups): pilot (the two pilot traps), agent (outputs/agent_targeted_facts.jsonl, all of the memory
tier plus a sample of the other tiers), trap (traps/catalog/*/*/spec.json, when it exists), and control (for
every fact, one random function or import from the same repo and base commit that no agent relied on, with
the same kind of edit).

Usage (the system python3 has mlx + mlx_lm; -I keeps downloaded files from shadowing modules):
  python3 -I scripts/probe_memory.py build                 # probe items -> data/probes_work/items.jsonl
        (add-only when items.jsonl exists; --catalog-anyway reads traps/catalog before READY; --fresh rebuilds)
  python3 -I scripts/probe_memory.py run MODEL [MODEL...]  # score all items with each model (resumable)
  python3 -I scripts/probe_memory.py report                # outputs/probes/probes.jsonl, summary.json, fig_*.csv
  python3 -I scripts/probe_memory.py check MODEL           # check that cached scoring = plain scoring
MODEL is a short name from MODELS below.
"""
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = ROOT / 'data/repos'
PARQUET = ROOT / 'data/meta/swebv/data/test-00000-of-00001.parquet'
FACTS = ROOT / 'outputs/agent_targeted_facts.jsonl'
CATALOG = ROOT / 'traps/catalog'
WORK = ROOT / 'data/probes_work'          # intermediate files (gitignored data folder)
OUT = ROOT / 'outputs/probes'
CONTEXT_LINES = 40
NO_FETCH = dict(os.environ, GIT_NO_LAZY_FETCH='1')   # never let git download a missing file from GitHub
SAMPLE_PER_TYPE = 6       # agent facts sampled per fact type from the 'unseen' tier and from the 'seen' tier
FACT_TYPES = ('arg_order', 'default', 'keyword', 'import')

# short name -> (Hugging Face id, size in billions of parameters, family)
MODELS = {
    'qwen2.5-0.5b': ('mlx-community/Qwen2.5-0.5B-Instruct-4bit', 0.5, 'general'),
    'qwen2.5-1.5b': ('mlx-community/Qwen2.5-1.5B-Instruct-4bit', 1.5, 'general'),
    'qwen2.5-3b': ('mlx-community/Qwen2.5-3B-Instruct-4bit', 3.0, 'general'),
    'qwen2.5-7b': ('mlx-community/Qwen2.5-7B-Instruct-4bit', 7.0, 'general'),
    'qwen2.5-coder-1.5b': ('mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit', 1.5, 'coder'),
    'qwen2.5-coder-7b': ('mlx-community/Qwen2.5-Coder-7B-Instruct-4bit', 7.0, 'coder'),
    'qwen2.5-7b-8bit': ('mlx-community/Qwen2.5-7B-Instruct-8bit', 7.0, 'general-8bit'),
    'qwen3-1.7b': ('mlx-community/Qwen3-1.7B-4bit', 1.7, 'qwen3'),
    'r1-distill-1.5b': ('mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit', 1.5, 'reasoning'),
}

PROJECT = {'django/django': 'Django', 'sympy/sympy': 'SymPy', 'astropy/astropy': 'Astropy',
           'matplotlib/matplotlib': 'Matplotlib', 'scikit-learn/scikit-learn': 'scikit-learn',
           'sphinx-doc/sphinx': 'Sphinx', 'pytest-dev/pytest': 'pytest', 'pydata/xarray': 'xarray',
           'psf/requests': 'Requests', 'pylint-dev/pylint': 'Pylint', 'mwaskom/seaborn': 'seaborn',
           'pallets/flask': 'Flask'}
# top-level package name used in absolute imports
PACKAGE = {'django/django': 'django', 'sympy/sympy': 'sympy', 'astropy/astropy': 'astropy',
           'matplotlib/matplotlib': 'matplotlib', 'scikit-learn/scikit-learn': 'sklearn',
           'sphinx-doc/sphinx': 'sphinx', 'pytest-dev/pytest': '_pytest', 'pydata/xarray': 'xarray',
           'psf/requests': 'requests', 'pylint-dev/pylint': 'pylint', 'mwaskom/seaborn': 'seaborn',
           'pallets/flask': 'flask'}

# the two pilot traps (notes/pilot-2026-11-13.md)
PILOTS = [
    dict(instance_id='django__django-11149', def_file='django/contrib/auth/__init__.py',
         qual='get_permission_codename', fact_type='arg_order', params=['action', 'opts']),
    dict(instance_id='sympy__sympy-13615', def_file='sympy/utilities/iterables.py',
         qual='sift', fact_type='arg_order', params=['seq', 'keyfunc']),
]

# plausible "other" values for string defaults (used only when a default is one of these)
STRING_FLIPS = {'raise': 'ignore', 'ignore': 'raise', 'strict': 'ignore', 'left': 'right', 'right': 'left',
                'upper': 'lower', 'lower': 'upper', 'auto': 'manual', 'utf-8': 'ascii', 'utf8': 'ascii',
                'ascii': 'utf-8', 'default': 'custom', 'all': 'any', 'any': 'all', 'first': 'last',
                'last': 'first', 'inner': 'outer', 'outer': 'inner', 'before': 'after', 'after': 'before',
                'mean': 'median', 'median': 'mean', 'linear': 'log', 'log': 'linear', 'center': 'edge',
                'edge': 'center', 'C': 'F', 'F': 'C', 'csr': 'csc', 'csc': 'csr', 'none': 'all',
                'GET': 'POST', 'POST': 'GET', 'html': 'text', 'text': 'html', 'r': 'rb', 'rb': 'r', 'w': 'wb',
                'warn': 'raise', '': ' ', ' ': '', ',': ';', '\n': ' ', 'replace': 'strict'}


# ----------------------------------------------------------------------------------------------- small helpers

def seed_of(*parts):
    """A stable random seed from some strings (so every run picks the same random items)."""
    return int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16)


def repo_dir(repo):
    return REPOS / repo.replace('/', '_')


def git(repo, *args):
    """Run git in one of the repo clones. Returns the output text, or None if git failed."""
    r = subprocess.run(['git', '-C', str(repo_dir(repo)), *args], capture_output=True, env=NO_FETCH)
    return r.stdout.decode('utf-8', errors='replace') if r.returncode == 0 else None


_show_cache = {}


def show(repo, commit, path):
    """The text of a file at a commit (None if missing or never downloaded)."""
    key = (repo, commit, path)
    if key not in _show_cache:
        _show_cache[key] = git(repo, 'show', f'{commit}:{path}')
    return _show_cache[key]


def is_test_path(path):
    parts = path.split('/')
    return (any(p in ('tests', 'test', 'testing') for p in parts[:-1])
            or parts[-1].startswith(('test_', 'conftest')) or parts[-1].endswith('_test.py'))


_files_cache = {}


def source_files(repo, commit):
    """Library .py files at a commit (no tests, docs, examples, benchmarks, setup scripts)."""
    key = (repo, commit)
    if key not in _files_cache:
        listing = git(repo, 'ls-tree', '-r', '--name-only', commit) or ''
        skip = ('doc/', 'docs/', 'examples/', 'example/', 'benchmarks/', 'asv_bench/', 'tools/', 'bin/',
                'release/', 'ci/', 'extern/', 'scripts/', 'galleries/', 'tutorials/', 'plot_types/')
        _files_cache[key] = sorted(p for p in listing.splitlines()
                                   if p.endswith('.py') and not is_test_path(p) and not p.startswith(skip)
                                   and '/_vendor' not in p and '/externals/' not in p and '/extern/' not in p
                                   and p.split('/')[-1] not in ('setup.py', 'conftest.py', 'setupext.py'))
    return _files_cache[key]


def module_of(path):
    """django/contrib/auth/__init__.py -> django.contrib.auth ; lib/matplotlib/axes/_base.py -> matplotlib.axes._base"""
    p = path[:-3] if path.endswith('.py') else path
    for prefix in ('lib/', 'src/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
    parts = p.split('/')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def module_file(repo, commit, module):
    """The repo path of a module at a commit (or None)."""
    files = set(source_files(repo, commit))
    base = module.replace('.', '/')
    for prefix in ('', 'lib/', 'src/'):
        for cand in (f'{prefix}{base}.py', f'{prefix}{base}/__init__.py'):
            if cand in files:
                return cand
    return None


# ------------------------------------------------------------------------------ reading Python without running it

def code_chars(text):
    """Walk Python source and yield (index, char, depth) for every character outside strings and comments.
    depth = number of open brackets, counting the bracket itself for '(' and ')'."""
    depth, i, n = 0, 0, len(text)
    while i < n:
        c = text[i]
        if c == '#':                                   # comment: jump to the end of the line
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if c in '"\'':                                 # string literal: jump over it
            q = text[i:i + 3] if text[i:i + 3] in ('"""', "'''") else c
            j = i + len(q)
            while j < n and text[j:j + len(q)] != q:
                if q == c and text[j] == '\n':         # unterminated one-line string: stop at the line end
                    break
                j += 2 if text[j] == '\\' else 1
            i = j + len(q)
            continue
        if c in '([{':
            depth += 1
        yield i, c, depth
        if c in ')]}':
            depth -= 1
        i += 1


def parse_header(text, need_colon=True):
    """Split a 'def name(...):' header into parts. Returns dict(open, close, colon, parts): text[open] is the
    '(' of the parameter list, text[close] its ')', text[colon] the ':' that ends the header, and parts are the
    (start, end) spans of the parameters, split at top-level commas. None if it is not a def header."""
    m = re.match(r'\s*(async\s+)?def\s+\w+\s*\(', text)
    if not m:
        return None
    open_i = m.end() - 1
    parts, start, close_i = [], open_i + 1, None
    for i, c, d in code_chars(text):
        if i <= open_i:
            continue
        if close_i is None:
            if c == ',' and d == 1:
                parts.append((start, i))
                start = i + 1
            elif c == ')' and d == 1:
                parts.append((start, i))
                close_i = i
                if not need_colon:
                    break
        elif c == ':' and d == 0:
            return dict(open=open_i, close=close_i, colon=i, parts=[p for p in parts if text[p[0]:p[1]].strip()])
        elif c == '\n' and d == 0 and not text[close_i:i].strip().startswith(')'):
            pass
    if close_i is not None and not need_colon:
        return dict(open=open_i, close=close_i, colon=None, parts=[p for p in parts if text[p[0]:p[1]].strip()])
    return None


def param_info(seg):
    """'name: type = default' -> (name, stars, default text or None). A bare '*' or '/' has name ''."""
    s = seg.strip()
    stars = len(s) - len(s.lstrip('*'))
    s = s.lstrip('*').strip()
    eq = next((i for i, c, d in code_chars(s) if c == '=' and d == 0), None)
    head = s if eq is None else s[:eq]
    name = re.match(r'\w*', head.strip()).group(0)
    return name, stars, (None if eq is None else s[eq + 1:].strip())


def header_params(header):
    h = parse_header(header)
    return None if h is None else [param_info(header[s:e]) for s, e in h['parts']]


def valid_header(header):
    """True if the header is valid Python (checked with ast, adding a dummy body)."""
    h = parse_header(header)
    if h is None:
        return False
    code = header[:h['colon'] + 1].lstrip()
    try:
        ast.parse(code + '\n    pass\n')
        return True
    except SyntaxError:
        return False


def functions(src):
    """Every function or method defined at module level or in a class (not inside another function).
    Returns [(qualname, node)]. Functions under 'if'/'try' blocks at the same level count too."""
    out = []

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((prefix + node.name, node))
            elif isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.If, ast.Try)):
                inner = node.body + node.orelse + getattr(node, 'finalbody', [])
                for handler in getattr(node, 'handlers', []):
                    inner += handler.body
                walk(inner, prefix)
    walk(ast.parse(src).body, '')
    return out


_parse_cache = {}


def file_functions(repo, commit, path):
    key = (repo, commit, path)
    if key not in _parse_cache:
        src = show(repo, commit, path)
        try:
            _parse_cache[key] = (src, functions(src)) if src else (None, [])
        except (SyntaxError, ValueError, RecursionError):
            _parse_cache[key] = (src, [])
    return _parse_cache[key]


def top_level_names(src):
    """Public functions and classes defined at module level (the names a module really defines)."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not n.name.startswith('_')]


def norm_default(text):
    """Normalize a default value so that 'raise' and "raise" compare equal."""
    if text is None:
        return None
    try:
        return repr(ast.literal_eval(text))
    except Exception:
        try:
            return ast.dump(ast.parse(text, mode='eval'))
        except Exception:
            return text.strip()


# --------------------------------------------------------------------------------------- making the 3 line versions

def rebuild(header, h, cores):
    """Put new parameter texts (cores, stripped) into the header, keeping the whitespace around each slot."""
    out, pos = [], 0
    for (s, e), core in zip(h['parts'], cores):
        seg = header[s:e]
        lead = seg[:len(seg) - len(seg.lstrip())]
        trail = seg[len(seg.rstrip()):]
        out += [header[pos:s], lead, core, trail]
        pos = e
    out.append(header[pos:])
    return ''.join(out)


def swap_params(header, p1, p2):
    """Trap for argument order: swap the two parameters. If only one of them has a default, swap just the
    names (otherwise the result would not be valid Python)."""
    h = parse_header(header)
    cores = [header[s:e].strip() for s, e in h['parts']]
    infos = [param_info(c) for c in cores]
    names = [i[0] if i[1] == 0 else None for i in infos]
    if p1 not in names or p2 not in names:
        return None
    a, b = names.index(p1), names.index(p2)
    if (infos[a][2] is None) == (infos[b][2] is None):
        cores[a], cores[b] = cores[b], cores[a]
    else:
        cores[a], cores[b] = p2 + cores[a][len(p1):], p1 + cores[b][len(p2):]
    return rebuild(header, h, cores)


def set_default(header, p, new_value):
    """Trap for a default value: give parameter p a different default."""
    h = parse_header(header)
    cores = [header[s:e].strip() for s, e in h['parts']]
    for k, c in enumerate(cores):
        name, stars, default = param_info(c)
        if name == p and stars == 0 and default is not None:
            eq = next(i for i, ch, d in code_chars(c) if ch == '=' and d == 0)
            space = ' ' if c[eq + 1:eq + 2] == ' ' else ''
            cores[k] = c[:eq + 1] + space + new_value
            return rebuild(header, h, cores)
    return None


def rename_param(header, p, new):
    """Trap for a keyword name: rename parameter p (only in the signature line)."""
    h = parse_header(header)
    cores = [header[s:e].strip() for s, e in h['parts']]
    for k, c in enumerate(cores):
        name, stars, _ = param_info(c)
        if name == p and stars == 0:
            cores[k] = new + c[len(p):]
            return rebuild(header, h, cores)
    return None


def add_kwargs(header):
    """Placebo: add a catch-all parameter. Every call that worked before still works the same way, so the
    order, names and defaults a caller relies on all stay true (the rr-targeted keyword placebo does the same)."""
    h = parse_header(header)
    infos = [param_info(header[s:e]) for s, e in h['parts']]
    if any(st == 2 for _, st, _ in infos):              # already has **kwargs: add *args before it, if missing
        if any(st == 1 for _, st, _ in infos):
            return None
        k = next(i for i, (_, st, _) in enumerate(infos) if st == 2)
        s, e = h['parts'][k]
        seg = header[s:e]
        lead = len(seg) - len(seg.lstrip())
        return header[:s + lead] + '*args, ' + header[s + lead:]
    if not h['parts']:
        return header[:h['open'] + 1] + '**kwargs' + header[h['open'] + 1:]
    s, e = h['parts'][-1]
    end = s + len(header[s:e].rstrip())
    return header[:end] + ', **kwargs' + header[end:]


def toggle_plural(name):
    """Trap name for a keyword: opts -> opt, code -> codes (a small, plausible rename)."""
    if name.endswith('ss') or not name.endswith('s'):
        return name + ('es' if name.endswith(('ss', 'x', 'ch', 'sh')) else 's')
    return name[:-1]


def trapped_default(text):
    """A plausible different value for a literal default, or None when we have no fair rule
    (None defaults mean 'not set', so changing them is not a fair trap)."""
    try:
        v = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(v, bool):
        return 'False' if v else 'True'
    if v is None:
        return None
    if isinstance(v, int):
        return '1' if v == 0 else ('0' if v == 1 else str(v * 2))
    if isinstance(v, float):
        return repr(v * 2) if v else '1.0'
    if isinstance(v, str) and v in STRING_FLIPS:
        q = text.strip()[-1]
        if q not in '"\'':
            return None
        new = STRING_FLIPS[v].replace('\\', '\\\\').replace('\n', '\\n')
        return q + new + q
    return None


# ------------------------------------------------------------------------------------------------- probe items

def context_before(repo, commit, path, src, lineno):
    """The prompt for the likelihood probe: a one-line file-path comment, then up to 40 real lines before line
    `lineno` (1-based)."""
    lines = src.splitlines(keepends=True)
    start = max(0, lineno - 1 - CONTEXT_LINES)
    body = ''.join(lines[start:lineno - 1])
    if body and not body.endswith('\n'):
        body += '\n'
    return f'# file: {path}\n' + body


def def_header(src, lineno):
    """The full 'def ...:' header starting at line `lineno` (1-based), possibly several lines long."""
    lines = src.splitlines(keepends=True)
    rest = ''.join(lines[lineno - 1:lineno - 1 + 60])
    h = parse_header(rest)
    if h is None:
        return None
    n_extra = rest.count('\n', 0, h['colon'])
    header = ''.join(lines[lineno - 1:lineno + n_extra])
    return header if header.endswith('\n') else header + '\n'


def strip_self(params):
    return [p for p in params if p[0] not in ('self', 'cls', 'mcs', 'klass') or p[1]]


def choose_params(fact_type, params):
    """For a control function: which parameter(s) the edit targets, or None if the function does not fit."""
    plain = [p for p in strip_self(params) if p[1] == 0 and p[0]]
    if fact_type == 'arg_order':
        if len(plain) >= 2 and plain[0][2] is None and plain[1][2] is None:
            return [plain[0][0], plain[1][0]]
    elif fact_type == 'default':
        for name, _, default in plain:
            if default is not None and trapped_default(default):
                return [name]
    elif fact_type == 'keyword':
        for name, _, default in plain:
            if default is not None:
                return [name]
    return None


def make_signature_item(base, path, qual, node, fact_type, params, spec=None):
    """Build a likelihood + recall probe for a fact about a function signature. base has the task fields.
    spec (optional, from the trap catalog) can fix the trapped value. Returns None if it cannot be built."""
    repo, commit = base['repo'], base['base_commit']
    src, _ = file_functions(repo, commit, path)
    if not src:
        return None
    header = def_header(src, node.lineno)
    if header is None or not valid_header(header):
        return None
    truth = header_params(header)
    names = {p[0]: p for p in truth}
    fact = {}
    if fact_type == 'arg_order':
        p1, p2 = params[:2]
        trapped = swap_params(header, p1, p2)
        fact = dict(params=[p1, p2])
        desc = f'{qual} takes ({p1}, {p2}) in this order'
    elif fact_type == 'default':
        p = params[0]
        if p not in names or names[p][2] is None:
            return None
        new = (spec or {}).get('trap_default') or trapped_default(names[p][2])
        if not new:
            return None
        trapped = set_default(header, p, new)
        fact = dict(param=p, default=names[p][2], trap_default=new)
        desc = f'{qual} has default {p}={names[p][2]}'
    elif fact_type == 'keyword':
        p = params[0]
        if p not in names or names[p][1]:
            return None
        new = (spec or {}).get('trap_name') or toggle_plural(p)
        trapped = rename_param(header, p, new)
        fact = dict(param=p, trap_name=new)
        desc = f'{qual} accepts keyword {p}='
        if (spec or {}).get('negate_default'):          # T8 flag inversion: also negate the bool default
            flipped = trapped_default(names[p][2]) if names[p][2] in ('True', 'False') else None
            if not flipped:
                return None
            trapped = set_default(trapped, new, flipped)
            fact.update(default=names[p][2], trap_default=flipped)
            desc = f'{qual} takes flag {p}={names[p][2]}'
    else:
        return None
    placebo = add_kwargs(header)
    if not trapped or trapped == header or not placebo or not valid_header(trapped) or not valid_header(placebo):
        return None
    context = context_before(repo, commit, path, src, node.lineno)
    name = qual.split('.')[-1]
    if name in ('__init__', '__new__'):
        name = qual.split('.')[-2]
    visible = bool(re.search(r'\b%s\b' % re.escape(name), context.split('\n', 1)[1]))
    dotted = module_of(path) + '.' + qual
    prompt = (f'In {PROJECT[repo]} {base["version"]} ({base["year"]}), what is the exact signature of '
              f'`{dotted}`? Reply with only the Python def line, nothing else.')
    return dict(kind='signature', fact_type=fact_type, repo=repo, instance_id=base['instance_id'],
                base_commit=commit, version=base['version'], year=base['year'], def_file=path, def_line=node.lineno,
                qualname=qual, dotted=dotted, fact=fact, fact_desc=desc, probe_file=path, probe_line=node.lineno,
                context=context, original=header, trapped=trapped, placebo=placebo,
                placebo_kind='add *args' if placebo.count('*args') > header.count('*args') else 'add **kwargs',
                name_in_context=visible, truth=dict(params=truth), recall_prompt=prompt)


_import_cache = {}


def import_lines(repo, commit):
    """Every one-line absolute 'from <package>... import a, b' statement at a commit:
    [(path, lineno, line text, module, [imported names])]."""
    key = (repo, commit)
    if key in _import_cache:
        return _import_cache[key]
    pkg = PACKAGE[repo]
    out = git(repo, 'grep', '-n', '-E', f'^ *from {pkg}(\\.[A-Za-z0-9_.]+)? import ', commit, '--', '*.py') or ''
    rows = []
    for line in out.splitlines():
        try:
            _, path, lineno, text = line.split(':', 3)
        except ValueError:
            continue
        m = re.match(r'^(\s*)from\s+([\w.]+)\s+import\s+([^()\\#]+?)\s*(#.*)?$', text)
        if not m or text.rstrip().endswith('\\'):
            continue
        names = [n.strip().split(' as ')[0].strip() for n in m.group(3).split(',') if n.strip()]
        if all(re.fullmatch(r'\w+', n) for n in names):
            rows.append((path, int(lineno), text + '\n', m.group(2), names))
    _import_cache[key] = rows
    return rows


def make_import_item(base, module, name, rng, used_names, trap_module=None):
    """Build a probe for 'name is imported from module'. The probe line is a real import line somewhere in the
    repo. Trap: the same line importing from a sibling module (which does not have the name). Placebo: the
    same line also importing one more real name from the module."""
    repo, commit = base['repo'], base['base_commit']
    cands = [r for r in import_lines(repo, commit) if r[3] == module and name in r[4]]
    if not cands:
        return None
    mfile = module_file(repo, commit, module)
    msrc = show(repo, commit, mfile) if mfile else None
    if not msrc:
        return None
    path, lineno, line, _, names = sorted(cands)[rng.randrange(len(cands))]
    # trap module: a sibling module in the same package that does not mention the name at all
    folder = mfile.rsplit('/', 1)[0] + '/'
    siblings = [p for p in source_files(repo, commit) if p.startswith(folder) and '/' not in p[len(folder):]
                and not p.endswith('__init__.py') and p != mfile]
    rng.shuffle(siblings)
    for sib in ([] if trap_module else siblings[:30]):
        ssrc = show(repo, commit, sib)
        if ssrc is not None and not re.search(r'\b%s\b' % re.escape(name), ssrc):
            trap_module = module_of(sib)
            break
    if trap_module is None:
        return None
    # placebo name: another public function/class defined in the module, not already in this line
    extra = [n for n in top_level_names(msrc) if n != name and n not in names and n not in used_names]
    if not extra:
        return None
    extra_name = sorted(extra)[rng.randrange(len(extra))]
    trapped = line.replace(f'from {module} import', f'from {trap_module} import', 1)
    placebo = re.sub(r'\b%s\b' % re.escape(name), f'{name}, {extra_name}', line.split(' import ', 1)[1], count=1)
    placebo = line.split(' import ', 1)[0] + ' import ' + placebo
    src = show(repo, commit, path)
    context = context_before(repo, commit, path, src, lineno)
    visible = bool(re.search(r'\b%s\b' % re.escape(name), context.split('\n', 1)[1]))
    prompt = (f'In {PROJECT[repo]} {base["version"]} ({base["year"]}), what is the Python import statement for '
              f'`{name}`? Reply with only the import line, nothing else.')
    return dict(kind='import', fact_type='import', repo=repo, instance_id=base['instance_id'], base_commit=commit,
                version=base['version'], year=base['year'], def_file=mfile, def_line=None, qualname=name,
                dotted=f'{module}.{name}', fact=dict(module=module, name=name, trap_module=trap_module,
                                                     placebo_name=extra_name),
                fact_desc=f'{name} is imported from {module}', probe_file=path, probe_line=lineno,
                context=context, original=line, trapped=trapped, placebo=placebo,
                placebo_kind=f'import one more name ({extra_name})', name_in_context=visible,
                truth=dict(module=module, name=name), recall_prompt=prompt)


def find_node(repo, commit, path, qual, line=None):
    """The function node for qualname `qual` in a file (matched by line first, then by name)."""
    _, funcs = file_functions(repo, commit, path)
    for q, node in funcs:
        if line is not None and node.lineno == line and q.split('.')[-1] == qual.split('.')[-1]:
            return q, node
    for q, node in funcs:
        if q == qual:
            return q, node
    return None


def load_tasks():
    import pyarrow.parquet as pq
    df = pq.read_table(PARQUET, columns=['instance_id', 'repo', 'base_commit', 'version', 'created_at']).to_pandas()
    return {r.instance_id: dict(instance_id=r.instance_id, repo=r.repo, base_commit=r.base_commit,
                                version=str(r.version), year=str(r.created_at)[:4]) for r in df.itertuples()}


def load_agent_facts():
    """Distinct (task, fact) pairs of the four probe-able types, with the best tier any model reached:
    memory (no definition and no use seen before writing it) > unseen (no definition seen) > seen."""
    groups = defaultdict(list)
    used = defaultdict(set)    # repo -> every name any agent's patch relied on (controls must avoid these)
    for line in open(FACTS):
        r = json.loads(line)
        for n in (r.get('def_name'), r.get('use_name'), (r.get('target_qual') or '').split('.')[-1]):
            if n:
                used[r['repo']].add(n)
        if r['fact_type'] in FACT_TYPES:
            groups[(r['instance_id'], r['fact_key'])].append(r)
    facts = []
    for (iid, key), rs in sorted(groups.items()):
        r0 = rs[0]
        tier = 'memory' if any(r['no_use'] for r in rs) else 'unseen' if any(r['unseen'] for r in rs) else 'seen'
        facts.append(dict(instance_id=iid, fact_key=key, tier=tier, fact_type=r0['fact_type'],
                          target_file=r0['target_file'], target_line=r0['target_line'], target_qual=r0['target_qual'],
                          params=r0['params'], detail=r0['detail'], models=sorted({r['model'] for r in rs}),
                          models_memory=sorted({r['model'] for r in rs if r['no_use']}),
                          models_unseen=sorted({r['model'] for r in rs if r['unseen']})))
    return facts, used


def build_fact_item(task, f, rng, used_names):
    """Turn one agent fact (or catalog spec, same fields) into a probe item, or None."""
    repo, commit = task['repo'], task['base_commit']
    if f['fact_type'] == 'import':
        module, name = f['fact_key'].split('|')[-1].split(':')
        return make_import_item(task, module, name, rng, used_names, (f.get('spec') or {}).get('trap_module'))
    if not f.get('target_qual') or f['target_qual'].startswith('<') or not f.get('params'):
        return None
    found = find_node(repo, commit, f['target_file'], f['target_qual'], f.get('target_line'))
    if not found:
        return None
    qual, node = found
    return make_signature_item(task, f['target_file'], qual, node, f['fact_type'], f['params'], f.get('spec'))


def pick_control(fact_item, task, used_names, taken):
    """One random function (or import) from the same repo and base commit that no agent relied on, probed with
    the same kind of edit as the fact. Random file first, then a random fitting function in it."""
    repo, commit = task['repo'], task['base_commit']
    rng = random.Random(seed_of('control', fact_item['id']))
    if fact_item['kind'] == 'import':
        rows = [r for r in import_lines(repo, commit) if not is_test_path(r[0])]
        rng.shuffle(rows)
        for path, lineno, line, module, names in rows[:400]:
            for name in names:
                if name.startswith('_') or name in used_names or (repo, commit, 'import', module, name) in taken:
                    continue
                mfile = module_file(repo, commit, module)
                msrc = show(repo, commit, mfile) if mfile else None
                if not msrc or name not in top_level_names(msrc):      # must be really defined there
                    continue
                item = make_import_item(task, module, name, random.Random(seed_of('imp', fact_item['id'], name)),
                                        used_names)
                if item:
                    taken.add((repo, commit, 'import', module, name))
                    return item
        return None
    files = list(source_files(repo, commit))
    rng.shuffle(files)
    for path in files[:300]:
        src, funcs = file_functions(repo, commit, path)
        funcs = list(funcs)
        rng.shuffle(funcs)
        for qual, node in funcs:
            name = node.name
            if name in used_names or (name.startswith('__') and name not in ('__init__', '__new__')):
                continue
            if name in ('__init__', '__new__') and qual.split('.')[-2] in used_names:
                continue
            if (repo, commit, path, qual) in taken:
                continue
            header = def_header(src, node.lineno)
            params = header_params(header) if header else None
            chosen = choose_params(fact_item['fact_type'], params) if params else None
            if not chosen:
                continue
            item = make_signature_item(task, path, qual, node, fact_item['fact_type'], chosen)
            if item:
                taken.add((repo, commit, path, qual))
                return item
    return None


def load_catalog():
    """Trap specs from traps/catalog/*/*/spec.json, turned into the same fields as agent facts. The catalog is
    written by another part of the project; we read whichever fields are there and skip what we cannot use."""
    specs = []
    for p in sorted(CATALOG.glob('*/*/spec.json')):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        s['_path'] = str(p.relative_to(ROOT))
        specs.append(s)
    return specs


def catalog_to_fact(s, task):
    """Map one catalog spec (scripts/build_trap_catalog.py format) to the fields build_fact_item uses.
    Probe-able here: T1 argument order, T2 default, T7 keyword rename, T8 flag inversion (keyword renamed and its
    bool default negated), T4 move (import from the new module). T3 renames, T5 exception swaps and T6 return
    conventions are not signature-line or import-line facts of the four kinds probed, so they are skipped."""
    t = s.get('transform') or {}
    init = t.get('init') or {}
    typ = s.get('type') or t.get('type')
    defs = t.get('defs') or [[s['target']['file'], s['target']['qualname']]]
    def_file, qual = defs[0]
    base = dict(target_file=def_file, target_line=None, target_qual=qual, detail=s.get('original_fact', ''), spec={})
    if typ == 'T1' and t.get('p0') and t.get('p1'):
        return dict(base, fact_type='arg_order', params=[t['p0'], t['p1']], fact_key=f'arg_order|{def_file}|{qual}')
    if typ == 'T2' and init.get('param'):
        return dict(base, fact_type='default', params=[init['param']], spec=dict(trap_default=init.get('new_default')),
                    fact_key=f'default|{def_file}|{qual}|{init["param"]}')
    if typ in ('T7', 'T8') and init.get('param') and init.get('new_param'):
        return dict(base, fact_type='keyword', params=[init['param']],
                    spec=dict(trap_name=init['new_param'], negate_default=typ == 'T8'),
                    fact_key=f'keyword|{def_file}|{qual}|{init["param"]}')
    if typ == 'T4' and init.get('new_file') and '.' not in qual:
        module = module_of(def_file)
        return dict(base, fact_type='import', params=None, spec=dict(trap_module=module_of(init['new_file'])),
                    fact_key=f'import|{def_file}|{qual}|{module}:{qual}')
    return None


def build(args):
    tasks = load_tasks()
    facts, used = load_agent_facts()
    items, taken = [], set()

    def add(item, group, extra):
        item.update(extra)
        item['group'] = group
        items.append(item)
        return item

    # (1) pilot traps
    for p in PILOTS:
        task = tasks[p['instance_id']]
        found = find_node(task['repo'], task['base_commit'], p['def_file'], p['qual'])
        item = make_signature_item(task, p['def_file'], found[0], found[1], p['fact_type'], p['params'])
        add(item, 'pilot', dict(id=f"pilot:{p['instance_id']}:{p['qual']}", tier='pilot'))
        taken.add((task['repo'], task['base_commit'], p['def_file'], found[0]))
    # (2) agent-targeted facts: all of the memory tier, a seeded sample of the other two tiers
    by_bucket = defaultdict(list)
    for f in facts:
        by_bucket[(f['tier'], f['fact_type'])].append(f)
    seen_keys = set()
    for (tier, ftype), bucket in sorted(by_bucket.items()):
        rng = random.Random(seed_of('sample', tier, ftype))
        bucket = sorted(bucket, key=lambda f: (f['instance_id'], f['fact_key']))
        rng.shuffle(bucket)
        limit = None if tier == 'memory' else SAMPLE_PER_TYPE
        n = 0
        for f in bucket:
            if limit is not None and n >= limit:
                break
            task = tasks.get(f['instance_id'])
            if task is None:
                continue
            key = (task['repo'], task['base_commit'], f['fact_key'].split('|', 1)[1])
            if key in seen_keys:
                continue
            item = build_fact_item(task, f, random.Random(seed_of('fact', f['instance_id'], f['fact_key'])),
                                   used[task['repo']])
            if item is None:
                print(f'  skip {tier} {ftype} {f["instance_id"]} {f["fact_key"]}', file=sys.stderr)
                continue
            seen_keys.add(key)
            add(item, 'agent', dict(id=f"agent:{f['instance_id']}:{f['fact_key']}", tier=tier,
                                    agent_detail=f['detail'], models_relied=f['models'],
                                    models_memory=f['models_memory'], models_unseen=f['models_unseen']))
            n += 1
    # (3) trap catalog (only once traps/catalog/READY exists, unless --catalog-anyway)
    if (CATALOG / 'READY').exists() or '--catalog-anyway' in args:
        for s in load_catalog():
            iid = s.get('instance_id') or s.get('task')
            task = tasks.get(iid)
            f = catalog_to_fact(s, task) if task else None
            item = build_fact_item(task, f, random.Random(seed_of('trap', s['_path'])), used[task['repo']]) if f else None
            if item is None:
                print(f'  skip catalog spec {s["_path"]}', file=sys.stderr)
                continue
            add(item, 'trap', dict(id=f"trap:{s['_path']}", tier='trap', spec_path=s['_path']))
    # (4) controls: one per fact, same repo + base commit + kind of edit
    for item in [i for i in items if i['group'] != 'control']:
        task = tasks[item['instance_id']]
        ctrl = pick_control(item, task, used[task['repo']], taken)
        if ctrl is None:
            print(f'  no control for {item["id"]}', file=sys.stderr)
            continue
        add(ctrl, 'control', dict(id=f"control:{item['id']}", tier='control', pair_id=item['id']))
        item['pair_id'] = ctrl['id']
    WORK.mkdir(parents=True, exist_ok=True)
    # Add-only: items that were already scored are kept exactly as they were (which import line gets picked can
    # change when more files of a repo get downloaded), and only items with new ids are added. --fresh rebuilds.
    if (WORK / 'items.jsonl').exists() and '--fresh' not in args:
        old = load_items()
        old_ids = {it['id'] for it in old}
        added = [it for it in items if it['id'] not in old_ids]
        print(f'kept {len(old)} existing items, added {len(added)} new ones')
        items = old + added
    with open(WORK / 'items.jsonl.tmp', 'w') as fh:
        for it in items:
            fh.write(json.dumps(it) + '\n')
    os.replace(WORK / 'items.jsonl.tmp', WORK / 'items.jsonl')
    c = Counter((i['group'], i['fact_type']) for i in items)
    print(f'{len(items)} items -> {WORK / "items.jsonl"}')
    for k in sorted(c):
        print('  ', k, c[k])
    print('   agent tiers:', dict(Counter(i['tier'] for i in items if i['group'] == 'agent')))


# ------------------------------------------------------------------------------------------ running one model

def load_items():
    return [json.loads(line) for line in open(WORK / 'items.jsonl')]


def output_layer(model, hidden):
    """Hidden states -> logits. Only needed for the last context position, which saves computing the huge
    vocabulary layer for every context token."""
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(hidden)
    return model.lm_head(hidden)


def score_lines(model, tok, context, lines, use_cache=True):
    """log P(line | context) for each line: the sum over the line's tokens of the log-probability the model gives
    each token (natural log). The context is run once; its key/value cache is reused for every line and trimmed
    back afterwards. If the tokenizer merges the last context token with the line (it should not, the context ends
    with a newline), the line is scored from the first differing token and boundary_ok is False."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    ctx = tok.encode(context)
    cache = make_prompt_cache(model)
    hidden = model.model(mx.array(ctx)[None], cache=cache)     # run the context once (fills the cache)
    last = output_layer(model, hidden[:, -1:, :])[0]           # logits that predict the line's first token
    mx.eval(last)
    out = []
    for line in lines:
        full = tok.encode(context + line)
        ok = full[:len(ctx)] == ctx
        if ok and use_cache:
            ids = full[len(ctx):]
            logits = model(mx.array(ids)[None], cache=cache)[0]
            logits = mx.concatenate([last, logits[:-1]], axis=0)
            trim_prompt_cache(cache, len(ids))
        else:                                   # plain full pass (also used by 'check')
            j = next((k for k, (a, b) in enumerate(zip(full, ctx)) if a != b), min(len(full), len(ctx)))
            ids = full[j:]
            logits = model(mx.array(full)[None])[0, j - 1:-1]
        logits = logits.astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token_lp = logp[mx.arange(len(ids)), mx.array(ids)]
        mx.eval(token_lp)
        out.append(dict(logp=round(float(token_lp.sum()), 4), n_tokens=len(ids), boundary_ok=ok,
                        min_token_logp=round(float(token_lp.min()), 4)))
    return out


def answer_complete(text, kind):
    """Stop generating once the answer holds a full def line (or import line)."""
    if kind == 'import':
        return bool(re.search(r'^\s*(from\s+[\w.]+\s+import\s+.+|import\s+[\w.]+.*)\n', text, re.M))
    m = re.search(r'(async\s+)?def\s+\w+\s*\(', text)
    if not m:
        return False
    rest = text[m.start():]
    if parse_header(rest) is not None:
        return True
    h = parse_header(rest, need_colon=False)
    return h is not None and '\n' in rest[h['close']:]


def answer_prefix(item):
    """The start of the answer is written for the model ('def name(' or 'from '), so it only has to produce the
    parameters (or the module). This keeps every model in the same format and saves generation time."""
    if item['kind'] == 'import':
        return 'from'          # no trailing space: the model writes ' django.utils...' as it normally would
    return 'def ' + item['qualname'].split('.')[-1] + '('


def too_long(text, item):
    """Stop rule for rambling answers (small models repeat parameters until the token limit): stop once the
    answer already lists 4 more parameters than the real signature. Such answers are wrong either way."""
    n_real = len(item['truth']['params'])
    n_commas = sum(1 for _, c, d in code_chars(text) if c == ',' and d == 1)
    return n_commas >= n_real + 4


def chat_prompt_ids(tok, item, family):
    """Token ids of the closed-book question in the model's chat format, with the answer prefix appended."""
    messages = [dict(role='user', content=item['recall_prompt'])]
    extra = dict(enable_thinking=False) if family == 'qwen3' else {}
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
    text += answer_prefix(item)
    add_special = tok.bos_token is None or not text.startswith(tok.bos_token)   # same rule as mlx_lm
    return tok.encode(text, add_special_tokens=add_special)


def cut_answer(tok, item, token_ids):
    """Replay the stop rule on the generated tokens: the answer ends at the first token after which the def
    (or import) line is complete, or the answer has 4 parameters too many. Same result as stopping early,
    because greedy decoding never looks ahead."""
    prefix = answer_prefix(item)
    for k in range(1, len(token_ids) + 1):
        text = prefix + tok.decode(token_ids[:k])
        if answer_complete(text, item['kind']) or (item['kind'] == 'signature' and too_long(text, item)):
            return text
    return prefix + tok.decode(token_ids)


def recall_batch(model, tok, items, family):
    """Closed-book questions for many items at once (greedy decoding, batched): returns the answer texts."""
    from mlx_lm.generate import batch_generate
    prompts = [chat_prompt_ids(tok, it, family) for it in items]
    max_tokens = [30 if it['kind'] == 'import' else 80 for it in items]
    # several facts can share one question (e.g. five facts about LogisticRegression.__init__): ask it once
    unique = list(dict.fromkeys(zip(map(tuple, prompts), max_tokens)))
    out = batch_generate(model, tok, [list(pr) for pr, _ in unique], max_tokens=[mt for _, mt in unique],
                         return_token_ids=True)
    generated = dict(zip(unique, out.token_ids))
    return [cut_answer(tok, it, generated[(tuple(pr), mt)]) for it, pr, mt in zip(items, prompts, max_tokens)]


def recall(model, tok, item, family):
    """One closed-book question at a time (the first runs used this; kept to compare with the batched answers)."""
    from mlx_lm import stream_generate
    messages = [dict(role='user', content=item['recall_prompt'])]
    extra = dict(enable_thinking=False) if family == 'qwen3' else {}
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
    text = answer_prefix(item)
    max_tokens = 30 if item['kind'] == 'import' else 80
    for resp in stream_generate(model, tok, prompt + text, max_tokens=max_tokens):
        text += resp.text
        if answer_complete(text, item['kind']) or (item['kind'] == 'signature' and too_long(text, item)):
            break
    return text


def run_model(short):
    """Score every item with one model, in two passes, both resumable:
    1. likelihood, item by item -> data/probes_work/results_<model>.jsonl
    2. closed-book recall, all missing items in one batched pass -> data/probes_work/recall_<model>.jsonl
    (the reasoning model only gets pass 1: its answers need long thinking first)."""
    from mlx_lm import load
    hf_id, size, family = MODELS[short]
    items = load_items()
    lik_path, rec_path = WORK / f'results_{short}.jsonl', WORK / f'recall_{short}.jsonl'
    done_lik = {json.loads(line)['id'] for line in open(lik_path)} if lik_path.exists() else set()
    done_rec = {json.loads(line)['id'] for line in open(rec_path)} if rec_path.exists() else set()
    todo_lik = [it for it in items if it['id'] not in done_lik]
    todo_rec = [] if family == 'reasoning' else [it for it in items if it['id'] not in done_rec]
    print(f'{short}: likelihood for {len(todo_lik)} items, recall for {len(todo_rec)} items', flush=True)
    if not todo_lik and not todo_rec:
        return
    t0 = time.time()
    model, tok = load(hf_id)
    print(f'  loaded {hf_id} in {time.time() - t0:.1f}s', flush=True)
    with open(lik_path, 'a') as fh:
        for k, item in enumerate(todo_lik):
            t1 = time.time()
            lik = score_lines(model, tok, item['context'], [item['original'], item['trapped'], item['placebo']])
            fh.write(json.dumps(dict(id=item['id'], model=short, original=lik[0], trapped=lik[1], placebo=lik[2],
                                     sec_lik=round(time.time() - t1, 3))) + '\n')
            fh.flush()
            if k % 25 == 0:
                print(f'  likelihood {k}/{len(todo_lik)}  {time.time() - t0:.0f}s', flush=True)
    if todo_rec:
        t1 = time.time()
        answers = recall_batch(model, tok, todo_rec, family)
        sec = (time.time() - t1) / len(todo_rec)
        with open(rec_path, 'a') as fh:
            for item, answer in zip(todo_rec, answers):
                fh.write(json.dumps(dict(id=item['id'], model=short, answer=answer, sec_recall=round(sec, 3))) + '\n')
        print(f'  recall for {len(todo_rec)} items in {time.time() - t1:.0f}s', flush=True)
    print(f'  {short} done in {time.time() - t0:.0f}s', flush=True)


def check(short, n=4):
    """Check that the cached scoring gives the same numbers as one plain pass per line."""
    from mlx_lm import load
    model, tok = load(MODELS[short][0])
    for item in load_items()[:n]:
        lines = [item['original'], item['trapped'], item['placebo']]
        a = score_lines(model, tok, item['context'], lines)
        b = score_lines(model, tok, item['context'], lines, use_cache=False)
        print(item['id'][:60], [(x['logp'], y['logp']) for x, y in zip(a, b)])


# ------------------------------------------------------------------------------------------- scoring + report

def parse_answer_signature(text):
    """Parameters [(name, stars, default)] of the first def line in an answer, or None. A cut-off answer
    (no closing parenthesis) is closed with '):' so that the parameters written so far still count."""
    m = re.search(r'(async\s+)?def\s+\w+\s*\(', text)
    if not m:
        return None
    rest = text[m.start():]
    h = parse_header(rest) or parse_header(rest, need_colon=False) or parse_header(rest + '):')
    if h is None:
        return None
    return [param_info(rest[s:e]) for s, e in h['parts']]


def score_signature(item, answer):
    """Score a recalled def line against the real one."""
    out = dict(parsed=False, fact_correct=False, fact_trap=False, names_exact=False, full_exact=False)
    params = parse_answer_signature(answer or '')
    if params is None:
        return out
    a, t = strip_self(params), strip_self(item['truth']['params'])
    a_names = ['*' * st + n for n, st, _ in a]
    t_names = ['*' * st + n for n, st, _ in t]
    out.update(parsed=True, names_exact=a_names == t_names)
    out['full_exact'] = out['names_exact'] and all(norm_default(x[2]) == norm_default(y[2]) for x, y in zip(a, t))
    plain = [n for n, st, _ in a if st == 0 and n]
    defaults = {n: d for n, st, d in a if st == 0 and n}
    fact = item['fact']
    if item['fact_type'] == 'arg_order':
        p1, p2 = fact['params']
        both = p1 in plain and p2 in plain
        out['fact_correct'] = both and plain.index(p1) < plain.index(p2)
        out['fact_trap'] = both and plain.index(p1) > plain.index(p2)
        out['names_present'] = both
    elif item['fact_type'] == 'default':
        d = defaults.get(fact['param'])
        out['fact_correct'] = d is not None and norm_default(d) == norm_default(fact['default'])
        out['fact_trap'] = d is not None and norm_default(d) == norm_default(fact['trap_default'])
        out['names_present'] = fact['param'] in plain
    elif item['fact_type'] == 'keyword':
        out['fact_correct'] = fact['param'] in plain
        out['fact_trap'] = fact['trap_name'] in plain and fact['param'] not in plain
        out['names_present'] = fact['param'] in plain
    return out


def score_import(item, answer):
    """Score a recalled import line: is the name imported from the real module?"""
    out = dict(parsed=False, fact_correct=False, fact_trap=False, names_exact=None, full_exact=False)
    m = re.search(r'from\s+([\w. ]+?)\s+import\s+([^\n]+)', answer or '')
    if not m:
        return out
    module = re.sub(r'\s*\.\s*', '.', m.group(1).strip())
    names = [n.strip().split(' as ')[0].strip(' ()') for n in m.group(2).split(',')]
    named = item['fact']['name'] in names
    out.update(parsed=True, module=module, names_present=named)
    out['fact_correct'] = out['full_exact'] = named and module == item['fact']['module']
    out['fact_trap'] = named and module == item['fact']['trap_module']
    # looser: the stated module exists at the base commit and mentions the name (defines or re-exports it)
    mfile = module_file(item['repo'], item['base_commit'], module) if named else None
    msrc = show(item['repo'], item['base_commit'], mfile) if mfile else None
    out['module_valid'] = bool(msrc and re.search(r'\b%s\b' % re.escape(item['fact']['name']), msrc))
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(xs, n_boot=2000, seed=0):
    """95% percentile bootstrap interval for the mean of xs."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def r3(x):
    return None if x is None else round(x, 3)


def fact_in_context(item):
    """True if the 40 context lines already show the probed fact (so the model can copy it instead of
    remembering it): the same two names in the same order ('f, g'), the same 'name=default', the keyword
    'name=', or the same module name for an import."""
    ctx = item['context'].split('\n', 1)[1]
    f = item['fact']
    if item['fact_type'] == 'arg_order':
        pat = r'\b%s\s*,\s*%s\b' % (re.escape(f['params'][0]), re.escape(f['params'][1]))
    elif item['fact_type'] == 'default':
        pat = r'\b%s\s*=\s*%s' % (re.escape(f['param']), re.escape(f['default']))
    elif item['fact_type'] == 'keyword':
        pat = r'\b%s\s*=' % re.escape(f['param'])
    else:
        pat = r'\b%s\b' % re.escape(f['module'])
    return bool(re.search(pat, ctx))


def subsets(item):
    """The reporting groups an item belongs to."""
    g = [item['group']]
    if item['group'] == 'agent':
        g.append('agent:' + item['tier'])
    if item['group'] == 'control':
        g.append('control:' + item['pair_id'].split(':', 1)[0])
    return g


def lik_stats(rows):
    d = [r['delta_trap'] for r in rows]
    fp = [r['fact_pref'] for r in rows]
    lo, hi = boot_ci(d)
    flo, fhi = boot_ci(fp)
    return dict(n=len(rows), mean_delta_trap=r3(mean(d)), ci_lo=r3(lo), ci_hi=r3(hi),
                median_delta_trap=r3(sorted(d)[len(d) // 2]) if d else None,
                pct_trap_less_likely=r3(mean([x > 0 for x in d])),
                mean_delta_placebo=r3(mean([r['delta_placebo'] for r in rows])),
                mean_fact_pref=r3(mean(fp)), fact_pref_ci_lo=r3(flo), fact_pref_ci_hi=r3(fhi),
                pct_placebo_beats_trap=r3(mean([x > 0 for x in fp])))


def recall_stats(rows):
    rows = [r for r in rows if r.get('recall')]
    if not rows:
        return dict(n=0)
    fc = [float(r['recall']['fact_correct']) for r in rows]
    lo, hi = boot_ci(fc)
    return dict(n=len(rows), parsed=r3(mean([float(r['recall']['parsed']) for r in rows])),
                fact_correct=r3(mean(fc)), ci_lo=r3(lo), ci_hi=r3(hi),
                fact_trap=r3(mean([float(r['recall']['fact_trap']) for r in rows])),
                names_present=r3(mean([float(bool(r['recall'].get('names_present'))) for r in rows])),
                full_exact=r3(mean([float(r['recall']['full_exact']) for r in rows])),
                # imports only: the stated module exists and mentions the name (a re-export counts; star
                # re-exports are missed, so this is a lower bound)
                import_valid=r3(mean([float(r['recall']['module_valid']) for r in rows if 'module_valid' in r['recall']])))


def report():
    items = {it['id']: it for it in load_items()}
    per_item = defaultdict(dict)      # item id -> model -> result
    seconds = defaultdict(float)
    for short in MODELS:
        path = WORK / f'results_{short}.jsonl'
        if not path.exists():
            continue
        rec_path = WORK / f'recall_{short}.jsonl'
        answers = {}
        if rec_path.exists():
            for line in open(rec_path):
                r = json.loads(line)
                answers[r['id']] = r
        for line in open(path):
            r = json.loads(line)
            it = items.get(r['id'])
            if it is None:
                continue
            res = dict(lp_original=r['original']['logp'], lp_trapped=r['trapped']['logp'],
                       lp_placebo=r['placebo']['logp'], n_tokens=[r[k]['n_tokens'] for k in ('original', 'trapped', 'placebo')],
                       boundary_ok=all(r[k]['boundary_ok'] for k in ('original', 'trapped', 'placebo')))
            res['delta_trap'] = round(res['lp_original'] - res['lp_trapped'], 4)
            res['delta_placebo'] = round(res['lp_original'] - res['lp_placebo'], 4)
            res['fact_pref'] = round(res['lp_placebo'] - res['lp_trapped'], 4)
            if r['id'] in answers:
                res['answer'] = answers[r['id']]['answer']
                res['recall'] = (score_import if it['kind'] == 'import' else score_signature)(it, res['answer'])
                seconds[short] += answers[r['id']]['sec_recall']
            if r.get('answer') is not None:          # first runs: one question at a time (kept to compare)
                res['answer_one_at_a_time'] = r['answer']
            per_item[r['id']][short] = res
            seconds[short] += r['sec_lik']
    models = [m for m in MODELS if any(m in v for v in per_item.values())]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'probes.jsonl', 'w') as fh:
        for iid, it in items.items():
            fh.write(json.dumps(dict(it, fact_in_context=fact_in_context(it), results=per_item.get(iid, {}))) + '\n')

    summary = dict(created=time.strftime('%Y-%m-%d %H:%M'), context_lines=CONTEXT_LINES,
                   models={m: dict(hf_id=MODELS[m][0], size_b=MODELS[m][1], family=MODELS[m][2],
                                   n_items=sum(m in v for v in per_item.values()),
                                   minutes=round(seconds[m] / 60, 1)) for m in models},
                   items=dict(total=len(items), by_group=dict(Counter(i['group'] for i in items.values())),
                              by_group_type={f'{g}|{t}': n for (g, t), n in sorted(Counter(
                                  (i['group'], i['fact_type']) for i in items.values()).items())},
                              agent_tiers=dict(Counter(i['tier'] for i in items.values() if i['group'] == 'agent')),
                              name_in_context=dict(Counter(f"{i['group']}|{i['name_in_context']}" for i in items.values())),
                              fact_in_context=dict(Counter(f"{i['group']}|{fact_in_context(i)}" for i in items.values())),
                              boundary_problems=sum(1 for v in per_item.values() for r in v.values() if not r['boundary_ok'])),
                   likelihood={}, recall={}, fact_vs_control={}, contrasts={}, pilot={}, by_type={})
    lik_rows, rec_rows, pair_rows, type_rows, pilot_rows, contrast_rows = [], [], [], [], [], []
    group_names = ['pilot', 'agent', 'agent:memory', 'agent:unseen', 'agent:seen', 'trap', 'control',
                   'control:agent', 'control:trap', 'control:pilot']
    for m in models:
        size, fam = MODELS[m][1], MODELS[m][2]
        rows_by_group = defaultdict(list)
        for iid, it in items.items():
            if m in per_item[iid]:
                for g in subsets(it):
                    rows_by_group[g].append(per_item[iid][m])
        summary['likelihood'][m], summary['recall'][m] = {}, {}
        for g in group_names:
            if rows_by_group.get(g):
                ls, rs = lik_stats(rows_by_group[g]), recall_stats(rows_by_group[g])
                summary['likelihood'][m][g], summary['recall'][m][g] = ls, rs
                lik_rows.append(dict(model=m, family=fam, size_b=size, group=g, **ls))
                if rs['n']:
                    rec_rows.append(dict(model=m, family=fam, size_b=size, group=g, **rs))
        # paired: each fact against its own control (same repo, commit and kind of edit)
        summary['fact_vs_control'][m] = {}
        for subset in ['all facts', 'pilot', 'agent', 'agent:memory', 'agent:unseen', 'agent:seen', 'trap',
                       'all facts, name not in context', 'all facts, fact not in context']:
            pairs = []
            for iid, it in items.items():
                if it['group'] == 'control' or not it.get('pair_id') or it['pair_id'] not in items:
                    continue
                ctrl = items[it['pair_id']]
                if not subset.startswith('all facts') and subset not in subsets(it):
                    continue
                if subset.endswith('name not in context') and (it['name_in_context'] or ctrl['name_in_context']):
                    continue
                if subset.endswith('fact not in context') and (fact_in_context(it) or fact_in_context(ctrl)):
                    continue
                a, b = per_item[iid].get(m), per_item[ctrl['id']].get(m)
                if a and b:
                    pairs.append((a, b))
            if not pairs:
                continue
            diff = [a['delta_trap'] - b['delta_trap'] for a, b in pairs]
            lo, hi = boot_ci(diff)
            row = dict(n_pairs=len(pairs), mean_delta_fact=r3(mean([a['delta_trap'] for a, _ in pairs])),
                       mean_delta_control=r3(mean([b['delta_trap'] for _, b in pairs])), mean_diff=r3(mean(diff)),
                       ci_lo=r3(lo), ci_hi=r3(hi), pct_fact_gt_control=r3(mean([x > 0 for x in diff])))
            rp = [(a, b) for a, b in pairs if a.get('recall') and b.get('recall')]
            if rp:
                rd = [float(a['recall']['fact_correct']) - float(b['recall']['fact_correct']) for a, b in rp]
                rlo, rhi = boot_ci(rd)
                row.update(recall_fact=r3(mean([float(a['recall']['fact_correct']) for a, _ in rp])),
                           recall_control=r3(mean([float(b['recall']['fact_correct']) for _, b in rp])),
                           recall_diff=r3(mean(rd)), recall_ci_lo=r3(rlo), recall_ci_hi=r3(rhi))
            summary['fact_vs_control'][m][subset] = row
            pair_rows.append(dict(model=m, family=fam, size_b=size, subset=subset, **row))
        # by fact type, facts (pilot+agent+trap) vs controls
        summary['by_type'][m] = {}
        for ftype in FACT_TYPES:
            for side in ('fact', 'control'):
                rows = [per_item[iid][m] for iid, it in items.items() if it['fact_type'] == ftype and m in per_item[iid]
                        and (it['group'] == 'control') == (side == 'control')]
                if rows:
                    ls, rs = lik_stats(rows), recall_stats(rows)
                    summary['by_type'][m][f'{ftype}|{side}'] = dict(likelihood=ls, recall=rs)
                    type_rows.append(dict(model=m, family=fam, size_b=size, fact_type=ftype, side=side, n=ls['n'],
                                          mean_delta_trap=ls['mean_delta_trap'], ci_lo=ls['ci_lo'], ci_hi=ls['ci_hi'],
                                          pct_trap_less_likely=ls['pct_trap_less_likely'],
                                          mean_fact_pref=ls['mean_fact_pref'],
                                          recall_fact_correct=rs.get('fact_correct'), recall_full_exact=rs.get('full_exact')))
        # do the two measures agree? memory strength when the closed-book answer got the fact right vs wrong
        both = [per_item[iid][m] for iid in items if m in per_item[iid] and per_item[iid][m].get('recall')]
        right = [r['delta_trap'] for r in both if r['recall']['fact_correct']]
        wrong = [r['delta_trap'] for r in both if not r['recall']['fact_correct']]
        summary.setdefault('agreement', {})[m] = dict(
            n_recall_right=len(right), n_recall_wrong=len(wrong), mean_delta_when_right=r3(mean(right)),
            mean_delta_when_wrong=r3(mean(wrong)),
            pct_trap_less_likely_when_right=r3(mean([x > 0 for x in right])),
            pct_trap_less_likely_when_wrong=r3(mean([x > 0 for x in wrong])))
        # batched vs one-at-a-time greedy answers (same prompts; only numerics can differ)
        # (signatures only: the first runs used the prefix 'from ' for imports, the final runs 'from')
        same = [' '.join(per_item[iid][m]['answer'].split()) == ' '.join(per_item[iid][m]['answer_one_at_a_time'].split())
                for iid, it in items.items() if it['kind'] == 'signature' and m in per_item[iid]
                and 'answer' in per_item[iid][m] and 'answer_one_at_a_time' in per_item[iid][m]]
        if same:
            summary.setdefault('batched_vs_one_at_a_time', {})[m] = dict(n=len(same), identical=r3(mean(same)))
        # the two pilot traps, one row each
        summary['pilot'][m] = {}
        for iid, it in items.items():
            if it['group'] == 'pilot' and m in per_item[iid]:
                r = per_item[iid][m]
                row = dict(lp_original=r['lp_original'], lp_trapped=r['lp_trapped'], lp_placebo=r['lp_placebo'],
                           delta_trap=r['delta_trap'], delta_placebo=r['delta_placebo'], fact_pref=r['fact_pref'],
                           recall_answer=r.get('answer'), recall_fact_correct=(r.get('recall') or {}).get('fact_correct'))
                summary['pilot'][m][it['qualname']] = row
                pilot_rows.append(dict(model=m, family=fam, size_b=size, fact=it['fact_desc'], **row))
    # gradients: paired per-item differences between two models (same items)
    contrasts = [('qwen2.5-7b', 'qwen2.5-0.5b'), ('qwen2.5-7b', 'qwen2.5-1.5b'), ('qwen2.5-3b', 'qwen2.5-0.5b'),
                 ('qwen2.5-coder-7b', 'qwen2.5-7b'), ('qwen2.5-coder-1.5b', 'qwen2.5-1.5b'),
                 ('qwen2.5-coder-7b', 'qwen2.5-coder-1.5b'), ('qwen3-1.7b', 'qwen2.5-1.5b'),
                 ('r1-distill-1.5b', 'qwen2.5-1.5b')]
    for big, small in contrasts:
        if big not in models or small not in models:
            continue
        for g in ['agent', 'agent:memory', 'control', 'pilot', 'trap']:
            pairs = [(per_item[iid][big], per_item[iid][small]) for iid, it in items.items()
                     if g in subsets(it) and big in per_item[iid] and small in per_item[iid]]
            if not pairs:
                continue
            d = [a['delta_trap'] - b['delta_trap'] for a, b in pairs]
            lo, hi = boot_ci(d)
            row = dict(contrast=f'{big} minus {small}', group=g, n=len(pairs), mean_diff_delta_trap=r3(mean(d)),
                       ci_lo=r3(lo), ci_hi=r3(hi))
            rp = [(a, b) for a, b in pairs if a.get('recall') and b.get('recall')]
            if rp:
                rd = [float(a['recall']['fact_correct']) - float(b['recall']['fact_correct']) for a, b in rp]
                rlo, rhi = boot_ci(rd)
                row.update(mean_diff_recall=r3(mean(rd)), recall_ci_lo=r3(rlo), recall_ci_hi=r3(rhi))
            contrast_rows.append(row)
            summary['contrasts'][f'{big} minus {small}|{g}'] = row
    with open(OUT / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    for name, rows in [('fig_likelihood_by_model', lik_rows), ('fig_recall_by_model', rec_rows),
                       ('fig_fact_vs_control', pair_rows), ('fig_by_type', type_rows), ('fig_pilot', pilot_rows),
                       ('fig_gradient_contrasts', contrast_rows)]:
        if not rows:
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT / f'{name}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (str(v).replace('\n', '\\n') if isinstance(v, str) else v) for k, v in r.items()})
    print(f'wrote {OUT}/probes.jsonl, summary.json and fig_*.csv for {len(models)} models: {", ".join(models)}')
    print(markdown_tables(summary, models))


def fmt(x, pct=False):
    if x is None:
        return '-'
    return f'{100 * x:.0f}%' if pct else f'{x:.2f}'


def markdown_tables(summary, models):
    """The main result tables as markdown text."""
    out = []
    lik, rec, pairs = summary['likelihood'], summary['recall'], summary['fact_vs_control']
    out.append('\n### Memory strength by group (mean logp(original) - logp(trapped), nats; 95% bootstrap CI; '
               'share of items where the trapped line is less likely)\n')
    groups = ['pilot', 'agent:memory', 'agent:unseen', 'agent:seen', 'trap', 'control']
    out.append('| model | ' + ' | '.join(groups) + ' |')
    out.append('|---|' + '---|' * len(groups))
    for m in models:
        cells = []
        for g in groups:
            st = lik[m].get(g)
            cells.append('-' if not st else f"{fmt(st['mean_delta_trap'])} [{fmt(st['ci_lo'])}, {fmt(st['ci_hi'])}] "
                                            f"{fmt(st['pct_trap_less_likely'], True)} (n={st['n']})")
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    out.append('\n### Placebo check (mean logp(placebo) - logp(trapped), nats; share where the placebo beats the trap; '
               'mean cost of the placebo edit logp(original) - logp(placebo))\n')
    groups2 = ['agent', 'trap', 'control']
    out.append('| model | ' + ' | '.join(groups2) + ' |')
    out.append('|---|' + '---|' * len(groups2))
    for m in models:
        cells = []
        for g in groups2:
            st = lik[m].get(g)
            cells.append('-' if not st else f"{fmt(st['mean_fact_pref'])} [{fmt(st['fact_pref_ci_lo'])}, "
                                            f"{fmt(st['fact_pref_ci_hi'])}] {fmt(st['pct_placebo_beats_trap'], True)}; "
                                            f"placebo cost {fmt(st['mean_delta_placebo'])}")
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    out.append('\n### Each fact against its own control (same repo, commit, kind of edit): memory strength fact - control '
               '(95% CI), and closed-book recall accuracy fact vs control\n')
    subs = ['all facts', 'agent:memory', 'agent:unseen', 'agent:seen', 'trap', 'all facts, fact not in context']
    out.append('| model | ' + ' | '.join(subs) + ' |')
    out.append('|---|' + '---|' * len(subs))
    for m in models:
        cells = []
        for sb in subs:
            st = pairs[m].get(sb)
            if not st:
                cells.append('-')
                continue
            rc = (f"; recall {fmt(st.get('recall_fact'), True)} vs {fmt(st.get('recall_control'), True)}"
                  if 'recall_fact' in st else '')
            cells.append(f"{fmt(st['mean_diff'])} [{fmt(st['ci_lo'])}, {fmt(st['ci_hi'])}] (n={st['n_pairs']}){rc}")
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    out.append('\n### Closed-book recall (fact correct: right order / default / keyword present / module; '
               'full exact = every parameter name, order and default right)\n')
    groups3 = ['pilot', 'agent:memory', 'agent:unseen', 'agent:seen', 'trap', 'control']
    out.append('| model | ' + ' | '.join(groups3) + ' |')
    out.append('|---|' + '---|' * len(groups3))
    for m in models:
        cells = []
        for g in groups3:
            st = rec[m].get(g)
            cells.append('-' if not st or not st.get('n') else
                         f"{fmt(st['fact_correct'], True)} (exact {fmt(st['full_exact'], True)}, n={st['n']})")
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    out.append('\n### Gradients: paired per-item differences between models (memory strength, nats; recall, points)\n')
    out.append('| contrast | group | n | memory strength diff [95% CI] | recall diff [95% CI] |')
    out.append('|---|---|---|---|---|')
    for key, st in summary['contrasts'].items():
        c, g = key.split('|')
        rc = (f"{fmt(st['mean_diff_recall'])} [{fmt(st['recall_ci_lo'])}, {fmt(st['recall_ci_hi'])}]"
              if 'mean_diff_recall' in st else '-')
        out.append(f"| {c} | {g} | {st['n']} | {fmt(st['mean_diff_delta_trap'])} [{fmt(st['ci_lo'])}, "
                   f"{fmt(st['ci_hi'])}] | {rc} |")
    out.append('\n### Pilot traps (logp original / trapped / placebo; memory strength; recalled answer)\n')
    out.append('| model | fact | lp orig | lp trap | lp placebo | strength | recall answer | order right? |')
    out.append('|---|---|---|---|---|---|---|---|')
    for m in models:
        for q, st in summary['pilot'].get(m, {}).items():
            ans = (st['recall_answer'] or '-').strip().replace('|', '\\|').replace('\n', ' ')[:70]
            out.append(f"| {m} | {q} | {fmt(st['lp_original'])} | {fmt(st['lp_trapped'])} | {fmt(st['lp_placebo'])} | "
                       f"{fmt(st['delta_trap'])} | `{ans}` | {st['recall_fact_correct']} |")
    return '\n'.join(out)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'build':
        build(sys.argv[2:])
    elif cmd == 'run':
        for name in sys.argv[2:]:
            run_model(name)
    elif cmd == 'check':
        check(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        print(__doc__)
