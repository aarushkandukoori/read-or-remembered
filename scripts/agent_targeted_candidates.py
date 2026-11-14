"""Agent-targeted memory-trap candidates.

A trap only bites if the agent relies on the trapped fact WITHOUT reading it (pilot lesson: in
django-11149 the agent copied the swapped call it saw next to its edit; in sympy-13615 the agents
never used the trapped function). So we mine the agents' OWN patches for repo facts they relied on
but never looked at.

For every model x task in the public mini-swe-agent runs (data/raw):
  1. Take the agent's final submitted patch. Rebuild each edited source file in memory
     (file at the task's base commit + the patch). Agent scripts, backups and test files are skipped.
  2. On the lines the patch ADDS, find every use of something the repo already defined before the
     patch, outside the code the patch changed: functions, classes, methods, attributes, exceptions.
  3. Write down the fact the new code relies on:
       arg_order        f(a, b): calls it with 2+ positional arguments (relies on parameter order)
       default          f(a): leaves a parameter out (relies on its default value)
       import           from X import Y (relies on Y being importable from X)
       raises           try: f(...) except E (relies on f raising E)
       exception_class  raise E / except E with a repo exception class
       attribute        obj.m / obj.m(...) on a repo class or module (relies on that name)
       keyword          f(k=v) (relies on f having a parameter called k)
     Plus four "memory error" checks that are not trap candidates: keyword_not_in_signature,
     too_many_positional, import_unresolved, module_attr_unresolved (the patch uses something that
     does not exist at the base commit).
  4. Look at the run. Find the step where the agent first wrote that line, then check every tool
     output before it:
       unseen     no output showed the definition (its signature, `class E`, the import, the
                  region of the file holding it, or an inspect.signature/help() call on it)
       no_use     unseen, AND no output showed another use of the name, AND the line did not just
                  keep a call that was already on the line it replaced  <- the memory tier
       strong     no_use, AND the definition never shows up anywhere in the run
     Spot checks showed that "unseen" facts were usually copied from a visible use, so trap
     candidates should come from the memory tier (no_use / strong).

Subcommands:
  extract  -> outputs/agent_targeted_facts.jsonl (one row per model x task x fact, seen or not)
  report   -> outputs/agent_targeted_candidates.jsonl + notes/agent-targeted-candidates.md
Usage (from the project folder):
  .venv/bin/python -I scripts/agent_targeted_candidates.py extract [INSTANCE_ID ...]
  .venv/bin/python -I scripts/agent_targeted_candidates.py report
"""
import ast, bisect, builtins, json, os, re, subprocess, sys, warnings
from collections import Counter, defaultdict

import pyarrow.parquet as pq

warnings.simplefilter('ignore')  # old repo files trigger "invalid escape sequence" warnings when parsed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, 'data/meta/swebv/data/test-00000-of-00001.parquet')
REPOS = os.path.join(ROOT, 'data/repos')
RAW = os.path.join(ROOT, 'data/raw')
FACTS = os.path.join(ROOT, 'outputs/agent_targeted_facts.jsonl')
CANDS = os.path.join(ROOT, 'outputs/agent_targeted_candidates.jsonl')
NOTE = os.path.join(ROOT, 'notes/agent-targeted-candidates.md')
RUNS = {  # model -> run folder in data/raw
    'deepseek-v3.2': '20251201_mini-v1.17.1_deepseek-v3.2-reasoner',
    'kimi-k2.5': '20260217_mini-v2.0.0_kimi-k2-5-high',
    'minimax-m2.5': '20260217_mini-v2.0.0_minimax-2-5-high',
}
# git must never download a file by itself while we read (blobs were batch-downloaded beforehand
# with scripts/prefetch_blobs.py); a file that is not there just reads as "missing"
NO_FETCH = dict(os.environ, GIT_NO_LAZY_FETCH='1')

# Method names that also exist on Python built-in types, numpy and pandas objects. When we cannot
# tell what object `x` is in `x.name(...)`, we only guess "it is the repo's class" for names NOT in
# this list (x.append(...) is far more likely a list than a repo class).
COMMON_ATTRS = set()
for _t in (str, bytes, bytearray, list, dict, set, frozenset, tuple, int, float, complex, object, type,
           BaseException, range, slice, memoryview):
    COMMON_ATTRS.update(dir(_t))
import io, pathlib, collections, datetime, decimal, fractions, logging, threading, argparse  # noqa: E402
for _t in (io.StringIO, io.BytesIO, pathlib.Path, collections.OrderedDict, collections.defaultdict,
           collections.Counter, collections.deque, datetime.datetime, datetime.date, datetime.timedelta,
           decimal.Decimal, fractions.Fraction, logging.Logger, threading.Thread, argparse.ArgumentParser,
           argparse.Namespace, type(re.compile('')), type(re.match('', ''))):
    COMMON_ATTRS.update(dir(_t))
try:
    import numpy, pandas  # only used for their method names
    for _t in (numpy.ndarray, numpy.dtype, pandas.DataFrame, pandas.Series, pandas.Index):
        COMMON_ATTRS.update(dir(_t))
except ImportError:
    pass
# Common method names of big dependencies that are not installed here (docutils for sphinx, scipy.sparse
# for scikit-learn, pluggy/py for pytest). Seaborn tasks also exclude every matplotlib method name,
# read from the local matplotlib clone (see extract()).
COMMON_ATTRS.update('''astext traverse findall next_node walk walkabout deepcopy pformat setup_child replace_self
    first_child_matching_class first_child_not_matching_class non_default_attributes update_basic_atts
    append_attr_list replace_attr update_all_atts starttag endtag emptytag shortrepr set_class rawsource children
    parent document tagname attributes asdom ids names classes backrefs dupnames settings reporter
    note_explicit_target note_implicit_target note_refname note_refid set_id has_name nameids
    tocsr tocsc tocoo tolil todok todense toarray nnz getnnz eliminate_zeros sum_duplicates sorted_indices
    sort_indices has_sorted_indices indices indptr getformat asformat multiply power setdiag getrow getcol
    call_historic get_hookimpls register unregister add_hookspecs get_plugin is_registered get_result
    force_result bestrelpath relto dirpath basename purebasename strpath ensure listdir visit isdir isfile'''.split())
BUILTIN_NAMES = set(dir(builtins))
EXTERNAL = {'kind': 'external'}  # marker: refers to something outside the repo (stdlib, numpy, builtins)
NOT_PACKAGES = {'setup', 'docs', 'doc', 'examples', 'example', 'benchmarks', 'benchmark', 'asv_bench', 'tools',
                'ci', 'conftest', 'build_tools', 'maint_tools', 'scripts', 'bin', 'release', 'galleries',
                'tutorials', 'plot_types', 'setupext', 'extra', 'util', 'utils'}


# ---------------------------------------------------------------- reading the repos ----------

def is_test_path(path):
    """True for files in test folders, test_*.py files and conftest.py."""
    parts = path.split('/')
    return (any(p in ('tests', 'test', 'testing') for p in parts[:-1])
            or parts[-1].startswith('test_') or parts[-1] == 'conftest.py')


class Git:
    """Folder listings and file contents of one blobless clone, read without downloading."""

    def __init__(self, path):
        self.path = path
        self.cat = subprocess.Popen(['git', '-C', path, 'cat-file', '--batch'], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, env=NO_FETCH)

    def tree(self, commit):
        """{path: blob id} for every file at a commit."""
        out = subprocess.run(['git', '-C', self.path, 'ls-tree', '-r', '-z', commit],
                             capture_output=True, text=True).stdout
        files = {}
        for entry in out.split('\0'):
            if '\t' in entry:
                meta, path = entry.split('\t', 1)
                mode, typ, oid = meta.split()
                if typ == 'blob':
                    files[path] = oid
        return files

    def read(self, oid):
        """Text of a file, or None if it was never downloaded."""
        self.cat.stdin.write(oid.encode() + b'\n')
        self.cat.stdin.flush()
        header = self.cat.stdout.readline().split()
        if len(header) < 3:
            return None  # "<id> missing"
        data = self.cat.stdout.read(int(header[2]))
        self.cat.stdout.read(1)  # newline after the contents
        return data.decode('utf-8', errors='replace')


def module_name(path):
    """django/db/models/query.py -> django.db.models.query; lib/matplotlib/axes/__init__.py -> matplotlib.axes"""
    p = path[:-3]
    for prefix in ('src/', 'lib/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    if p.endswith('/__init__') or p == '__init__':
        p = p[:-len('/__init__')]
    return p.replace('/', '.')


def absolute_module(path, level, module):
    """Turn a relative import (from ..x import y, inside `path`) into an absolute module name."""
    if not level:
        return module or ''
    mod = module_name(path)
    pkg = mod if path.endswith('__init__.py') else mod.rpartition('.')[0]
    parts = pkg.split('.') if pkg else []
    if level > 1:
        parts = parts[:max(0, len(parts) - (level - 1))]
    return '.'.join(parts + ([module] if module else []))


# ------------------------------------------------- what each source file defines ------------

def literal(node):
    """repr of a plain default value (None, True, 3, 'x', -1), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (type(None), bool, int, float, str)):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, (int, float)):
        return '-' + repr(node.operand.value)
    return None


def deco_name(d):
    d = d.func if isinstance(d, ast.Call) else d
    return d.id if isinstance(d, ast.Name) else d.attr if isinstance(d, ast.Attribute) else ''


def signature(fn):
    """Parameters of a def: positional names in order, keyword-only names, defaults, *args/**kwargs."""
    a = fn.args
    pos = [p.arg for p in a.posonlyargs + a.args]
    defaults = {}
    for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        defaults[p] = literal(d)  # None here means "has a default, but not a plain value"
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            defaults[p.arg] = literal(d)
    return dict(pos=pos, kwonly=[p.arg for p in a.kwonlyargs], defaults=defaults, vararg=bool(a.vararg),
                kwarg=bool(a.kwarg), decos=[deco_name(d) for d in fn.decorator_list],
                line=fn.lineno, end=fn.end_lineno)


def summarize_class(node):
    methods, attrs = {}, set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[item.name] = signature(item)
            for n in ast.walk(item):  # instance attributes: self.x = ... anywhere in a method
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        for x in ast.walk(t):
                            if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == 'self':
                                attrs.add(x.attr)
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            for t in (item.targets if isinstance(item, ast.Assign) else [item.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        attrs.add(x.id)
        elif isinstance(item, ast.ClassDef):
            attrs.add(item.name)
    bases = [ast.unparse(b) for b in node.bases]
    decos = [deco_name(d) for d in node.decorator_list]
    meta = (any(k.arg == 'metaclass' for k in node.keywords) or 'add_metaclass' in decos
            or any('with_metaclass' in b for b in bases))
    return dict(bases=bases, methods=methods, attrs=sorted(attrs), decos=decos, meta=meta,
                line=node.lineno, end=node.end_lineno)


def summarize(text):
    """Top-level functions, classes, variables and imports of one module (None if it does not parse)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    s = dict(funcs={}, classes={}, names=set(), imports={}, stars=[])

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s['funcs'][node.name] = signature(node)
            elif isinstance(node, ast.ClassDef):
                s['classes'][node.name] = summarize_class(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            s['names'].add(x.id)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == '*':
                        s['stars'].append((node.level, node.module))
                    else:
                        s['imports'][a.asname or a.name] = ('from', node.level, node.module, a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        s['imports'][a.asname] = ('import', 0, a.name, None)
                    else:  # "import a.b.c" binds the name "a"
                        top = a.name.split('.')[0]
                        s['imports'].setdefault(top, ('import', 0, top, None))
            elif isinstance(node, ast.If):  # e.g. if PY3: def f(): ...
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):  # e.g. try: from x import y / except ImportError: ...
                visit(node.body)
                for h in node.handlers:
                    visit(h.body)
                visit(node.orelse)
                visit(node.finalbody)
    visit(tree.body)
    return s


class CommitIndex:
    """Every definition in the repo's source files at one commit, with lookups that follow imports."""

    def __init__(self, git, commit, cache):
        self.git = git
        self.tree = git.tree(commit)
        self.files = {}
        for path, oid in self.tree.items():
            if path.endswith('.py') and not is_test_path(path):
                if oid not in cache:
                    text = git.read(oid)
                    cache[oid] = summarize(text) if text is not None else None
                if cache[oid]:
                    self.files[path] = cache[oid]
        self.modules = {}
        for p in sorted(self.files, key=len):
            self.modules.setdefault(module_name(p), p)
        self.packages = {m.split('.')[0] for m in self.modules} - NOT_PACKAGES
        self._members = None
        self._sources = {}
        self.extra_common = set()  # extra method names to never guess for (a dependency's API)
        self._ctor_map = None

    def source(self, path):
        """Text of a source file at this commit."""
        if path not in self._sources:
            self._sources[path] = self.git.read(self.tree[path]) if path in self.tree else None
        return self._sources[path]

    def raises_in_body(self, t, exc):
        """True if the body of function/method `t` has an explicit `raise exc` (or raise module.exc)."""
        text = self.source(t['path'])
        if not text or not t.get('line'):
            return False
        body = '\n'.join(text.split('\n')[t['line'] - 1:t.get('end') or t['line']])
        return re.search(rf'\braise\s+(?:[\w.]+\.)?{re.escape(exc)}\b', body) is not None

    def is_repo_module(self, module):
        return bool(module) and module.split('.')[0] in self.packages

    def lookup(self, module, name, depth=0):
        """Where is `name`, imported from `module`, really defined? Follows re-exports in __init__.py.
        Returns a target dict (kind function/class/module/name) or None."""
        if depth > 6:
            return None
        path = self.modules.get(module)
        sub = module + '.' + name
        if path is None:
            return target('module', self.modules[sub], sub) if sub in self.modules else None
        s = self.files[path]
        if name in s['funcs']:
            return target('function', path, name, s['funcs'][name])
        if name in s['classes']:
            return target('class', path, name, line=s['classes'][name]['line'])
        if name in s['imports']:
            kind, level, mod, orig = s['imports'][name]
            absmod = absolute_module(path, level, mod) if kind == 'from' else mod
            if not self.is_repo_module(absmod):
                return target('reexport', path, name)  # e.g. requests.compat re-exporting urlparse
            if kind == 'from':
                r = self.lookup(absmod, orig, depth + 1)
                if r:
                    return r
            elif mod in self.modules:
                return target('module', self.modules[mod], mod)
        if name in s['names']:
            return target('name', path, name)
        if sub in self.modules:
            return target('module', self.modules[sub], sub)
        for level, mod in s['stars']:
            absmod = absolute_module(path, level, mod)
            if not self.is_repo_module(absmod):
                return target('reexport', path, name)  # from numpy import *: cannot tell, assume it exists
            r = self.lookup(absmod, name, depth + 1)
            if r:
                return r
        return None

    def dynamic(self, module):
        """True if a module can create names at run time: it writes to globals(), or has a module-level
        __getattr__ (PEP 562)."""
        path = self.modules.get(module)
        text = self.source(path) if path else None
        return bool(text) and ('globals()' in text or '__getattr__' in self.files[path]['funcs'])

    def resolve_in_file(self, path, dotted):
        """Resolve a name written in file `path` (like a base class 'models.Model') to a target."""
        head, *rest = dotted.split('.')
        s = self.files.get(path)
        if s is None:
            return None
        if head in s['classes']:
            r = target('class', path, head, line=s['classes'][head]['line'])
        elif head in s['funcs']:
            r = target('function', path, head, s['funcs'][head])
        elif head in s['imports']:
            kind, level, mod, orig = s['imports'][head]
            if kind == 'from':
                r = self.lookup(absolute_module(path, level, mod), orig)
            else:
                r = target('module', self.modules[mod], mod) if mod in self.modules else None
        else:
            r = None
        for part in rest:
            if r is None or r['kind'] != 'module':
                return None
            r = self.lookup(r['qual'], part)
        return r

    def member(self, path, cname, attr, depth=0):
        """Find method/attribute `attr` on class `cname` (defined in `path`), searching repo base classes.
        Returns a target dict (kind method or attr) or None."""
        if depth > 8:
            return None
        c = self.files.get(path, {}).get('classes', {}).get(cname)
        if not c:
            return None
        if attr in c['methods']:
            return target('method', path, f'{cname}.{attr}', c['methods'][attr], name=attr)
        if attr in c['attrs']:
            return target('attr', path, f'{cname}.{attr}', line=c['line'], name=attr)
        for b in c['bases']:
            r = self.resolve_in_file(path, b)
            if r and r['kind'] == 'class':
                m = self.member(r['path'], r['name'], attr, depth + 1)
                if m:
                    return m
        return None

    def constructor(self, path, cname, depth=0):
        """The __init__ (or, if a class has none, __new__) that Foo(...) runs, searching repo base classes."""
        c = self.files.get(path, {}).get('classes', {}).get(cname)
        if not c or depth > 8 or 'dataclass' in c['decos'] or c.get('meta'):
            return None
        for m in ('__init__', '__new__'):
            if m in c['methods']:
                return target('method', path, f'{cname}.{m}', c['methods'][m], name=m)
        for b in c['bases']:
            r = self.resolve_in_file(path, b)
            if r and r['kind'] == 'class':
                t = self.constructor(r['path'], r['name'], depth + 1)
                if t:
                    return t
        return None

    def classes_by_constructor(self):
        """(file, 'Base.__init__') -> names of every class whose Foo(...) call runs that constructor."""
        if self._ctor_map is None:
            m = defaultdict(set)
            for path, s in self.files.items():
                for cname in s['classes']:
                    t = self.constructor(path, cname)
                    if t:
                        m[(t['path'], t['qual'])].add(cname)
            self._ctor_map = m
        return self._ctor_map

    def members_index(self):
        """name -> every repo class that defines a method with that name, plus every class that has
        a plain attribute with that name (built once per commit, only if needed)."""
        if self._members is None:
            idx = defaultdict(list)
            for path, s in self.files.items():
                for cname, c in s['classes'].items():
                    for m, sig in c['methods'].items():
                        idx[m].append(target('method', path, f'{cname}.{m}', sig, name=m))
                    for a in c['attrs']:
                        if a not in c['methods']:
                            idx[a].append(target('attr', path, f'{cname}.{a}', line=c['line'], name=a))
            self._members = idx
        return self._members

    def by_unique_name(self, attr):
        """Guess the target of x.attr when we do not know what x is: only if exactly one repo class
        defines `attr` (or up to 10 classes define it with the identical parameter list)."""
        if attr.startswith('__') or attr in COMMON_ATTRS or attr in self.extra_common or len(attr) < 3:
            return None
        sites = self.members_index().get(attr, [])
        # plain attributes are often set in ways we cannot see (setattr, metaclasses, Django's _meta),
        # so a guess is only made for methods
        if len(sites) == 1 and sites[0]['kind'] == 'method':
            return dict(sites[0], how='unique_name', n_sites=1)
        if 1 < len(sites) <= 10 and all(t['kind'] == 'method' for t in sites):
            if len({json.dumps(t['sig']['pos']) + json.dumps(t['sig']['defaults']) for t in sites}) == 1:
                return dict(sites[0], how='same_signature', n_sites=len(sites), qual=f'<{len(sites)} classes>.{attr}')
        return None


def target(kind, path, qual, sig=None, line=None, name=None):
    """A repo definition: kind (function/class/method/attr/module/name), file, qualified name, signature."""
    return dict(kind=kind, path=path, qual=qual, name=name or qual.split('.')[-1], sig=sig,
                line=line if line is not None else (sig['line'] if sig else None),
                end=sig['end'] if sig else None, how=None, n_sites=1)


# --------------------------------------------------------------- patches ---------------------

def parse_diff(text):
    """Split a unified diff into {path: [(old_start, old_len, new_start, new_len, lines)]}.
    Also returns the set of files the diff creates."""
    files, new_files = {}, set()
    lines = text.split('\n')
    i, cur, old_path = 0, None, None
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            cur, old_path = None, None
        elif line.startswith('--- '):
            old_path, cur = line[4:].split('\t')[0].strip(), None
        elif line.startswith('+++ ') and old_path is not None:
            p = line[4:].split('\t')[0].strip()
            if p != '/dev/null':
                cur = p[2:] if p.startswith('b/') else p
                files[cur] = []
                if old_path == '/dev/null':
                    new_files.add(cur)
        elif line.startswith('@@') and cur is not None:
            m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if m:
                os_, ol = int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 1
                ns, nl = int(m.group(3)), int(m.group(4)) if m.group(4) is not None else 1
                body, o, n = [], ol, nl
                i += 1
                while i < len(lines) and (o > 0 or n > 0):  # read exactly the lines the header promises
                    l = lines[i]
                    if l.startswith('\\'):  # "\ No newline at end of file"
                        i += 1
                        continue
                    tag = l[:1] or ' '
                    if tag == ' ':
                        o, n = o - 1, n - 1
                    elif tag == '-':
                        o -= 1
                    elif tag == '+':
                        n -= 1
                    else:
                        break
                    body.append(tag + l[1:])
                    i += 1
                files[cur].append((os_, ol, ns, nl, body))
                continue
        i += 1
    return files, new_files


def apply_hunks(base_text, hunks):
    """Rebuild the patched file. Returns (new_text, added, removed, inserts) or None if it does not fit.
    added   line numbers (new file) of added lines
    removed line numbers (base file) of removed lines
    inserts base line numbers p where new lines were inserted between line p and p+1"""
    old = base_text.split('\n')
    new, added, removed, inserts, pos = [], set(), set(), set(), 0
    hunk_of, removed_text = {}, []  # added line -> hunk number; text each hunk removed
    for h, (os_, ol, ns, nl, body) in enumerate(hunks):
        removed_text.append('\n'.join(l[1:] for l in body if l.startswith('-')))
        start = os_ - 1 if ol > 0 else os_
        if start < pos or start > len(old):
            return None
        new.extend(old[pos:start])
        pos = start
        for l in body:
            tag, text = l[0], l[1:]
            if tag in ' -':
                if pos >= len(old) or old[pos].rstrip('\r') != text.rstrip('\r'):
                    return None
                if tag == ' ':
                    new.append(old[pos])
                else:
                    removed.add(pos + 1)
                pos += 1
            else:
                new.append(text)
                added.add(len(new))
                hunk_of[len(new)] = h
                inserts.add(pos)
    new.extend(old[pos:])
    return '\n'.join(new), added, removed, inserts, hunk_of, removed_text


# ------------------------------------------------------------ trajectories --------------------

def load_steps(traj):
    """Split a run into steps. Step 0 is the task text; step k (1..N) is the agent's k-th message.
    Returns a list of dicts: cmd (commands it wrote), think (reasoning + visible text), obs (tool output
    it got back). The final submission (the diff) is not counted as an observation."""
    msgs = traj['messages']
    submission = ((traj.get('info') or {}).get('submission') or '').strip()
    task = next((m.get('content') for m in msgs if m.get('role') == 'user'), '') or ''
    steps = [dict(cmd='', think='', obs=str(task))]
    for m in msgs:
        role = m.get('role')
        content = m.get('content') or ''
        if isinstance(content, list):
            content = '\n'.join(c.get('text', '') for c in content if isinstance(c, dict))
        if role == 'assistant':
            resp = (m.get('extra') or {}).get('response') or {}
            msg = ((resp.get('choices') or [{}])[0].get('message') or {}) if isinstance(resp, dict) else {}
            reasoning = m.get('reasoning') or msg.get('reasoning_content') or msg.get('reasoning') or ''
            cmds = re.findall(r'```(?:bash|sh)?\n(.*?)```', content, re.S)
            for tc in (m.get('tool_calls') or msg.get('tool_calls') or []):
                try:
                    cmds.append(json.loads(tc['function']['arguments']).get('command', ''))
                except Exception:
                    pass
            steps.append(dict(cmd='\n'.join(cmds), think=str(reasoning) + '\n' + content, obs=''))
        elif len(steps) > 1 and role != 'exit' and m is not msgs[1]:
            if content.strip() and content.strip() == submission:
                continue  # the final diff is the agent's own patch, not something it read
            steps[-1]['obs'] += '\n' + content
    return steps


class Run:
    """Fast "first step where X appears" searches over one trajectory."""

    def __init__(self, steps):
        self.steps = steps
        self.n = len(steps) - 1
        self.obs, self.obs_starts = self._join([s['obs'] for s in steps])
        self.agent, self.agent_starts = self._join([s['think'] + '\n' + s['cmd'] for s in steps])
        self.cmds, self.cmd_starts = self._join([s['cmd'] for s in steps])
        self.cmd_norm = [norm(s['cmd']) for s in steps]
        self.cache = {}

    @staticmethod
    def _join(texts):
        starts, parts, pos = [], [], 0
        for t in texts:
            starts.append(pos)
            parts.append(t)
            pos += len(t) + 1
        return '\n'.join(parts), starts

    def first(self, pattern, where='obs', flags=re.S | re.M):
        """First step whose tool output (or, with where='agent', whose own text) matches the regex."""
        key = (pattern, where)
        if key not in self.cache:
            text, starts = {'obs': (self.obs, self.obs_starts), 'agent': (self.agent, self.agent_starts),
                            'cmd': (self.cmds, self.cmd_starts)}[where]
            m = re.search(pattern, text, flags)
            self.cache[key] = bisect.bisect_right(starts, m.start()) - 1 if m else None
        return self.cache[key]

    def write_step(self, line_text, segment):
        """First step whose command contains this added line (whitespace and backslashes ignored).
        Falls back to the code segment (e.g. the call text). None if never found."""
        for key in (norm(line_text), norm(segment or '')):
            if len(key) >= 10:
                for k in range(1, len(self.steps)):
                    if key in self.cmd_norm[k]:
                        return k
        return None


def norm(s):
    return re.sub(r'[\s\\]', '', s)


# ---------------------------------------------------------- facts in a patch ------------------

class EditContext:
    """Name lookups from inside one patched file, backed by the base-commit index."""

    def __init__(self, index, path, new_summary, touched):
        self.index, self.path = index, path
        self.base = index.files.get(path)
        self.new = new_summary
        self.touched = touched  # {path: (removed lines, insert points)} for every file the patch edits

    def untouched(self, t):
        """True if the patch did not change the definition `t` (only matters for edited files)."""
        if t is None or t['path'] not in self.touched or t.get('line') is None:
            return True
        removed, inserts = self.touched[t['path']]
        a, b = t['line'], t.get('end') or t['line']
        return not any(a <= r <= b for r in removed) and not any(a <= p < b for p in inserts)

    def resolve_name(self, name):
        """What a bare name in the patched file refers to."""
        b, n = self.base or dict(funcs={}, classes={}), self.new
        if name in b['funcs'] and name in n['funcs']:
            t = dict(target('function', self.path, name, b['funcs'][name]), how='same_file')
            return t if self.untouched(t) else None
        if name in b['classes'] and name in n['classes']:
            t = dict(target('class', self.path, name, line=b['classes'][name]['line']), how='same_file')
            return t if self.untouched(dict(t, end=t['line'])) else None
        if name in n['funcs'] or name in n['classes']:
            return None  # defined by the patch itself
        if name in n['imports']:
            kind, level, mod, orig = n['imports'][name]
            if kind == 'from':
                absmod = absolute_module(self.path, level, mod)
                if not self.index.is_repo_module(absmod):
                    return EXTERNAL
                r = self.index.lookup(absmod, orig)
            else:
                if not self.index.is_repo_module(mod):
                    return EXTERNAL
                r = target('module', self.index.modules[mod], mod) if mod in self.index.modules else None
            if r and r['kind'] == 'reexport':
                return EXTERNAL
            return dict(r, how='import') if r and self.untouched(r) else None
        if name in BUILTIN_NAMES:
            return EXTERNAL
        return None  # local variable, parameter, star import, ...

    def resolve_expr(self, expr):
        """Target for a receiver expression like `models.Field` or `np`; EXTERNAL; or None (unknown)."""
        if isinstance(expr, ast.Name):
            return None if expr.id in ('self', 'cls') else self.resolve_name(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self.resolve_expr(expr.value)
            if base is EXTERNAL:
                return EXTERNAL
            if base is None:
                return None
            if base['kind'] == 'module':
                r = self.index.lookup(base['qual'], expr.attr)
                if r and r['kind'] == 'reexport':
                    return EXTERNAL
                return dict(r, how='module') if r else None
            if base['kind'] == 'class':
                r = self.index.member(base['path'], base['name'], expr.attr)
                return dict(r, how='class') if r else None
            return None
        if isinstance(expr, (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp,
                             ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.BinOp)):
            return EXTERNAL
        return None

    def member_of_class(self, cname, attr, skip_own=False):
        """self.attr inside class `cname` of the patched file (skip_own: start at the base classes, for super())."""
        newc = self.new['classes'].get(cname)
        basec = (self.base or {}).get('classes', {}).get(cname)
        if newc and attr in newc['methods'] and not (basec and attr in basec['methods']) and not skip_own:
            return None  # the patch added this method itself
        if basec and not skip_own:
            r = self.index.member(self.path, cname, attr)
        else:
            r = None
            for b in (newc or basec or {}).get('bases', []):
                c = self.index.resolve_in_file(self.path, b)
                if c and c['kind'] == 'class':
                    r = self.index.member(c['path'], c['name'], attr)
                    if r:
                        break
        if r and self.untouched(r):
            return dict(r, how='super' if skip_own else 'self')
        return None


class FactFinder(ast.NodeVisitor):
    """Collects facts from the added lines of one patched file."""

    def __init__(self, ctx, added, text, new_modules, patch_defs):
        self.ctx, self.added, self.text, self.lines = ctx, added, text, text.split('\n')
        self.new_modules = new_modules  # modules the patch itself creates (imports from them are not facts)
        self.patch_defs = patch_defs  # {module: summary of the patched file} for every file the patch edits
        self.classes = []
        self.facts = []
        self.done_attr = set()  # Attribute nodes already handled as a call

    def add(self, ftype, node, t, detail, key, line=None, params=None, use='call', def_name=None, use_name=None,
            extra=None):
        extra = dict(extra or {})
        line = line or node.lineno
        self.facts.append(dict(
            fact_type=ftype, line=line, code=self.lines[line - 1].strip()[:200],
            segment=(ast.get_source_segment(self.text, node) or '')[:300],
            target_kind=t['kind'], target_qual=t['qual'], target_file=t['path'], target_line=t.get('line'),
            target_end=t.get('end'),
            how=t.get('how'), n_sites=t.get('n_sites', 1), detail=detail,
            fact_key=f"{ftype}|{t['path']}|{t['qual']}|{key}", params=params, use=use,
            def_name=def_name or t['name'], use_name=use_name or t['name'], **(extra or {})))

    @staticmethod
    def merge(names, more=None):
        """Keyword arguments for add(): names plus any extra fields."""
        out = {k: v for k, v in names.items() if k != 'extra'}
        out['extra'] = dict(names.get('extra') or {}, **(more or {}))
        return out

    def patch_defines(self, module, name):
        """True if the patch itself creates module.name (new module, or a name added to an edited file)."""
        if f'{module}.{name}' in self.new_modules or module in self.new_modules:
            return True
        s = self.patch_defs.get(module)
        return bool(s) and (name in s['funcs'] or name in s['classes'] or name in s['names'] or name in s['imports'])

    # --- scopes
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    # --- what does a call / attribute refer to?
    def callee(self, func):
        """(target, bound): bound=True when the first parameter (self/cls) is filled automatically."""
        if isinstance(func, ast.Name):
            return self.ctx.resolve_name(func.id), False
        if not isinstance(func, ast.Attribute):
            return None, False
        recv = func.value
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name) and recv.func.id == 'super':
            return (self.ctx.member_of_class(self.classes[-1], func.attr, skip_own=True) if self.classes else None), True
        if isinstance(recv, ast.Name) and recv.id in ('self', 'cls') and self.classes:
            return self.ctx.member_of_class(self.classes[-1], func.attr), True
        r = self.ctx.resolve_expr(recv)
        if r is EXTERNAL:
            return None, False
        if r is not None and r['kind'] == 'module':
            t = self.ctx.index.lookup(r['qual'], func.attr)
            if t is None:
                if self.patch_defines(r['qual'], func.attr) or self.ctx.index.dynamic(r['qual']):
                    return None, False
                return dict(kind='missing_module_attr', module=r['qual'], attr=func.attr), False
            if t['kind'] == 'reexport':
                return None, False
            return (dict(t, how='module') if self.ctx.untouched(t) else None), False
        if r is not None and r['kind'] == 'class':
            t = self.ctx.index.member(r['path'], r['name'], func.attr)
            return (dict(t, how='class') if t else None), False
        if isinstance(recv, ast.Name) and recv.id in ('self', 'cls'):
            return None, False  # self outside a class
        t = self.ctx.index.by_unique_name(func.attr)
        return (t if t and self.ctx.untouched(t) else None), True

    def signature_for(self, t, bound):
        """(parameter names a caller fills by position, signature, owner target) or None."""
        if t['kind'] == 'function' and t['sig']:
            return t['sig']['pos'], t['sig'], t
        if t['kind'] == 'method' and t['sig']:
            s = t['sig']
            if {'property', 'cached_property'} & set(s['decos']):
                return None
            if 'staticmethod' in s['decos']:
                return s['pos'], s, t
            if bound or 'classmethod' in s['decos']:
                return s['pos'][1:], s, t
            return None  # plain method called through its class (Foo.bar(obj, ...)): skip
        if t['kind'] == 'class':
            init = self.ctx.index.constructor(t['path'], t['name'])
            if init and init['sig'] and self.ctx.untouched(init):
                return init['sig']['pos'][1:], init['sig'], dict(init, how=t['how'])
        return None

    def visit_Call(self, node):
        if node.lineno in self.added:
            self.handle_call(node)
        else:  # an existing call: the patch may still add keyword arguments to it
            for kw in node.keywords:
                if kw.arg and getattr(kw, 'lineno', None) in self.added:
                    self.handle_call(node, only_keyword=kw)
        self.generic_visit(node)

    def handle_call(self, node, only_keyword=None):
        t, bound = self.callee(node.func)
        if t is None or t is EXTERNAL:
            return
        if t['kind'] == 'missing_module_attr':
            if not only_keyword:
                self.facts.append(dict(fact_type='module_attr_unresolved', line=node.lineno,
                                       code=self.lines[node.lineno - 1].strip()[:200],
                                       detail=f"{t['module']} has no {t['attr']} at the base commit",
                                       fact_key=f"module_attr_unresolved|{t['module']}|{t['attr']}"))
            return
        is_attr = isinstance(node.func, ast.Attribute)
        if is_attr:
            self.done_attr.add(id(node.func))
        if is_attr and not only_keyword and t['kind'] in ('method', 'attr', 'function', 'class'):
            where = t['qual'] if t['kind'] in ('method', 'attr') else f"{module_name(t['path'])}.{t['qual']}"
            self.add('attribute', node, t, f"{where} exists (code calls .{node.func.attr}(...))", t['qual'],
                     use='attr')
        sig = self.signature_for(t, bound)
        if sig is None:
            return
        params, s, owner = sig
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            return  # f(*args) / f(**kwargs): we cannot tell which parameters are filled
        ctor = t['kind'] == 'class'
        names = dict(def_name=owner['name'], use_name=t['name'],
                     use='ctor' if ctor else ('method' if is_attr else 'call'))
        if ctor and owner['qual'].split('.')[0] != t['name']:
            # Foo(...) runs an inherited Base.__init__: a call to any class sharing it shows the same signature
            sib = sorted(self.ctx.index.classes_by_constructor().get((owner['path'], owner['qual']), ()))
            names['extra'] = dict(use_alt=sib[:300])
        label = f"{t['qual']}(...)" if ctor else owner['qual']
        npos = len(node.args)
        for kw in ([only_keyword] if only_keyword else node.keywords):
            if kw.arg in params or kw.arg in s['kwonly']:
                self.add('keyword', node, owner, f'{label} accepts keyword {kw.arg}=', kw.arg,
                         line=getattr(kw, 'lineno', None), params=[kw.arg], **self.merge(names))
            elif not s['kwarg']:
                self.add('keyword_not_in_signature', node, owner, f'{label} has no parameter {kw.arg}', kw.arg,
                         line=getattr(kw, 'lineno', None), params=[kw.arg], **self.merge(names))
        if only_keyword:
            return
        if npos >= 2 and len(params) >= 2:
            self.add('arg_order', node, owner, f'{label} takes ({params[0]}, {params[1]}, ...) in this order; '
                     f'code passes {npos} positional args', f'{params[0]},{params[1]}', params=params[:2],
                     **self.merge(names))
        if npos > len(params) and not s['vararg']:
            self.add('too_many_positional', node, owner, f'{label} takes {len(params)} positional args, '
                     f'code passes {npos}', str(npos), params=params, **self.merge(names))
        passed = set(params[:npos]) | {k.arg for k in node.keywords}
        for p, d in s['defaults'].items():
            if p not in passed and d is not None and (p in params[npos:] or p in s['kwonly']):
                self.add('default', node, owner, f'{label} leaves out {p}, relies on default {p}={d}', p, params=[p],
                         **self.merge(names, dict(default=d, n_defaults=len(s['defaults']))))

    def visit_Attribute(self, node):
        if node.lineno in self.added and id(node) not in self.done_attr and isinstance(node.ctx, ast.Load):
            recv = node.value
            t = None
            if isinstance(recv, ast.Name) and recv.id in ('self', 'cls') and self.classes:
                t = self.ctx.member_of_class(self.classes[-1], node.attr)
            else:
                r = self.ctx.resolve_expr(recv)
                if r is not None and r is not EXTERNAL and r['kind'] == 'module':
                    t = self.ctx.index.lookup(r['qual'], node.attr)
                    t = dict(t, how='module') if t and t['kind'] not in ('module', 'reexport') else None
                elif r is not None and r is not EXTERNAL and r['kind'] == 'class':
                    t = self.ctx.index.member(r['path'], r['name'], node.attr)
                    t = dict(t, how='class') if t else None
                elif r is None and not (isinstance(recv, ast.Name) and recv.id in ('self', 'cls')):
                    t = self.ctx.index.by_unique_name(node.attr)
                    t = t if t and self.ctx.untouched(t) else None
            if t:
                where = t['qual'] if t['kind'] in ('method', 'attr') else f"{module_name(t['path'])}.{t['qual']}"
                self.add('attribute', node, t, f'{where} exists (code reads .{node.attr})', t['qual'], use='attr')
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for a in node.names:
            line = getattr(a, 'lineno', node.lineno)
            if line not in self.added or a.name == '*':
                continue
            mod = absolute_module(self.ctx.path, node.level, node.module)
            if not self.ctx.index.is_repo_module(mod):
                continue
            t = self.ctx.index.lookup(mod, a.name)
            written = f"from {'.' * node.level}{node.module or ''} import {a.name}"
            if t is None and (self.patch_defines(mod, a.name) or self.ctx.index.dynamic(mod)):
                continue
            if t is not None and (t['kind'] == 'reexport' or not self.ctx.untouched(t)):
                continue
            if t is None:
                self.facts.append(dict(fact_type='import_unresolved', line=line, code=self.lines[line - 1].strip()[:200],
                                       detail=f'{written}: {a.name} not found in {mod} at the base commit',
                                       fact_key=f'import_unresolved|{mod}|{a.name}'))
                continue
            self.add('import', node, dict(t, how='import'), f'{a.name} importable from {mod} ({written})',
                     f'{mod}:{a.name}', line=line, use='import', extra=dict(module=mod, imported=a.name))
        self.generic_visit(node)

    def visit_Raise(self, node):
        if node.lineno in self.added and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            t = self.exception_target(exc)
            if t:
                self.add('exception_class', node, t, f"raises repo exception {t['qual']}", 'raise', use='exc')
        self.generic_visit(node)

    def exception_target(self, expr):
        t = self.ctx.resolve_name(expr.id) if isinstance(expr, ast.Name) else \
            self.ctx.resolve_expr(expr) if isinstance(expr, ast.Attribute) else None
        return t if t and t is not EXTERNAL and t['kind'] == 'class' else None

    def visit_Try(self, node):
        for h in node.handlers:
            if h.lineno not in self.added or h.type is None:
                continue
            types = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
            for ty in types:
                name = ast.unparse(ty)
                if name in ('Exception', 'BaseException'):
                    continue
                t = self.exception_target(ty)
                if t:
                    self.add('exception_class', h, t, f"catches repo exception {t['qual']}", 'catch', line=h.lineno,
                             use='exc')
                if t is None and name not in BUILTIN_NAMES:
                    continue  # not a builtin and not resolved: skip
                exc = name.split('.')[-1]
                for call in calls_in(node.body):
                    c, bound = self.callee(call.func)
                    # only a real fact if the called repo function itself raises it (a defensive
                    # "except (TypeError, KeyError)" around code that never raises them is not)
                    if c and c is not EXTERNAL and c['kind'] in ('function', 'method') \
                            and self.ctx.index.raises_in_body(c, exc):
                        self.add('raises', call, c, f"{c['qual']}(...) raises {exc} (code catches it)", exc,
                                 line=h.lineno, use='raises', extra=dict(exception=exc, call_line=call.lineno))
        self.generic_visit(node)

    visit_TryStar = visit_Try


def calls_in(body):
    """Calls inside a try body, not looking into nested functions or classes."""
    out, todo = [], list(body)
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Call):
            out.append(n)
        todo.extend(ast.iter_child_nodes(n))
    return out


# ------------------------------------------------------- did the agent see it? -----------------

def evidence_patterns(f):
    """(definition regex, use regex) for one fact. A match in a tool output means the agent could have
    read the fact there. Both are loose on purpose: a loose match marks a fact as seen, so the facts
    we keep as unseen are the safe ones."""
    d, u = re.escape(f.get('def_name') or ''), re.escape(f.get('use_name') or '')
    p = [re.escape(x) for x in (f.get('params') or [])]
    ft = f['fact_type']
    sep = r'[\s\d:|\-]*?'  # spaces, newlines and line-number prefixes (cat -n, grep -n) inside a signature
    inside = r'(?:[^()]|\([^()]*\)){0,800}?'  # stays inside the def's parentheses
    use_call = {'call': rf'(?<![\w.])(?<!def )(?<!class ){u}\s*\(', 'method': rf'\.{u}\s*\(',
                'ctor': rf'(?<![\w.])(?<!class ){u}\s*\('}.get(f.get('use'), rf'\.{u}\b')
    if f.get('use') == 'ctor' and f.get('use_alt'):  # any class that runs the same inherited constructor
        alt = '|'.join(re.escape(n) for n in sorted(set(f['use_alt']) | {f['use_name']}))
        use_call = rf'(?<![\w])(?<!class )(?:{alt})\s*\('
    ann = r'(?::(?:[^=(),\[\]]|\[[^\]]*\])*)?'  # an optional type annotation, e.g. ": Dict[str, int]"
    if ft == 'arg_order':  # def NAME(...p0 ... p1...): both parameters inside the def's parentheses, in order
        return rf'\bdef\s+{d}\s*\({inside}\b{p[0]}\b{inside}\b{p[1]}\b', use_call
    if ft == 'too_many_positional':
        return rf'\bdef\s+{d}\s*\(', use_call
    if ft == 'default':
        return rf'\bdef\s+{d}\s*\({inside}\b{p[0]}\s*{ann}\s*=', use_call
    if ft in ('keyword', 'keyword_not_in_signature'):
        names = u
        if f.get('use') == 'ctor' and f.get('use_alt'):
            names = '(?:' + '|'.join(re.escape(n) for n in sorted(set(f['use_alt']) | {f['use_name']})) + ')'
        return rf'\bdef\s+{d}\s*\({inside}\b{p[0]}\b', rf'(?<![\w]){names}\s*\({inside}\b{p[0]}\s*='
    if ft == 'attribute':  # the fact is only "this name exists", so any mention of the name (a dir() list,
        # an error message, docs) counts as a use
        return rf'\bdef\s+{d}\s*\(|^\s*{d}\s*[:=]|\bself\.{d}\s*=|^\s*class\s+{d}\b', rf'\b{d}\b'
    if ft == 'exception_class':
        return rf'\bclass\s+{d}\b', rf'\b{d}\b'
    if ft == 'raises':
        e = re.escape(f['exception'])
        return (rf'\bdef\s+{d}\s*\(.{{0,3000}}?\braise\s+(?:[\w.]+\.)?{e}\b|\bin\s+{d}\b\n.{{0,400}}?\b{e}\b',
                rf'\b{u}\s*\(.{{0,400}}?\bexcept\b[^\n]*\b{e}\b')
    if ft == 'import':
        mod, y = f['module'], re.escape(f['imported'])
        last = re.escape(mod.split('.')[-1])
        return (rf'\bfrom\s+{re.escape(mod)}\s+import\s*\(?[^)]{{0,400}}?\b{y}\b'
                rf'|\bfrom\s+\.+(?:[\w.]*\.)?{last}\s+import\s*\(?[^)]{{0,400}}?\b{y}\b'
                rf'|\b{re.escape(mod)}\.{y}\b', rf'\b{y}\b')
    return None, None


def seen_location(run, f):
    """Import facts: also count a step whose tool output shows `class Y`/`def Y` while the step's command
    or output names the file it lives in (e.g. grep -n 'class Q' django/db/models/query_utils.py)."""
    y = re.escape(f['imported'])
    paths = {f['target_file'], f['target_file'].replace('/__init__.py', '')}
    for k, s in enumerate(run.steps):
        if re.search(rf'(?:class|def)\s+{y}\b', s['obs']):
            text = s['cmd'] + '\n' + s['obs']
            if any(pth in text for pth in paths):
                return k
    return None


INTROSPECT = ('signature(', 'getfullargspec', 'getargspec', 'help(', 'getsource', '__doc__', '__code__', 'pydoc',
              '__signature__', '__defaults__', '__kwdefaults__', 'co_varnames', '__annotations__')


def introspected(run, f):
    """First step whose command asks Python about the target (inspect.signature, help(), getsource,
    __doc__, dir() for attributes): its output shows the definition even without a `def` line."""
    marks = INTROSPECT + (('dir(', 'hasattr(', 'vars(') if f['fact_type'] == 'attribute' else ())
    names = [n for n in {f.get('def_name'), f.get('use_name')} if n and n not in ('__init__', '__new__')]
    for k in range(1, len(run.steps)):
        c = run.steps[k]['cmd']
        if any(m in c for m in marks) and any(re.search(rf'\b{re.escape(n)}\b', c) for n in names):
            return k
    return None


def region_seen(run, f):
    """First step that printed the definition's region (its def line and the 25 lines after it) of the
    target file: judged from sed -n 'A,Bp' / head / awk ranges in a command that names the file, or from
    line numbers at the start of output lines (cat -n, grep -n, nl). This is the "file region containing
    the definition" rule; it also catches docstrings and bodies that name the parameters."""
    path, line = f.get('target_file'), f.get('target_line')
    if not path or not line or f.get('target_kind') == 'attr':
        return None
    lo, hi = line, min(f.get('target_end') or line + 25, line + 25)
    base = path.rsplit('/', 1)[-1]
    for k in range(1, len(run.steps)):
        cmd, obs = run.steps[k]['cmd'], run.steps[k]['obs']
        if base not in cmd:
            continue
        for a, b in re.findall(r'(\d+)\s*,\s*(\d+)\s*p', cmd) + re.findall(r'NR\s*>=?\s*(\d+)\s*&&\s*NR\s*<=?\s*(\d+)', cmd):
            if int(a) <= hi and int(b) >= lo:
                return k
        m = re.search(r'\bhead\s+(?:-n\s*)?-?(\d+)', cmd)
        if m and int(m.group(1)) >= lo:
            return k
        nums = [int(x) for x in re.findall(r'^\s*(\d+)[\t:\-]', obs, re.M)]
        if any(lo <= n <= hi for n in nums):
            return k
    return None


def kept_from_old_line(f, removed):
    """True if the lines this added line replaced already used the same thing: the agent edited existing
    code (e.g. return f(a, b) -> x = f(a, b)) rather than writing the use itself."""
    if not removed:
        return False
    u = re.escape(f.get('use_name') or '')
    ft = f['fact_type']
    if ft in ('arg_order', 'default', 'too_many_positional'):
        return re.search(rf'(?<![\w]){u}\s*\(', removed) is not None
    if ft in ('keyword', 'keyword_not_in_signature'):
        return (re.search(rf'(?<![\w]){u}\s*\(', removed) is not None
                and re.search(rf'\b{re.escape(f["params"][0])}\s*=', removed) is not None)
    if ft == 'attribute':
        return re.search(rf'\.{u}\b', removed) is not None
    if ft == 'import':
        return 'import' in removed and re.search(rf'\b{re.escape(f["imported"])}\b', removed) is not None
    if ft == 'exception_class':
        return re.search(rf'\b{u}\b', removed) is not None
    if ft == 'raises':
        return re.search(rf'except[^\n]*\b{re.escape(f["exception"])}\b', removed) is not None
    return False


def judge(run, f):
    """Add write step, first-seen steps and the unseen/strong verdicts to a fact."""
    w = run.write_step(f['code'], f.get('segment'))
    f['write_step'], f['write_found'], f['n_steps'] = w, w is not None, run.n
    cutoff = w if w is not None else run.n + 1  # not found: count every output as "before" (cautious)
    dpat, upat = evidence_patterns(f)
    if dpat is None:
        return f
    ds = run.first(dpat)
    extra = [seen_location(run, f)] if f['fact_type'] == 'import' else [region_seen(run, f)]
    extra.append(introspected(run, f))
    seen_at = [x for x in [ds] + extra if x is not None]
    ds = min(seen_at) if seen_at else None
    us = run.first(upat) if upat else None
    name = re.escape(f.get('use_name') or f.get('def_name') or '')
    f['def_seen_step'], f['use_seen_step'] = ds, us
    f['first_obs_mention'] = run.first(rf'\b{name}\b') if name else None
    f['first_agent_mention'] = run.first(rf'\b{name}\b', where='agent') if name else None
    f['seen_def_before_write'] = ds is not None and ds < cutoff
    f['seen_use_before_write'] = us is not None and us < cutoff
    u = re.escape(f.get('use_name') or '')
    probe = (rf'import[^\n]*\b{re.escape(f["imported"])}\b' if f['fact_type'] == 'import' else
             rf'\.{u}\b' if f['fact_type'] == 'attribute' else rf'(?<![\w]){u}\s*\(')
    p = run.first(probe, where='cmd') if u or f['fact_type'] == 'import' else None
    f['probe_step'] = p if p is not None and p < cutoff else None  # wrote and ran the use in a test first
    f['unseen'] = not f['seen_def_before_write']
    # no_use: also no other use of the name shown before, and not just kept from the line it edited
    f['no_use'] = f['unseen'] and not f['seen_use_before_write'] and not f.get('kept_from_old_line')
    f['strong'] = f['no_use'] and ds is None  # ...and the definition never shows up in the whole run
    return f


TRAP_TYPES = ('arg_order', 'default', 'keyword', 'import', 'raises', 'exception_class', 'attribute')
DIAG_TYPES = ('keyword_not_in_signature', 'too_many_positional', 'import_unresolved', 'module_attr_unresolved')


def facts_for_patch(index, submission, run):
    """All facts in one agent patch. Returns (facts, notes about skipped files)."""
    files, new_files = parse_diff(submission)
    edited = {}
    notes = []
    for path, hunks in files.items():
        if not path.endswith('.py') or path in new_files or is_test_path(path) or path not in index.tree:
            continue
        text = index.source(path)
        if text is None:
            notes.append(f'{path}: base file not downloaded')
            continue
        applied = apply_hunks(text, hunks)
        if applied is None:
            notes.append(f'{path}: patch does not apply to the base file')
            continue
        edited[path] = applied
    touched = {p: (a[2], a[3]) for p, a in edited.items()}
    new_modules = {module_name(p) for p in new_files if p.endswith('.py')}
    patch_defs = {module_name(p): summarize(a[0]) or dict(funcs={}, classes={}, names=set(), imports={})
                  for p, a in edited.items()}
    facts = []
    for path, (new_text, added, removed, inserts, hunk_of, removed_text) in edited.items():
        # an added line that already exists in the base file (moved or re-indented code) is old code
        base_lines = {l.strip() for l in index.source(path).split('\n') if l.strip()}
        new_lines = new_text.split('\n')
        added = {n for n in added if new_lines[n - 1].strip() not in base_lines}
        new_summary = summarize(new_text)
        if new_summary is None:
            notes.append(f'{path}: patched file does not parse')
            continue
        tree = ast.parse(new_text)
        finder = FactFinder(EditContext(index, path, new_summary, touched), added, new_text, new_modules, patch_defs)
        finder.visit(tree)
        for f in finder.facts:
            f['edit_file'] = path
            f['cross_file'] = f.get('target_file') not in (None, path)
            f['kept_from_old_line'] = kept_from_old_line(f, removed_text[hunk_of[f['line']]] if f['line'] in hunk_of else '')
            facts.append(judge(run, f) if f['fact_type'] in TRAP_TYPES + DIAG_TYPES[:2] else f)
    # the same fact on the same line can be found twice (e.g. a call inside a try body); keep one
    out, seen = [], set()
    for f in facts:
        k = (f['fact_key'], f['edit_file'], f['line'])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out, notes


def extract(only_ids):
    tasks = pq.read_table(PARQUET).to_pandas()
    if only_ids:
        tasks = tasks[tasks.instance_id.isin(only_ids)]
    tasks = tasks.sort_values(['repo', 'base_commit'])
    out = open(FACTS if not only_ids else FACTS.replace('.jsonl', '.sample.jsonl'), 'w')
    status = Counter()
    all_tasks = pq.read_table(PARQUET).to_pandas()
    for repo, group in tasks.groupby('repo', sort=True):
        git = Git(os.path.join(REPOS, repo.replace('/', '_')))
        cache = {}  # parsed files by blob id, shared by all commits of this repo
        extra = set()
        if repo == 'mwaskom/seaborn':  # seaborn code calls matplotlib objects all the time
            mpl = all_tasks[all_tasks.repo == 'matplotlib/matplotlib'].sort_values('created_at').iloc[-1]
            mpl_index = CommitIndex(Git(os.path.join(REPOS, 'matplotlib_matplotlib')), mpl.base_commit, {})
            extra = set(mpl_index.members_index())
        for r in group.itertuples():
            index = None
            for model, folder in RUNS.items():
                tf = os.path.join(RAW, folder, 'trajs', r.instance_id, r.instance_id + '.traj.json')
                rf = os.path.join(RAW, folder, 'logs', r.instance_id, 'report.json')
                if not os.path.exists(tf):
                    status['no trajectory'] += 1
                    continue
                traj = json.load(open(tf))
                resolved = None
                if os.path.exists(rf):
                    rep = json.load(open(rf))
                    rep = rep.get(r.instance_id, rep)
                    resolved = rep.get('resolved')
                submission = (traj.get('info') or {}).get('submission') or ''
                base = dict(model=model, instance_id=r.instance_id, repo=repo, resolved=resolved,
                            exit_status=(traj.get('info') or {}).get('exit_status'))
                if not submission.strip():
                    status['empty patch'] += 1
                    out.write(json.dumps(dict(base, fact_type='none', note='empty patch')) + '\n')
                    continue
                if index is None:
                    index = CommitIndex(git, r.base_commit, cache)
                    index.extra_common = extra
                run = Run(load_steps(traj))
                facts, notes = facts_for_patch(index, submission, run)
                status['patches'] += 1
                status['patches with facts'] += bool(facts)
                for n in notes:
                    status[n.split(': ', 1)[1]] += 1
                if not facts:
                    out.write(json.dumps(dict(base, fact_type='none', note='; '.join(notes) or 'no repo facts',
                                              n_steps=run.n)) + '\n')
                for f in facts:
                    out.write(json.dumps(dict(base, **f)) + '\n')
            out.flush()
        print(repo, dict(status), flush=True)
    out.close()
    print('done', dict(status))


# ------------------------------------------------------------------ report ------------------------

MODELS = list(RUNS)
SHORT = {'deepseek-v3.2': 'D', 'kimi-k2.5': 'K', 'minimax-m2.5': 'M'}
TYPE_W = {'arg_order': 3, 'default': 3, 'keyword': 3, 'raises': 3, 'import': 2, 'attribute': 1, 'exception_class': 1}
HOW_W = {'same_file': 2, 'import': 2, 'self': 2, 'super': 2, 'class': 2, 'module': 2, 'unique_name': 1,
         'same_signature': 0}


def per_model_fact(rows):
    """One row per model x task x fact: its earliest-written occurrence (most likely to be unseen)."""
    best = {}
    for r in rows:
        k = (r['model'], r['instance_id'], r['fact_key'])
        cut = r['write_step'] if r.get('write_step') is not None else (r.get('n_steps') or 0) + 1
        if k not in best or cut < best[k][0]:
            best[k] = (cut, r)
    return [v[1] for v in best.values()]


def aggregate(facts, tasks):
    """Group model-level facts into candidates: one per task x fact, with which models relied on it."""
    groups = defaultdict(list)
    for f in facts:
        groups[(f['instance_id'], f['fact_key'])].append(f)
    cands = []
    for (iid, key), fs in groups.items():
        unseen = [f for f in fs if f['unseen']]
        if not unseen:
            continue
        no_use = [f for f in fs if f['no_use']]
        strong = [f for f in fs if f['strong']]
        f0 = sorted(fs, key=lambda f: (not f['strong'], not f['no_use'], not f['unseen'], MODELS.index(f['model'])))[0]
        gold = tasks.loc[iid, 'patch']
        gold_added = '\n'.join(l for l in gold.split('\n') if l.startswith('+') and not l.startswith('+++'))
        gold_files = set(re.findall(r'^\+\+\+ b/(\S+)', gold, re.M))
        name = f0.get('use_name') or f0.get('def_name') or ''
        c = dict(
            instance_id=iid, repo=f0['repo'], fact_type=f0['fact_type'], fact_key=key, detail=f0['detail'],
            target_qual=f0['target_qual'], target_file=f0['target_file'], target_line=f0.get('target_line'),
            use_name=name, use=f0.get('use'), how=f0.get('how'), n_sites=f0.get('n_sites'),
            cross_file=any(f['cross_file'] for f in fs),
            in_gold_patch=bool(name) and re.search(rf'\b{re.escape(name)}\b', gold_added) is not None,
            agent_edited_gold_file=any(f['edit_file'] in gold_files for f in fs),
            models_relied=sorted((f['model'] for f in fs), key=MODELS.index),
            models_unseen=sorted((f['model'] for f in unseen), key=MODELS.index),
            models_no_use=sorted((f['model'] for f in no_use), key=MODELS.index),
            models_strong=sorted((f['model'] for f in strong), key=MODELS.index),
            per_model={f['model']: dict(
                resolved=f['resolved'], unseen=f['unseen'], no_use=f['no_use'], strong=f['strong'],
                kept_from_old_line=f.get('kept_from_old_line'), probe_step=f.get('probe_step'),
                edit_file=f['edit_file'],
                line=f['line'], code=f['code'], write_step=f['write_step'], n_steps=f['n_steps'],
                def_seen_step=f['def_seen_step'], use_seen_step=f['use_seen_step'],
                first_agent_mention=f.get('first_agent_mention'), first_obs_mention=f.get('first_obs_mention'))
                for f in fs})
        c['n_unseen'], c['n_no_use'], c['n_strong'] = len(unseen), len(no_use), len(strong)
        c['n_resolved_no_use'] = sum(1 for f in no_use if f['resolved'])
        c['default'], c['n_defaults'], c['target_kind'] = f0.get('default'), f0.get('n_defaults'), f0['target_kind']
        c['type_weight'] = trap_weight(c)
        cands.append(c)
    rescore(cands)
    return cands


def trap_weight(c):
    """How usable the fact is for a behavior-preserving trap (0-3)."""
    ft = c['fact_type']
    if ft == 'default':
        if c['default'] == 'None':
            return -6  # "not set" defaults: changing them is not a clean trap
        return -6 if (c['n_defaults'] or 0) > 6 else 3  # Field() relies on 20 defaults at once: not a trap
    if ft == 'attribute' and c['how'] == 'module' and c['target_kind'] in ('class', 'name', 'module'):
        return -6  # models.Model, models.Q, matplotlib.rcParams: public names, not realistic to change
    if ft == 'import' and '.' not in c['fact_key'].split('|')[3].split(':')[0]:
        return -6  # from sympy import asin: the package's public top-level API, not realistic to move
    return TYPE_W.get(ft, 0)


def site_adjust(sites):
    """Smaller rewrites make easier traps; very common names are not realistic to change."""
    if not sites:
        return 0
    n = sites['src'] + sites['tests'] + sites['docs']
    return 1 if n <= 60 else 0 if n <= 200 else -3 if n <= 600 else -6


def rescore(cands):
    for c in cands:
        # memory signal first: per model, 6 if strong, 4 if no use seen, 1 if it only missed the definition
        # (spot check: those last ones almost always copied a visible call)
        memory = 6 * c['n_strong'] + 4 * (c['n_no_use'] - c['n_strong']) + (c['n_unseen'] - c['n_no_use'])
        c['score'] = (memory + c['type_weight'] + HOW_W.get(c['how'], 0) + 2 * c['cross_file']
                      + c['n_resolved_no_use'] + site_adjust(c.get('call_sites')))
    cands.sort(key=lambda c: (-c['score'], c['instance_id'], c['fact_key']))


def best_per_task(cands):
    best, order = {}, []
    for c in cands:
        if c['instance_id'] not in best:
            best[c['instance_id']] = c
            order.append(c['instance_id'])
    return best, order


def site_pattern(c):
    """git grep -E pattern for the places a trap on this fact would have to rewrite."""
    n = re.escape(c['use_name']).replace('\\_', '_')
    word_end = '([^A-Za-z0-9_]|$)'
    if c['fact_type'] == 'attribute':
        return rf'\.{n}{word_end}'
    if c['fact_type'] == 'import':  # import lines, and lines inside a parenthesized "from x import (...)" list
        return rf'(import.*[^A-Za-z0-9_.]{n}{word_end})|(^[[:space:]]*[A-Za-z0-9_, ]*(^|[^A-Za-z0-9_.]){n}[A-Za-z0-9_, ]*\)?[[:space:]]*$)'
    if c['fact_type'] == 'exception_class':
        return rf'(^|[^A-Za-z0-9_]){n}{word_end}'
    if c.get('use') == 'method':
        return rf'\.{n}[[:space:]]*\('
    return rf'(^|[^A-Za-z0-9_]){n}[[:space:]]*\('  # f(...) and module.f(...)


def count_sites(c, commit):
    """Lines that use the name at the base commit (definitions excluded), split into source/tests/docs."""
    rp = os.path.join(REPOS, c['repo'].replace('/', '_'))
    r = subprocess.run(['git', '-C', rp, 'grep', '-n', '-I', '-E', site_pattern(c), commit, '--',
                        '*.py', '*.rst', '*.txt'], capture_output=True, text=True, env=NO_FETCH)
    if r.returncode not in (0, 1):
        return None  # some file was not downloaded: no count
    out = Counter()
    name = re.escape(c['use_name'])
    for line in r.stdout.splitlines():
        parts = line.split(':', 3)
        if len(parts) < 4 or re.search(rf'\b(def|class)\s+{name}\b', parts[3]):
            continue
        path = parts[1]
        out['docs' if not path.endswith('.py') else 'tests' if is_test_path(path) else 'src'] += 1
    return dict(src=out['src'], tests=out['tests'], docs=out['docs'])


def report(n_top=40, n_count=60):
    rows = [json.loads(l) for l in open(FACTS)]
    tasks = pq.read_table(PARQUET).to_pandas().set_index('instance_id')
    resolved = {(r['model'], r['instance_id']): r.get('resolved') for r in rows}
    facts = per_model_fact([r for r in rows if r['fact_type'] in TRAP_TYPES])
    diags = per_model_fact([r for r in rows if r['fact_type'] in DIAG_TYPES and r.get('fact_key')])
    cands = aggregate(facts, tasks)

    # call-site counts (git grep at the base commit) for the top tasks; this needs all their files, so
    # those are downloaded first. Counting changes the ranking, so repeat until the top tasks all have counts.
    counted = set()
    idfile = os.path.join(ROOT, 'outputs/agent_targeted_top_ids.txt')
    for _ in range(4):
        best, order = best_per_task(cands)
        todo = [iid for iid in order[:n_count] if iid not in counted]
        if not todo:
            break
        open(idfile, 'w').write('\n'.join(todo) + '\n')
        subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'scripts/prefetch_blobs.py'), PARQUET, REPOS,
                        'all', idfile], check=True)
        for c in cands:
            if c['instance_id'] in todo:
                c['call_sites'] = count_sites(c, tasks.loc[c['instance_id'], 'base_commit'])
        counted.update(todo)
        rescore(cands)
    open(idfile, 'w').write('\n'.join(sorted(counted)) + '\n')  # every task that got call-site counts
    best, order = best_per_task([c for c in cands if c['type_weight'] >= 0])
    with open(CANDS, 'w') as out:
        for rank, c in enumerate(cands, 1):
            out.write(json.dumps(dict(rank=rank, **c)) + '\n')
    write_note(rows, facts, diags, cands, best, order[:n_top], resolved)
    print(f'{len(cands)} candidates -> {CANDS}; note -> {NOTE}')


def pct(a, b):
    return f'{100 * a / b:.0f}%' if b else '-'


def write_note(rows, facts, diags, cands, best, top_ids, resolved):
    L = []
    w = L.append
    n_tasks = len({r['instance_id'] for r in rows})
    w('# Agent-targeted trap candidates (Nov 13, 2026)\n')
    w('Script: `scripts/agent_targeted_candidates.py` (extract, then report). Data: the public mini-swe-agent '
      'runs on SWE-bench Verified in `data/raw` (DeepSeek-V3.2, Kimi K2.5, MiniMax M2.5), repos at each task\'s '
      'base commit. No API calls.\n')
    w('**Question:** which repo facts does each agent\'s own final patch rely on without the agent ever having '
      'looked at them? Those are the facts a trap should change (pilot lesson: generic traps on facts the agent '
      'can see or never uses do not bite).\n')
    w('**How:** for every added line of the patch (moved or re-indented old lines excluded), find uses of things '
      'the repo defined before the patch, outside the code the patch changed. Record the exact fact relied on. '
      'Then find the step where the agent first wrote that line, and check every tool output before it for the '
      'definition (signature with those parameters, `class E`, the import, the `raise`, or an `inspect.signature`/'
      '`help()` call on it). Three levels:\n')
    w('- **unseen**: no tool output before the write showed the definition.')
    w('- **no use seen** (the memory tier): unseen, and no other use of the name was shown before the write either, '
      'and the line did not just keep a call that was already on the line it replaced.')
    w('- **strong**: no use seen, and the definition never shows up anywhere in the run.\n')
    bottom_line(w, rows, facts, diags, cands)

    # coverage
    w('## Coverage\n')
    w('| model | tasks with a run | non-empty patch | patch has 1+ repo fact | facts (model x task x fact) | unseen | no use seen | strong |')
    w('|---|---|---|---|---|---|---|---|')
    for m in MODELS:
        mr = [r for r in rows if r['model'] == m]
        withp = {r['instance_id'] for r in mr if r.get('note') != 'empty patch'}
        withf = {r['instance_id'] for r in mr if r['fact_type'] in TRAP_TYPES}
        mf = [f for f in facts if f['model'] == m]
        w(f"| {m} | {len({r['instance_id'] for r in mr})} | {len(withp)} | {len(withf)} | {len(mf)} | "
          f"{sum(f['unseen'] for f in mf)} ({pct(sum(f['unseen'] for f in mf), len(mf))}) | "
          f"{sum(f['no_use'] for f in mf)} ({pct(sum(f['no_use'] for f in mf), len(mf))}) | "
          f"{sum(f['strong'] for f in mf)} ({pct(sum(f['strong'] for f in mf), len(mf))}) |")
    w('')

    # by fact type and model
    w('## Facts by type and model (relied on / unseen / no use seen / strong)\n')
    w('| fact type | ' + ' | '.join(MODELS) + ' | all |')
    w('|---|' + '---|' * (len(MODELS) + 1))
    for ft in TRAP_TYPES:
        cells = []
        for m in MODELS + [None]:
            sel = [f for f in facts if f['fact_type'] == ft and (m is None or f['model'] == m)]
            cells.append(f"{len(sel)} / {sum(f['unseen'] for f in sel)} / {sum(f['no_use'] for f in sel)} / "
                         f"{sum(f['strong'] for f in sel)}")
        w(f'| {ft} | ' + ' | '.join(cells) + ' |')
    w('')

    # agreement
    w('## Cross-model agreement\n')
    n_t = lambda key, k: len({c['instance_id'] for c in cands if c[key] >= k})
    w(f"{len(cands)} candidates (task x fact unseen by at least one model) in {n_t('n_unseen', 1)} of {n_tasks} "
      f"tasks; {sum(1 for c in cands if c['n_no_use'])} of them are in the memory tier (no use seen) for at least one "
      f"model, in {n_t('n_no_use', 1)} tasks.\n")
    w('| k models | candidates unseen by exactly k | tasks (>= k) | no use seen by exactly k | tasks (>= k) | strong by exactly k | tasks (>= k) |')
    w('|---|---|---|---|---|---|---|')
    for k in (1, 2, 3):
        w(f"| {k} | {sum(1 for c in cands if c['n_unseen'] == k)} | {n_t('n_unseen', k)} | "
          f"{sum(1 for c in cands if c['n_no_use'] == k)} | {n_t('n_no_use', k)} | "
          f"{sum(1 for c in cands if c['n_strong'] == k)} | {n_t('n_strong', k)} |")
    w('')
    by_type = Counter(c['fact_type'] for c in cands if c['n_no_use'] >= 2)
    w('Memory-tier candidates shared by 2+ models, by type: ' +
      (', '.join(f'{t} {n}' for t, n in by_type.most_common()) or 'none') + '.\n')
    cross = [c for c in cands if c['cross_file']]
    w(f"Target defined in a different file from the edit: {len(cross)} of {len(cands)} candidates "
      f"({pct(len(cross), len(cands))}). Resolution of the target name: " +
      ', '.join(f'{h} {n}' for h, n in Counter(c['how'] for c in cands).most_common()) + '.\n')
    ru = [(f['resolved'], f['no_use'], f['unseen']) for f in facts if f['resolved'] is not None]
    w(f"Share of model-level facts that sit in a run whose patch resolved the task: memory tier "
      f"{pct(sum(1 for r, n, u in ru if r and n), sum(1 for r, n, u in ru if n))}, unseen "
      f"{pct(sum(1 for r, n, u in ru if r and u), sum(1 for r, n, u in ru if u))}, definition seen "
      f"{pct(sum(1 for r, n, u in ru if r and not u), sum(1 for r, n, u in ru if not u))}.\n")

    # memory errors
    w('## Memory errors that reached the final patch (not trap candidates)\n')
    w('Uses of things that do not exist at the base commit, as written: a keyword the function does not take, '
      'more positional arguments than it accepts, an import of a name the module does not have, or a module '
      'attribute that is not there. These are leads for "wrong-version recall"; each needs a manual look '
      '(dynamic attributes and `__getattr__` can fool the check).\n')
    w('| check | ' + ' | '.join(MODELS) + ' |')
    w('|---|' + '---|' * len(MODELS))
    for dt in DIAG_TYPES:
        w(f'| {dt} | ' + ' | '.join(str(sum(1 for d in diags if d['fact_type'] == dt and d['model'] == m))
                                     for m in MODELS) + ' |')
    w('')
    for d in sorted(diags, key=lambda d: (d['fact_type'], d['instance_id']))[:25]:
        res = 'no run report' if d['resolved'] is None else 'resolved' if d['resolved'] else 'not resolved'
        w(f"- {d['model']} `{d['instance_id']}` ({res}): "
          f"{d['detail']} - `{d['code'][:90]}`")
    w('')

    # top table
    w(f'## Top {len(top_ids)} tasks (best trap-usable candidate per task)\n')
    w('Score = memory signal (per model: 6 strong, 4 no use seen, 1 only unseen) + trap usability (arg_order/keyword/raises 3, default 3 '
      '[-6 if the function has more than 6 defaults or the default is None], import 2 [-6 for a package\'s top-level '
      'API like `from sympy import asin`], attribute/exception 1 [-6 for public module names like models.Model]) '
      '+ name resolution (direct 2, unique method name 1, '
      'same-signature overrides 0) + 2 if defined in another file + 1 per memory-tier model whose run resolved the task '
      '+ rewrite size (<= 60 sites +1, <= 200 0, <= 600 -3, more -6). Models: D = DeepSeek-V3.2, '
      'K = Kimi K2.5, M = MiniMax M2.5. "Relied on it" lists every model whose patch uses the fact: upper case = '
      'memory tier (no definition, no use seen before writing it), lower case = definition unseen but a use was '
      'visible (likely copied), in brackets = definition seen. Sites = lines using the name at the base commit '
      '(source/tests/docs), a rough size of the trap rewrite.\n')
    w('| # | task | type | fact the patch relies on | defined in | relied on it | strong | resolved | sites src/tests/docs | other cands |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    n_per_task = Counter(c['instance_id'] for c in cands)
    for i, iid in enumerate(top_ids, 1):
        c = best[iid]
        models = ''.join(SHORT[m] if m in c['models_no_use'] else SHORT[m].lower() if m in c['models_unseen']
                         else f'({SHORT[m].lower()})' for m in c['models_relied'])
        strong = ''.join(SHORT[m] for m in c['models_strong']) or '-'
        res = ' '.join(f"{SHORT[m]}{'+' if resolved.get((m, iid)) else '-'}" for m in c['models_relied'])
        cs = c.get('call_sites')
        sites = f"{cs['src']}/{cs['tests']}/{cs['docs']}" if cs else 'n/a'
        if c['use'] == 'ctor' and not c['target_qual'].startswith(c['use_name'] + '.'):
            sites += f" (runs inherited `{c['target_qual']}`: every subclass call too)"
        where = f"`{c['target_file']}:{c['target_line']}`" + (' (other file)' if c['cross_file'] else '')
        w(f"| {i} | {iid} | {c['fact_type']} | {c['detail'].replace('|', '/')} | {where} | {models} | {strong} | "
          f"{res} | {sites} | {n_per_task[iid] - 1} |")
    w('')
    # memory-tier facts that are not clean traps
    bad = [c for c in cands if c['n_no_use'] and c['type_weight'] < 0]
    w('## Memory-tier facts that are not clean traps\n')
    w('Relied on without reading, but changing them would not make a fair, behavior-preserving trap: '
      '`None` defaults ("not set"), constructors with more than 6 defaults, public module names.\n')
    for c in bad:
        w(f"- `{c['instance_id']}` {c['detail']} ({''.join(SHORT[m] for m in c['models_no_use'])})")
    w('')
    spot_check_section(w)
    caveats(w)
    w('Reproduce: `.venv/bin/python -I scripts/prefetch_blobs.py data/meta/swebv/data/test-00000-of-00001.parquet '
      'data/repos src`, then `.venv/bin/python -I scripts/agent_targeted_candidates.py extract` and `... report` '
      '(the report downloads the test and doc files of the top tasks for the call-site counts). Files: '
      '`outputs/agent_targeted_facts.jsonl` (every fact), `outputs/agent_targeted_candidates.jsonl` (ranked '
      'candidates), `outputs/agent_targeted_spot_check.json` (hand verdicts).')
    open(NOTE, 'w').write('\n'.join(L) + '\n')


def bottom_line(w, rows, facts, diags, cands):
    """Key numbers first, computed from the same data as the tables below."""
    patches = {(r['model'], r['instance_id']) for r in rows if r.get('note') != 'empty patch'}
    with_facts = {(f['model'], f['instance_id']) for f in facts}
    mem = [f for f in facts if f['no_use']]
    usable = [c for c in cands if c['n_no_use'] and c['type_weight'] >= 0]
    shared = [c for c in usable if c['n_no_use'] >= 2]
    probed = sum(1 for f in mem if f.get('probe_step') is not None)
    path = os.path.join(ROOT, 'outputs/agent_targeted_spot_check.json')
    checks = json.load(open(path)) if os.path.exists(path) else []
    last = [c for c in checks if c['round'] == max((x['round'] for x in checks), default=0)]
    first = [c for c in checks if c['round'] == 1]
    w('## Bottom line\n')
    w(f"- {len(patches)} final patches, {len(with_facts)} of them rely on at least one earlier repo definition "
      f"({len(facts)} model-level facts).")
    n_unseen = sum(f['unseen'] for f in facts)
    w(f"- **The agent never saw the definition before writing for {n_unseen} facts "
      f"({pct(n_unseen, len(facts))}).** But {n_unseen - len(mem)} of those ({pct(n_unseen - len(mem), n_unseen)}) had "
      f"another use of the same name in an earlier output, or kept the call from the line being edited. In the first "
      f"spot check, {sum(1 for c in first if c['def_unseen_correct'] and not c['memory'])} of {len(first)} unseen facts "
      f"were copied that way, the same thing that made the pilot's django-11149 trap fail.")
    w(f"- **Real memory reliance is rare: {len(mem)} facts ({pct(len(mem), len(facts))}) are in the memory tier** "
      f"(no definition and no use shown before the agent wrote it), in {len({f['instance_id'] for f in mem})} tasks. "
      f"{len(usable)} candidates in {len({c['instance_id'] for c in usable})} tasks pass the trap-usability filter "
      f"(not a `None` default or a public module name); each still needs a manual look (see caveats).")
    w(f"- **Models rarely agree on them:** {len(shared)} trap-usable memory-tier candidate(s) shared by 2+ models"
      + (': ' + '; '.join(f"`{c['instance_id']}` {c['detail'].split(';')[0]} "
                          f"({''.join(SHORT[m] for m in c['models_no_use'])})" for c in shared) if shared else '') + '.')
    if last:
        ok = sum(c['def_unseen_correct'] and c['memory'] for c in last)
        bound = f"; with {len(last)} checks the 95% lower bound is {100 * 0.025 ** (1 / len(last)):.0f}%" if ok == len(last) else ''
        w(f"- **Spot check of the final memory tier: {ok}/{len(last)} correct** (definition and uses really never shown "
          f"before the write{bound}). Earlier versions of the heuristic missed `inspect.signature` output, sibling-class "
          f"calls and `sed -n` ranges; all three are handled now.")
    w(f"- {probed} of the {len(mem)} memory-tier facts were first exercised in a test command before the patch "
      f"line was written, so a trap on them would likely surface as an error during that probe (detected and "
      f"repaired, like the pilot's sift run) rather than as a silent wrong patch.")
    w(f"- Memory errors that reached a final patch (a call or import that cannot work at the base commit): "
      f"{len(diags)}.\n")


def caveats(w):
    w('## Caveats and what this means for the trap design\n')
    w('- Only final patches are analyzed. A fact recalled in an earlier attempt, or only in the reasoning, is not counted.')
    w('- Recall of the heuristic was not measured. The memory tier is cautious by design: any visible use of the '
      'name disqualifies a fact, so some real memory facts are surely excluded.')
    w('- When the receiver of `x.method()` cannot be resolved, the target is guessed only if exactly one repo class '
      'defines a method with that name ("unique_name"), excluding names that exist on builtins, numpy, pandas, '
      'docutils, scipy.sparse and pluggy (and all matplotlib names for seaborn). One wrong guess (seaborn '
      '`formatter.format_data` is matplotlib\'s, not seaborn\'s) was found and fixed; other guesses are not all checked.')
    w('- Classes built by metaclasses, star imports, module `__getattr__` and `globals()` tricks are skipped, which '
      'keeps false memory errors out but loses some facts.')
    w('- Argument-order traps only bite if the two parameters are not interchangeable (`dmp_mul(f, g)` is commutative; '
      '`dmp_div`, `dmp_sub` are not). Default traps only bite if the code path depends on the default (xarray-4094 '
      'guards the call so `errors=` never matters).')
    w('- Call-site counts are lines matching the name at the base commit (docs included), so they overcount common '
      'names. For inherited constructors (`Exact(...)` runs `Lookup.__init__`) they undercount: the real rewrite '
      'covers every subclass.')
    w('- The public runs had network access (a `pip install pytest` succeeded in at least one trajectory), but none '
      'of the memory-tier runs fetched upstream code (no curl/wget/git clone/pip download of the repo).')
    w('- **Design implication:** facts that agents rely on from memory are rare and idiosyncratic (almost no '
      'cross-model agreement), so traps cannot be picked once for all models from public runs. The plan from the '
      'pilot stands, with this script as its second step: run each model on the original repo with network off, '
      'several times, mine its own patches with `agent_targeted_candidates.py`, and trap facts in the memory tier '
      'that recur across runs of that model.\n')


def spot_check_section(w):
    """Hand-checked verdicts, kept in outputs/agent_targeted_spot_check.json."""
    path = os.path.join(ROOT, 'outputs/agent_targeted_spot_check.json')
    if not os.path.exists(path):
        return
    checks = json.load(open(path))
    w('## Spot check by hand\n')
    w('For each candidate: open the trajectory, find the step that first wrote the line, and read every earlier '
      'tool output and command for the definition (or the region of the file holding it, or an '
      '`inspect`/`help` call) and for other uses of the name. Two verdicts: is "definition unseen" right, and '
      'did the agent really rely on memory (neither the definition nor a use was visible)? '
      '`scripts/spot_check_candidate.py` prints what is needed for one candidate.\n')
    for r in sorted({c['round'] for c in checks}):
        rows = [c for c in checks if c['round'] == r]
        n = len(rows)
        w(f"**Round {r}** ({rows[0]['heuristic']}): definition unseen correct "
          f"{sum(c['def_unseen_correct'] for c in rows)}/{n}, real memory reliance {sum(c['memory'] for c in rows)}/{n}.\n")
        w('| model | task | fact | definition unseen? | memory? | what the trajectory shows |')
        w('|---|---|---|---|---|---|')
        for c in rows:
            w(f"| {c['model']} | {c['instance_id']} | {c['fact']} | {'yes' if c['def_unseen_correct'] else 'NO'} | "
              f"{'yes' if c['memory'] else 'no'} | {c['why']} |")
        w('')

if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(sys.argv[2:])
    elif sys.argv[1] == 'report':
        report()
