"""Git lookups for RQ1 (fact provenance): does a file path or a code symbol exist in a repo at a
given commit, and if not, does it exist in an older or newer version of the repo?

Used by rq1_fact_provenance.py. Works on blobless clones (git clone --filter=blob:none): they
have every commit and folder listing, but file contents ("blobs") must be downloaded. We never
let git download a missing file on its own (that is one slow request per file); instead
fetch_missing() asks GitHub for all missing files of a commit in one batch.

What "a symbol is defined" means here (deliberately simple, regex/ast based, no type checking):
  Python      def/class names at any depth, names assigned at module or class level,
              names imported at module level (so re-exports count), parameter lists of defs
  Go          func/method names, type names, var/const names, struct fields, interface methods
  TypeScript  function/class/interface/type/enum/namespace names, const/let/var names,
  JavaScript  class and object method names, names in export { ... } lists
  Rust        fn/struct/enum/trait/type/const/static/mod/macro names, enum variants

Versions: besides the task's base commit we index a few "snapshot" commits of the same repo
(the other tasks' base commits, the newest commit, and older commits picked by date). A symbol
missing at the base commit but defined in an older snapshot is "stale"; only in newer ones,
"future"; in none, "never". Paths use the full history of when each path was added.
"""
import ast
import json
import os
import re
import subprocess
import warnings
from collections import defaultdict

# never let git download a missing file by itself
NO_FETCH = dict(os.environ, GIT_NO_LAZY_FETCH='1')

SOURCE_EXT = {'.py': 'py', '.pyi': 'py', '.go': 'go', '.rs': 'rs',
              '.ts': 'ts', '.tsx': 'ts', '.mts': 'ts', '.cts': 'ts',
              '.js': 'js', '.jsx': 'js', '.mjs': 'js', '.cjs': 'js'}
# folders whose files are third-party, generated or data, not the repo's own code
SKIP_DIRS = {'node_modules', 'vendor', 'dist', 'build', 'target', '.git', '__pycache__', 'site-packages',
             'testdata', 'fixtures', '__snapshots__', 'third_party', 'third-party', 'externals'}
MAX_BLOB_BYTES = 1_500_000


def git(repo, *args, stdin=None):
    # log.mailmap=false: otherwise git log/show try to read the .mailmap file, whose contents a
    # blobless clone does not have, and fail with lazy fetching turned off
    r = subprocess.run(['git', '-C', repo, '-c', 'log.mailmap=false', *args], input=stdin, capture_output=True,
                       text=True, env=NO_FETCH, errors='replace')
    return r.stdout


def source_lang(path):
    """'py', 'go', 'rs', 'ts', 'js' for a source file we index, else None."""
    if path.endswith('.min.js'):
        return None
    ext = os.path.splitext(path)[1]
    if ext not in SOURCE_EXT:
        return None
    if any(part in SKIP_DIRS for part in path.split('/')[:-1]):
        return None
    return SOURCE_EXT[ext]


# ----------------------------------------------------------------------------------------------
# Parsing one file into the names it defines
# ----------------------------------------------------------------------------------------------

def _params_from_ast(args):
    """Ordered parameter names of a Python def (self/cls kept; the caller drops them)."""
    pos = [a.arg for a in getattr(args, 'posonlyargs', [])] + [a.arg for a in args.args]
    return {'pos': pos, 'kwonly': [a.arg for a in args.kwonlyargs],
            'varargs': bool(args.vararg), 'varkw': bool(args.kwarg)}


def parse_python(text):
    """Return {'defs': [...], 'top': [...], 'star': [...], 'sigs': {name: [params, ...]}}.
    defs = every def/class name at any depth plus class-level assignments
    top  = names bound at module level (defs, classes, assignments, imports, __all__)
    star = modules imported with 'from X import *' at module level, as (level, 'X')"""
    out = {'defs': set(), 'top': set(), 'star': [], 'sigs': defaultdict(list), 'dyn': bool(PY_DYN_RE.search(text))}
    imported = set()  # names bound only by import statements (not definitions of the repo)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # old files trigger 'invalid escape sequence' warnings
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return parse_python_regex(text)

    def bind_target(t, into):
        if isinstance(t, ast.Name):
            into.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                bind_target(e, into)

    for node in tree.body:  # module level
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out['top'].add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                bind_target(t, out['top'])
            # __all__ = ['a', 'b'] lists public names
            if any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets) and \
                    isinstance(node.value, (ast.List, ast.Tuple)):
                out['top'].update(e.value for e in node.value.elts
                                  if isinstance(e, ast.Constant) and isinstance(e.value, str))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            bind_target(node.target, out['top'])
        elif isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.asname or a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == '*':
                    out['star'].append((node.level, node.module or ''))
                else:
                    imported.add(a.asname or a.name)
        elif isinstance(node, (ast.If, ast.Try)):
            # names bound inside module-level if/try blocks (e.g. try: import x except: x = None)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out['top'].add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        bind_target(t, out['top'])
                elif isinstance(sub, ast.Import):
                    for a in sub.names:
                        imported.add(a.asname or a.name.split('.')[0])
                elif isinstance(sub, ast.ImportFrom):
                    for a in sub.names:
                        if a.name != '*':
                            imported.add(a.asname or a.name)

    for node in ast.walk(tree):  # any depth
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out['defs'].add(node.name)
            out['sigs'][node.name].append(_params_from_ast(node.args))
        elif isinstance(node, ast.ClassDef):
            out['defs'].add(node.name)
            for item in node.body:  # class attributes, e.g. name = models.CharField()
                if isinstance(item, ast.Assign):
                    for t in item.targets:
                        bind_target(t, out['defs'])
                elif isinstance(item, ast.AnnAssign):
                    bind_target(item.target, out['defs'])
    out['defs'] |= out['top']        # module-level definitions and assignments
    out['top'] |= imported           # importable from this module, but defined elsewhere
    return out


# a module that creates names at run time (globals()[...] = ..., module __getattr__, exec):
# a name missing from it is "unknown", not "never"
PY_DYN_RE = re.compile(r'\bglobals\(\)|^def __getattr__\(|sys\.modules\[|\bexec\(|\bsetattr\(sys\.modules', re.M)
PY_DEF_RE = re.compile(r'^[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)[ \t]*\(([^)]*)\)', re.M)
PY_CLASS_RE = re.compile(r'^[ \t]*class[ \t]+(\w+)', re.M)
PY_TOP_RE = re.compile(r'^(?:(?:async[ \t]+)?def|class)[ \t]+(\w+)|^(\w+)[ \t]*(?::[^=\n]*)?=(?!=)', re.M)
PY_FROM_RE = re.compile(r'^from[ \t]+(\.*[\w.]*)[ \t]+import[ \t]+\(?([^)\n]*)', re.M)
PY_IMPORT_RE = re.compile(r'^import[ \t]+([\w., \t]+)', re.M)


def split_top(raw):
    """Split a parameter list at commas that are not inside brackets: 'a, b: Dict[x, y]' -> ['a', 'b: Dict[x, y]']."""
    out, depth, cur = [], 0, ''
    for ch in raw:
        if ch in '([{<':
            depth += 1
        elif ch in ')]}>':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    out.append(cur.strip())
    return [p for p in out if p]


def split_params(raw):
    """Python/TypeScript/Rust style 'a, b=1, *args, c: int = 2, **kw, ...rest' -> parameter names
    (with * and ** markers kept). Destructured parameters ({a, b}: T) are skipped."""
    names = []
    for p in split_top(raw):
        if p == '/':
            continue
        p = re.sub(r'^\.\.\.', '*', p)                  # TS rest parameter
        p = re.sub(r'^(?:&\s*)?(?:mut\s+)?', '', p)       # Rust &self / mut x
        m = re.match(r'(\*{0,2})\s*([A-Za-z_$][\w$]*)', p)
        if m:
            names.append(m.group(1) + m.group(2))
        elif p == '*':
            names.append('*')
    return names


def params_from_names(names):
    pos, kwonly, varargs, varkw, after_star = [], [], False, False, False
    for n in names:
        if n.startswith('**'):
            varkw = True
        elif n.startswith('*'):
            varargs = varargs or n != '*'
            after_star = True
        elif after_star:
            kwonly.append(n)
        else:
            pos.append(n)
    return {'pos': pos, 'kwonly': kwonly, 'varargs': varargs, 'varkw': varkw}


def parse_python_regex(text):
    """Fallback for files ast cannot parse (e.g. Python 2 code)."""
    out = {'defs': set(), 'top': set(), 'star': [], 'sigs': defaultdict(list), 'dyn': bool(PY_DYN_RE.search(text))}
    for m in PY_DEF_RE.finditer(text):
        out['defs'].add(m.group(1))
        out['sigs'][m.group(1)].append(params_from_names(split_params(m.group(2))))
    out['defs'].update(PY_CLASS_RE.findall(text))
    for a, b in PY_TOP_RE.findall(text):
        out['top'].add(a or b)
    out['defs'] |= out['top']
    for mod, names in PY_FROM_RE.findall(text):
        for n in names.split(','):
            n = n.strip().split(' as ')[-1].strip()
            if n == '*':
                level = len(mod) - len(mod.lstrip('.'))
                out['star'].append((level, mod.lstrip('.')))
            elif re.fullmatch(r'\w+', n):
                out['top'].add(n)
    for names in PY_IMPORT_RE.findall(text):
        for n in names.split(','):
            n = n.strip()
            if ' as ' in n:
                out['top'].add(n.split(' as ')[-1].strip())
            elif n:
                out['top'].add(n.split('.')[0])
    return out


# other languages: regexes over the whole file
GO_PATTERNS = [
    re.compile(r'^func[ \t]+(?:\([^)]*\)[ \t]*)?(\w+)', re.M),          # func F / func (r *T) M
    re.compile(r'^[ \t]*type[ \t]+(\w+)', re.M),                         # type T ...
    re.compile(r'^(?:var|const)[ \t]+(\w+)', re.M),                      # var x / const y
]
GO_SIG_RE = re.compile(r'^func[ \t]+(?:\([^)]*\)[ \t]*)?(\w+)(?:\[[^\]]*\])?\(([^)]*)\)', re.M)
TS_PATTERNS = [
    re.compile(r'\b(?:function\*?|class|interface|type|enum|namespace)[ \t]+([A-Za-z_$][\w$]*)'),
    re.compile(r'\b(?:const|let|var)[ \t]+([A-Za-z_$][\w$]*)'),
    # method shorthand at the start of a line: foo(...) {  /  async foo(...)  /  static get foo()
    re.compile(r'^[ \t]*(?:(?:public|private|protected|static|readonly|async|override|abstract|declare|get|set)'
               r'[ \t]+)*\*?[ \t]*#?([A-Za-z_$][\w$]*)[ \t]*(?:<[^>\n]*>)?[ \t]*\([^)\n]*\)[ \t]*(?::[^{\n]*)?\{[ \t]*$', re.M),
    # property: foo: (...) => / foo = (...) => / foo: function
    re.compile(r'^[ \t]*(?:(?:public|private|protected|static|readonly)[ \t]+)*([A-Za-z_$][\w$]*)[ \t]*[?!]?[ \t]*[:=]'
               r'[ \t]*(?:async[ \t]*)?(?:function|\([^)\n]*\)[ \t]*(?::[^=\n]*)?=>|[A-Za-z_$][\w$]*[ \t]*=>)', re.M),
    # object keys and interface members: foo: ... or foo?: ...
    re.compile(r'^[ \t]+(?:readonly[ \t]+)?([A-Za-z_$][\w$]*)[ \t]*\??[ \t]*:', re.M),
]
TS_EXPORT_LIST_RE = re.compile(r'\bexport[ \t]*(?:type[ \t]*)?\{([^}]*)\}')
TS_SIG_RE = re.compile(r'\bfunction\*?[ \t]+([A-Za-z_$][\w$]*)[ \t]*(?:<[^>\n]*>)?[ \t]*\(([^)]*)\)')
RS_PATTERNS = [
    re.compile(r'\b(?:fn|struct|enum|trait|type|const|static|mod|union)[ \t]+([A-Za-z_]\w*)'),
    re.compile(r'\bmacro_rules![ \t]*([A-Za-z_]\w*)'),
]
RS_SIG_RE = re.compile(r'\bfn[ \t]+([A-Za-z_]\w*)[ \t]*(?:<[^>{]*>)?[ \t]*\(([^)]*)\)')
TS_KEYWORDS = {'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'constructor', 'super', 'new',
               'typeof', 'await', 'import', 'export', 'else', 'do', 'try', 'with', 'yield', 'delete', 'void'}


def _block_members(text, opener_re):
    """Names declared one level inside blocks opened by opener_re (Go structs/interfaces/const
    groups, Rust enums): the first identifier on each line at depth 1."""
    names = set()
    for m in opener_re.finditer(text):
        depth, i, line_start = 1, m.end(), True
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch in '{(':
                depth += 1
            elif ch in '})':
                depth -= 1
            elif ch == '\n':
                line_start = True
                i += 1
                continue
            if line_start and depth == 1 and not ch.isspace():
                w = re.match(r'[A-Za-z_]\w*', text[i:i + 80])
                if w:
                    names.add(w.group(0))
                line_start = False
            i += 1
    return names


GO_BLOCK_RE = re.compile(r'\b(?:struct|interface)[ \t]*\{|^(?:const|var)[ \t]*\(', re.M)
RS_ENUM_RE = re.compile(r'\benum[ \t]+\w+[^{;]*\{')


def go_param_names(raw):
    """Go params 'a, b int, c ...string' -> ['a', 'b', 'c'] (types dropped). A list of bare types
    such as 'int, string' has no names and gives []."""
    parts = split_top(raw)
    if not any(len(p.split()) >= 2 for p in parts):
        return []
    return [p.split()[0] for p in parts if re.match(r'[A-Za-z_]\w*', p.split()[0])]


def parse_other(text, lang):
    out = {'defs': set(), 'top': set(), 'star': [], 'sigs': defaultdict(list)}
    if lang == 'go':
        for pat in GO_PATTERNS:
            out['defs'].update(pat.findall(text))
        out['defs'].update(_block_members(text, GO_BLOCK_RE))
        for name, raw in GO_SIG_RE.findall(text):
            out['sigs'][name].append(params_from_names(go_param_names(raw)))
    elif lang in ('ts', 'js'):
        for pat in TS_PATTERNS:
            out['defs'].update(n for n in pat.findall(text) if n not in TS_KEYWORDS)
        for names in TS_EXPORT_LIST_RE.findall(text):
            for n in names.split(','):
                n = n.strip().split(' as ')[-1].strip()
                if re.fullmatch(r'[A-Za-z_$][\w$]*', n):
                    out['defs'].add(n)
        for name, raw in TS_SIG_RE.findall(text):
            out['sigs'][name].append(params_from_names(split_params(raw)))
    elif lang == 'rs':
        for pat in RS_PATTERNS:
            out['defs'].update(pat.findall(text))
        out['defs'].update(_block_members(text, RS_ENUM_RE))
        for name, raw in RS_SIG_RE.findall(text):
            names = [n for n in split_params(raw) if n not in ('self', 'mut')]
            out['sigs'][name].append(params_from_names(names))
    return out


def parse_file(text, lang):
    return parse_python(text) if lang == 'py' else parse_other(text, lang)


# ----------------------------------------------------------------------------------------------
# One repository
# ----------------------------------------------------------------------------------------------

class Repo:
    """Path and symbol lookups for one blobless clone. Results are cached in memory and in
    cache_dir (so a second run does not redo git work)."""

    def __init__(self, path, cache_dir, allow_fetch=True, old_adds_cache=None, readonly=False):
        self.path = path
        self.name = os.path.basename(path.rstrip('/'))
        self.cache_dir = cache_dir
        self.allow_fetch = allow_fetch
        self.readonly = readonly          # True: only read the parse cache, never add to it
        self.old_adds_cache = old_adds_cache  # read-only cache from the pilot, if present
        os.makedirs(cache_dir, exist_ok=True)
        self._trees, self._src, self._blobsets, self._times, self._parsed = {}, {}, {}, {}, {}
        self._adds = None
        self._name_index = defaultdict(set)   # symbol name -> blob ids that define it
        self._import_index = defaultdict(set)  # name -> blob ids that import it at module level (Python)
        self._indexed_blobs = set()
        self.fetch_log = []

    # --- commits and trees ---
    def commit_time(self, commit):
        if commit not in self._times:
            # (git log, not git show: show may try to read file contents a blobless clone lacks)
            out = git(self.path, 'log', '-1', '--format=%ct', commit).strip()
            self._times[commit] = int(out) if out.isdigit() else None
        return self._times[commit]

    def has_commit(self, commit):
        return self.commit_time(commit) is not None

    def tree(self, commit):
        """{path: (blob id, 0)} for every file at the commit (sizes are checked when a file is read).
        Only the 24 most recently used commits are kept in memory."""
        if commit in self._trees:
            self._trees[commit] = self._trees.pop(commit)  # mark as recently used
            return self._trees[commit]
        files = {}
        for line in git(self.path, 'ls-tree', '-r', commit).splitlines():
            meta, path = line.split('\t', 1)
            mode, typ, oid = meta.split()
            if typ == 'blob':
                files[path] = (oid, 0)
        self._trees[commit] = files
        while len(self._trees) > 24:
            self._trees.pop(next(iter(self._trees)))
        return files

    def paths(self, commit):
        return self.tree(commit).keys()

    def head(self):
        return git(self.path, 'rev-parse', 'HEAD').strip()

    def commit_before(self, timestamp):
        """Newest first-parent commit of HEAD at or before a time (None if none)."""
        out = git(self.path, 'rev-list', '-1', '--first-parent', f'--before={int(timestamp)}', 'HEAD').strip()
        return out or None

    # --- path history (when each path was first added, over all refs) ---
    def path_adds(self):
        """{path: sorted list of commit times where the path was added}."""
        if self._adds is None:
            cache = os.path.join(self.cache_dir, self.name + '.adds.json')
            for f in (self.old_adds_cache, cache):
                if f and os.path.exists(f):
                    with open(f) as fh:
                        self._adds = json.load(fh)
                    break
            else:
                out = git(self.path, 'log', '--all', '--no-renames', '--diff-filter=A', '--name-only', '--format=@@%ct')
                adds, t = defaultdict(list), 0
                for line in out.split('\n'):
                    if line.startswith('@@'):
                        t = int(line[2:])
                    elif line:
                        adds[line].append(t)
                self._adds = {p: sorted(v) for p, v in adds.items()}
                with open(cache, 'w') as fh:
                    json.dump(self._adds, fh)
            self._adds_by_name = defaultdict(list)
            for p in self._adds:
                self._adds_by_name[p.rsplit('/', 1)[-1]].append(p)
        return self._adds

    # --- file contents ---
    def source_blobs(self, commit):
        """{path: blob id} for the source files we index at a commit (kept for every commit)."""
        if commit not in self._src:
            self._src[commit] = {p: oid for p, (oid, _) in self.tree(commit).items() if source_lang(p)}
        return self._src[commit]

    def missing(self, oids):
        if not oids:
            return []
        out = git(self.path, 'cat-file', '--batch-check', stdin='\n'.join(oids) + '\n')
        return [l.split()[0] for l in out.splitlines() if l.endswith(' missing')]

    def fetch_missing(self, commits):
        """Download, in one request, the source files at these commits that are not local yet."""
        want = set()
        for c in commits:
            want.update(self.source_blobs(c).values())
            # small manifest files: package names (package.json), Go module path (go.mod), Rust crates
            want.update(oid for p, (oid, _) in self.tree(c).items()
                        if p.rsplit('/', 1)[-1] in ('package.json', 'go.mod', 'Cargo.toml')
                        and 'node_modules' not in p and p.count('/') <= 3)
        missing = self.missing(sorted(want))
        status = 'nothing to fetch'
        if missing and self.allow_fetch:
            r = subprocess.run(['git', '-C', self.path, '-c', 'fetch.negotiationAlgorithm=noop', '-c', 'gc.auto=0',
                                '-c', 'maintenance.auto=false', 'fetch', 'origin', '--no-tags',
                                '--no-write-fetch-head', '--recurse-submodules=no', '--filter=blob:none', '--stdin'],
                               input='\n'.join(missing) + '\n', capture_output=True, text=True)
            status = 'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr[-300:]
        elif missing:
            status = 'not allowed to fetch'
        self.fetch_log.append({'repo': self.name, 'commits': len(commits), 'wanted': len(want),
                               'missing_before': len(missing), 'status': status})
        return status

    def read_blobs(self, oids):
        """{blob id: text} for blobs that are present locally (missing ones are skipped)."""
        out = {}
        if not oids:
            return out
        r = subprocess.run(['git', '-C', self.path, 'cat-file', '--batch'], input=('\n'.join(oids) + '\n').encode(),
                           capture_output=True, env=NO_FETCH)
        data, i = r.stdout, 0
        while i < len(data):
            nl = data.index(b'\n', i)
            header = data[i:nl].decode(errors='replace').split()
            i = nl + 1
            if len(header) < 3 or header[1] == 'missing':
                continue
            size = int(header[2])
            out[header[0]] = data[i:i + size].decode('utf-8', errors='replace')
            i += size + 1
        return out

    # --- symbol definitions ---
    def index_commit(self, commit):
        """Parse every source file at the commit that is not parsed yet (cached per blob)."""
        cache_file = os.path.join(self.cache_dir, self.name + '.parsed.jsonl')
        if not getattr(self, '_cache_loaded', False):  # load earlier work once
            self._cache_loaded = True
            if os.path.exists(cache_file):
                with open(cache_file) as fh:
                    for line in fh:
                        d = json.loads(line)
                        self._add_parsed(d['oid'], d)
        if self.readonly:
            return
        todo = {oid: p for p, oid in self.source_blobs(commit).items() if oid not in self._indexed_blobs}
        if not todo:
            return
        texts = {}
        oids = sorted(todo)
        for i in range(0, len(oids), 2000):
            texts.update(self.read_blobs(oids[i:i + 2000]))
        with open(cache_file, 'a') as fh:
            for oid, path in todo.items():
                if oid not in texts:
                    continue  # not downloaded; lookups treat this file as unknown
                if len(texts[oid]) > MAX_BLOB_BYTES:  # huge (often generated) files: recorded as empty
                    p = {'defs': set(), 'top': set(), 'star': [], 'sigs': {}, 'dyn': True}
                else:
                    p = parse_file(texts[oid], source_lang(path))
                d = {'oid': oid, 'defs': sorted(p['defs']), 'top': sorted(p['top']), 'star': p['star'],
                     'sigs': {k: v for k, v in p['sigs'].items()}, 'dyn': p.get('dyn', False)}
                fh.write(json.dumps(d) + '\n')
                self._add_parsed(oid, d)

    def _add_parsed(self, oid, d):
        self._parsed[oid] = {'top': set(d['top']), 'star': [tuple(s) for s in d['star']], 'sigs': d['sigs'],
                             'dyn': d.get('dyn', False)}
        for n in d['defs']:
            self._name_index[n].add(oid)
        for n in set(d['top']) - set(d['defs']):
            self._import_index[n].add(oid)
        self._indexed_blobs.add(oid)

    def coverage(self, commit):
        """Share of the commit's source files whose contents we could parse."""
        blobs = set(self.source_blobs(commit).values())
        return (len(blobs & self._indexed_blobs) / len(blobs)) if blobs else None

    def defined(self, name, commit):
        """True if some indexed source file at the commit defines this name."""
        oids = self._name_index.get(name)
        if not oids:
            return False
        at = self._blobset(commit)
        return any(o in at for o in oids)

    def imported(self, name, commit):
        """True if some Python file at the commit imports this name at module level (so the name
        is usable in the repo even though a library defines it, e.g. FastAPI's TestClient)."""
        oids = self._import_index.get(name)
        if not oids:
            return False
        at = self._blobset(commit)
        return any(o in at for o in oids)

    def _blobset(self, commit):
        if commit not in self._blobsets:
            self._blobsets[commit] = set(self.source_blobs(commit).values())
        return self._blobsets[commit]

    def signatures(self, name, commit):
        """Parameter lists of every def/func/fn with this name at the commit."""
        out = []
        at = self._blobset(commit)
        for oid in self._name_index.get(name, ()):
            if oid in at:
                out.extend(self._parsed[oid]['sigs'].get(name, []))
        return out

    # --- Python modules ---
    def py_module_file(self, dotted, commit):
        """'a.b.c' -> ('a/b/c.py' or 'a/b/c/__init__.py' or the src/ variants, 'module'|'package'), or None."""
        files = self.source_blobs(commit)
        base = dotted.replace('.', '/')
        for root in ('', 'src/', 'lib/'):
            if root + base + '.py' in files:
                return root + base + '.py', 'module'
            if root + base + '/__init__.py' in files:
                return root + base + '/__init__.py', 'package'
            if root + base + '.pyi' in files:
                return root + base + '.pyi', 'module'
        return None

    def py_module_names(self, module_path, commit, depth=0):
        """Names bound at module level in a Python file (following 'from X import *' up to 3 levels).
        Returns None if the file's contents are not available. The set has the extra entry
        '<dynamic>' if the module (or a star-imported one) creates names at run time."""
        files = self.source_blobs(commit)
        if module_path not in files:
            return None
        oid = files[module_path]
        info = self._parsed.get(oid)
        if info is None:
            return None
        names = set(info['top'])
        if info.get('dyn'):
            names.add('<dynamic>')
        if depth < 3:
            for level, mod in info['star']:
                if level:  # relative: from .x import *
                    pkg = module_path.rsplit('/', 1)[0].split('/')
                    pkg = pkg[:len(pkg) - (level - 1)] if level > 1 else pkg
                    target = '.'.join(pkg + ([mod] if mod else []))
                    target = re.sub(r'^(src|lib)\.', '', target)
                else:
                    target = mod
                hit = self.py_module_file(target, commit)
                if hit:
                    sub = self.py_module_names(hit[0], commit, depth + 1)
                    if sub:
                        names |= {n for n in sub if not n.startswith('_') or n == '<dynamic>'}
        return names

    def py_top_packages(self, commit):
        """Top-level importable Python package/module names of the repo (e.g. {'django'})."""
        tops = set()
        for p in self.tree(commit):
            parts = p.split('/')
            if parts[0] in ('src', 'lib') and len(parts) >= 3 and parts[2] == '__init__.py':
                tops.add(parts[1])
            elif len(parts) == 2 and parts[1] == '__init__.py':
                tops.add(parts[0])
        # a few repos ship a top-level module file instead of a package (e.g. dateutil is a package,
        # but six.py-style single modules exist); include them if they look like a library module
        return {t for t in tops if t not in ('tests', 'test', 'docs', 'doc', 'examples', 'benchmarks', 'scripts')}

    def snapshots(self, base_commits, years_back=(2, 5)):
        """Commits to index for 'other version' checks: the given base commits, HEAD, and
        first-parent commits some years before the oldest base commit."""
        times = [self.commit_time(c) for c in base_commits if self.commit_time(c)]
        snaps = set(c for c in base_commits if self.commit_time(c))
        head = self.head()
        if head:
            snaps.add(head)
        if times:
            for y in years_back:
                c = self.commit_before(min(times) - y * 365.25 * 86400)
                if c:
                    snaps.add(c)
        return sorted(snaps, key=lambda c: self.commit_time(c) or 0)
