"""Which SWE-bench tasks allow a memory trap on a function the gold fix itself calls?

For each task we apply the gold patch to the files it touches (at the base commit), find the calls
on the lines the fix adds, and resolve each callee to its definition (same file, or a module the file
imports). Two kinds of trap are counted:
  argswap  - the fix passes 2+ positional arguments to a repo function. Swapping those two parameters
             in the definition and at every call site keeps behavior the same, but an agent that writes
             the call from memory passes them in the old order.
  default  - the fix relies on a default value of a repo function. Changing the default and writing
             the old value at every call site that relied on it keeps behavior the same, but an agent
             that leaves the argument out from memory gets the new default.
Usage: python3 -I scripts/trap_candidates.py SWEBV_PARQUET REPOS_DIR OUT_JSONL
"""
import ast, json, os, re, subprocess, sys
import pyarrow.parquet as pq


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout


def parse_patch(patch):
    """{path: [(old_start, old_len, new_start, new_len, lines)]} for each file in a unified diff."""
    files, cur = {}, None
    for line in patch.split('\n'):
        if line.startswith('+++ b/'):
            cur = line[6:]
            files[cur] = []
        elif line.startswith('@@') and cur:
            m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            files[cur].append([int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1), []])
        elif cur and files[cur] and line[:1] in (' ', '+', '-'):
            files[cur][-1][4].append(line)
    return files


def apply_hunks(text, hunks):
    """Apply hunks to text; return (new_text, set of added line numbers in the new text)."""
    old = text.split('\n')
    out, added, pos = [], set(), 0
    for os_, ol, ns, nl, lines in hunks:
        start = os_ - 1 if ol else os_
        out.extend(old[pos:start])
        pos = start
        for l in lines:
            if l.startswith(' '):
                out.append(old[pos]); pos += 1
            elif l.startswith('-'):
                pos += 1
            else:
                out.append(l[1:]); added.add(len(out))
    out.extend(old[pos:])
    return '\n'.join(out), added


def module_file(mod, tree):
    for base in (mod.replace('.', '/'), 'src/' + mod.replace('.', '/'), 'lib/' + mod.replace('.', '/')):
        for cand in (base + '.py', base + '/__init__.py'):
            if cand in tree:
                return cand
    return None


def defs_in(source):
    """{name: FunctionDef} for top-level functions, plus {Class.method: FunctionDef}."""
    try:
        t = ast.parse(source)
    except SyntaxError:
        return {}
    out = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f'{n.name}.{m.name}'] = m
    return out


def enclosing_class(tree, lineno):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.lineno <= lineno <= getattr(n, 'end_lineno', n.lineno):
            return n.name
    return None


def classify(call, fdef, is_method):
    a = fdef.args
    params = [p.arg for p in a.posonlyargs + a.args]
    if is_method and params and params[0] in ('self', 'cls'):
        params = params[1:]
    n_def = len(a.defaults)
    defaults = params[len(params) - n_def:] if n_def else []
    if any(isinstance(x, ast.Starred) for x in call.args) or any(k.arg is None for k in call.keywords):
        return []
    out = []
    npos = len(call.args)
    if npos >= 2 and len(params) >= 2 and not a.vararg:
        out.append(('argswap', params[0], params[1]))
    passed = set(params[:npos]) | {k.arg for k in call.keywords}
    lit = lambda d: isinstance(d, ast.Constant) and d.value is not None and not isinstance(d.value, bool) or \
        isinstance(d, ast.Constant) and isinstance(d.value, bool)
    for p, d in zip(defaults, a.defaults):
        if p not in passed and lit(d):
            out.append(('default', p, repr(d.value)))
    return out


def main():
    parquet, repos_dir, out_path = sys.argv[1:4]
    t = pq.read_table(parquet).to_pandas()
    n_tasks = n_with = 0
    with open(out_path, 'w') as out:
        for r in t.itertuples():
            rp = os.path.join(repos_dir, r.repo.replace('/', '_'))
            tree = set(git(rp, 'ls-tree', '-r', '--name-only', r.base_commit).split('\n'))
            n_tasks += 1
            cands = []
            for path, hunks in parse_patch(r.patch).items():
                if not path.endswith('.py') or '/tests/' in path or path.startswith('tests/'):
                    continue
                base = git(rp, 'show', f'{r.base_commit}:{path}')
                try:
                    new, added = apply_hunks(base, hunks)
                except IndexError:
                    continue  # new file or context mismatch
                try:
                    tr = ast.parse(new)
                except SyntaxError:
                    continue
                local = defs_in(new)
                imports = {}
                pkg = path.rsplit('/', 1)[0].split('/')
                if pkg and pkg[0] in ('src', 'lib'):
                    pkg = pkg[1:]
                for n in ast.walk(tr):
                    if isinstance(n, ast.ImportFrom) and (n.module or n.level):
                        if n.level:  # relative import: resolve against this file's package
                            up = pkg[:len(pkg) - (n.level - 1)] if n.level > 1 else pkg
                            mod = '.'.join(up + ([n.module] if n.module else []))
                        else:
                            mod = n.module
                        for a in n.names:
                            imports[a.asname or a.name] = (mod, a.name)
                base_defs = defs_in(base)  # defs that already existed before the fix
                cache = {}
                for n in ast.walk(tr):
                    if not isinstance(n, ast.Call) or n.lineno not in added:
                        continue
                    f, fdef, method, where = n.func, None, False, None
                    if isinstance(f, ast.Name):
                        if f.id in local and f.id in base_defs:
                            fdef, where = local[f.id], path
                        elif f.id in imports:
                            mod, name = imports[f.id]
                            mf = module_file(mod, tree)
                            if mf:
                                if mf not in cache:
                                    cache[mf] = defs_in(git(rp, 'show', f'{r.base_commit}:{mf}'))
                                fdef, where = cache[mf].get(name), mf
                        name = f.id
                    elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ('self', 'cls'):
                        cls = enclosing_class(tr, n.lineno)
                        if cls and f'{cls}.{f.attr}' in local and f'{cls}.{f.attr}' in base_defs:
                            fdef, method, where = local[f'{cls}.{f.attr}'], True, path
                        name = f.attr
                    if fdef is None:
                        continue
                    for kind, x, y in classify(n, fdef, method):
                        cands.append(dict(kind=kind, callee=name, defined_in=where, a=x, b=y,
                                          cross_file=where != path, call=ast.unparse(n)[:120], fix_file=path))
            n_with += bool(cands)
            out.write(json.dumps(dict(instance_id=r.instance_id, repo=r.repo, candidates=cands)) + '\n')
    print(f'{n_with} of {n_tasks} tasks have at least one candidate -> {out_path}')


if __name__ == '__main__':
    main()
