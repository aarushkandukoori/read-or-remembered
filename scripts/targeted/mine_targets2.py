"""Broader agent-targeted mining: every repo function/method a model's own patch calls (added lines),
resolved to a definition in the repo at the base commit, and whether the model ever saw that
definition in the run. Lists which trap types would apply:
  argswap  - 2+ positional args, first two params required
  kwrename - the call passes a keyword that is a named parameter of the definition
  default  - the call leaves out a parameter whose default is a literal
  rename   - any call (rename the function everywhere)
Only django 3.0 and sympy 1.1 tasks (their Docker layers are already pulled).
Usage: python -I mine_targets2.py RR_ROOT OUT_JSONL RUN [RUN ...]"""
import ast, json, os, re, subprocess, sys
from collections import defaultdict
import pyarrow.parquet as pq

rr, out_path, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = pq.read_table(os.path.join(rr, 'data/meta/swebv/data/test-00000-of-00001.parquet')).to_pandas().set_index('instance_id')
KEEP = None  # all django and sympy versions
SKIP = {'print', 'len', 'isinstance', 'str', 'int', 'list', 'dict', 'set', 'tuple', 'super', 'getattr', 'setattr',
        'hasattr', 'range', 'enumerate', 'zip', 'sorted', 'any', 'all', 'type', 'repr', 'min', 'max', 'map', 'filter',
        'bool', 'float', 'next', 'iter', 'format', 'join', 'append', 'get', 'items', 'keys', 'values', 'update',
        'split', 'strip', 'startswith', 'endswith', 'replace', 'lower', 'upper', 'extend', 'pop', 'add', 'copy'}


def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True).stdout


index_cache = {}


def def_index(repo, commit):
    """name -> list of (path, args) for every function/method def (non-test files), built once per
    repo version from a checked-out worktree (signatures barely change between nearby commits)."""
    wt = os.path.join(HERE, 'work', f"index-{task_repo.split('/')[1]}-{task_version}")
    key = wt
    if key in index_cache:
        return index_cache[key]
    idx = defaultdict(list)
    files = []
    for root, dirs, fs in os.walk(wt):
        dirs[:] = [d for d in dirs if d not in ('.git', 'tests')]
        files += [os.path.relpath(os.path.join(root, f), wt) for f in fs if f.endswith('.py')]
    for p in files:
        src = open(os.path.join(wt, p), encoding='utf-8', errors='replace').read()
        try:
            tr = ast.parse(src)
        except Exception:
            continue
        for n in ast.walk(tr):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                idx[n.name].append((p, n.args))
    index_cache[key] = idx
    return idx


rows = []
for run in runs:
    model = run.split('_')[-1]
    root = os.path.join(rr, 'data/raw', run, 'trajs')
    for iid in sorted(os.listdir(root)):
        if iid not in t.index or t.loc[iid].repo not in ('django/django', 'sympy/sympy'):
            continue
        task = t.loc[iid]
        f = os.path.join(root, iid, iid + '.traj.json')
        if not os.path.exists(f):
            continue
        traj = json.load(open(f))
        patch = traj.get('info', {}).get('submission') or ''
        seen = '\n'.join(str(m.get('content') or '') for m in traj['messages'] if m.get('role') in ('user', 'tool'))
        repo = os.path.join(HERE, 'repos', task.repo.replace('/', '_'))
        task_repo, task_version = task.repo, task.version
        tree_files = set(git(repo, 'ls-tree', '-r', '--name-only', task.base_commit).split('\n'))
        added, cur = [], None
        for l in patch.split('\n'):
            if l.startswith('+++ b/'):
                cur = l[6:]
            elif l.startswith('+') and not l.startswith('+++') and cur and cur.endswith('.py') \
                    and cur in tree_files and '/tests/' not in cur and not cur.startswith('tests/'):
                added.append(l[1:])
        idx = def_index(repo, task.base_commit)
        for line in added:
            code = line.strip()
            if not code or code.startswith('#'):
                continue
            try:
                tree = ast.parse(code.rstrip(':') if not code.endswith(':') else code + ' pass')
            except SyntaxError:
                try:
                    tree = ast.parse(re.sub(r'^(return|elif|if|while|assert)\s+', '', code.rstrip(':')))
                except SyntaxError:
                    continue
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                name = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else None
                if not name or name in SKIP or name.startswith('__') or name not in idx or len(idx[name]) > 3:
                    continue
                if isinstance(n.func, ast.Attribute) and not (isinstance(n.func.value, ast.Name) and
                                                              n.func.value.id in ('self', 'cls', 'transaction', 'timezone', 'models', 'connection', 'connections')):
                    pass  # keep, but marked as attribute call below
                defs = idx[name]
                path, args = defs[0]
                params = [a.arg for a in args.posonlyargs + args.args]
                is_method = bool(params) and params[0] in ('self', 'cls')
                if is_method:
                    params = params[1:]
                n_def = len(args.defaults)
                required = params[:len(params) - n_def] if n_def else params
                defaults = dict(zip(params[len(params) - n_def:], args.defaults)) if n_def else {}
                npos = len(n.args)
                kws = [k.arg for k in n.keywords if k.arg]
                types = ['rename']
                if npos >= 2 and len(required) >= 2 and not any(isinstance(a, ast.Starred) for a in n.args):
                    types.append('argswap')
                if any(k in params for k in kws):
                    types.append('kwrename')
                passed = set(params[:npos]) | set(kws)
                if any(p not in passed and isinstance(d, ast.Constant) for p, d in defaults.items()):
                    types.append('default')
                kind = 'name' if isinstance(n.func, ast.Name) else ('self' if isinstance(n.func.value, ast.Name) and n.func.value.id in ('self', 'cls') else 'attr')
                rows.append(dict(model=model, instance_id=iid, func=name, def_file=path, n_defs=len(defs), call_kind=kind,
                                 method=is_method, params=params[:4], call=ast.unparse(n)[:100], types=types,
                                 saw_definition=f'def {name}(' in seen))
with open(out_path, 'w') as o:
    for r in rows:
        o.write(json.dumps(r) + '\n')
print(len(rows), 'calls ->', out_path)
