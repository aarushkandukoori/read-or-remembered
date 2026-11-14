"""Agent-targeted trap candidates: repo functions that a model's OWN submitted patch calls with
2+ positional arguments, defined in another file, whose definition the model never saw in that run.
Reads public mini-swe-agent trajectories (read-only). Usage:
  python -I mine_targets.py RR_ROOT OUT_JSONL RUN [RUN ...]
"""
import ast, json, os, re, subprocess, sys
import pyarrow.parquet as pq

rr, out_path, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
t = pq.read_table(os.path.join(rr, 'data/meta/swebv/data/test-00000-of-00001.parquet')).to_pandas().set_index('instance_id')
KEEP = {('django/django', '3.0'), ('sympy/sympy', '1.1')}


def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True).stdout


def module_file(mod, tree):
    for base in (mod.replace('.', '/'), 'src/' + mod.replace('.', '/')):
        for c in (base + '.py', base + '/__init__.py'):
            if c in tree:
                return c


def parse_patch(patch):
    files, cur = {}, None
    for line in patch.split('\n'):
        if line.startswith('+++ b/'):
            cur = line[6:]; files[cur] = []
        elif cur and line.startswith('+') and not line.startswith('+++'):
            files[cur].append(line[1:])
    return files


def text_of(m):
    c = m.get('content') or ''
    return c if isinstance(c, str) else json.dumps(c)


rows = []
for run in runs:
    model = run.split('_')[-1]
    root = os.path.join(rr, 'data/raw', run, 'trajs')
    for iid in sorted(os.listdir(root)):
        if iid not in t.index or (t.loc[iid].repo, t.loc[iid].version) not in KEEP:
            continue
        task = t.loc[iid]
        f = os.path.join(root, iid, iid + '.traj.json')
        if not os.path.exists(f):
            continue
        traj = json.load(open(f))
        patch = traj.get('info', {}).get('submission') or ''
        seen = '\n'.join(text_of(m) for m in traj['messages'] if m.get('role') in ('user', 'tool'))
        repo = os.path.join(rr, 'data/repos', task.repo.replace('/', '_'))
        tree = set(git(repo, 'ls-tree', '-r', '--name-only', task.base_commit).split('\n'))
        for path, added in parse_patch(patch).items():
            if not path.endswith('.py') or 'tests/' in path or path not in tree:
                continue
            base_src = git(repo, 'show', f'{task.base_commit}:{path}')
            try:
                tr = ast.parse(base_src)
            except SyntaxError:
                continue
            imports = {}
            for n in ast.walk(tr):
                if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    for a in n.names:
                        imports[a.asname or a.name] = (n.module, a.name)
            # local imports written inside the patch itself
            for line in added:
                m = re.match(r'\s*from ([\w.]+) import (.+)', line)
                if m:
                    for nm in re.split(r'\s*,\s*', m.group(2).strip('() ')):
                        nm = nm.split(' as ')
                        imports[nm[-1].strip()] = (m.group(1), nm[0].strip())
            for line in added:
                for cm in re.finditer(r'(?<![\w.])([A-Za-z_]\w*)\((?!\))', line):
                    name = cm.group(1)
                    if name not in imports:
                        continue
                    mod, real = imports[name]
                    mf = module_file(mod, tree)
                    if not mf or mf == path:
                        continue
                    try:
                        call = ast.parse(line.strip().rstrip(':').split(' = ', 1)[-1].lstrip('return ').strip(), mode='eval')
                    except SyntaxError:
                        call = None
                    npos = None
                    if call is not None:
                        for n in ast.walk(call):
                            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
                                npos = len(n.args) if not any(isinstance(a, ast.Starred) for a in n.args) else None
                    src = git(repo, 'show', f'{task.base_commit}:{mf}')
                    dm = re.search(rf'^def {real}\(([^)]*)\)', src, re.M)
                    if not dm or npos is None or npos < 2:
                        continue
                    params = [p.strip() for p in dm.group(1).split(',') if p.strip()]
                    if len(params) < 2 or any(c in params[0] + params[1] for c in '=*'):
                        continue
                    rows.append(dict(model=model, instance_id=iid, version=task.version, func=real, def_file=mf,
                                     fix_file=path, params=params[:3], call=line.strip()[:120],
                                     saw_definition=f'def {real}(' in seen,
                                     calls_in_fix_file_at_base=len(re.findall(rf'(?<![\w.]){real}\(', base_src)),
                                     resolved=None))
with open(out_path, 'w') as o:
    for r in rows:
        o.write(json.dumps(r) + '\n')
print(len(rows), 'candidate calls ->', out_path)
