"""For model-typed tool outputs, classify the repo file paths they list: already seen in a real
observation, or new and real / future-only / stale / never existed at the base commit.
Usage: python3 -I scripts/fake_output_paths.py RUN_DIR SWEBV_PARQUET REPOS_DIR"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_pilot import PATH_RE, norm, Repo, resolve
import pyarrow.parquet as pq
run, parquet, repos_dir = sys.argv[1:4]
t = pq.read_table(parquet).to_pandas()
tasks = {r.instance_id: dict(repo=r.repo, base_commit=r.base_commit) for r in t.itertuples()}
repos = {}; c = collections.Counter(); ex = collections.defaultdict(list)
for iid in sorted(os.listdir(os.path.join(run, 'trajs'))):
    f = os.path.join(run, 'trajs', iid, iid + '.traj.json')
    task = tasks.get(iid)
    if not task or not os.path.exists(f): continue
    rp = os.path.join(repos_dir, task['repo'].replace('/', '_'))
    repo = repos.setdefault(rp, Repo(rp)); paths, by_name = repo.tree(task['base_commit'])
    d = json.load(open(f)); seen = '\n'.join(str(m.get('content') or '') for m in d['messages'][:2])
    for m in d['messages']:
        cont = m.get('content') or ''
        if not isinstance(cont, str): continue
        if m.get('role') != 'assistant':
            seen += '\n' + cont; continue
        if not (re.search(r'<returncode>\s*-?\d+\s*</returncode>', cont) or re.search(r'<output>[\s\S]{20,}</output>', cont)):
            continue
        for raw in set(PATH_RE.findall(cont)):
            p = norm(raw)
            if '/' not in p or p.startswith('/'): continue
            if p in seen: c['already seen'] += 1; continue
            if resolve(p, paths, by_name): k = 'new, real'
            else:
                ev = repo.history(); hit = resolve(p, ev, repo.by_name)
                if not hit: k = 'new, never existed'
                else:
                    t0 = task.setdefault('_t', repo.commit_time(task['base_commit']))
                    k = 'new, stale' if ev[hit][0] < t0 else 'new, future-only'
            c[k] += 1
            if len(ex[k]) < 5: ex[k].append(f'{iid}: {p}')
tot = sum(c.values())
print(run.split('/')[-1], 'paths inside typed tool outputs:', tot)
for k, v in c.most_common(): print(f'  {k}: {v} ({100*v/tot:.1f}%)')
for k in ('new, future-only', 'new, stale', 'new, never existed'):
    print(' ', k, 'examples:', ex[k])
