"""Print what a human needs to check one candidate by hand: the patch line, the definition at the base
commit, the step where the agent first wrote the line, every command before that step that touched the
definition's file or name, and every tool output (before and after) that mentions the name.

Usage: .venv/bin/python -I scripts/spot_check_candidate.py MODEL INSTANCE_ID FACT_KEY_SUBSTRING
"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))  # our own script folder (trusted), for load_steps()
from agent_targeted_candidates import RUNS, RAW, REPOS, PARQUET, load_steps  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

model, iid, key = sys.argv[1:4]
facts = [json.loads(l) for l in open(os.path.join(ROOT, 'outputs/agent_targeted_facts.jsonl'))]
f = next(x for x in facts if x['model'] == model and x['instance_id'] == iid and key in x.get('fact_key', ''))
task = pq.read_table(PARQUET).to_pandas().set_index('instance_id').loc[iid]
print('FACT:', f['detail'])
print('patch line', f['line'], 'in', f['edit_file'], ':', f['code'])
print('heuristic: write step', f['write_step'], '| def seen step', f['def_seen_step'], '| use seen step',
      f['use_seen_step'], '| unseen', f['unseen'], '| strong', f['strong'], '| steps', f['n_steps'])
rp = os.path.join(REPOS, task.repo.replace('/', '_'))
src = subprocess.run(['git', '-C', rp, 'show', f"{task.base_commit}:{f['target_file']}"], capture_output=True,
                     text=True, env=dict(os.environ, GIT_NO_LAZY_FETCH='1')).stdout.split('\n')
if f.get('target_line'):
    a = f['target_line'] - 1
    print(f"--- definition at base: {f['target_file']}:{f['target_line']}")
    print('\n'.join(src[a:a + 6]))
traj = json.load(open(os.path.join(RAW, RUNS[model], 'trajs', iid, iid + '.traj.json')))
steps = load_steps(traj)
name = f.get('def_name') or ''
use = f.get('use_name') or name
fname = f['target_file'].rsplit('/', 1)[-1]
w = f['write_step'] or len(steps)
print(f'--- commands before step {w} that mention {fname} or {use}:')
for k in range(1, w):
    c = steps[k]['cmd']
    if fname in c or re.search(rf'\b{re.escape(use)}\b', c) or (name and re.search(rf'\b{re.escape(name)}\b', c)):
        print(f'  step {k}: {c[:300]!r}')
print(f'--- tool outputs that mention {use} / {name} (+-2 lines), all steps:')
for k, s in enumerate(steps):
    lines = s['obs'].split('\n')
    hits = [i for i, l in enumerate(lines) if re.search(rf'\b{re.escape(use)}\b', l)
            or (name not in ('__init__', '__new__') and name and re.search(rf'\bdef\s+{re.escape(name)}\b', l))]
    if hits:
        print(f'  [step {k}{" (before write)" if k < w else ""}] {len(hits)} lines')
        for i in hits[:4]:
            print('     ' + ' | '.join(x.strip()[:140] for x in lines[max(0, i - 1):i + 2]))
