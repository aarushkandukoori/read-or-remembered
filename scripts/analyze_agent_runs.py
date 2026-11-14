"""What did each agent run do around the trapped function?

For each trajectory in outputs/agent_runs (or RR_RUNS): did the agent open the definition of
get_permission_codename, did it see the call sites in options.py, what argument order did its
reasoning state, and what order did its submitted patch use?
Usage: .venv/bin/python -I scripts/analyze_agent_runs.py
"""
import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'outputs' / os.environ.get('RR_RUNS', 'agent_runs')
FUNC = 'get_permission_codename'
CALL = re.compile(FUNC + r'\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)')
DEF_FILE = 'django/contrib/auth/__init__.py'


def order(a, b):
    a, b = a.strip(), b.strip()
    if '=' in a or '=' in b:
        return 'keywords'
    if 'opts' in a or '_meta' in a:
        return 'opts first'
    if 'opts' in b or '_meta' in b:
        return 'action first'
    return f'? ({a}, {b})'


def steps(traj):
    out = []
    msgs = traj['messages']
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        cmds = []
        for tc in m.get('tool_calls') or []:
            try:
                cmds.append(json.loads(tc['function']['arguments']).get('command', ''))
            except Exception:
                pass
        obs = '\n'.join(str(n.get('content') or '') for n in msgs[i + 1:i + 1 + max(1, len(cmds))]
                        if n.get('role') in ('tool', 'user'))
        out.append(dict(reasoning=m.get('reasoning') or m.get('reasoning_content') or '',
                        text=m.get('content') or '', cmds=cmds, obs=obs))
    return out


rows = []
for f in sorted(RUNS.glob('*.traj.json')):
    traj = json.load(open(f))
    cond = f.name.split('-')[0]
    st = steps(traj)
    patch = traj.get('info', {}).get('submission') or ''
    added = '\n'.join(l for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
    first_def = first_site = first_claim = first_write = None
    claims = []
    for k, s in enumerate(st, 1):
        joined_cmd = '\n'.join(s['cmds'])
        if first_def is None and (f'def {FUNC}' in s['obs'] or (DEF_FILE in joined_cmd and FUNC in s['obs'])):
            first_def = k
        if first_site is None and CALL.search(s['obs']):
            first_site = k
        for a, b in CALL.findall(s['reasoning'] + '\n' + s['text']):
            claims.append((k, order(a, b)))
        if first_write is None and FUNC in joined_cmd and re.search(r'(cat\s*>|sed -i|python3? -c|python3? - <<|>>|tee )', joined_cmd):
            first_write = k
    patch_orders = sorted({order(a, b) for a, b in CALL.findall(added)})
    rows.append(dict(run=f.name.replace('.traj.json', ''), steps=len(st),
                     cost=round(traj.get('info', {}).get('model_stats', {}).get('instance_cost', 0), 4),
                     saw_definition_at=first_def, saw_call_site_at=first_site,
                     first_reasoning_order=claims[0] if claims else None,
                     patch_call_order=patch_orders or ['no call in patch']))
for r in rows:
    print(json.dumps(r))
json.dump(rows, open(RUNS / 'analysis.json', 'w'), indent=1)
