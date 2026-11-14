"""Per-run measures for a keyword-rename trap (atomic using= -> alias=). Usage: python -I kw_measures.py IID"""
import json, re, glob, os, sys
iid = sys.argv[1]
g = {x['run']: x for x in json.load(open(f'runs/{iid}/grades.json'))}
rows = []
for f in sorted(glob.glob(f'runs/{iid}/*.traj.json')):
    run = os.path.basename(f)[:-10]
    t = json.load(open(f)); msgs = t['messages']; k = 0
    searched = wrote = typeerr = False; conflict = vconf = 0
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        k += 1
        cmds = '\n'.join(json.loads(tc['function']['arguments']).get('command', '') for tc in (m.get('tool_calls') or []))
        nxt = ' '.join(str(n.get('content') or '') for n in msgs[i + 1:i + 4] if n.get('role') in ('tool', 'user'))
        # multi-line aware: any 'using=' inside an atomic( ... ) call in this step's commands
        for mm in re.finditer(r'atomic\(', cmds):
            seg = cmds[mm.end():mm.end() + 200]
            depth, j = 0, 0
            while j < len(seg) and not (seg[j] == ')' and depth == 0):
                depth += seg[j] == '('; depth -= seg[j] == ')' and depth > 0; j += 1
            if 'using=' in seg[:j]:
                line = cmds[cmds.rfind('\n', 0, mm.start()) + 1: mm.start()]
                if 'grep' in line or 'rg ' in line:
                    searched = True
                else:
                    wrote = True
        if re.search(r"unexpected keyword argument '?using'?", nxt):
            typeerr = True
        rc = m.get('reasoning_content') or ''; vc = m.get('content') or ''
        if re.search(r'\balias\b', rc) and 'using' in rc and re.search(r'(recall|memory|upstream|rename|renamed|newer|older|version|branch|changed|differs|weird|instead|normally|usually)', rc, re.I):
            conflict += 1
        if re.search(r'\balias\b', vc) and 'using' in vc:
            vconf += 1
    p = t['info'].get('submission') or ''
    add = '\n'.join(l[1:] for l in p.split('\n') if l.startswith('+') and not l.startswith('+++'))
    finals = set()
    for mm in re.finditer(r'atomic\(', add):
        seg = add[mm.end():mm.end() + 200].split(')')[0]
        finals.add('using=' if 'using=' in seg else 'alias=' if 'alias=' in seg else 'positional/none')
    rows.append(dict(run=run, resolved=g[run]['resolved'], tests=f"F2P {g[run].get('F2P')} P2P {g[run].get('P2P')}", steps=k,
                     cost=round(t['info']['model_stats']['instance_cost'], 4), searched_old_kw=searched,
                     ran_or_wrote_old_kw=wrote, typeerror_seen=typeerr, hidden_conflict_steps=conflict,
                     visible_conflict_steps=vconf, final_patch=sorted(finals) or ['no atomic call']))
    print(json.dumps(rows[-1]))
json.dump(rows, open(f'runs/{iid}/kw_measures.json', 'w'), indent=1)
