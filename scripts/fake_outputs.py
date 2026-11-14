"""Find assistant messages where the model typed a tool result itself (<returncode>/<output> block).
Usage: python3 -I scripts/fake_outputs.py RUN_DIR"""
import json, os, sys, re, collections
run = sys.argv[1]
n_tasks = n_msgs = n_block = 0
tasks_with = set(); starts = 0; examples = []
for iid in sorted(os.listdir(os.path.join(run, 'trajs'))):
    f = os.path.join(run, 'trajs', iid, iid + '.traj.json')
    if not os.path.exists(f): continue
    d = json.load(open(f)); n_tasks += 1
    ms = d['messages']
    for i, m in enumerate(ms):
        if m.get('role') != 'assistant': continue
        c = m.get('content') or ''
        if not isinstance(c, str): continue
        n_msgs += 1
        if re.search(r'<returncode>\s*-?\d+\s*</returncode>', c) or re.search(r'<output>[\s\S]{20,}</output>', c):
            n_block += 1; tasks_with.add(iid)
            if c.lstrip().startswith('<returncode>') or c.lstrip().startswith('<output>'): starts += 1
            nxt = (ms[i+1].get('content') or '')[:80] if i + 1 < len(ms) else ''
            if len(examples) < 6: examples.append((iid, c[:260].replace('\n', ' | '), nxt.replace('\n', ' ')))
print(f'{run.split("/")[-1]}: tasks {n_tasks}, assistant msgs {n_msgs}, msgs with a typed tool-result block {n_block} '
      f'({len(tasks_with)} tasks), of which the message starts with it: {starts}')
for e in examples: print('  ', e)
