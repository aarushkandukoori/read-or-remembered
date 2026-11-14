"""Summarize provenance_pilot.py outputs. Usage: python3 -I scripts/summarize_pilot.py outputs/*.jsonl"""
import json, sys, collections, os

def pct(a, b):
    return f'{100 * a / b:.1f}%' if b else '-'

rows_by_model = {os.path.basename(f)[:-6]: [json.loads(l) for l in open(f)] for f in sys.argv[1:]}
print('| model | file mentions | named before seen | of those: real | future-only | stale | never existed | hedged |')
print('|---|---|---|---|---|---|---|---|')
for model, rows in rows_by_model.items():
    rows = [r for r in rows if not r.get('in_fake_output')]
    un = [r for r in rows if r['prov'] == 'unseen']
    c = collections.Counter(r['status'] for r in un)
    print(f"| {model} | {len(rows)} | {len(un)} ({pct(len(un), len(rows))}) | {pct(c['real'], len(un))} | "
          f"{c['future']} | {c['stale']} | {c['invented']} | {pct(sum(r['hedged'] for r in un), len(un))} |")

print('\nTasks with at least one future-only or stale file named before seen, and their resolve rates:')
for model, rows in rows_by_model.items():
    rows = [r for r in rows if not r.get('in_fake_output')]
    tasks = {}
    for r in rows:
        tasks.setdefault(r['instance_id'], r['resolved'])
    drift = {r['instance_id'] for r in rows if r['prov'] == 'unseen' and r['status'] in ('future', 'stale')}
    inv = {r['instance_id'] for r in rows if r['prov'] == 'unseen' and r['status'] == 'invented'}
    res = lambda ids: pct(sum(1 for i in ids if tasks.get(i)), len(ids))
    other = set(tasks) - drift - inv
    print(f'  {model}: drift tasks {len(drift)} (resolved {res(drift)}), invented-only tasks {len(inv - drift)} '
          f'(resolved {res(inv - drift)}), clean tasks {len(other)} (resolved {res(other)})')

print('\nFake tool outputs typed by the model (steps whose own message contains <output>):')
for model, rows in rows_by_model.items():
    t = {}
    for r in rows:
        t[r['instance_id']] = r.get('n_fake_outputs', 0)
    print(f"  {model}: {sum(1 for v in t.values() if v)} of {len(t)} tasks")

print('\nFuture-only paths named before seen (examples):')
for model, rows in rows_by_model.items():
    ex = [r for r in rows if r['prov'] == 'unseen' and r['status'] == 'future' and not r.get('in_fake_output')]
    print(f'  [{model}] {len(ex)}')
    for r in ex[:8]:
        print(f"    {r['instance_id']} | {r['path']} | added {r['years']:+.1f} y after base | {r['field']} | "
              f"{r['context'][:110].strip()!r}")
