"""Batches for LLM provenance adjudication of faulty claims (RQ1).

Scope (You Wang's suggestion): the 257 Misstatement + Misreference claims that You Wang et al. confirmed by hand
in the two no-network DeepSeek runs. Each item joins the claim, the verifier's evidence and our deterministic
features (was the cited code shown before the claim? do its names exist at the base commit or only in other
versions?). Batches of 22; the first 3 batches are also given to a second annotator (agreement check).
Usage: python3 -I scripts/make_provenance_batches.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/provenance/batches'
tidy = {}
for i, line in enumerate(open(ROOT / 'data/faultyclaims/claims_tidy.jsonl')):
    tidy[f'c{i:05d}'] = json.loads(line)
feats = {}
for line in open(ROOT / 'outputs/rq1/claims_features.jsonl'):
    d = json.loads(line)
    feats[d['cid']] = d

items = []
for cid, f in feats.items():
    if f['error_form'] in ('Misstatement', 'Misreference') and f['network_none'] and f['manual_confirmed']:
        t = tidy[cid]
        items.append(dict(
            cid=cid, model=f['model'], instance_id=f['instance_id'], repo=f['repo'], base_commit=f['base_commit'],
            step=f['step'], message_index=f['message_index'], n_steps=f['n_steps'], reasoning_type=f['ogsf_type'],
            error_form=f['error_form'], claim=t['claim'], quote=t['quote'],
            verifier_finding=t['evidence_finding'], verifier_rationale=t['rationale'],
            verifier_evidence=str(t['evidence'])[:2000], trajectory_path=t['trajectory_path'],
            evidence_seen_before=f['evidence_seen_before'], file_seen_before=f['file_seen_before'],
            n_cited_lines=f['n_cited_lines'], n_lines_seen=f['n_lines_seen'], citations=f['citations'][:1500],
            identifiers=f['identifiers'][:12], unexplained_names=f['unexplained'], ident_class=f['ident_class'],
            cites_dependency=f['cites_dependency']))
items.sort(key=lambda d: (d['instance_id'], d['model'], d['message_index']))
size = 22
names = []
for b in range(0, len(items), size):
    name = f'p_{b // size:02d}'
    with open(OUT / f'{name}.jsonl', 'w') as fh:
        for it in items[b:b + size]:
            fh.write(json.dumps(it) + '\n')
    names.append(name)
print(len(items), 'claims in', len(names), 'batches:', ' '.join(names))
