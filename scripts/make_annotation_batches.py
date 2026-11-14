"""Make annotation batches from You Wang et al.'s labeled claims (data/faultyclaims/claims_tidy.jsonl).

calib_*: the 773 refuted claims their authors labeled by hand with an error form (label hidden here),
         so we can measure how well our LLM annotators agree with their human labels.
new_*:   600 refuted claims from the two no-network DeepSeek runs that nobody labeled yet (300 per model).
Each batch has 40 claims. The hidden answer key goes to outputs/annotation/answer_key.jsonl.
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/annotation'
rows = [json.loads(l) for l in open(ROOT / 'data/faultyclaims/claims_tidy.jsonl')]
def yes(x):
    return x is True or x == 'True'

for i, r in enumerate(rows):
    r['cid'] = f'c{i:05d}'          # our own unique id (row number in claims_tidy.jsonl)

def item(r):
    # what the annotator sees: the claim, where it came from, and the verifier's evidence; no model name, no labels
    return dict(cid=r['cid'], claim=r['claim'], quote=r['quote'], reasoning_type=r['ogsf_type'],
                related_files=r['related_files'], verifier_finding=r['evidence_finding'],
                verifier_rationale=r['rationale'], verifier_evidence=str(r['evidence'])[:1500],
                instance_id=r['instance_id'], step=r['step'], trajectory_path=r['trajectory_path'])

calib = [r for r in rows if yes(r['faulty']) and r['error_form'] not in (None, 'None')]
pool = [r for r in rows if yes(r['faulty']) and yes(r['network_none']) and not yes(r['manual_reviewed'])]
rng = random.Random(0)
new = []
for m in ('deepseek-v32', 'flash'):
    ms = [r for r in pool if r['model'] == m]
    new += rng.sample(ms, 300)
rng.shuffle(calib)
with open(OUT / 'answer_key.jsonl', 'w') as f:
    for r in calib + new:
        f.write(json.dumps(dict(cid=r['cid'], set='calib' if r in calib else 'new', model=r['model'],
                                error_form=r['error_form'], proof_pattern=r['proof_pattern'],
                                network_none=r['network_none'])) + '\n')
for name, items in (('calib', calib), ('new', new)):
    for b in range(0, len(items), 40):
        with open(OUT / 'batches' / f'{name}_{b // 40:02d}.jsonl', 'w') as f:
            for r in items[b:b + 40]:
                f.write(json.dumps(item(r)) + '\n')
print(len(calib), 'calib claims,', len(new), 'new claims')
