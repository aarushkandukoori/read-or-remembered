"""How well do our LLM annotators agree with You Wang et al.'s human error-form labels, and what do the
new labels say about the no-network DeepSeek claims nobody had labeled?

Reads outputs/annotation/labels/*.jsonl (our labels) and outputs/annotation/answer_key.jsonl (hidden key).
Writes outputs/annotation/agreement.json.
Usage: python3 -I scripts/annotation_agreement.py
"""
import collections as C
import glob
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'outputs/annotation'
FORMS = ['Misreference', 'Misstatement', 'Omission', 'Overgeneralization']


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def kappa(pairs, cats):
    """Cohen's kappa for a list of (label_a, label_b) pairs."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = C.Counter(a for a, _ in pairs)
    cb = C.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return (po - pe) / (1 - pe)


def boot_kappa(pairs, cats, reps=2000, seed=0):
    """Bootstrap 95% interval for kappa (resampling claims)."""
    rng = random.Random(seed)
    ks = sorted(kappa([rng.choice(pairs) for _ in pairs], cats) for _ in range(reps))
    return [round(ks[int(0.025 * reps)], 3), round(ks[int(0.975 * reps)], 3)]


key = {}
for line in open(A / 'answer_key.jsonl'):
    d = json.loads(line)
    key[d['cid']] = d

labels = {}
for f in sorted(glob.glob(str(A / 'labels/*.jsonl'))):
    for line in open(f):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('cid') in key and d.get('error_form') in FORMS:
            labels[d['cid']] = d

out = {'n_labeled': len(labels), 'n_key': len(key)}

# 1) calibration: our label vs the human label on the 773 claims they labeled by hand
cal = [(key[c]['error_form'], labels[c]['error_form'], labels[c]) for c in labels if key[c]['set'] == 'calib']
pairs = [(h, m) for h, m, _ in cal]
n = len(pairs)
agree = sum(h == m for h, m in pairs)
mm = lambda x: 'MM' if x in ('Misreference', 'Misstatement') else 'other'
pairs2 = [(mm(h), mm(m)) for h, m in pairs]
out['calibration'] = {
    'n': n,
    'agreement': round(agree / n, 4), 'agreement_ci': wilson(agree, n),
    'kappa': round(kappa(pairs, FORMS), 3), 'kappa_ci': boot_kappa(pairs, FORMS),
    'confusion_rows_human_cols_llm': {h: {m: sum(1 for a, b in pairs if a == h and b == m) for m in FORMS} for h in FORMS},
    'mm_vs_other_agreement': round(sum(a == b for a, b in pairs2) / n, 4),
    'mm_vs_other_kappa': round(kappa(pairs2, ['MM', 'other']), 3),
    'by_confidence': {c: {'n': len(g), 'agreement': round(sum(h == m for h, m, _ in g) / len(g), 4)}
                      for c in ('high', 'medium', 'low')
                      for g in [[x for x in cal if x[2]['confidence'] == c]] if g},
    'verdict_doubt_rate': round(sum(1 for _, _, d in cal if d['verdict_doubt']) / n, 4),
    'by_model': {},
}
for model in sorted({key[c]['model'] for c in labels if key[c]['set'] == 'calib'}):
    g = [(key[c]['error_form'], labels[c]['error_form']) for c in labels if key[c]['set'] == 'calib' and key[c]['model'] == model]
    out['calibration']['by_model'][model] = {'n': len(g), 'agreement': round(sum(a == b for a, b in g) / len(g), 4)}

# 2) new labels on 600 unlabeled refuted claims from the no-network DeepSeek runs
new = [(key[c], labels[c]) for c in labels if key[c]['set'] == 'new']
dist = C.Counter(d['error_form'] for _, d in new)
out['new'] = {
    'n': len(new),
    'distribution': {f: dist[f] for f in FORMS},
    'share': {f: round(dist[f] / len(new), 4) for f in FORMS} if new else {},
    'mm_share': round((dist['Misreference'] + dist['Misstatement']) / len(new), 4) if new else None,
    'mm_share_ci': wilson(dist['Misreference'] + dist['Misstatement'], len(new)),
    'verdict_doubt_rate': round(sum(1 for _, d in new if d['verdict_doubt']) / len(new), 4) if new else None,
    'by_model': {m: dict(C.Counter(d['error_form'] for k, d in new if k['model'] == m)) for m in ('deepseek-v32', 'flash')},
}
# the human-labeled shares in the same two models, for comparison
hum = C.Counter(key[c]['error_form'] for c in key if key[c]['set'] == 'calib' and key[c]['model'] in ('deepseek-v32', 'flash'))
tot = sum(hum.values())
out['human_share_deepseek_models'] = {f: round(hum[f] / tot, 4) for f in FORMS}

(A / 'agreement.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('n_labeled',)}, indent=1))
print('calibration:', {k: out['calibration'][k] for k in ('n', 'agreement', 'agreement_ci', 'kappa', 'kappa_ci', 'mm_vs_other_agreement', 'mm_vs_other_kappa')})
print('new:', out['new']['n'], out['new']['share'], 'MM share', out['new']['mm_share'], out['new']['mm_share_ci'])
print('human share (DeepSeek models):', out['human_share_deepseek_models'])
