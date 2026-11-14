"""RecallAudit v0: a small open classifier for explicit memory recall in coding-agent reasoning.

TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression, trained on the LLM-labeled TRAIN items
(SWE-bench Verified runs + You Wang et al.'s CC-BY-4.0 trajectories; DeepSWE is never used for training),
evaluated on held-out tasks (test) and on DeepSWE (ood), and compared with two regex lexicon baselines from
scripts/rq3_disclosure.py: family A (explicit memory / real upstream fix) and any lexicon match (A or B).
It reads every label file in data/recall_audit/labels/ (4 of the 15 sampled batches are labeled).
Fix (Nov 14, 2026): the first version of this script scored the lexicon as bool(lexicon_match), which is True
for the string 'none' too, so its "lexicon" flagged every item (the lexicon numbers in release v0.1.0).
The classifier and its metrics are unchanged by the fix.
Usage: .venv/bin/python -I scripts/recall_audit_train.py
"""
import collections, glob, json, pickle, random
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score

VERSION = 'v0'
ROOT = Path(__file__).resolve().parent.parent
label_files = sorted(glob.glob(str(ROOT / 'data/recall_audit/labels/*.jsonl')))
n_batches = len(glob.glob(str(ROOT / 'data/recall_audit/batches/*.jsonl')))
lab = {}
for f in label_files:
    for l in open(f):
        if l.strip():
            d = json.loads(l); lab[d['item_id']] = d['label']
items = []
for f in sorted(glob.glob(str(ROOT / 'data/recall_audit/batches/*.jsonl'))):
    for l in open(f):
        d = json.loads(l)
        if d['item_id'] in lab:
            d['label'] = lab[d['item_id']]; items.append(d)
split = {s: [d for d in items if d['split'] == s] for s in ('train', 'test', 'ood')}
y = lambda ds: np.array([d['label'] != 'NONE' for d in ds])   # binary target: any explicit recall
# lexicon_match is the string 'A', 'B' or 'none' (scripts/recall_audit_sample.py)
BASELINES = {'lexicon_A': lambda d: d['lexicon_match'] == 'A',
             'lexicon_any': lambda d: d['lexicon_match'] in ('A', 'B')}

model = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                 TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
                      LogisticRegression(C=4.0, class_weight='balanced', max_iter=2000, random_state=0))
model.fit([d['text'] for d in split['train']], y(split['train']))


def boot_f1(yt, yp, reps=2000):
    """95% bootstrap interval for F1, resampling items."""
    rng = random.Random(0); n = len(yt); out = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(f1_score(yt[idx], yp[idx], zero_division=0))
    out.sort(); return [round(out[int(.025 * reps)], 3), round(out[int(.975 * reps)], 3)]


def composition(ds):
    """Label classes and lexicon strata of a set of items (the sample is stratified by the lexicon)."""
    return {'n': len(ds), 'recall_items': int(y(ds).sum()),
            'labels': dict(collections.Counter(d['label'] for d in ds).most_common()),
            'lexicon_match': dict(collections.Counter(d['lexicon_match'] for d in ds).most_common()),
            'recall_items_by_lexicon_match': dict(collections.Counter(d['lexicon_match'] for d in ds
                                                                      if d['label'] != 'NONE').most_common())}


res = {'version': VERSION, 'n_labeled': len(items),
       'label_files': [Path(f).name for f in label_files],
       'labels': f'LLM-annotated (Claude subagents); {len(label_files)} of {n_batches} sampled batches were labeled',
       'counts': {s: composition(v) for s, v in split.items()},
       'sampling_note': ('items were sampled 45% from lexicon matches (A or B) and 55% from random non-matching '
                         'text (scripts/recall_audit_sample.py), so lexicon hits are over-represented relative to '
                         'agent text in general; recall measured on this sample overstates recall on all text'),
       'lexicon_matched_share_of_labeled': round(sum(d['lexicon_match'] != 'none' for d in items) / len(items), 4)}
for s in ('test', 'ood'):
    ds = split[s]; yt = y(ds)
    p = model.predict_proba([d['text'] for d in ds])[:, 1]; yp = p >= 0.5
    r = {}
    preds = [('recallaudit', yp)] + [(name, np.array([rule(d) for d in ds])) for name, rule in BASELINES.items()]
    for name, pred in preds:
        pr, rc, f, _ = precision_recall_fscore_support(yt, pred, average='binary', zero_division=0)
        r[name] = {'precision': round(pr, 3), 'recall': round(rc, 3), 'f1': round(f, 3), 'f1_ci': boot_f1(yt, pred),
                   'n_flagged': int(pred.sum())}
    r['recallaudit']['auroc'] = round(roc_auc_score(yt, p), 3)
    res[s] = r
(ROOT / 'outputs/recall_audit/metrics.json').write_text(json.dumps(res, indent=1))
pickle.dump(model, open(ROOT / f'models/recall-audit/recall_audit_{VERSION}.pkl', 'wb'))
print(json.dumps(res, indent=1))
