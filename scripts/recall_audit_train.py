"""RecallAudit v0: a small open classifier for explicit memory recall in coding-agent reasoning.

TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression, trained on the LLM-labeled TRAIN items
(SWE-bench Verified runs + You Wang et al.'s CC-BY-4.0 trajectories; DeepSWE is never used for training),
evaluated on held-out tasks (test) and on DeepSWE (ood), and compared with the regex lexicon baseline.
Usage: .venv/bin/python -I scripts/recall_audit_train.py
"""
import glob, json, pickle, random
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
lab = {}
for f in glob.glob(str(ROOT / 'data/recall_audit/labels/*.jsonl')):
    for l in open(f):
        d = json.loads(l); lab[d['item_id']] = d['label']
items = []
for f in sorted(glob.glob(str(ROOT / 'data/recall_audit/batches/*.jsonl'))):
    for l in open(f):
        d = json.loads(l)
        if d['item_id'] in lab:
            d['label'] = lab[d['item_id']]; items.append(d)
split = {s: [d for d in items if d['split'] == s] for s in ('train', 'test', 'ood')}
y = lambda ds: np.array([d['label'] != 'NONE' for d in ds])   # binary target: any explicit recall

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


res = {'n_labeled': len(items), 'counts': {s: {'n': len(v), 'recall_items': int(y(v).sum())} for s, v in split.items()},
       'labels': 'LLM-annotated (Claude subagents); 4 of 15 sampled batches were labeled'}
for s in ('test', 'ood'):
    ds = split[s]; yt = y(ds)
    p = model.predict_proba([d['text'] for d in ds])[:, 1]; yp = p >= 0.5
    lex = np.array([bool(d['lexicon_match']) for d in ds])
    r = {}
    for name, pred in (('recallaudit', yp), ('lexicon', lex)):
        pr, rc, f, _ = precision_recall_fscore_support(yt, pred, average='binary', zero_division=0)
        r[name] = {'precision': round(pr, 3), 'recall': round(rc, 3), 'f1': round(f, 3), 'f1_ci': boot_f1(yt, pred)}
    r['recallaudit']['auroc'] = round(roc_auc_score(yt, p), 3)
    res[s] = r
(ROOT / 'outputs/recall_audit/metrics.json').write_text(json.dumps(res, indent=1))
pickle.dump(model, open(ROOT / 'models/recall-audit/recall_audit_v0.pkl', 'wb'))
print(json.dumps(res, indent=1))
