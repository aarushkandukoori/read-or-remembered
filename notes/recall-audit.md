# RecallAudit v0 (Nov 14, 2026)

A small open classifier that flags explicit recall of training knowledge in coding-agent text (the upstream fix, or repository/library/version facts "from memory"), for auditing benchmark runs. Model card: models/recall-audit/MODEL_CARD.md (shipped in the release zip).

Commands: `.venv/bin/python -I scripts/recall_audit_sample.py` (sampling, already run), then `.venv/bin/python -I scripts/recall_audit_train.py` (reads every file in data/recall_audit/labels/, writes outputs/recall_audit/metrics.json and models/recall-audit/recall_audit_v0.pkl). Inference: `.venv/bin/python -I scripts/recall_audit.py --help`.

## Data and labels

- LLM-annotated (Claude subagents); 4 of 15 sampled batches were labeled: **400 labeled items** of the 1,500 sampled (15 batches of 100). The other 11 batches are unlabeled: the labeling of further batches did not complete, so there is no v1.
- Labels: SOLUTION_RECALL, KNOWLEDGE_RECALL or NONE, collapsed to binary (any recall vs none). LLM-annotated, not human-checked.
- train (SWE-bench Verified runs + You Wang et al. trajectories): 249 items, 50 with recall ({'NONE': 199, 'SOLUTION_RECALL': 30, 'KNOWLEDGE_RECALL': 20}); lexicon strata {'none': 140, 'B': 61, 'A': 48}; recall items by stratum {'A': 29, 'B': 19, 'none': 2}
- test (held-out tasks, same sources): 71 items, 23 with recall ({'NONE': 48, 'SOLUTION_RECALL': 12, 'KNOWLEDGE_RECALL': 11}); lexicon strata {'none': 36, 'B': 19, 'A': 16}; recall items by stratum {'A': 11, 'B': 10, 'none': 2}
- ood (DeepSWE, never used for training): 80 items, 27 with recall ({'NONE': 53, 'SOLUTION_RECALL': 23, 'KNOWLEDGE_RECALL': 4}); lexicon strata {'none': 44, 'A': 32, 'B': 4}; recall items by stratum {'A': 24, 'B': 3}

## Results

Held-out tasks (n=71, 23 recall items):

| method | precision | recall | F1 [95% bootstrap CI] | flagged | AUROC |
|---|---|---|---|---|---|
| RecallAudit v0 (p >= 0.5) | 0.818 | 0.391 | 0.529 [0.300, 0.711] | 11 | 0.908 |
| lexicon family A (explicit memory) | 0.688 | 0.478 | 0.564 [0.345, 0.731] | 16 | - |
| any lexicon match (A or B) | 0.600 | 0.913 | 0.724 [0.566, 0.839] | 35 | - |

DeepSWE (n=80, 27 recall items):

| method | precision | recall | F1 [95% bootstrap CI] | flagged | AUROC |
|---|---|---|---|---|---|
| RecallAudit v0 (p >= 0.5) | 0.714 | 0.556 | 0.625 [0.444, 0.776] | 21 | 0.867 |
| lexicon family A (explicit memory) | 0.750 | 0.889 | 0.814 [0.681, 0.909] | 32 | - |
| any lexicon match (A or B) | 0.750 | 1.000 | 0.857 [0.746, 0.939] | 36 | - |

## What this shows

- On held-out tasks v0 is the most precise (0.82 vs 0.69 for family A) but finds fewer recall items (recall 0.39 vs 0.48); its F1 does not beat either lexicon, and all intervals overlap.
- On DeepSWE the lexicon is better on both precision (0.75 vs 0.71) and recall (0.89 vs 0.56).
- So v0 does not yet replace the lexicon. With 249 training items, more labels are the next step.

## Correction to release v0.1.0

The first version of recall_audit_train.py scored the lexicon as `bool(lexicon_match)`. lexicon_match is the string 'A', 'B' or 'none', and `bool('none')` is True, so that "lexicon" flagged every item: precision 0.324 and 0.338 (exactly the share of recall items, 23/71 and 27/80), recall 1.0, F1 0.489 and 0.505. Those numbers went out in the v0.1.0 model card, README and draft. The fixed script scores family A and any-match separately. Retraining with the fix gives a model with identical predictions on all 400 labeled items (max probability difference 0.0), so the v0 weights in the release are unchanged and the RecallAudit numbers above are the same as in v0.1.0.

## Caveats

- The sample is stratified by the lexicon: 45% of the labeled items are lexicon matches and the rest random non-matching text. Among the non-matching items, recall is rare (test 2 of 36, DeepSWE 0 of 44), so the lexicon's recall here is measured almost only on its own matches. On agent text in general, where non-matching text dominates, the lexicon's recall would be lower, and so the any-match recall of 1.0 on DeepSWE is partly an artifact of the sampling.
- Small test sets (71 and 80 items): F1 intervals are wide and overlap for all three methods.
- Labels are LLM-annotated; RQ3's LLM-judged check found a strict precision of about 50% for family A on its own matches (notes/rq3-disclosure.md), a stricter definition than the any-recall label used here.
- No analysis applies RecallAudit to the full corpora yet (no recall-rate estimates from the classifier).
