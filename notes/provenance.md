# RQ1 provenance of the 257 Misstatement/Misreference claims (Nov 14, 2026)

**Question.** For the faulty claims that You Wang et al. confirmed by hand as Misstatement or Misreference in the two
network-isolated DeepSeek runs (257 claims: 216 Misstatements, 41 Misreferences; 112 DeepSeek-V3.2, 145
DeepSeek-V4.1-Flash), where does each claim come from?

**Method.** Each claim gets the first matching provenance class of the codebook (paper Appendix, "Annotation
Codebook"): (1) Propagated from an earlier faulty claim in the same run; (2) Recalled-other-version: false at the base
commit but true in another commit of the repository or another version of a pinned dependency (claims that use names
introduced by the task's own upstream fix count here, with the fixed version as the other version); (3)
Convention-guess; (4) Read-but-misread: contradicts the task text or an earlier tool output; (5) Unsupported. Flags:
contradicted_by_prior_output, named_before_seen, channel, hedged, uses_upstream_fix, plus confidence and a short
evidence note. Annotators see the claim, its trajectory up to the claim, the assessment agent's evidence and our
deterministic features (outputs/provenance/batches/*.jsonl), with read access to the repository history.

- **Annotators are LLMs:** Claude Opus 5.5 subagents run through Claude Code. All labels are LLM-annotated; none were
  checked by a human.
- **Annotator A** labeled all 257 claims (outputs/provenance/labels_A/p_00..p_11.jsonl). The first 44 claims (batches
  p_00, p_01) were labeled in an earlier run (Nov 13) with the same codebook; the other 213 on Nov 14.
- **Annotator B**, an independent subagent with no access to A's labels, labeled the first 44 claims again
  (labels_B/p_00, p_01) to measure agreement. There was no adjudication: every reported class is annotator A's.
- Coverage check: all 257 claim ids in outputs/provenance/batches/*.jsonl have an A label; no duplicate, unknown-class
  or stray labels (summary.json, `coverage`).

Command: `.venv/bin/python -I scripts/provenance_summary.py` (writes outputs/provenance/summary.json and prints the
tables below).

## Results

Coverage: 257 of 257 claims labeled by A; missing 0 ; B labeled 44 (p_00, p_01); problems A 0, B 0; stray labels 0

| class | All (n=257) | DeepSeek-V3.2 (n=112) | DeepSeek-V4.1-Flash (n=145) | Misstatement (n=216) | Misreference (n=41) |
|---|---|---|---|---|---|
| Propagated | 95 (37.0%) | 36 (32.1%) | 59 (40.7%) | 78 (36.1%) | 17 (41.5%) |
| Recalled-other-version | 25 (9.7%) | 14 (12.5%) | 11 (7.6%) | 22 (10.2%) | 3 (7.3%) |
| Convention-guess | 33 (12.8%) | 17 (15.2%) | 16 (11.0%) | 19 (8.8%) | 14 (34.1%) |
| Read-but-misread | 69 (26.8%) | 31 (27.7%) | 38 (26.2%) | 63 (29.2%) | 6 (14.6%) |
| Unsupported | 35 (13.6%) | 14 (12.5%) | 21 (14.5%) | 34 (15.7%) | 1 (2.4%) |

95% Wilson intervals (all claims): Propagated 31.3%-43.0%; Recalled-other-version 6.7%-14.0%; Convention-guess 9.3%-17.5%; Read-but-misread 21.8%-32.6%; Unsupported 10.0%-18.4%

| flag (annotator A) | count | share | 95% CI |
|---|---|---|---|
| contradicted_by_prior_output | 152/257 | 59.1% | 53.0%-65.0% |
| named_before_seen | 38/257 | 14.8% | 11.0%-19.6% |
| hedged | 103/257 | 40.1% | 34.3%-46.2% |
| uses_upstream_fix | 19/257 | 7.4% | 4.8%-11.3% |
| channel | {'hidden_reasoning': 257} | | |
| confidence | {'medium': 127, 'high': 74, 'low': 56} | | |

| class | contradicted_by_prior_output | named_before_seen | hedged | uses_upstream_fix |
|---|---|---|---|---|
| Propagated (n=95) | 59 | 21 | 49 | 7 |
| Recalled-other-version (n=25) | 15 | 2 | 12 | 7 |
| Convention-guess (n=33) | 9 | 10 | 18 | 3 |
| Read-but-misread (n=69) | 69 | 2 | 12 | 1 |
| Unsupported (n=35) | 0 | 3 | 12 | 1 |

| class x evidence_seen_before | all | some | none | no_repo_evidence |
|---|---|---|---|---|
| Propagated | 42 | 37 | 1 | 15 |
| Recalled-other-version | 13 | 8 | 2 | 2 |
| Convention-guess | 8 | 15 | 5 | 5 |
| Read-but-misread | 35 | 24 | 3 | 7 |
| Unsupported | 2 | 14 | 4 | 15 |
| total | 100 | 98 | 15 | 44 |

Agreement A vs B on 44 claims: raw 81.8% (36/44), kappa (5 classes) 0.761 (bootstrap 95% CI 0.587 to 0.906); memory vs not: raw 95.5%, kappa 0.645 (CI 0.0 to 1.0, 30 undefined resamples skipped); A memory 4, B memory 2, both 2

| A \ B | Propagated | Recalled-other-version | Convention-guess | Read-but-misread | Unsupported |
|---|---|---|---|---|---|
| Propagated | 11 | 0 | 0 | 2 | 1 |
| Recalled-other-version | 1 | 2 | 1 | 0 | 0 |
| Convention-guess | 0 | 0 | 4 | 0 | 1 |
| Read-but-misread | 1 | 0 | 0 | 9 | 1 |
| Unsupported | 0 | 0 | 0 | 0 | 10 |

Flag agreement A vs B: contradicted_by_prior_output 95.5%, named_before_seen 100.0%, hedged 97.7%, uses_upstream_fix 97.7%


Reading the tables:
- **Most claims are not memory:** propagation (37.0%) and misreading (26.8%) together are 63.8% of the 257.
  Recalled-other-version is 25 claims (9.7%, 95% CI 6.7-14.0); 7 of them use names from the task's own upstream
  fix, and 15 of the 25 contradict an output the agent had already seen.
- 59.1% of all claims contradict something already shown (by construction all 69 read-but-misread claims do);
  14.8% name identifiers before any output showed them.
- Misreferences are more often convention guesses (14 of 41) than Misstatements (19 of 216).
- All 257 claims are in hidden reasoning: the claims were extracted from reasoning text, so the channel flag carries
  no information here.

## Caveats

- One LLM annotator decides the class of 213 of the 257 claims; agreement is measured on 44 claims only, and those
  are the first two batches, not a random sample.
- The memory-vs-not kappa rests on 4 (A) and 2 (B) recalled-other-version labels on the overlap; its bootstrap
  interval spans 0 to 1, so it says little.
- Propagated comes first in the decision order: a propagated claim inherits whatever the origin of the earlier claim
  was, so the memory share counts only claims that introduce the remembered fact themselves.
- Wilson intervals treat claims as independent; claims cluster in 20 trajectories, so the intervals are too narrow.
- The claims are faulty per You Wang et al.'s human labels, but which claims exist at all was decided by their
  LLM assessment agent.
