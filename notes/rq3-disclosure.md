# RQ3 disclosure: does the reasoning reveal reliance on memory? (Nov 13, 2026)

Script: `scripts/rq3_disclosure.py`. Outputs: `outputs/rq3/` (summary JSON + `fig_*.csv`).
Everything here is free (local data, no API calls). Lexicon checks were judged by Claude reading each
snippet, so they are **LLM-judged, not hand-labeled**.

## Bottom line

1. **Memory talk lives in the hidden reasoning, not in the visible message.** In every dataset the
   hidden channel has many more recall statements per step than the visible text, and most trajectories
   with a hidden recall statement show none in the visible text (SWE-bench Verified: 1 of 63; DeepSWE: 17%
   do show one; You Wang et al.'s runs: 48% do).
2. **How much an agent talks about memory depends hugely on the model and the task set.** On SWE-bench
   Verified it is rare (0.13% of steps). On You Wang et al.'s SWE-bench Pro runs, DeepSeek-V4.1-Flash
   mentions memory or the upstream fix in 37% of its hidden steps (every one of its 100 runs), Qwen3.8-27B
   in 14%, while DeepSeek-V3.2 and Qwen3.6-27B almost never do (0.3%, 0.1%).
3. **Facts named before they were seen are almost always stated plainly.** For You Wang et al.'s claims
   whose names had not appeared yet, 84% of the *wrong* (refuted) ones carry no hedge and no recall flag.
   Wrong claims are hedged a bit more often than right ones (13.5% vs 8.3%, +5.2 points, 95% CI 3.0 to
   7.4), so hedging is a weak signal, not a warning light. For file paths named before seen (DeepSeek-V3.2)
   wrong paths were not hedged more often than correct ones (never-existed paths were hedged less: 16% vs
   31%), and no sentence flagged recall at all. Kimi K2.5 and MiniMax M2.5 name 96% / 98% of such paths
   only inside a command, with no sentence to hedge.
4. **The recall lexicon is conservative but not precise in the strict sense.** About half of its matches
   (50%) are true memory statements; 88% are memory statements or references to the real upstream fix/PR.
   It misses about as many explicit statements as it catches (estimated recall 49%, very uncertain), so
   all rates below are lower bounds on explicit memory talk.
5. **Our trap runs repeat the pilot's gap.** Across all 33 runs (15 pilot + 18 targeted runs of the other
   session, DeepSeek `deepseek-flash`), 293 steps talk about memory in the hidden reasoning vs 33 in the
   visible text; every run has more hidden than visible memory steps. The pilot's own numbers reproduce
   (django-11149: 9 to 14 hidden steps per run, 0 to 6 visible; the pilot note said 10 to 14 and 0 to 6).

## Data

| dataset | what | trajectories | agent steps |
|---|---|---|---|
| swebv | public mini-swe-agent runs on SWE-bench Verified: DeepSeek-V3.2 (reasoner), Kimi K2.5, MiniMax M2.5 | 1,500 | 78,108 |
| faultyclaims | You Wang et al.'s 400 trajectories (SWE-bench Pro): DeepSeek-V3.2, DeepSeek-V4.1-Flash, Qwen3.6-27B, Qwen3.8-27B | 400 | 28,001 |
| deepswe | DeepSWE trials with any reasoning text, 33 models | 1,225 | 116,905 |
| trap runs | `outputs/agent_runs`, `outputs/runs/*`, and (read only) `~/Projects/rr-targeted/runs/*` | 33 | |

- DeepSWE: 443 trials with no reasoning text and 36 gemini-3-7-flash trials (reasoning = copy of the
  visible message) were excluded. Several DeepSWE models (Claude Fable 5, Opus 5, Sonnet 5, Haiku 4.5,
  grok-build, muse-spark) never expose reasoning, so they are not in this analysis.
- faultyclaims: both DeepSeek models ran with no network and pruned git history; both Qwen models ran
  with network (from each trajectory's `run_args`). DeepSeek-V3.2 in these runs has no visible text at all
  (0 of 8,834 steps), so its hidden/visible contrast is undefined.
- swebv: Kimi and MiniMax rarely write visible text (3,138 of 24,687 and 6,345 of 30,219 steps).
- The targeted runs were read as a snapshot: 18 files, newest modified 2026-11-13 05:18 (all
  `Submitted`, all `--network none`).

## Method

**Channels.** Each agent step is split into hidden reasoning (`reasoning_content` / `reasoning`), visible
message (`content` with ``` code blocks removed) and command (the shell commands). For hidden and visible
text, fenced code blocks are dropped and DeepSeek's typed fake tool outputs (`<returncode>`, `<output>`)
are removed (995 steps); if the visible text is an exact copy of the hidden text it is dropped (104 steps).
Text is split into sentences (line breaks, or `. ! ?` followed by a space and a capital); a step counts for
a channel if any of its sentences matches.

**Recall lexicon** (all regexes are in the script and in `disclosure_summary.json`). Family A is the main
measure; family B is weaker evidence.

| pattern | family | catches | deliberately skips |
|---|---|---|---|
| i_recall | A | "I recall", "I remember", "I vaguely remember", "I can't recall" | "I remember to ...", "I remember we saw ...", "I recall the earlier output" |
| if_i_recall | A | "if I recall correctly", "as far as I know", "to my knowledge", "IIRC", "AFAIK" | |
| lets_recall | A | "let me recall ...", "let's try to remember ..." | "let me recall the issue / what I was doing / the current state" |
| from_memory | A | "from memory", "from my memory" | "load/free/read from memory" (computer memory) |
| my_memory | A | "my memory", "my knowledge", "my training data", "prior/general knowledge", "a vague memory", "knowledge cutoff" | |
| search_memory | A | "search memory", "search my knowledge" | |
| memorized | A | "memorized", "memorize" | "memoize" (caching) |
| upstream | A | "upstream" (the real project) | paths/identifiers (`plugins/upstream`), "somewhere upstream", "upstream callers/servers" |
| actual_pr_commit | A | "the actual PR", "a real commit", "the merged PR" | "the actual fix/patch/change" and "the original PR/commit" (usually the agent's own fix, the PR text, or a local git commit) |
| based_on_real | A | "based on a real PR", "derived from the actual commit" | |
| actual_project_fix | A | "the actual Django fix", "the real fix in Django", "the official upstream patch" | |
| newer_versions | B | "in newer versions", "in later releases", "from an older version" (needs a preposition) | "install an older version" |
| modern_lib | B | "in modern Django", "in newer Python", "in older versions of numpy" | "current Django" (= this repo) |
| lib_version_range | B | "Django 4.x", "Python 3.8+", "sympy 1.12 and later" | numbers without a library name ("runs of 3+") |
| changed_in_version | B | "removed in Python 3.10", "introduced in Django 3.2" | |
| historically | B | "historically" | "in the past", "used to" (too many code uses) |

Two extra rules, added after reading matches and before drawing the validation sample:
- Rule 1: the first-person patterns (i_recall, if_i_recall, lets_recall) do not count when the same
  sentence has an in-run cue ("earlier", "above", "I saw", "the instructions", "PR says", "the output", ...).
- Rule 2: lib_version_range / changed_in_version do not count when every dotted version number in the
  match already appeared in the task text or an earlier tool output of that trajectory (it could be read).
  Rule 2 is not applied in the calibration part.

**Hedge lexicon:** probably; likely / unlikely; maybe, perhaps, possibly, presumably, supposedly,
apparently; I think / believe / guess / suspect / assume / suppose, my guess; not sure / not certain /
unsure / uncertain / if I'm not mistaken; seems / appears to; might, could be, may be, may have;
"should be" only in the location sense ("should be in/at/defined/located ...").

**Rates.** Per trajectory and channel: share of all agent steps with a hit. Reported as the mean over
trajectories with a 95% bootstrap CI (2,000 resamples of trajectories, seed 0), plus the share of steps
with text in that channel and the share of trajectories with at least one hit. Channel contrasts are
paired per trajectory; model contrasts resample each model's trajectories independently.

**Calibration.**
- Pilot file paths (`outputs/{deepseek-v3.2,kimi-k2.5,minimax-m2.5}.jsonl`): rows with `prov == unseen`
  and not inside a fake output. The sentence is re-extracted from the raw trajectory (the pilot's saved
  `context` can be cut inside a path). Correct = exists at the base commit; wrong = other version
  (stale/future) or never existed.
- You Wang et al.'s claims (`claims_tidy.jsonl` + `claims_seen_before.jsonl`): "named before seen" = some
  identifier or related file of the claim had not appeared before the claim. The sentence is the one(s)
  around the quote, found by its character offsets (75,156 claims) or the quote alone (1,381). Wrong =
  refuted, right = supported (unresolved reported separately). A quote-only version is also in the CSV.
- Wrong-vs-right differences use a bootstrap that resamples trajectories once for both groups.

**Validation (LLM-judged).** A fixed sample (seed 0, reservoir sampling over unique hidden/visible
sentences of swebv, faultyclaims and deepswe): 100 family-A matches (40 swebv / 30 faultyclaims / 30
deepswe), 50 family-B-only matches (20/15/15), 100 unmatched sentences (50 that contain a broad
memory-ish word such as version, original, actual, commit, recall; 50 others), plus 80 hedge matches and
40 non-matches. Recall labels: `mem_repo` (remembered knowledge about the task's own project, incl. "I
can't remember the upstream code"), `mem_general` (remembered knowledge about other software, languages,
math), `upstream_ref` (refers to the real upstream fix/PR as something to match, find or guess, without
claiming to remember it), `read_upstream` (upstream content actually obtained in the run), `no`,
`unclear`. Strict positive = mem_repo + mem_general; lenient = + upstream_ref. Hedge labels: `hedge`
(epistemic hedge or tentative claim/plan) vs `no` (e.g. "x might be None" describing code behavior).
Precision and recall are weighted by each stratum's pool size. Snippet text is kept only in
`data/rq3_validation/sample.jsonl` (DeepSWE text may not be redistributed); the labels CSV has ids only.

## Commands (all from the project root)

    .venv/bin/python -I scripts/rq3_disclosure.py      # final run: 2026-11-13 06:27:08 to 06:32:46

    # read-only check for the caveat on summarized reasoning (characters per reasoning token per DeepSWE model)
    .venv/bin/python -I - <<'EOF'
    import csv, json, os, collections
    man = list(csv.DictReader(open('data/deepswe/sample_manifest.csv')))
    chars, toks = collections.Counter(), collections.Counter()
    for m in man:
        p = os.path.join('data/deepswe/trials', m['version'], m['trial_name'], 'trajectory.json')
        if not os.path.exists(p) or m['model'] == 'gemini-3-7-flash':
            continue
        t = json.load(open(p))
        c = sum(len((s.get('reasoning_content') or '').strip()) for s in t.get('steps', []) if s.get('source') == 'agent')
        tok = ((t.get('final_metrics') or {}).get('extra') or {}).get('total_reasoning_tokens')
        if c and tok:
            chars[m['model']] += c
            toks[m['model']] += tok
    for k in sorted(chars, key=lambda k: chars[k] / toks[k]):
        print(f'{k:28s} {chars[k] / toks[k]:5.2f}')
    EOF

The script was run four times: run 1 (first lexicon, no labels), run 2 (rules 1 and 2 added; this run
drew the validation sample), then I judged all 370 snippets and wrote the labels into
`outputs/rq3/fig_lexicon_validation.csv`, then runs 3 and 4 (scoring code and wrong-vs-right bootstrap
added; same sample, the script checks the ids match). Lexicon exploration used throwaway scripts in the
session scratchpad (pattern counts and random examples); nothing from them is reported.

## Results

### 1. Recall statements per step and channel (family A, mean % of steps per trajectory, 95% CI)

| dataset | model | hidden | visible | trajectories with a hidden hit | of those, also a visible hit |
|---|---|---|---|---|---|
| swebv | DeepSeek-V3.2 | 0.35 [0.24, 0.48] | 0.01 [0.00, 0.03] | 10.8% | 1 of 54 |
| swebv | Kimi K2.5 | 0.03 [0.01, 0.06] | 0 | 1.0% | 0 of 5 |
| swebv | MiniMax M2.5 | 0.01 [0.00, 0.02] | 0 | 0.8% | 0 of 4 |
| swebv | all 3 | 0.13 [0.09, 0.17] | 0.00 [0.00, 0.01] | 4.2% | 1 of 63 |
| faultyclaims | DeepSeek-V3.2 | 0.31 [0.16, 0.49] | no visible text | 17% | 0 of 17 |
| faultyclaims | DeepSeek-V4.1-Flash | 37.28 [35.16, 39.43] | 2.68 [2.17, 3.21] | 100% | 71% |
| faultyclaims | Qwen3.6-27B | 0.11 [0.03, 0.21] | 0 | 6% | 0 of 6 |
| faultyclaims | Qwen3.8-27B | 13.88 [12.32, 15.41] | 2.02 [1.27, 2.83] | 99% | 36% |
| faultyclaims | all 4 | 12.90 [11.33, 14.54] | 1.17 [0.93, 1.45] | 55.5% | 48% |
| deepswe | all 33 models | 1.35 [1.18, 1.56] | 0.23 | 27.9% | 17% |

- Paired hidden-minus-visible differences are positive with CIs above 0 for every swebv and faultyclaims
  model except MiniMax M2.5, whose 4 hidden hits give a CI that touches 0 (`fig_channel_contrasts.csv`);
  e.g. faultyclaims all: +11.7 points [10.3, 13.3].
- Model gaps: swebv DeepSeek-V3.2 minus Kimi +0.32 points [0.21, 0.45]; faultyclaims DeepSeek-V4.1-Flash
  minus Qwen3.8-27B +23.4 points [20.6, 26.1] (`fig_model_differences.csv`).
- Commands almost never carry a recall statement (family A in 5 of 76,049 swebv command steps, 28 of
  28,001 faultyclaims, 27 of 116,905 deepswe). A look at 15 random command matches: mostly the word
  "upstream" inside code (identifiers, comments, echo labels) plus a few curl attempts to fetch upstream
  files, i.e. not statements. Command-channel matches were not part of the validation sample.
- Version talk (family B) follows the same pattern: swebv 0.84% of hidden steps vs 0.04% visible;
  faultyclaims 2.10% vs 0.08%.
- Hedges are everywhere in hidden reasoning (swebv DeepSeek-V3.2: 57.9% of steps; faultyclaims
  DeepSeek-V4.1-Flash: 69.3%) and rare in visible text (7.0% and 3.6%).
- DeepSWE, highest hidden rates: gemini-3-1-pro-preview 10.7%, gemini-3-flash-preview 9.3%,
  deepseek-v4-pro 9.1%, gemini-3-5-flash 7.2%, deepseek-v4-flash 6.1%, glm-5-3 4.7%, glm-5-3-flash 4.3%.
  All four Gemini models have 0 visible hits. Claude models are at or below 1.0% hidden; GPT-5.x models
  0.3% to 1.5%. Gemini's hits are almost all "I recall / I remembered" in narrated thought summaries, and
  in the validation sample 2 of 7 Gemini matches were true memory statements, so Gemini rates are
  inflated.
- What drives the high faultyclaims rates (hidden steps, DeepSeek-V4.1-Flash): `upstream` 2,035,
  `i_recall` 978, `actual_pr_commit` 721, `my_memory` 682, `lets_recall` 674, `actual_project_fix` 475,
  `from_memory` 457, `search_memory` 444 (`fig_pattern_counts.csv`). These runs had no network, and the
  model keeps trying to reconstruct the upstream PR and its tests from memory.

Example (swebv DeepSeek-V3.2, django-16333, step 6, opened by hand): hidden reasoning says "In ModelForm,
the save method does (from the Django source code I recall):", while the visible text of the same step
only says it will run a sed command. Three random swebv DeepSeek trajectories with hidden hits were
opened this way; none mentioned the memory in the visible text (one of the hidden hits, "in a real PR we
would update the test", is a false positive).

### 2. Calibration: facts named before seen

You Wang et al.'s claims, all four models (`fig_calibration_claims.csv`, `fig_calibration_contrasts.csv`):

| names | verdict | n | hedged | recall flag (A) | any flag | plain |
|---|---|---|---|---|---|---|
| named before seen | refuted (wrong) | 1,321 | 13.5% [11.2, 16.0] | 2.7% | 15.9% | 84.1% |
| named before seen | supported (right) | 5,340 | 8.3% [7.3, 9.4] | 0.9% | 9.2% | 90.8% |
| named before seen | unresolved | 922 | 16.9% | 9.5% | 24.8% | 75.2% |
| all seen before | refuted | 5,493 | 13.0% | 1.6% | 14.4% | 85.7% |
| all seen before | supported | 44,789 | 6.6% | 0.5% | 7.1% | 92.9% |

- Wrong minus right, named before seen: hedged +5.2 points [3.0, 7.4]; recall flag +1.8 [0.9, 2.9];
  any flag +6.7 [4.5, 9.1]. For names already seen the gap is similar (hedged +6.4 [5.3, 7.5]), so the
  hedge tracks uncertainty in general, not memory in particular.
- Manually confirmed faulty claims (You Wang et al.'s 40 reviewed trajectories), named before seen:
  130 claims, 9.2% hedged, 3.1% recall flag, 87.7% plain.
- Quote-only text gives the same picture with lower rates (named before seen: refuted 12.0% vs supported
  7.0% hedged).

Pilot file paths named before seen (`fig_calibration_paths.csv`):
- DeepSeek-V3.2 (1,072 paths): 639 in a plain sentence, 326 inside a code block in the reasoning, 106
  only in a command, 1 where the path could not be re-found (pilot context used).
- In plain sentences: correct 31.4% hedged [26.3, 36.9] (n=474), wrong other-version 20.0% [6.7, 34.5]
  (n=30), never existed 16.3% [10.3, 23.3] (n=135). Wrong minus correct: -11.4 points [-26.0, +4.8] and
  -15.1 points [-22.8, -7.0]. No sentence had a recall flag (0 of 639).
- Kimi K2.5: 283 of 294 unseen paths (96.3%) appear only in a command; MiniMax M2.5: 487 of 497 (98.0%).

### 3. Lexicon validation (LLM-judged, 370 snippets)

| part | n | strict precision (memory) | lenient (+ upstream refs) |
|---|---|---|---|
| family A, swebv | 40 | 55% (22) | 72.5% |
| family A, faultyclaims | 30 | 53% (16) | 90% |
| family A, deepswe | 30 | 33% (10) | 80% |
| family A, weighted | 100 | 50.2% | 88.3% |
| family B only, weighted | 50 | 57.7% | 73.2% |
| A or B, weighted | 150 | 50.7% | 87.2% |

- Main false positives: memories of the run that rule 1 missed ("I recall the system is reminding me to
  format"), "in a real PR we would update the tests", "the actual fix in the Django code" meaning the
  agent's own fix, commits in the local git history, a repo module literally named `upstream`, quoted task
  text, version facts restated from the PR text, and 4 cases of upstream code the agent had actually
  fetched or installed (`read_upstream`).
- Unmatched: 3 of 50 near-miss sentences were true memory statements (an upstream PR URL named from
  memory, a statement that it lacks the upstream PR number, a Node version fact); 0 of 50 others.
  Estimated recall of A or B: 48.7% strict (55.1% lenient). If the miss rates sat at the top of their 95%
  intervals, recall could be as low as 4.9% (strict), because the "other" pool is 2.66 million sentences.
  Read all rates as lower bounds.
- Hedge lexicon: precision 86.3% (69 of 80; the misses are "might/could" describing what code may do);
  1 of 40 unmatched sentences had a hedge ("not necessarily"); estimated recall 77.5%.

### 4. Trap runs (`fig_trap_runs.csv`)

| run set | runs | hidden steps with family A | visible steps with family A |
|---|---|---|---|
| pilot django-11149 (network on) | 3 | 8 to 14 | 0 to 6 |
| pilot sympy-13615 (network on) | 6 | 3 to 11 | 0 to 7 |
| pilot sympy-13615 (network off) | 6 | 4 to 16 | 0 to 2 |
| targeted django-15563 (network off) | 8 | 5 to 12 | 0 to 1 |
| targeted django-16100 (network off) | 8 | 5 to 16 | 0 to 2 |
| targeted sympy-15345, sympy-19040 | 1 + 1 | 12, 15 | 3, 0 |

- With the pilot's own regex (`scripts/trap_pipeline.py`): django-11149 hidden 9 to 14, visible 0 to 6;
  sympy-13615 network off hidden 5 to 16, visible 0 to 3 (same as its `analysis.json`).
- All 33 runs: 293 hidden vs 33 visible family-A steps; 33 of 33 runs have more hidden than visible;
  14 runs have any visible family-A step. Mean hidden family-A steps per run: original 8.55, placebo 9.64,
  trap 8.45 (11 runs each); visible about 1 in each. With 11 runs per condition there is no sign that the
  trap changes how much the model talks about memory.

## Caveats

- Lexicon matching only finds memory talk that is said out loud. Most memory use is silent (Part 2:
  84 to 98% of facts named before seen carry no flag), so these rates are not rates of memory use.
- Strict precision is about 50%: roughly half of family-A hits are true memory statements, most of the
  rest are references to the upstream fix/PR. Precision differs by source (deepswe lowest, driven by
  Gemini narration). Recall is low and poorly bounded.
- The validation was judged by one LLM reader (Claude), once, without a second annotator; judgments on
  "memory of this run vs training memory" are sometimes guesses (3 snippets marked unclear). The lexicon
  rules were tuned on other random examples from the same data before the sample was drawn.
- Hidden reasoning is not the same thing across providers. Characters of reasoning text per reported
  reasoning token (DeepSWE trials with reasoning, pooled per model; command below): GPT-5.x 0.79 to 1.13,
  grok 0.92 to 1.20, gemini-3-flash-preview 0.41, vs 3.8 to 4.4 for most open models and 4.6 to 5.0 for
  Claude, so GPT/grok reasoning text is very likely a summary. Gemini returns narrated thought summaries,
  and Kimi/MiniMax on swebv write almost no visible text. A low visible rate can mean "nothing visible to
  say it in".
- faultyclaims mixes network-off (DeepSeek) and network-on (Qwen) runs; Qwen3.8's memory talk includes
  some upstream code it actually fetched. Model, harness and task set are confounded across datasets.
- File-path "never existed" includes paths the agent plans to create or uses as examples, so it is not a
  clean "wrong memory" group; the other-version group is small (30 sentences).
- Calibration uses the lexicons without rule 2; the claim sentence is the one around the LLM-extracted
  quote (1,381 claims fell back to the quote alone).
- The targeted runs come from another session and are a snapshot (18 runs as of 05:18); rerunning later
  may include more runs.
- DeepSWE trajectories have no redistribution license: outputs contain only ids, counts and short
  paraphrases; the validation snippets with text stay in `data/rq3_validation/` (gitignored).

## Files

- `scripts/rq3_disclosure.py` (the analysis; lexicons documented inline)
- `outputs/rq3/disclosure_summary.json` (key numbers, lexicons, validation, trap summary)
- `outputs/rq3/fig_recall_rates.csv`, `fig_channel_contrasts.csv`, `fig_model_differences.csv`,
  `fig_traj_recall_counts.csv`, `fig_pattern_counts.csv` (Part 1)
- `outputs/rq3/fig_calibration_claims.csv`, `fig_calibration_paths.csv`, `fig_calibration_contrasts.csv` (Part 2)
- `outputs/rq3/fig_lexicon_validation.csv` (Part 3: sample ids + LLM-judged labels, no text)
- `outputs/rq3/fig_trap_runs.csv` (Part 4)
- `data/rq3_validation/sample.jsonl` (validation snippets with text; stays local)
