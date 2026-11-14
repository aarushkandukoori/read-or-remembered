# RQ1 at scale: do agents name repo facts before reading them, and are they right? (Nov 13, 2026)

Deterministic extension of the pilot (`notes/pilot-2026-11-13.md`, file paths only) to symbols and
signatures, and to the DeepSWE sample. No paid APIs, no LLM labeling in the pipeline. Every number
below comes from `outputs/rq1/` (written by the commands in "Commands"). The hand check at the end
is LLM-judged (Claude read the cases), not hand-labeled by a person.

## Bottom line

- **Agents state many repo facts before any tool output showed them, and most are right.**
  SWE-bench Verified: 24.0% of 35,876 judged first mentions were named before seen (NBS), and 90.5%
  of those were true at the base commit (facts that had been seen: 97.5%). DeepSWE (39 models,
  newer repos): 7.2% of 60,173 mentions were NBS, 80.7% true (seen: 96.8%).
- **DeepSeek-V3.2 recalls the most and is wrong the most** (31.8% NBS, 86.6% right) vs Kimi K2.5
  (17.2%, 92.5%) and MiniMax M2.5 (21.9%, 95.5%). 559 of the 818 wrong NBS facts on Verified are
  DeepSeek's: 351 first stated in its hidden reasoning, 88 inside tool outputs it typed itself
  (invented file listings and signatures), 120 in commands.
- **Wrong recalls are mostly "never existed", not "other version".** Verified: of 818 wrong NBS
  facts, 65 were stale (true only in older commits), 42 future (only newer), 635 never in any
  indexed version, 76 signature/keyword mismatches. DeepSWE: 832 wrong, 42 stale, 17 future, 765 never.
  Real version-drift examples exist (django.test.simple, django.contrib.auth.tests, matplotlib's old
  `ylabel(s, ...)`, sympy.testing on old SymPy, `astropy.utils.parsing.LRParser`, which only newer
  astropy has; checked with git).
- **Most wrong recalls are never contradicted.** Only 23.4% (Verified) and 38.9% (DeepSWE) of wrong
  NBS facts meet a later tool output that contradicts them; 18.3% / 33.8% are followed by a change
  of course. Usually the wrong fact sits in reasoning and is never acted on, so nothing corrects it.
- **Association with resolution is weak and mixed.** Verified: trajectories with at least one wrong
  NBS fact resolve 64.0% vs 71.4% without (-7.3 points; DeepSeek -2.3, Kimi +2.0, MiniMax -10.8).
  DeepSWE: 35.3% vs 35.5%. Association only (longer runs have more facts and different difficulty).
- Precision of the pipeline (LLM-judged): 17 of 20 random NBS mentions fully correct (85%,
  Wilson 95% CI 64-95%); 7 of 10 random *wrong*-labeled NBS mentions really wrong (70%, 40-89%).
  So the error rates above are upper-leaning: roughly 3 in 10 "wrong" labels are extraction noise.

## Method (scripts/rq1_fact_provenance.py, helpers rq1_repo_index.py, rq1_summarize.py)

Data: the three public mini-swe-agent runs on all 500 SWE-bench Verified tasks (DeepSeek-V3.2
reasoner, Kimi K2.5, MiniMax M2.5; all open weights, reasoning visible), and the 1,704 DeepSWE trials
(39 models; 1,434 DeepSWE tasks in TypeScript, Go, Python, Rust, JavaScript, plus 270 SWE-bench Pro
trials from the cross-bench slice). Steps come from the trajectory: hidden reasoning, visible text,
commands, and the tool output after the step (DeepSWE via `summarize_deepswe.iter_steps`).

Fact types, counted once per trajectory at the first mention:
- `path`: repo file paths (pilot regex, more file types, DeepSWE's `/app/` prefix). Skipped:
  absolute or `../` paths, third-party/build folders (node_modules, site-packages, build, dist,
  target, _build, ...), relative module specifiers inside import statements, and files the agent
  writes itself (pilot rule; real files it overwrites stay counted).
- `dotted` (Python): dotted module paths such as `django.utils.encoding.force_str`, from the agent's
  text, its imports, and attributes of its import aliases. Only the module path plus the first name
  inside it is checked (deeper attributes are often set at run time); dunder attributes are dropped.
- `code_name`: function/class/method names called in code the agent runs (heredocs, python -c,
  sed replacement text, echo/printf into files, the strings of Python edit scripts in non-Python
  repos). Not counted: builtins, standard library, numpy, pandas, pytest names (13,367 names), calls
  on other libraries' aliases, names the agent defines itself (in its commands or its final patch),
  and, for Go/TS/JS/Rust, comments, string literals and method calls on untyped receivers.
  Fenced code inside the agent's reasoning is a draft, so it is not used for code facts.
- `sig_def`: a signature line in the agent's text (`def f(a, b)`, `func F(a int)`, `function f(a)`,
  `fn f(a)`), compared with the base definition's parameter names and order. Skipped: functions the
  agent itself (re)defines in commands or in its final patch, drafts written as heredocs, test names.
- `sig_call`: a call-shaped statement in the text whose arguments are all parameter names of f
  (e.g. `get_permission_codename(action, opts)`); judges order only.
- `kwarg` (Python): keyword names in plain calls to repo functions in code the agent runs.

Named before seen (NBS): no earlier sighting in the task text or in an earlier tool output (the
agent's own earlier words do not count). Lenient on purpose, so NBS is a lower bound: for paths the
full path, the file name, the dotted module, or the extensionless module path seen counts as seen;
for names, the name seen anywhere as a word counts as seen.

Truth at the base commit: paths with `git ls-tree` (tail matches allowed, as in the pilot), and the
full add history of each path for stale/future. Symbols: every source file at the base commit is
parsed (Python with `ast`, regex fallback; Go/TS/JS/Rust with regexes for definitions); module
names include re-exports and one level of `import *`; a module that builds names at run time
(`globals()`, module `__getattr__`) makes a missing name "unknown", not "never". Other versions for
symbols = snapshots: all base commits of that repo in the benchmark, the newest commit in the clone,
and first-parent commits 2 and 5 years (Verified) or 1, 2 and 4 years (DeepSWE) before the oldest
base. Parse coverage of base commits was 100% (mean_base_coverage = 1.0 in facts_summary.json).

Not judged (kept in the JSONL, excluded from rates): `external` code names (not in this repo but
defined by another studied repo, or a method on an untyped receiver), `planned` mentions (planning
words such as "create", "add", "new", "I'll" within 100 characters in the agent's text; applied to
right and wrong mentions alike), `outside` paths (first folder never existed anywhere in the repo,
e.g. scratch apps). Verified: 1,764 planned, 126 external, 195 outside. DeepSWE: 1,680 / 994 / 288.

Self-correction, for each wrong NBS fact: the first tool output from the mention step on with an
error line naming the fact (No such file, ImportError, AttributeError, NameError, TypeError,
undefined:, Cannot find name, unresolved import, ...) or an empty search for it; "changed course"
= contradicted and the fact never appears in a later command.

Groups: model; reasoning visible vs hidden (model label = most of its trajectories carry reasoning
text; 10 DeepSWE models hidden, 29 visible); open vs closed weights by vendor family (open: DeepSeek,
Kimi, MiniMax, GLM, MiMo = 12 DeepSWE models; Qwen Plus/Max counted closed). The weights labels are
NOT verified per release (several models are newer than my knowledge).

## Commands (run from the project folder, in this order)

    .venv/bin/python -I scripts/rq1_clone_deepswe.py data/deepswe data/repos_deepswe   # 54 blobless clones, 591 MB
    .venv/bin/python -I scripts/rq1_fact_provenance.py index swebv     # batch-fetch + parse source files (run twice; 2nd after a fix)
    .venv/bin/python -I scripts/rq1_fact_provenance.py index deepswe
    .venv/bin/python -I scripts/rq1_fact_provenance.py vocab           # library names + per-repo names
    .venv/bin/python -I scripts/rq1_fact_provenance.py run swebv --workers 4
    .venv/bin/python -I scripts/rq1_fact_provenance.py run deepswe --workers 3
    .venv/bin/python -I scripts/rq1_fact_provenance.py summarize
    .venv/bin/python -I scripts/rq1_handcheck.py sample 10 20261114          # final hand check, 20 random NBS
    .venv/bin/python -I scripts/rq1_handcheck.py sample 5 99 wrong           # final hand check, 10 wrong NBS
    (earlier rounds, before fixes: sample 10 20261113; sample 6 777 wrong)

The run step was repeated several times while fixing extraction errors found by inspection and by
the earlier hand-check rounds; the numbers here are from the final run (06:46) and the final
summarize (07:30). Verdicts are stored in `data/repos_deepswe/_rq1_cache/handcheck_judgments.json`
and copied into `facts_summary.json` -> `hand_check`.

## Results: SWE-bench Verified (1,500 trajectories, 37,961 fact rows, 35,876 judged)

| model | mentions | NBS | NBS rate | NBS accuracy (95% CI) | seen accuracy |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | 13,095 | 4,163 | 31.8% | 86.6% (85.5-87.6) | 96.2% |
| Kimi K2.5 | 11,341 | 1,956 | 17.2% | 92.5% (91.3-93.6) | 97.5% |
| MiniMax M2.5 | 11,440 | 2,504 | 21.9% | 95.5% (94.6-96.2) | 98.8% |
| all | 35,876 | 8,623 | 24.0% | 90.5% (89.9-91.1) | 97.5% |

By fact type (all three models): path 7,363 mentions, 1,704 NBS (23.1%), 80.7% right; dotted 20,232,
5,070 (25.1%), 93.1%; code_name 6,918, 1,533 (22.2%), 95.9%; sig_def 767, 257 (33.5%), 70.8%;
sig_call 307, 43, 100%; kwarg 289, 16, 93.8%.

Pilot check (paths): DeepSeek 2,713 judged path mentions, 983 NBS (36.2%), 73.7% real (pilot: 3,243,
1,072, 33.1%, 67.0%); Kimi 2,353 / 269 (11.4%) / 84.4% (pilot 11.7%, 81.0%); MiniMax 2,297 / 452
(19.7%) / 93.8% (pilot 20.1%, 88.9%). Same picture; accuracy is higher now mainly because planned and
outside paths are set aside.

Where NBS facts first appear: DeepSeek 41.5% in hidden reasoning, 11.3% in visible text, 47.3% in a
command; Kimi 0.5% / 0% / 99.5%; MiniMax 0.3% / 0.04% / 99.7%. 480 DeepSeek NBS facts first appear
inside tool outputs it typed itself (fake outputs); 88 of them are wrong (e.g. `set_annotation_mask(mask)`
where Django's parameter is `names`). Hedging words near NBS mentions in text: 180 of 2,214 (8.1%);
among wrong ones 29 of 445 (6.5%), so wrong recalls are not hedged more.

Wrong NBS facts (818): stale 65, future 42, never 635, signature names differ 73, order wrong 2,
keyword invalid 1. Most frequent stale: `tests/settings.py` (Django, 20 mentions, a generic guess),
`testing/test_python.py` (pytest, 3), `django.test.simple`, `django.contrib.auth.tests`. Most frequent
future: `sympy.testing` on SymPy versions before it existed (8). For stale/future paths with an add
date (n = 58) the median distance from the base commit is -6.0 years. Of 75 wrong sig_def facts
(74 DeepSeek), 8 match an older and 8 a newer definition (e.g. `call_command(name)`, `ylabel(s)`).

Self-correction: 191 of 818 wrong NBS facts (23.4%) were contradicted by a later tool output (98 in
the same step, median 0 steps), 150 (18.3%) followed by a change of course, 627 never contradicted;
27 appear in the final patch (18 of them never contradicted). Per model contradicted / changed:
DeepSeek 16.3% / 12.7%, Kimi 46.6% / 34.3%, MiniMax 28.3% / 25.7%. Paths 24.9% / 23.7%; signature
claims 2.7% / 0% (almost never tested).

Resolution (association only; 7 DeepSeek runs without a report excluded): with a wrong NBS fact
439 trajectories, 64.0% resolved; without 1,054, 71.4%. DeepSeek 59.8% (n=256) vs 62.0% (237);
Kimi 72.4% (105) vs 70.4% (395); MiniMax 66.7% (78) vs 77.5% (422).

## Results: DeepSWE sample (1,704 trajectories, 63,135 fact rows, 60,173 judged)

Overall: 4,316 NBS (7.2% of mentions), 80.7% right (79.5-81.9); seen accuracy 96.8%. By type: path
27,298 / 2,309 NBS (8.5%) / 74.1%; code_name 26,906 / 1,744 (6.5%) / 87.9%; dotted 5,230 / 220 (4.2%)
/ 93.6%; sig_def 198 / 34 / 79.4%; sig_call 153 / 9 / 88.9%; kwarg 388 / 0 NBS.

| group | trajectories | NBS rate | NBS accuracy (95% CI) |
|---|---|---|---|
| reasoning hidden (10 models) | 516 | 8.9% | 86.6% (84.7-88.3) |
| reasoning visible (29 models) | 1,188 | 6.6% | 78.0% (76.5-79.5) |
| closed weights (27 models) | 1,460 | 7.1% | 82.4% (81.1-83.6) |
| open weights (12 models) | 244 | 7.3% | 73.1% (69.9-76.1) |
| DeepSWE tasks | 1,434 | 7.6% | 81.5% |
| SWE-bench Pro slice | 270 | 4.2% | 69.6% |
| Go / Python / Rust / TS+JS | 508 / 471 / 115 / 610 | 7.4 / 7.0 / 4.6 / 7.7% | 70.5 / 81.6 / 73.9 / 85.9% |

The visible-vs-hidden and open-vs-closed gaps come mostly from paths (visible 71.5% vs hidden 81.0%;
open 62.5% vs closed 76.6%). Visible reasoning gives more room for hypothetical example paths, which
the hand check found to be a false-"wrong" source, so part of this gap is measurement, not memory.

Per model (all fact types; n is small for 20-trajectory models, see CIs in fig_nbs_rate_accuracy.csv):
highest accuracy claude-fable-5 95.1% (246 NBS), claude-opus-4-8 94.5% (218), claude-opus-5 92.7%
(246); lowest deepseek-v4-pro 55.4% (65), muse-spark-1-2 58.1% (43), deepseek-v4-flash 61.3% (62).
NBS rate ranges from 3.8% (gpt-5-4) to 14.9% (gemini-3-1-pro-preview).

Wrong NBS facts (832): stale 42, future 17, never 765, names differ 7, order wrong 1. Stale examples:
`packages/query-core/vitest.config.ts` (TanStack, 4), `pkg/chartutil/coalesce.go` and
`pkg/chart/chart.go` (Helm, old layout), `static/katex.css` (KaTeX). Future: the module
`ansible.module_utils.common.respawn` (a module-level check, so solid). Code-name "future" is weaker:
it only means the name is defined somewhere in a newer commit. Example: an agent called
`Encoding.decodeUtf8` in Effect; newer Effect defines a `decodeUtf8` helper, but not as that
Encoding API, and `model_copy` (LangChain) is really a pydantic method. Median distance for
stale/future paths: -2.66 years (n = 43). So for these post-2025 tasks, the old-version recalls are
few; most wrong facts are invented or convention guesses (e.g. test files the agent expects to exist).

Self-correction: 324 of 832 (38.9%) contradicted (279 in the same step), 281 (33.8%) changed course,
508 never contradicted; 152 appear in the final patch (135 never contradicted). Paths 43.6% / 41.6%,
code names 28.9% / 14.7%. NBS facts are first stated in a command 93.5% of the time (reasoning 5.6%,
text 0.9%); in visible-reasoning models 7.2% first appear in reasoning. Hedged: 64 of 281 NBS text
mentions (22.8%), 27 of 121 wrong ones.

Resolution (association only): with a wrong NBS fact 459 trajectories, 35.3% resolved; without
1,245, 35.5%. Hidden reasoning 25.2% vs 30.6%; visible 38.5% vs 37.9%; closed 34.4% vs 36.4%; open
38.8% vs 28.8% (open-model runs with wrong facts are longer runs; no causal reading).

## Hand check (LLM-judged by Claude; cases printed by scripts/rq1_handcheck.py)

"Correct" = the mention is a statement/use of a fact about this repo, it was not visible earlier in
the run (checked by a substring search of the task text and all earlier tool outputs), and the truth
label is right (checked with git ls-tree / git grep -P at the base commit).

- Final, 20 random judged NBS mentions (10 per benchmark, seed 20261114): **17/20 correct (85%;
  Wilson 95% CI 64-95%)**. Errors: a pydantic method (`model_dump`) counted as a repo name because a
  repo test defines a method with that name (label harmless, but not a repo fact); a hypothetical
  example path in reasoning; a path inside a made-up stack-trace string in test data. All 10 Verified
  cases were correct (mostly imports and settings strings the agent wrote from memory, e.g.
  `django.test.override_settings`, `django.db.backends.sqlite3`).
- Final, 10 random NBS mentions labeled wrong (seed 99): **7/10 really wrong (70%; 40-89%)**.
  Real ones: Django test modules that do not exist (`tests.foreign_key`,
  `django/contrib/admin/tests/test_options.py`), a hallucinated `IPv6AddressValidator`, a `find`
  output DeepSeek imagined in its reasoning, test files DeepSWE agents ran that do not exist, an old
  KaTeX CSS location. Noise: an SQL type inside a string read as a call, two paths inside made-up
  stack traces.
- Earlier rounds on previous versions: 18/20 random correct and 8/12 wrong-labeled really wrong.
  Errors found there were fixed before the final run: TS modules seen as extensionless imports were
  labeled NBS; relative import specifiers read as repo paths; a Go `x, ok :=` local not recognized as
  the agent's own; a chained method call split across lines read as a plain call; hypothetical paths
  under folders the repo never had (now set aside as `outside`).

## Caveats

- NBS is not proof of memory: it includes guesses from naming conventions and from similar code seen
  in the same run (e.g. a sibling file). It is a lower bound on unseen mentions because "seen" is lenient.
- Precision: about 85% for random NBS mentions and about 70% for wrong labels (small samples, wide
  CIs). Known noise: paths inside test-data strings (stack traces), hypothetical example paths in
  reasoning, SQL or text inside Python strings read as calls, library methods on objects of unknown
  type (set aside as `external` when not found, but counted as right when the repo happens to define
  the same name), and planning statements without planning words.
- `code_name` checks only that some file defines the name (no types), so right/stale/future for code
  names are name-level (a newer helper with the same name counts), and Go/TS/Rust definitions come
  from regexes. Signature checks use parameter names, not types. Keyword checks skip functions with
  `**kwargs`, classes and method calls.
- Other versions for symbols are snapshots (base commits, newest commit, 2-3 older dates), so some
  "never" symbols existed between snapshots; paths use full history, so path and symbol stale rates
  are not directly comparable (e.g. `tests/settings.py` is a stale path, `tests.settings` a never symbol).
- DeepSWE clones are single-branch (default branch only): history on other branches is missing.
  DeepSWE tasks ask for new features, so many unseen names are things the agent adds; the pipeline
  removes names defined in the agent's commands and final patch and planned mentions, but some remain.
- The library vocabulary only covers builtins, the standard library, numpy, pandas and pytest; other
  dependencies (starlette, pydantic, docutils, astroid, matplotlib for seaborn, ...) are only caught
  through the "another studied repo defines it" rule.
- Reasoning visibility is a model-level label from the data (gemini-3-1-pro-preview counts as hidden
  because 60 of its 80 trials have no reasoning). Several "visible" closed models give summaries, not
  raw reasoning. Open/closed labels are by vendor family and unverified for new releases.
- Resolution comparisons are associations confounded by trajectory length and task difficulty.
- `data/repos_deepswe/` (1.2 GB incl. a 480 MB parse cache) is not in the top-level .gitignore; the
  folder has its own `.gitignore` containing `*`. Blobs for the Verified snapshots were fetched into
  the shared `data/repos/` clones (additive, about 26,600 objects).

## Files

- scripts/rq1_fact_provenance.py (main), scripts/rq1_repo_index.py (git + definitions index),
  scripts/rq1_summarize.py (tables), scripts/rq1_clone_deepswe.py (clones), scripts/rq1_handcheck.py
- outputs/rq1/facts_swebv.jsonl, facts_deepswe.jsonl (one row per fact; join `trial` with
  fig_trajectories.csv for task, repo, language, resolved, groups; DeepSWE rows carry no trajectory text)
- outputs/rq1/facts_summary.json (headline numbers, unjudged counts, hand check)
- outputs/rq1/fig_nbs_rate_accuracy.csv, fig_wrong_breakdown.csv, fig_self_correction.csv,
  fig_resolved_assoc.csv, fig_first_field.csv, fig_trajectories.csv
