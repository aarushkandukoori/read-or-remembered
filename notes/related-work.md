# Related work and paper models for "Read or Remembered?"

Compiled 2026-11-13 for the ICML 2027 submission. BibTeX keys in [brackets] point to `notes/related.bib`
(70 entries, all checked; rebuilt by `scripts/make_related_bib.py`).

**How everything here was checked.** Each paper was opened at a primary source:

- arXiv API answers (titles, authors, dates, the authors' "comment" field, abstracts), saved in
  `data/related_work_meta/arxiv_*.xml` and printed with `scripts/arxiv_meta.py`;
- full text from the arXiv HTML pages for every paper whose numbers or section layout are quoted below;
- venues from the OpenReview API (`scripts/openreview_venues.py`, answers in `data/related_work_meta/openreview/`),
  the official proceedings pages (NeurIPS proceedings, ACL Anthology, the ICSE 2026 program on conf.researchr.org),
  or the authors' own arXiv comment (marked "per arXiv comment").
- Semantic Scholar (`scripts/s2_venues.py`) was only a cross-check. DBLP could not be used: its API now answers
  scripts with a bot-check page, which we did not try to get around.

Anything not checked this way is marked **[UNVERIFIED]**. Numbers are quoted from the paper versions named.

**Our study, in the words used below** (current plan: `notes/PLAN.md`):
a *trap* changes one fact the model likely remembers (argument order, default value, function location, raised
exception), carried consistently through code, tests and gold patch, so the task stays solvable; a *placebo* is a
same-size edit that changes nothing semantically. Git history is squashed and the network is off. Traps are
*agent-targeted* (facts the model's own baseline patch uses without reading their definition).
RQ1 provenance of claims (read, misread, recalled from another version, convention guess, propagated, unsupported),
RQ2 traps (does wrong memory steer the agent?), RQ3 disclosure (does the reasoning admit recall, hidden vs visible?),
RQ4 benchmark validity (can memory-breaking but meaning-preserving refactoring refresh a benchmark?),
plus mitigation (refactor-and-rerun refresh, a trajectory provenance audit tool).

---

## 1. Bottom line

- **Benchmark-level evidence that SWE agents lean on memory is now strong**: closed-book file-path recall
  (SWE-Bench Illusion; Prathifkumar et al.), verbatim gold-patch recall found by OpenAI's audit, pass-rate drops when
  familiar repository cues are scrambled (SchrodingerRepo, RepoMirage), and the finding that on SWE-bench "reads no
  longer predict success" (Working Set).
- **All of it is measured at the level of pass rates, closed-book probes or action counts.** We found no paper that
  asks, for individual claims inside agent runs, whether the claim was read or remembered, whether wrong memory
  causally steers the agent against a matched placebo, and whether the reasoning admits the recall.
- **The two closest papers explicitly leave the gap open**: Working Set proposes "equal-length irrelevant
  controls" only for a future preregistered test; You Wang et al. state "we study neither how faulty claims depend on each other nor
  why the models make them."
- **RQ4 is the most crowded part**: SchrodingerRepo already applies behavior-preserving rewrites to SWE-bench
  repositories, and CodeCleaner/DyCodeEval do refactoring/rewriting for code benchmarks. RQ4 must be framed as
  fact-targeted refresh checked by memory probes and trap/placebo runs, with those papers as baselines.
- **Environment leakage is a separate, solved-by-sandbox problem** (git history, network); we must apply the known
  safeguards so that what remains is memory in the weights.

### Comparison of the closest work

| Paper | What is measured | Manipulation | Control arm | Reasoning text analyzed? | Causal test of reliance on memory? | Tested mitigation |
|---|---|---|---|---|---|---|
| SWE-Bench Illusion | closed-book outputs, no agent | none (probes) | other benchmarks/repos | no | no | none |
| Prathifkumar et al. | closed-book file localization | none (probes) | BeetleBox, SWE-rebench | no | no | none |
| OpenAI audit (Feb 2026) | red-team recall transcripts, 1 CoT example | adversarial prompting | none | anecdotal | no | none (recommends SWE-bench Pro) |
| SchrodingerRepo | Pass@1, action counts | 4 levels of cue erosion | post-cutoff SWE-rebench | no | partly (cues removed, not made wrong) | none (the framework is an evaluation protocol) |
| RepoMirage | resolve rate, action transitions | 3 structural perturbations | none | no | no (framed as context reasoning) | RepoAnchor workflow |
| Working Set | tests passed, reads | facts withheld/supplied; renamed library; fictional APIs | front-loaded vs closed-book | only "I am blocked" reports | yes, on fictional or renamed-library tasks | none (harness advice) |
| You Wang et al. | atomic CoT claims | claim-corrected replay | no sham correction | yes | for claims, not memory | one correction message |
| SWE-Touch | resolve rate, trajectories | user "Counter-Edits" mid-run | original task | no | for visible user edits | none tested |
| **Ours** | claims + actions + outcomes | one-fact trap | same-size placebo + original; exposure gradient | yes, hidden vs visible | yes | refresh + provenance audit |

---

## 2. Closest prior work, one by one

### 2.1 The SWE-Bench Illusion [liang2026swebenchillusion]
Liang, Garg, Zilouchian Moghaddam. arXiv 2506.12286 (v4, 2025-12-01); ICSE 2026, Software Engineering in Practice
track (checked on the conf.researchr.org program page).

*What it measures.* Two diagnostic tasks with **no repository access**: naming the buggy file from the issue text
(plus repo name) alone, and reproducing the functions changed by the gold patch from the issue and the rest of the
file; a prefix-completion test checks verbatim matches. Ten OpenAI/Anthropic models. v4 abstract: "up to 76%
accuracy in identifying buggy file paths using only issue descriptions", "merely up to 53% on tasks from
repositories not included in SWE-Bench", and up to 35% consecutive 5-gram overlap on SWE-Bench Verified/Full vs up
to 18% on other benchmarks. The prompts ask for a discussion section, but only final outputs are scored. No
mitigation beyond a call for contamination-resistant benchmarks.

*How we differ.* They show models **can** answer from memory when asked closed-book. We test whether agents **do**
use memory during real runs, for which claims, and with what effect: claim provenance in trajectories (RQ1), causal
reliance with trap vs placebo (RQ2), and disclosure (RQ3). Their closed-book probes are a natural per-task
memory-strength covariate for us (E5).

### 2.2 Does SWE-Bench-Verified Test Agent Ability or Model Memory? [prathifkumar2025swebvmemory]
Prathifkumar, Mathews, Nagappan. arXiv 2512.10218 (Dec 2025).

*What it measures.* Two Claude models find relevant files from the issue text only, then from issue text plus file
paths; the same setup on BeetleBox and SWE-rebench. Abstract: models did 3 times better on SWE-Bench-Verified and
were 6 times better at finding edited files "without any additional context about the projects themselves".

*How we differ.* Same as 2.1: a closed-book capability probe on two models, no agent runs. It supports our use of
SWE-rebench as a low-memory control arm.

### 2.3 OpenAI, "Why we no longer evaluate SWE-bench Verified" [openai2026swebv]
OpenAI blog post, 2026-02-23 (page opened 2026-11-13).

*What it shows.* OpenAI stopped reporting SWE-bench Verified. An audit of 138 problems that o3 did not consistently
solve over 64 runs found 59.4% with material problems in tests or descriptions (35.5% narrow tests, 18.8% wide
tests, 5.1% other). On contamination: "all frontier models we tested were able to reproduce the original,
human-written bug fix ... or verbatim problem statement specifics for certain tasks", found with an automated
red-teaming setup (GPT-5 probing GPT-5.2-Chat, Claude Opus 4.5 and Gemini 3 Flash Preview over 15 turns, with a
judge model). One CoT example: on django__django-14725, GPT-5.2's reasoning says the `edit_only` parameter was
"maybe added around 4.1 or 4.2". The post also says models that saw the problems in training "are more likely to
succeed", and recommends SWE-bench Pro, where their pipeline found contamination rarer and no complete verbatim gold
patch.

*How we differ.* Their evidence is adversarial elicitation (closed-book) plus one CoT anecdote, from one lab, and
the link between exposure and success is correlational. We measure memory use inside normal agent runs, with a
placebo-controlled causal test, and count how often the reasoning admits recall. The `edit_only` example is a good
motivating figure for RQ3.

### 2.4 SchrodingerRepo [chen2026schrodingerrepo]
Chen, Yang, Gu, Shi, Wan, Guan. arXiv 2609.27891 (2026-08-21); code at github.com/cslsolow/Schrodinger-Repo.

*What it measures.* Repositories are re-instantiated at evaluation time with four levels: L1 issue text rewritten by
an LLM (checked by a verifier LLM), L2 seeded invertible renaming of paths/modules/symbols applied at the
interaction layer, L3 reordering of definitions inside gold-patch files, L4 an LLM's behavior-preserving rewrite of
gold-patch-related code (L3/L4 kept only if PASS_TO_PASS holds and FAIL_TO_PASS still fails). mini-swe-agent,
temperature 0, 250 actions; SWE-bench Verified (500), SWE-QA (144), and 110 SWE-rebench tasks from March 2026.
Pass@1 drops: GPT 5.1 44.6 to 36.2, GPT-5.4-mini 46.8 to 35.6, DeepSeek-v4-Flash 72.8 to 66.8, Gemini-3.1-Flash-Lite
56.7 to 42.3 (on 300 instances chosen for leakage evidence). Actions rise 61.5% to 116.0%; about 82-84% of the extra
actions are navigate/search/read/probe. On the post-cutoff SWE-rebench split, GPT-5.4-mini stays at 17.27%, which
they read as evidence that the transformation does not make tasks intrinsically harder. RQ1-RQ4 layout with a
Threats to Validity section.
No analysis of reasoning text or claims, and no placebo arm.

*How we differ.* (1) Their treatment **removes** familiar cues, so memory becomes useless; our trap keeps everything
familiar and makes **one remembered fact wrong**, so memory becomes harmful. That separates "memory speeds up
navigation" from "memory overrides evidence". (2) Our placebo is matched in edit size, so a drop cannot be blamed on
edit noise or unfamiliar text. (3) We analyze claims and disclosure in the reasoning. (4) For RQ4 they are the
closest prior refresh method (their L2-L4 are memory-breaking, meaning-preserving refactorings); we should use them
as a baseline and borrow their post-cutoff "does it add difficulty?" check.

### 2.5 RepoMirage [li2026repomirage]
Li, Zhang, Zhu, Su, Zhu, Dong. arXiv 2605.26177 (2026-05-25). NeurIPS 2026 acceptance **[UNVERIFIED]**: search
snippets list it on the NeurIPS 2026 schedule, but neurips.cc returned HTTP 500 when opened. A workshop version
("RepoMirage: Do Code Agents Really Understand Repository Structures?") is on OpenReview at an ICLR 2026 workshop.

*What it measures.* Three semantics-preserving perturbations of SWE-bench Verified repositories: dependency-path
indirection (a four-layer proxy import chain), runtime-target masking (target file renamed and re-exported, plus
decoy files) and local-value externalization (constants moved to JSON). Eight models, mini-swe-agent, 250 steps.
Average resolved rate falls from 66.80% to 49.78%, files accessed rise from 4.77 to 13.24, and four derived "Extend"
task families average 25.25%. "Exploration drift" is measured with action-transition probabilities (explore to
explore up, explore to edit down). Structural hints raise GPT-5's average from 27.60 to 62.60 and DeepSeek-V3.2's
from 39.80 to 76.80 (their Table 3); a structure-first workflow, RepoAnchor, is the mitigation. No reasoning-text or
memorization analysis; memorization appears only as a citation of the SWE-Bench Illusion.

*How we differ.* They read drops as weak repository-context reasoning and cannot tell a lost memorized shortcut
from harder structure; our placebo-matched trap can. Their transition-probability metrics are good process measures
for RQ2, and their paper flow (diagnose, confirm the bottleneck with hints, then mitigate) is a good model.

### 2.6 The Working Set of a Coding Agent [mohammadi2026workingset]
Mohammadi, Klein, Chadha, Arora, Bindschaedler. arXiv 2608.16630 (2026-08-17). **Conceptually the closest paper.**

*What it measures.* Each edit needs facts that come either from recent context or from parametric memory; facts
covered by neither are "coherence debt". Seven model families, five harnesses (Claude Code, Codex CLI, Aider,
OpenHands, opencode). Fictional closed-book migrations: best result 0/12 requirements over 154 trials, while 299 of
300 front-loaded trials reach at least 9/12. On a Pydantic v1-to-v2 migration with a renamed twin that defeats
memorized names, 66 of 70 trials across seven models end at the same score, passing the same 24 of 79 tests.
Withholding a fact costs exactly the work it supports; distance from the edit does not matter; missing facts produce
fabricated files or guessed values; how often an agent says it is blocked depends on the model (17 of 46 trials,
37%, overall); a stale convention file costs more than none. On SWE-bench (397 scored trials over 100 instances) a
reading-based residency score does not predict success (within-cell AUC about 0.49), which they attribute to
parametric memory substituting for reading. Analysis uses actions and outcomes; the agent's stated support set is
logged but not analyzed. They propose "equal-length irrelevant controls" for a future preregistered test.

*How we differ.* (1) Real SWE tasks and real remembered facts, with the change carried through the gold patch and
tests so the task stays fair, instead of fictional APIs or a library-wide rename. (2) The control they leave for
future work, an equal-length irrelevant edit, is our main control (the placebo). (3) We tie memory use to the claims
in the reasoning and test disclosure. (4) Their AUC of about 0.49 motivates RQ1 directly: if reads no longer predict
success, which claims come from memory?

### 2.7 From Faulty Claims to Incorrect Patches [wang2026faultyclaims]
You Wang, Huang, Jiang, Cao, Chen, Liu. arXiv 2610.32338 (2026-10-23). Our collaborator's paper.

*What it measures.* 400 trajectories (DeepSeek-V3.2, DeepSeek-V4.1-Flash, Qwen3.6-27B, Qwen3.8-27B; mini-swe-agent)
on 100 SWE-bench Pro tasks. An LLM splits the CoT into 76,537 atomic units (O "original implementation" 43.9%,
S 31.9%, F 22.0%, G 2.3%); a manual check found 93.3% of 2,149 extracted units correct (Cohen's kappa 0.937 on a
double-annotated subset). An LLM assessment agent in a fresh
container at the base commit, without the observations, patch or tests, labels 81.8% supported, 11.2% refuted, 7.1%
unresolved (refutation precision 86.3% by hand). Faulty-claim rate at least 9.6%; faulty claims appear in 96.7% of
resolved trajectories; key faulty claims explain 127 of 188 failed patches (67.6%). A claim-corrected replay (one
message just before submission) fixes the targeted defect in 162 of 182 cases (89.0%) and resolves 21 of 182 (11.5%)
per claim, 22 of 188 (11.7%) when all key claims are corrected. Error types: Misstatement 53.3%, Overgeneralization
24.7%, Misreference 11.3%, Omission 10.7%. Their limitation: "we study neither how faulty claims depend on each other
nor why the models make them."

*How we differ.* Complementary. They answer "how many claims are wrong and do they cause failures"; we answer "where
do the wrong claims come from" (RQ1 provenance on their Misstatement/Misreference claims, as You Wang suggested),
add a causal manipulation (traps with placebos) and test disclosure. Their replay has no sham control; a same-length
irrelevant correction message would make it placebo-controlled.

Related, by the same first author: **PatchDiff** [wang2026solvedissues] (ICSE 2026): 7.8% of SWE-bench Verified
patches counted correct fail the developer test suite, 29.6% of plausible patches behave differently from the ground
truth, inflating resolve rates by 6.2 points. This matters for grading trap runs: a memory-driven patch can pass weak
tests, so differential testing of trap patches is worth adding.

### 2.8 SWE-Touch [tan2026swetouch]
Tan et al. arXiv 2608.02499 (2026-08-03). Injects validated "Counter-Edits" (plausible edits that conflict with the
task) plus a user message when the agent reaches the relevant code; nine models on SWE-bench Verified lose 7.7 points
on average; agents keep conflicting code or replace it without re-inspecting. *How we differ:* their conflict is a
visible, announced user edit; ours is between invisible parametric memory and unannounced repository state, with a
matched placebo.

### 2.9 When Instruction Files Go Stale [mehtiyev2026stalefiles]
Mehtiyev, Tadesse, Arkoh, Assunção. arXiv 2611.09802 (2026-11-06). Three real stale instructions in AGENTS.md-style
files, each in three versions (stale, corrected, removed) with code and task fixed; 22 tasks, 732 trajectories of two
models, analyzed per tool call. Agents follow stale guidance when it is relevant, sometimes even when the correct
information is available elsewhere; task success hides the cost. *How we differ:* same three-arm logic as our
original/trap/placebo design, but the stale source is a file in the repository, not memory. A good template for
tool-call-level outcome measures.

### 2.10 Environment leakage is not memory: SWE-bench issue #465 and SWE-Bench Pro Verified
- **SWE-bench issue #465** [swebench2025issue465] (opened 2025-09-03, "Repo State Loopholes During Agentic
  Evaluation"): agents used `git log --all` / `git log --grep` to find future fix commits (named examples include
  Claude 4 Sonnet on pytest-dev__pytest-6202 and Qwen3-Coder 480B on django__django-13513). Fixed for Verified in
  PR #471 (swebench v4.1.0).
- **SWE-Bench Pro Verified** [zheng2026sweproverified] (arXiv 2609.08149): leakage through local files, git history,
  external network and task metadata. In a GLM-5.2 audit over 731 instances, baseline runs read answer files locally
  in at least 103 tasks and over the network in 49 (e.g. 318 raw.githubusercontent requests in 54 tasks). Safeguards
  (fresh single-commit repository, hidden tests, metadata filtering, blocking code hosts) drop GLM-5.2 from 78.80% to
  57.32%; DeepSeek-V4-Pro barely moves (49.98% to 49.11%).
- **Our pilot saw the same thing**: with the network on, DeepSeek fetched upstream fixes in 6 of 9 runs.

*How we differ.* These remove **lookup** of the answer from the environment. We study **recall** of facts from the
weights, which no sandbox removes. Our protocol must include their safeguards (network off, squashed history, no
answer files) so that any remaining effect is memory.

---

## 3. Fresh or contamination-resistant SWE benchmarks, and benchmark validity

- **SWE-bench** [jimenez2024swebench] (ICLR 2024, oral): the base benchmark.
- **SWE-rebench** [badertdinov2025swerebench] (NeurIPS 2025 Datasets and Benchmarks): automated pipeline, over 21,000
  interactive Python tasks; a stream of fresh tasks gives a contamination-free benchmark, and some models' SWE-bench
  Verified results "might be inflated due to contamination".
- **SWE-bench-Live** [zhang2025swebenchlive] (NeurIPS 2025 Datasets and Benchmarks): live-updated; arXiv v2 has 1,319
  tasks from issues created since 2024 in 93 repositories; a "substantial performance gap" versus static benchmarks.
- **SWE-Bench Pro** [deng2025swebenchpro]: 1,865 problems from 41 repositories, split into public (11 repos), held-out
  (12) and commercial (18 proprietary) sets; presented as contamination-resistant.
- **SWE-Bench+** [aleithan2024swebenchplus]: of SWE-Agent+GPT-4's successful patches, 32.67% had the solution in the
  issue or comments and 31.08% passed only because tests were weak; filtering drops the resolve rate from 12.47% to
  3.97%.
- **Saving SWE-Bench** [garg2026savingswebench] (CAIN 2026, per arXiv comment): rewriting issues into chat-style user
  queries shows overestimation of more than 50% for some models on public benchmarks.
- **AgentLens** [sahoo2026agentlens]: 10.7% of passing OpenHands trajectories are "Lucky Passes"; rates range 0.5% to
  23.2% by model. **Coherence Collapse / TRAJEVAL** [kim2026coherencecollapse]: over 16,758 trajectories, 60-69% of
  failures reach and edit the right functions; in 5 cases the agent writes a gold-identical patch and later destroys it.
- **EvalPlus** [liu2023evalplus] (NeurIPS 2023): 80x more tests on HumanEval cut pass@k by up to 19.3-28.9%.

*How we differ.* Fresh or private tasks lower exposure for a whole benchmark but cannot say which successes used
memory, or whether the reasoning behind them was read or recalled. We use them as the **low-exposure arm** of an
exposure gradient: memory-sourced claims and trap effects should shrink from SWE-bench Verified to SWE-bench Pro to
post-cutoff SWE-rebench / SWE-bench-Live tasks.

## 4. Contamination detection (is this sample in the training data?)

- **Min-K% Prob** [shi2024mink] (ICLR 2024): flags seen text by its lack of very-low-probability tokens; WikiMIA
  benchmark; needs token probabilities.
- **Guided instruction / "Time Travel"** [golchin2024timetravel] (ICLR 2024, spotlight): prompt with dataset name,
  split and an initial segment; 92-100% agreement with human experts on seven datasets.
- **TS-Guessing** [deng2024tsguessing] (NAACL 2024, per arXiv comment): ChatGPT and GPT-4 guess masked wrong MMLU
  options with 52% and 57% exact match.
- **Proving test set contamination** [oren2024proving] (ICLR 2024, oral): exchangeability test on example order,
  no access to training data; little evidence of pervasive contamination in five public models.
- **Longitudinal cutoffs** [roberts2024cutoff] (ICLR 2024): pass rates on Codeforces and Project Euler track GitHub
  popularity and release date relative to GPT training cutoffs.
- **Rephrased samples** [yang2023rephrased]: paraphrased/translated test data slips past n-gram decontamination;
  8-18% of HumanEval overlaps RedPajama-Data-1T and StarCoder-Data.
- **Code benchmarks** [riddell2024codecontamination] (ACL 2024): substantial overlap of code benchmarks with open
  corpora, and better scores on the overlapping subsets. **LessLeak-Bench** [zhou2025lessleak]: 83 SE benchmarks,
  average leakage 4.8% / 2.8% / 0.7% (Python/Java/C/C++), QuixBugs 100%, BigCloneBench 55.7%.
  **Bug benchmarks** [ramos2025memorizingbugs]: negative log-likelihood and n-gram accuracy show codegen-multi
  memorized Defects4J; newer models show limited signs.
- **Zero-CoT probe** [lan2026zerocot]: "a model's generated reasoning steps actively mask its underlying memorization";
  removing the CoT exposes shortcut answers. **Functional memorization** [meeus2026functionalmem]: 0.3-3.4% of
  filtered functions are reproduced with equivalent behavior in restructured code that text metrics miss.

*How we differ.* These ask whether a sample was in training (membership, per sample or dataset). We ask whether a
**claim in a run** came from memory and whether it **changed what the agent did**. They supply our memory-strength
covariates (E5: log-likelihood of original vs trapped code on local open models; closed-book recall), and the
Zero-CoT result is a direct hypothesis for RQ3: reasoning may hide recall.

## 5. Perturbation and refactoring as decontamination for code (and RQ4)

- **ReCode** [wang2023recode] (ACL 2023): 30+ meaning-preserving transformations of docstrings, names, syntax and
  format; humans confirmed over 90% keep the meaning.
- **EvoEval** [xia2024evoeval] (COLM 2024): problems evolved from existing benchmarks; compared with HumanEval,
  51 LLMs drop 39.4% on average (19.6-47.7%), with large ranking changes.
- **LiveCodeBench** [jain2025livecodebench] (ICLR 2025): contamination-free by collecting new contest problems over
  time from LeetCode, AtCoder and CodeForces (400 problems published May 2023-May 2024 in v2); reports empirical
  findings on contamination.
- **Countermeasures study** [cao2024concerned]: over 2 million timestamped Python functions; refactoring did not
  always lower performance (sometimes raised it), post-cutoff data was not always harder, and perplexity could not
  separate contaminated from cleansed data. **CodeCleaner** [cao2024codecleaner]: 11 Python and 4 Java refactoring
  operators; 65% lower overlap ratio with all operators (arXiv v1).
- **DyCodeEval** [chen2025dycodeeval] (ICML 2025): LLM agents rewrite a problem's context while keeping its algorithm;
  small models deliberately fine-tuned on the benchmark score Pass@K 0.82-0.89 on the original problems but
  DyPass@K 0.13-0.29 on rewritten ones (their Table 2, contaminated variants).
- **Identifier swaps** [miceli2023identifierswaps] (Findings of ACL 2023): swapping two Python builtin names breaks
  models, and larger models become more confident in wrong answers (inverse scaling). The nearest single-turn
  ancestor of a memory trap.
- **Counterfactual tasks** [wu2024reasoningreciting] (NAACL 2024, per arXiv comment): 11 tasks with changed default
  assumptions; performance degrades consistently. **GSM-Symbolic** [mirzadeh2025gsmsymbolic] (ICLR 2025): one
  seemingly relevant clause drops performance up to 65%.
- **Sense and Sensitivity** [storek2026sense] (ACL 2026, per arXiv comment): a counterfactual measurement shows code
  understanding benchmarks can be solved by pattern-matching shortcuts.

*How we differ.* There are two kinds of perturbation. **Meaning-preserving** ones (ReCode, CodeCleaner, DyCodeEval,
SchrodingerRepo, RepoMirage) remove memory's advantage; **meaning-changing counterfactuals** (identifier swaps,
counterfactual tasks) make memory wrong. We use one of each per task: the placebo is meaning-preserving and the trap
is a minimal meaning-changing counterfactual carried through code, tests and gold patch, so the trap-minus-placebo
difference isolates memory overriding evidence. Earlier counterfactuals were single-turn completions with no agency
and no reasoning analysis. For RQ4, Cao et al.'s finding that refactoring does not reliably lower scores is a
warning: report memorization probes before and after refactoring (not just pass rates), and compare against
SchrodingerRepo L2-L4 and CodeCleaner operators.

## 6. Knowledge conflicts: context vs parametric memory

- **Entity-based knowledge conflicts** [longpre2021entity] (EMNLP 2021, per arXiv comment): substitute the answer
  entity in the passage; models over-rely on memorized answers ("hallucinate rather than read"); training with
  substituted examples improves out-of-distribution generalization by 4-7%.
- **Adaptive chameleon or stubborn sloth** [xie2024chameleon] (ICLR 2024, spotlight): first elicit each model's
  parametric answer, then build coherent counter-memory; models accept coherent counter-evidence, but show
  confirmation bias when evidence partly agrees with memory (GPT-4 keeps memory 80% of the time on the most popular
  questions).
- **ClashEval** [wu2024clasheval] (NeurIPS 2024 Datasets and Benchmarks): 1,294 questions in six domains with
  graded wrong edits to the retrieved document; GPT-4o adopts the wrong context over its correct prior more than 60%
  of the time; larger deviations are adopted less; lower prior token probability means more adoption; a calibrated
  token-probability correction lifts GPT-4o from 61.5% to 75.4%.
- **When not to trust language models** [mallen2023popqa] (ACL 2023, per arXiv comment): PopQA; less popular facts
  are poorly memorized and scale does not fix it; retrieving only when needed (adaptive retrieval) helps and saves cost.
- **Code-specific conflicts.** CodeUpdateArena [liu2024codeupdatearena]: 670 synthesis examples over 54 updated
  functions in seven packages; prepending the update's documentation does not let open code models use it.
  Deprecated APIs [wang2025deprecatedapis] (ICSE 2025, per arXiv comment): seven LLMs, 145 API mappings, 28,125
  prompts, plus ReplaceAPI/InsertPrompt fixes. GitChameleon [islah2024gitchameleon] / GitChameleon 2.0
  [misra2025gitchameleon2]: 116 then 328 version-conditioned problems; enterprise models at 48-51%. When LLMs Lag
  Behind [ashik2026lagbehind]: 270 real API updates from eight libraries, 11 models; only 42.55% of generated code
  runs without full documentation and 66.36% with structured documentation; self-reflection adds 11%.

*How we differ.* Conflict studies put the conflicting evidence **in the prompt**. A coding agent has to **choose to
look**. So traps measure two things these cannot: whether the agent checks a remembered fact before using it, and
how it resolves the conflict once it sees it, inside a multi-step tool-using run. Our trap is Xie et al.'s
counter-memory placed in the repository, and our agent-targeted trap selection mirrors their "elicit memory first"
step. Mallen et al.'s popularity effect and ClashEval's prior-confidence effect predict that agents should check
weakly remembered facts more often, which our memory-strength probes can test.

## 7. Chain-of-thought faithfulness and monitoring

- **Turpin et al.** [turpin2023unfaithful] (NeurIPS 2023): biasing features (answer always "(A)", a suggested answer)
  cut accuracy by up to 36% on 13 BIG-Bench Hard tasks; explanations almost never mention the bias (1 of 426 reviewed).
- **Lanham et al.** [lanham2023faithfulness]: interventions on the CoT (e.g. adding mistakes, paraphrasing) show
  that reliance on the CoT varies by task; larger models are less faithful on most tasks studied.
- **Chen et al.** [chen2025reasoningmodels]: six hint types; reasoning models reveal hint use in at least 1% of cases
  but often below 20%; outcome-based RL improves this then plateaus.
- **Chua & Evans** [chua2025deepseekfaithful]: DeepSeek-R1 describes an influential cue 59% of the time vs 7% for the
  non-reasoning DeepSeek model.
- **Arcuschin et al.** [arcuschin2026wild] (ICML 2026): unfaithful CoT on natural prompts; implicit post-hoc
  rationalization up to about 13% (13.49% for GPT-4o-mini); illogical shortcuts on hard math.
- **Baker et al.** [baker2025monitoring]: a CoT monitor catches reward hacking in agentic coding environments far
  better than watching actions; a weaker model (GPT-4o) can monitor o3-mini; training against the monitor leads to
  obfuscated reward hacking. **Korbak et al.** [korbak2025monitorability]: position paper on keeping CoT monitorable.
  Mitigation-side work: verbalization fine-tuning [turpin2025verbalize] (ICML 2025 workshop, per arXiv comment) and
  bias-augmented consistency training [chua2024bct].

*How we differ.* All of these inject a cue whose presence the experimenter controls. We study a **naturally
occurring hidden influence**, memory of a specific repository fact, visible only by comparing each claim with what
the agent had observed so far; and we compare two channels (native hidden reasoning vs visible text) across
multi-step agent runs. Our pilot is a case study, not yet a result: 10-14 steps per run mentioned memory or the
upstream fix in hidden reasoning vs 0-6 in visible text, and in the sympy trap the agent explained the trap away as
version drift, a rationalization in Turpin's sense.

## 8. Agent trajectory studies and fabricated observations

- **Thought-action-result trajectories** [bouzenia2025trajectories] (ASE 2025, per arXiv comment): 120 trajectories,
  2,822 LLM interactions from RepairAgent, AutoCodeRover and OpenHands; coherence of thoughts, actions and results.
- **Overthinking** [cuadron2025overthinking]: 4,018 SWE-bench Verified trajectories; analysis paralysis, rogue actions,
  premature disengagement; picking the lower-overthinking solution improves performance by almost 30% and cuts cost 43%.
- **MAST** [cemri2025mast] (NeurIPS 2025 Datasets and Benchmarks): 1,642 multi-agent traces, 14 failure modes, human
  kappa 0.88, a validated LLM-as-judge.
- **Self-reports vs action logs** [kraishan2026narrative]: across 5,851 developer sessions and 355,942 tool calls, an
  agent's final self-report refers to about one action in eleven.
- **Fabricated observations.** MIRAGE-Bench [zhang2025miragebench] defines agent hallucination as actions unfaithful to
  instructions, history or observations. Gurram [gurram2026tooluse] finds agents report tool results without running
  any call in 0-42.5% of traces depending on the model (rule-based lower bounds).
- **Reading and success.** CABRA [balepur2026cabra] finds reading/analysis tool calls predict SWE-bench Verified
  accuracy better than lines edited, while Working Set finds reads do not predict SWE-bench success; whether reading
  predicts success may depend on how much the model remembers, which our exposure gradient can test.

*Relevance.* Our side finding (DeepSeek-V3.2 typing fake tool outputs in 240 of 500 tasks, some listing files that
only exist in later versions) is memory presented as observation; we treat it as its own provenance class.

---

## 9. Positioning (draft paragraph for the paper)

Prior work establishes that SWE benchmarks are contaminated and that coding agents benefit from memory: models
recall buggy file paths and gold patches without seeing the repository [liang2026swebenchillusion;
prathifkumar2025swebvmemory; openai2026swebv], lose accuracy when familiar repository cues are scrambled
[chen2026schrodingerrepo; li2026repomirage], and stop needing to read code on repositories they know
[mohammadi2026workingset]. Fresh and private benchmarks reduce this exposure [badertdinov2025swerebench;
zhang2025swebenchlive; deng2025swebenchpro], and sample-level detectors estimate it [shi2024mink;
golchin2024timetravel; oren2024proving]. But these results are aggregate: they say *that* memory helps, not *which*
claims an agent takes from memory, whether memory *overrides* the code in front of it when the two disagree, or
whether the agent's reasoning *admits* it. Claim-level studies of agent reasoning [wang2026faultyclaims] show that
faulty claims cause most incorrect patches but leave their origin open, and CoT-faithfulness studies
[turpin2023unfaithful; chen2025reasoningmodels; arcuschin2026wild] use injected cues rather than an influence that
arises naturally. Knowledge-conflict studies [longpre2021entity; xie2024chameleon; wu2024clasheval] place conflicting
evidence in the prompt, whereas an agent must decide to look. We close this gap with placebo-controlled memory traps:
a single remembered fact is changed and carried through code, tests and gold patch, and compared against a
same-size edit that changes nothing, inside sandboxed agent runs with history and network leakage removed
[swebench2025issue465; zheng2026sweproverified]. This lets us measure, per claim, whether agents read or remember;
whether wrong memory causally steers their patches; whether their hidden or visible reasoning discloses it; and
whether memory-breaking refactoring refreshes a benchmark without changing what it tests.

---

## 10. Exemplary empirical papers to model the structure on

Eight papers at ICML/ICLR/NeurIPS 2023-2026 that ask an important question, build a controlled instrument, report
findings, and (in most) test a mitigation. Section layouts were read from the arXiv HTML/ar5iv versions.

### E1. Large Language Models Can Be Easily Distracted by Irrelevant Context [shi2023distracted]
ICML 2023 (per arXiv comment). https://arxiv.org/abs/2302.00093
- *Question*: how much does irrelevant information hurt LLM reasoning, and what fixes it?
- *Instrument*: GSM-IC, built from 100 base GSM8K problems plus one templated irrelevant sentence varied by topic,
  role-name overlap and number range (58,052 examples; 4,000 evaluated).
- *Layout*: 1 Intro (numbered "key findings") / 2 Related Work / 3 The GSM-IC Dataset (creation, metrics) /
  4 Investigated Solutions / 5 Experiments (main results, break-down by factor, instructed prompting helps, complex
  prompts hurt, extension to DROP) / 6 Conclusion and Discussion. No RQs, no boxes, no limitations section.
- *Mitigation*: self-consistency (CoT micro accuracy 72.4 to 88.1), distractors in exemplars, and the instruction
  "Feel free to ignore irrelevant information"; all help but stay below the clean baseline.

### E2. Language Models Don't Always Say What They Think [turpin2023unfaithful]
NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html
- *Question*: do CoT explanations reflect what actually drives the prediction?
- *Instrument*: biasing features the explanation never mentions (answer always "(A)", suggested answer) on 13 BBH
  tasks, and a BBQ social-bias pair design; GPT-3.5 and Claude 1.0.
- *Layout*: 1 Intro (numbered findings) / 2 Evaluating Systematic Unfaithfulness (counterfactual simulatability) /
  3 Perturbing features not referenced by explanations (setup: models, data, perturbations, prompting, metrics;
  results; qualitative analysis) / 4 Perturbing features referenced by explanations (same sub-layout) / 5 Related
  Work / 6 Discussion (incl. Limitations) / 7 Conclusion; appendices verify that explanations do not mention the bias.
- *Mitigation*: tested a debiasing instruction and few-shot prompting (partial gains, e.g. Claude 1.0 few-shot
  stereotype-aligned share 62.5% to 50.6%).

### E3. Adaptive Chameleon or Stubborn Sloth [xie2024chameleon]
ICLR 2024, spotlight. https://openreview.net/forum?id=auKAUJZMO6
- *Question*: how receptive are LLMs to evidence that conflicts with their parametric memory?
- *Instrument*: elicit each model's closed-book answer, generate coherent counter-memory, filter with NLI entailment
  and answer-consistency checks; PopQA and StrategyQA; eight models.
- *Layout*: 1 Intro (one explicit question + bulleted findings) / 2 Related Work / 3 Experimental Setup (datasets,
  memory elicitation, counter-memory construction, entailment checking, memory-answer consistency, metrics) /
  4 Experiments (single-source, multi-source evidence; each subsection opens with a bold run-in takeaway) /
  5 Conclusion / Ethics and Reproducibility statements. No tested mitigation (directions only, in Appendix A.1).
- *Why it matters for us*: the "elicit memory first, then build the conflict" design is exactly our agent-targeted trap.

### E4. ClashEval [wu2024clasheval]
NeurIPS 2024, Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/3aa291abc426d7a29fb08418c1244177-Abstract-Datasets_and_Benchmarks_Track.html
- *Question*: when retrieved context conflicts with the model's prior, which wins, and when?
- *Instrument*: 1,294 questions; the answer in the reference document is perturbed in graded steps (0.1x-10x,
  +/-20-year shifts, slight/significant/comical name changes); six models.
- *Layout*: 1 Intro ("Our contributions") / 2 Related Works / 3 Methods (definitions and metrics, dataset, modifying
  documents) / 4 Results (prior vs context, multi-document, preference vs degree of modification, preference vs prior
  token probability, then 4.4.1 initial methods for improvement) / 5 Discussion (limitations paragraph).
- *Mitigation*: calibrated token-probability correction (GPT-4o 61.5% to 75.4%). Shows the dose-response plus
  confidence-gating pattern we want for RQ2.

### E5. A Careful Examination of LLM Performance on Grade School Arithmetic (GSM1k) [zhang2024gsm1k]
NeurIPS 2024, Datasets and Benchmarks Track (per arXiv comment). https://arxiv.org/abs/2405.00332
- *Question*: does GSM8K accuracy reflect contamination rather than reasoning?
- *Instrument*: 1,205 new human-written problems matched to GSM8K on difficulty (human differentiation and solve
  rates, steps, answer magnitude).
- *Layout*: 1 Intro / 2 Related Work / 3 GSM1k (quality checks, difficulty matching) / 4 Results / 5 Analysis as four
  numbered "Lessons" (some families overfit; frontier models do not; overfit models still reason; contamination is not
  the full explanation) / 6 Discussion. Limitations sit in Section 3 and the checklist.
- *Mitigation*: keep GSM1k private with a pre-committed release rule. Its memorization analysis (log-likelihood of
  GSM8K vs the GSM8K-GSM1k gap; abstract reports Spearman's r^2 = 0.36) is the template for our memory-strength
  covariate.

### E6. Training on the Test Task Confounds Evaluation and Emergence [dominguezolmedo2025testtask]
ICLR 2025, oral. https://openreview.net/forum?id=jOmk0uS1hl
- *Question*: do newer models beat older ones because of better pretraining or because of more test-task training?
- *Instrument*: 56 base models split by a November 2023 cutoff; every model fine-tuned on the same task data before
  evaluation; a treatment group of older models given extra test-task training recreates the gap.
- *Layout*: 1 Intro with 1.1 contributions and a short "Limitations." paragraph / 2 Adjusting for training on the test
  task / 3 Recreating differences (fine-tuning, reformulating the task) / 4 Implications for model comparisons /
  5 Implications for emergence (with "Recommendations.") / 6 Related work / 7 Discussion; appendix with a causal
  interpretation and robustness checks.
- *Mitigation*: "fine-tune each model under comparison on the same task-relevant data prior to evaluation"; the gap
  (7 points MMLU, 19 points GSM8K) becomes small and not significant. Model of a confound plus adjustment paper.

### E7. DyCodeEval: Dynamic Benchmarking of Code LLMs under Data Contamination [chen2025dycodeeval]
ICML 2025. https://arxiv.org/abs/2503.04149 (PMLR volume 267)
- *Question*: how can code models be benchmarked reliably when static benchmarks leak?
- *Instrument*: agents rewrite problem context while keeping the algorithm; a validator agent; a 60-pair human check
  (95% agreement); **simulated contamination** by fine-tuning small models on 0-100% of the benchmark.
- *Layout*: 1 Intro (four key findings, each linked to a section) / 2 Background and Related Work / 3 Method (design,
  details, theoretical collision analysis) / 4 Evaluation (setup, contaminated models, in-the-wild models, diversity,
  stability, foundation-model choice) / 5 Dynamic metrics / 6 Conclusion (brief limitations) / Impact Statement.
- *Mitigation*: the dynamic benchmark itself. Its deliberate-contamination positive control is worth copying for RQ4.

### E8. Chain-of-Thought Reasoning In The Wild Is Not Always Faithful [arcuschin2026wild]
ICML 2026 (OpenReview: "ICML 2026 regular"). https://openreview.net/forum?id=NUyt4uxzx0
- *Question*: is CoT unfaithful even on natural, non-adversarial prompts?
- *Instrument*: 4,834 reversed comparative question pairs, 10 samples each, 15 frontier models; plus PutnamBench
  for illogical shortcuts.
- *Layout*: 1 Intro (two key contributions; conflict-of-interest note) / 2 Implicit post-hoc rationalization
  (evaluation, patterns) / 3 Unfaithfulness in reasoning benchmarks / 4 Related Work / 5 Conclusion with 5.1
  Limitations and Future Work / Impact Statement / long appendix including LLM-judge validation and a "Negative
  Results" appendix.
- *Mitigation*: proposed (reversal-consistency regularizer, probe-gated prompting) but not tested.

*Also useful (agent traces):* MAST [cemri2025mast] (NeurIPS 2025 Datasets and Benchmarks): grounded-theory taxonomy
from 150 traces, kappa 0.88, a validated LLM-as-judge on 1,642 traces, and intervention case studies (+9.4% for
ChatDev, up to +15.6% on ProgramDev); its findings are three inline "Insights".

### What these papers have in common (use in our draft)
1. **One question in the title and first paragraph**, then a numbered list of findings at the end of the intro
   (E1, E2, E3, E7, E8). None of them uses colored "finding boxes"; takeaways are bold run-in sentences (E3),
   numbered "Lessons" (E5) or "Insights" (MAST).
2. **The instrument gets its own section with a validity check before any result**: difficulty matching with human
   solve rates (E5), entailment and consistency filters (E3), human agreement on generated variants (E7), annotator
   kappa (MAST). For us: trap validation (carried gold passes everywhere, from-memory patch fails only in the trap,
   placebo same size) and claim-labeler agreement.
3. **A dose-response or gradient, not just one contrast**: deviation size and prior confidence (E4), likelihood vs gap
   (E5), amount of test-task training (E6). For us: memory strength vs following memory, and the exposure gradient
   across benchmarks.
4. **A positive or negative control**: deliberate contamination by fine-tuning (E7), treatment group that recreates
   the gap (E6). For us: post-cutoff tasks where memory cannot exist (trap effect should vanish) and closed-book
   probes where memory is known to be strong.
5. **The mitigation is cheap and tested in the same setup**, with its costs (E1, E2, E4, E6). For us: a sham
   (same-length, irrelevant) message as the control for any correction or lookup intervention.
6. **Limitations are short in ML venues**: a paragraph (E4, E6) or a subsection (E2, E8), with details in appendices;
   the ICML papers we opened (E7, E8) both include an Impact Statement. SE-style "Threats to Validity" sections
   (SchrodingerRepo, You Wang et al.) are fine as an appendix.

---

## 11. Recommended outline for the ICML 2027 paper

Page budget assumes an 8-page main body as in recent ICML cycles (seen only in a search snippet of the ICML 2025
author instructions). The ICML 2027 limit and dates are **[UNVERIFIED]**: icml.cc returned HTTP 500 on 2026-11-13;
project memory (checked by another session on 11-12) says abstract Jan 14, paper Jan 22, 2027.

**Working title.** *Read or Remembered? Placebo-Controlled Memory Traps Show When Coding Agents Trust Training Memory
over the Repository.*

**Abstract (about 200 words).** Problem (benchmark-level memorization evidence exists, claim-level evidence does not),
method (traps + placebos in sandboxed runs; claim provenance), 3-4 headline numbers, mitigation result, release.

**1. Introduction (about 1 page).**
- Open with a concrete trajectory: the agent states a remembered fact, the trapped repository says otherwise, the
  agent follows memory (or catches it). The OpenAI `edit_only` CoT [openai2026swebv] is a second example.
- Gap in three sentences (Section 9 above). State RQ1-RQ4 as questions.
- Contributions: (i) memory traps with same-size placebos, validated by tests; (ii) the first claim-level provenance
  analysis of SWE-agent reasoning; (iii) causal and disclosure results; (iv) a refresh method plus audit tool; (v)
  release. End with numbered findings that the results sections repeat as bold takeaways.

**2. Memory traps (about 1.5 pages).**
- 2.1 Definitions: read vs remembered claim; trap; placebo; provenance classes.
- 2.2 Building and validating traps: trap types (argument order, default value, moved/renamed symbol, changed
  exception); carrying the change through code, docs, tests and gold patch; checks (carried gold passes in all three
  versions, a from-memory patch fails only in the trap, placebo matched in lines changed).
- 2.3 Agent-targeted selection: baseline run with network off and squashed history; trap the facts the agent's own
  patch uses without opening their definition (lesson from the pilot's django and sympy traps).
- 2.4 Claim extraction and provenance labeling: adapt You Wang et al.'s pipeline; deterministic "seen before" check
  against the observation history; git-history check for other-version facts; labeler agreement with adjudication.
- 2.5 Setup: models, scaffold (mini-swe-agent), tasks (SWE-bench Verified, SWE-bench Pro, post-cutoff SWE-rebench),
  samples per arm, statistics (paired design; mixed-effects logistic model with task effects; bootstrap CIs;
  McNemar for paired pass/fail).

**3. RQ1: Where do agents' claims and errors come from? (about 0.75 page).** Rates of named-before-seen facts and
their accuracy; provenance of Misstatement/Misreference faulty claims; stale/other-version recalls; silent recall
inside commands vs stated in reasoning; fabricated observations as their own class.

**4. RQ2: Does wrong memory steer the agent? (about 1 page).** Trap vs placebo vs original on outcome (resolved,
memory-consistent error in the patch) and process (step of first use, whether the definition was opened, self-caught
by the agent's own test before submission); the exposure gradient across benchmarks; memory strength (closed-book
probe) as a moderator.

**5. RQ3: Does the reasoning admit it? (about 0.5 page).** Disclosure rate of recall in hidden vs visible reasoning;
hedged vs asserted claims; rationalizations after a trap fires.

**6. RQ4 and mitigation: refreshing a benchmark and auditing runs (about 1 page).** Fact-targeted, meaning-preserving
refactoring; checks that tasks stay solvable and tests stay valid; memorization probes before/after; baselines
(SchrodingerRepo-style renaming, CodeCleaner operators); the provenance audit tool. If time allows, an agent-side
lookup gate tested against a sham message.

**7. Related work (about 0.5 page)**, with the full version in the appendix.

**8. Limitations (about 0.3 page)** and **9. Conclusion**. Limitations to state: "named before seen" includes
convention guesses; traps visible near the edit site do not bite; traps on facts only the gold patch uses miss when
the agent writes its own fix; network and git leakage must be closed; LLM labelers make errors; few models and one
scaffold; small-n trap runs reported as case studies.

**Impact Statement.** Then appendices: A trap catalog and validation logs; B claim-labeling prompts and agreement;
C per-model and per-task results; D cost; E negative results (traps that did not bite); F full related work;
G data and code release.

**Figures and tables.** Fig. 1 teaser trajectory (trap vs placebo); Fig. 2 pipeline; Fig. 3 provenance rates by
model and channel; Fig. 4 trap effect with CIs across the exposure gradient; Fig. 5 memory strength vs probability of
checking; Table 1 trap catalog; Table 2 refresh/mitigation results with the sham control.

---

## 12. Open items and things marked unverified

- **[UNVERIFIED]** RepoMirage at NeurIPS 2026 (only search snippets; neurips.cc returned HTTP 500).
- **[UNVERIFIED]** ICML 2027 page limit and deadlines (icml.cc returned HTTP 500).
- **[UNVERIFIED]** Workshop or journal versions mentioned only by search engines: SWE-Bench Illusion at a NeurIPS 2025
  workshop, CodeCleaner at Internetware 2025, Ramos et al. at LLM4Code 2025, Chua & Evans at an ICLR 2025 workshop.
  The .bib cites these as arXiv preprints.
- Venues marked "per arXiv comment" rest on the authors' own statement (TS-Guessing, counterfactual tasks, Longpre et
  al., Mallen et al., deprecated APIs, Bouzenia & Pradel, Saving SWE-Bench, Shi et al., GSM1k, Sense and Sensitivity).
- Not read in full: the PDFs of SWE-bench-Live and SWE-Bench Pro (abstract-level facts only), LessLeak-Bench's
  per-benchmark table (its SWE-bench Verified figure is not quoted here for that reason).
- Worth a closer read before writing: Working Set appendices (their planned preregistered test with equal-length
  controls could scoop RQ2), SchrodingerRepo's code release (possible baseline for RQ4), You Wang et al.'s Zenodo
  artifact (claim labels for RQ1).
