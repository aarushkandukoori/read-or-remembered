# Read or Remembered? Full-project plan (Nov 13, 2026)

Owner: this folder. Collaborator: You Wang (ZJU), who agreed on 11-13 3:46 AM ET to work on this together.
Budget: paid API total = the $1.62 left on DeepSeek, spent only by the targeted-trap runs in ~/Projects/rr-targeted
(other Claude session, network off). Everything else here must be free: public data, local Docker, local MLX models,
Claude subagents as annotators.

## Research questions (revised after You Wang's 11-13 feedback)

- RQ1 Provenance. Where do coding agents' explicit factual errors come from? Scope = Misstatement and Misreference
  faulty claims (You Wang's suggestion; Omission/Overgeneralization excluded because imprecise wording can't be told
  apart from misunderstanding). Provenance classes: read-but-misread (contradicts something already observed),
  recalled-other-version (true in another commit of the repo), convention-guess (true in similar libraries or by
  naming convention, never in this repo), propagated (follows an earlier faulty claim), unsupported.
  Also at scale: how often do agents state repo facts (paths, symbols, signatures) before observing them, how often
  are those right, do they self-correct, and does it matter for resolution? Data: SWE-bench Verified mini-swe-agent
  runs (DeepSeek-V3.2, Kimi K2.5, MiniMax M2.5), DeepSWE trials (newer models), You Wang's labeled SWE-bench Pro claims.
- RQ2 Traps. When a remembered fact is made false, do agents act on memory? Where in the trajectory do trap-induced
  errors appear, and are they caught by the agent's own reproduction tests before submission (intermediate vs final)?
  Simple traps (argument order, default value) vs refactoring traps (moved/renamed symbol, changed exception),
  each with a same-size placebo. Design rules from You Wang: traps must make sense (plausible refactorings) and must
  matter for the task (test patch sensitive to the trap).
- RQ3 Disclosure. Does the reasoning reveal reliance on memory? Hidden vs visible reasoning; hedged vs asserted.
- RQ4 Benchmark validity (replaces "repair"). Can semantics-preserving, memory-breaking refactoring refresh a
  benchmark: tasks stay solvable, tests stay valid, and the advantage of memorized code goes away? Validate with
  solvability checks, closed-book memorization probes on local models, and the small agent runs.

## Experiments

- E1 fact provenance at scale (deterministic; paths + symbols + signatures), SWE-bench Verified + DeepSWE.
- E2 faulty-claim provenance (Misstatement/Misreference), two independent LLM annotators + adjudication, agreement reported.
- E3 trap catalog: simple + refactoring traps with placebos, validated in the official SWE-bench Docker images.
- E4 agent runs on targeted traps (other session, DeepSeek, network off) + pilot runs (django-11149, sympy-13615).
- E5 memorization probes with local MLX models (log-likelihood original vs trapped code; closed-book recall).
- E6 disclosure analysis (hidden vs visible; hedged vs asserted) on the same trajectories.
- E7 mitigation: automated refactor-and-rerun refresh at scale + a trajectory provenance audit tool.

## Deliverables

- Public GitHub repo aarushkandukoori/read-or-remembered: code, small derived data, results, docs, tests, CI,
  release with data bundle (+ weights if a classifier is trained). Paper source stays out of the public repo until
  You Wang has seen it (paper/ is gitignored).
- Paper: ICML-style anonymous submission draft (PDF + LaTeX).
- One combined email to You Wang (reply to the 3:46 AM message), with PDF + repo link; answer the CMU workload question.

## Honesty rules for every output

No invented numbers or citations. Small-n results are labeled as case studies. Annotations made by LLM annotators are
called that (never "hand-labeled"). Others' ideas are credited (You Wang's feedback shaped RQ1 scope, the refactoring
traps and the benchmark-validity framing).
