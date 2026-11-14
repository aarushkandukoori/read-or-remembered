# Memory traps: design rules and catalog

A memory trap changes one fact about a repository that a model has probably memorized from training, and carries
that change consistently through the whole repo (definition, every call site, docs, doctests, the task's tests and
the gold patch). The repo still works and the task is still solvable; only the fact the model remembers is now false.
Every trap has a placebo: an edit of the same size, in the same places, that keeps the remembered fact true.

## Three rules (the first two come from You Wang's feedback, Nov 13 2026)

1. **It must make sense.** The trap must look like a refactoring a real project could have done (reordered
   parameters, a renamed helper, a moved module, a stricter default). Arbitrary semantic changes just confuse the model.
2. **It must matter for the task.** The task's tests must give a different result depending on whether the agent
   used the remembered fact or the real one. Check: the upstream gold patch applied verbatim (= the fix a model would
   write from memory) must FAIL the target tests in the trap version and PASS in the placebo version.
3. **It must not be visible next to the edit.** If a call site that shows the new fact sits next to where the agent
   edits, the agent just copies it (seen in the django-11149 pilot). Prefer facts the agent's own past patches relied
   on without opening the definition ("agent-targeted" traps).

## Catalog

| id | trap | example | loud or quiet | placebo |
|---|---|---|---|---|
| T1 | argument order | `f(action, opts)` becomes `f(opts, action)`, all positional calls swapped | quiet if types match | keywordize the same calls |
| T2 | default value | `strict=False` becomes `strict=True`; calls that relied on the default now pass `strict=False` | quiet | add the same explicit `strict=False` without changing the default |
| T3 | rename | `get_permission_codename` becomes `permission_codename_for` everywhere, no alias | loud (NameError) | rewrite the same call sites as `module.f(...)` style, name unchanged |
| T4 | move | function moves from `a.py` to `b.py`, imports updated, no re-export | loud (ImportError) | add a re-export module and point imports at it, function stays in `a.py` |
| T5 | exception swap | function raises `TypeError` instead of `ValueError`; every `except` updated | quiet | wrap the same raise/except lines in a no-op helper |
| T6 | return convention | returns `(b, a)` instead of `(a, b)`; every caller updated | quiet | rename the local unpacking variables at the same callers |
| T7 | keyword rename | parameter `opts` becomes `options`; keyword calls updated | loud (TypeError) | reformat the same keyword calls |
| T8 | flag inversion | `case_sensitive=True` becomes `ignore_case=False` | loud by name, quiet by value | reformat the same calls |

Loud traps fail fast, so they test whether the agent notices and repairs (errors show up mid-trajectory).
Quiet traps can survive to the final patch unless the tests catch them. You Wang's point: simple traps may only show
up at intermediate points because the agent's own reproduction tests catch them. So every run is scored at three
levels: (a) did a claim or action follow the remembered fact, (b) did an error appear mid-trajectory, (c) did the
final patch still carry it.

## Validation (free, local Docker, official SWE-bench images)

For each (task, trap) and its placebo:
1. PASS_TO_PASS tests still pass with the carried gold patch (repo still works).
2. FAIL_TO_PASS tests fail with no fix and pass with the carried gold patch (task still solvable).
3. The verbatim upstream gold patch fails FAIL_TO_PASS in the trap and passes in the placebo (trap matters, placebo doesn't).
4. Git history is squashed in images used for agent runs (no `git log` shortcut to the upstream fix), and runs use
   `--network none` (pilot: 6 of 9 network-on runs downloaded the upstream fix).
