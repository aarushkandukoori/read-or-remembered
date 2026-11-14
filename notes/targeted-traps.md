# Agent-targeted memory traps, network off (Nov 13, 2026)

Model: DeepSeek `deepseek-flash` (the official API's V4 Flash), mini-swe-agent 2.4.6, SWE-bench Verified
Docker images. Every agent container ran with `--network none`, and git history was squashed to one
commit in every version. Code: `scripts/targeted.py` (build/run/grade/validate),
`scripts/make_argswap_trap.py`, `scripts/make_kwrename_trap.py`, `scripts/analyze_targeted.py`.
Per-run data: `runs/<task>/*.traj.json`, `runs/<task>/grades.json`, `outputs/targeted_summary.json`.

## 1. How targets were chosen

The protocol picks traps from what the model itself does: run the model on the original repo, find a
repo function its own patch calls without having opened the definition, then trap that function.

Step A (free, public data): in the public mini-swe-agent runs of DeepSeek-V3.2, Kimi K2.5 and MiniMax
M2.5 on all 306 django and sympy tasks of SWE-bench Verified, the fix code (non-test source files that
existed before) contains 603 calls to repo functions. 253 of them call a function whose definition the
agent never opened. But in every one of the 30 cases where a clean trap was possible, the agent had
already seen other calls to that function in the run (1 to 158 of them). So agents mostly copy usage
from code they read. `scripts/mine_targets2.py`, `outputs/targets4.jsonl`.

Step B (paid, network off): DeepSeek baseline runs on 4 candidate tasks. All 4 were solved. In 2 of them
its own fix called a function whose definition it never opened:
- django__django-15563: `query.add_filter("pk__in", ...)` (saw 26 other calls first)
- django__django-16100: `transaction.atomic(using=router.db_for_write(self.model))` (saw 8 other calls)

In sympy__sympy-15345 and sympy__sympy-19040 it read the definitions first, so no targeted trap there.

## 2. Trap A: django__django-15563, `Query.add_filter(filter_lhs, filter_rhs)` swapped

- Trap: definition becomes `add_filter(self, filter_rhs, filter_lhs)`; all 7 calls (3 files) are
  rewritten so behavior is unchanged. 19 changed lines.
- Placebo: the same 7 calls become keyword calls (`filter_lhs=..., filter_rhs=...`), definition
  unchanged. 17 changed lines.
- Validation (task tests in Docker):

| version | no fix | gold fix carried over | DeepSeek's own fix carried over | DeepSeek's own fix as written (original order) |
|---|---|---|---|---|
| original | F2P 0/2 | 2/2, P2P 29/29 | 2/2, P2P 29/29 | (same) |
| trap | 0/2 | 2/2, 29/29 | 2/2, 29/29 | **0/2, P2P 28/29** |
| placebo | 0/2 | 2/2, 29/29 | 2/2, 29/29 | 2/2, 29/29 |

- Runs (3 original, 5 trap, 5 placebo): **all 13 resolved.** In every written call, the form matched
  the version's existing call sites. Original runs wrote lookup first (`add_filter("pk__in", ids)`),
  trap runs wrote value first (`add_filter(ids, "pk__in")`, 4 to 10 writes per run, never the
  remembered order), and placebo runs wrote keywords. No intermediate errors from the trapped fact.
  Steps 31 to 63, cost $0.016 to $0.065 per run.

## 3. Trap B: django__django-16100, `transaction.atomic(using=...)` renamed to `alias=`

- Trap: keyword parameter `using` of `atomic()` renamed to `alias` (definition, body, the comment that
  named it), and all 39 calls in 20 files rewritten. A call written from memory with `using=` raises
  `TypeError`. 91 changed lines.
- Placebo: the identical visible edit, but the definition still accepts `using=` through a two-line
  alias (`**_legacy_kwargs`). 92 changed lines.
- Validation:

| version | no fix | gold fix carried over | DeepSeek's own fix carried over | DeepSeek's own fix as written (`using=`) |
|---|---|---|---|---|
| original | F2P 0/1 | 1/1, P2P 59/59 | 1/1, 59/59 | (same) |
| trap | 0/1 | 1/1, 59/59 | 1/1, 59/59 | **0/1, P2P 58/59** |
| placebo | 0/1 | 1/1, 59/59 | 1/1, 59/59 | 1/1, 59/59 |

- Runs (3 original, 6 trap, 6 placebo):

| version | resolved | searched for `atomic(using=` | ran or wrote `using=` | TypeError seen and fixed | hidden reasoning notes the conflict | visible text notes it | final patch |
|---|---|---|---|---|---|---|---|
| original | 3/3 | 0 | 3 (correct there) | 0 | 0 | 1 | `using=` |
| trap | **6/6** | 3/6 | 1/6 | 1/6 | **6/6** (1 to 14 steps) | 4/6 (1 to 2 steps) | `alias=` in 6/6 |
| placebo | 2/6 | 2/6 | 0/6 | 0 | 6/6 | 2/6 | `alias=` in 6/6 |

- In the trap, the hidden reasoning recalls the old API and then explains the mismatch away as a version
  difference, for example: "I recall exact upstream ... Existing code uses
  transaction.atomic(alias=router.db_for_write(self.model)), newer deprecation?" and "There is
  atomic(using=None, savepoint=True, durable=False), no alias? But source uses alias maybe branch from
  5?". The final patches follow the repo.
- The 4 failed placebo runs all failed the same existing test (a query-count test, "list_editable edits
  use a filtered queryset to limit memory usage"). They wrapped a larger block of `changelist_view` in
  the transaction (patch at line 1912 instead of 2011), which adds queries. Their `atomic` call itself
  was correct. We cannot explain why this placement shows up only in the placebo; it is unrelated to the
  trapped keyword and needs more samples.

## 4. What this says

1. **With network off, the targeted traps never reached a final patch: 11/11 trap runs were solved.**
   Simple API traps get caught or avoided, which matches You Wang's prediction that their effect would
   show up only at intermediate points.
2. **The agent copies usage from code it reads.** For an internal API (`add_filter`), the form it wrote
   matched the visible call sites in every written call of all 13 runs, and the remembered order never
   appeared in a trap run.
3. **For a very common idiom (`atomic(using=)`), memory does surface.** It shows up as searches for the
   old keyword, one TypeError, and conflict notes in the hidden reasoning of every trap run. The visible
   text is quieter (4 of 6 runs, 1 to 2 steps). The repo wins in the end.
4. Together with the earlier pilot, the larger threat to benchmark validity in our runs was not API-level
   memory. It was retrieval of the upstream fix: with network on, 6 of 9 runs downloaded newer upstream
   code (GitHub raw files, `pip download` of later releases) and copied the fix. Network-off runs still
   tried to download it every time.

Caveats: one model, two trapped tasks, 5 to 6 runs per arm. The heuristics for "searched" and "wrote"
are regex based. The conflict counts come from keyword matching in the reasoning text, not hand labels.

## 5. Spend

DeepSeek balance went from $2.00 to $0.98. This experiment used about $0.62 of that: baselines $0.22,
trap/placebo/original runs $0.33 + $0.17, plus test calls. The earlier pilot runs used the rest. Every
run cost $0.003 to $0.13.

## 6. Trap C: django__django-12741, same `atomic(using=)` rename, keyword on the edited line (Nov 14)

- Baseline (network off): DeepSeek solved it ($0.003). Its fix writes `transaction.atomic(using=self.connection.alias, ...)`
  without opening the definition (5 other calls seen). Target confirmed.
- Trap/placebo built with the same tools as Trap B (39 calls in 20 files; Django 3.1). Here the repo's own
  line that the fix edits reads `transaction.atomic(alias=using, savepoint=...)`, so the new keyword is
  visible exactly where the agent works.
- Validation: in the trap, DeepSeek's own fix written with `using=` fails 0/2 F2P + 3 P2P; carried over
  it passes 2/2, 67/67. Placebo clean. Gold passes everywhere.
- Runs (network off):

| version | n | resolved | ran or wrote `using=` | TypeError seen | final patch |
|---|---|---|---|---|---|
| original | 3 | 3 | 3 (correct there) | 0 | `using=` 3/3 |
| trap | 20 | **19** | **5** | 4 | `alias=` 19, `using=` 1 (the one failure) |
| placebo | 20 | 20 | 3 | 0 | `alias=` 18, `using=` 2 (works, alias accepted) |

- So memory "corrected" visibly correct code in 8 of 40 trap/placebo runs (20%). In the trap that was
  caught by the agent's own runs in 4 of 5 cases, and once it reached the final patch and failed the
  task (trap 19/20 vs placebo 20/20). Hidden reasoning says things like "Wait transaction.atomic
  signature uses using not alias!" and "Current uses alias=using incorrectly?".
- The keyword-matching "conflict" count is not reported here: `self.connection.alias` is part of the fix,
  so the word "alias" appears regardless.
- Cost: about $0.10 for all 44 runs ($0.002 to $0.004 each). DeepSeek balance now $0.88.
