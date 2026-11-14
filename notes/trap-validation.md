# Trap catalog: Docker validation (Nov 14, 2026)

Script: `.venv/bin/python -I scripts/docker_validate_traps.py run --deadline 03:03` (started 00:41, finished 01:08 ET, one container at a time, official SWE-bench images, `--network none`, diffs applied with `git apply` at /testbed, the task's official test command, SWE-bench log parser), then `... report` (writes outputs/traps/validation.json and validation.csv). Raw logs: work/trap_validation/logs/ (gitignored).

Checks per pair (VALID = a, b, c pass and every d check that ran):
- (a) trap + tests_trap + gold_trap: all FAIL_TO_PASS and PASS_TO_PASS pass (task still solvable)
- (b) trap + tests_trap + verbatim upstream fix: at least one FAIL_TO_PASS test fails (the trap matters for the task)
- (c) placebo + tests_placebo + verbatim upstream fix: all FAIL_TO_PASS pass (the placebo is harmless)
- (d) original + gold passes; trap with no fix fails; placebo + carried gold passes

| task | trap | type | (a) solvable | (b) upstream fix fails in trap | (c) placebo harmless | (d) | verdict |
|---|---|---|---|---|---|---|---|
| django__django-13551 | T3-get_email_field_name | T3 | ok (F2P 2/2) | ok (F2P 0/2) | ok (F2P 2/2) | ok | VALID |
| django__django-14011 | T3-close_all | T3 | ok (F2P 17/17) | FAIL (F2P 17/17) | ok (F2P 17/17) | ok | INVALID |
| django__django-15037 | T1-get_primary_key_column | T1 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| django__django-15127 | T4-get_level_tags | T4 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| django__django-15161 | T7-deconstructible-path | T7 | ok (F2P 3/3) | ok (F2P 0/3) | ok (F2P 3/3) | ok | VALID |
| django__django-15252 | T8-_create_project_state | T8 | ok (F2P 2/2) | ok (F2P 0/2) | ok (F2P 2/2) | ok | VALID |
| django__django-15957 | T3-clear_limits | T3 | ok (F2P 4/4) | ok (F2P 0/4) | ok (F2P 4/4) | ok | VALID |
| django__django-16263 | T2-split_having_qualify-must_group_by | T2 | ok (F2P 3/3) | FAIL (F2P 3/3) | ok (F2P 3/3) | ok | INVALID |
| django__django-16612 | T8-get_full_path | T8 | ok (F2P 2/2) | ok (F2P 0/2) | ok (F2P 2/2) | ok | VALID |
| django__django-16631 | T7-salted_hmac-secret | T7 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| sympy__sympy-12419 | T4-KroneckerDelta | T4 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| sympy__sympy-13877 | T6-_find_reasonable_pivot | T6 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| sympy__sympy-14248 | T2-_keep_coeff-sign | T2 | ok (F2P 2/2) | FAIL (F2P 2/2) | ok (F2P 2/2) | ok | INVALID |
| sympy__sympy-14248 | T3-_keep_coeff | T3 | ok (F2P 2/2) | ok (F2P 1/2) | ok (F2P 2/2) | ok | VALID |
| sympy__sympy-14248 | T6-as_coeff_mmul | T6 | ok (F2P 2/2) | ok (F2P 1/2) | ok (F2P 2/2) | ok | VALID |
| sympy__sympy-18199 | T1-mod_inverse | T1 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| sympy__sympy-18211 | T4-ConditionSet | T4 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| sympy__sympy-18211 | T5-solve_univariate_inequality | T5 | ok (F2P 1/1) | ok (F2P 0/1) | ok (F2P 1/1) | ok | VALID |
| django__django-11149 | pilot-T1-get_permission_codename | T1 | ok (F2P 2/2) | ok (F2P 0/2) | ok (F2P 2/2) | ok | VALID |

**Result: 15 of 18 catalog pairs VALID (last table row is the pilot, counted separately).** The pilot trap django__django-11149 (get_permission_codename argument order) is also VALID in its rr/django-11149 images, replacing the earlier local Python 3.8 check.

All 3 INVALID pairs fail only check (b): the verbatim upstream fix still passes the target tests in the trap version, so the trapped fact does not matter for the task's tests. These are exactly the traps You Wang's second rule ("the changes need to matter for the task") excludes, and the executable check catches them automatically. Every other check passed for all 18 catalog pairs and the pilot.

## Breakdown by trap type and loud/quiet

Script: `.venv/bin/python -I scripts/trap_validation_breakdown.py` (reads outputs/traps/validation.json and the types in traps/catalog/index.json; writes outputs/traps/validation_breakdown.json).

| type | title | built | valid | invalid pairs |
|---|---|---|---|---|
| T1 | argument order | 2 | 2 | - |
| T2 | default value | 2 | 0 | django-16263 (must_group_by), sympy-14248 (_keep_coeff sign) |
| T3 | rename | 4 | 3 | django-14011 (close_all) |
| T4 | move | 3 | 3 | - |
| T5 | exception swap | 1 | 1 | - |
| T6 | return convention | 2 | 2 | - |
| T7 | keyword rename | 2 | 2 | - |
| T8 | flag inversion | 2 | 2 | - |
| all | | 18 | 15 | |

| effect | built | valid |
|---|---|---|
| loud | 9 | 8 |
| quiet | 7 | 5 |
| both (T8: loud by keyword name, quiet by positional value) | 2 | 2 |

- Both default-value traps (T2) are invalid: changing a default that the task's tests never exercise does not matter for the task. The third invalid pair is a loud rename (django-14011, close_all): the upstream fix passes all 17 target tests in the trap.
- The 15 valid pairs pass all 8 checks, including the two extra no-fix sanity checks. In no pair does the upstream fix break a regression test in the placebo.
- Regression tests broken by the upstream fix in the trap (valid pairs): 0 to 208 per pair (208 of 208 in django-15161, the deconstructible path keyword rename); see `p2p_broken_by_upstream_fix_in_trap` in validation_breakdown.json.
- Docker vs the earlier local check (traps/catalog/index.json `local_check`, 17 pairs run, 14 all good): three pairs differ. django-14011 T3 passed locally but is INVALID in Docker; django-15127 T4 was not run locally and is VALID; sympy-14248 T3 failed locally (placebo) and is VALID. The Docker result is the authority.
