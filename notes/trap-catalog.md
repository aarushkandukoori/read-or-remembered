# Memory-trap catalog (E3), Nov 13 2026

**18 complete (task, trap) pairs on 15 SWE-bench Verified tasks** (10 Django pairs on 10 tasks, 8 SymPy pairs on
5 tasks), **all eight trap types T1-T8 covered**: T1 2, T2 2, T3 4, T4 3, T5 1, T6 2, T7 2, T8 2.
Docker validation in the official SWE-bench images: see `notes/trap-validation.md` (verdict per pair in the table
below and in `traps/catalog/index.json`).

The pairs were picked by the builder agent (an LLM session), not sampled: they are a case-study set chosen to cover
every trap type, not an estimate of how often traps apply. How often each type could apply is the static coverage
below.

## What a pair is

`traps/catalog/<instance_id>/<trap_id>/` holds:

| file | what it is |
|---|---|
| `trap.diff` | the trap edit of the repo at the task's base commit (one remembered fact made false, carried through definition, call sites, imports, docs, doctests) |
| `placebo.diff` | an edit of the same kind at the same places that keeps the remembered fact true |
| `tests_trap.diff`, `gold_trap.diff` | the task's test patch and gold fix carried through the trap edit: edit(base + patch) minus edit(base); they apply on top of `trap.diff` |
| `tests_placebo.diff`, `gold_placebo.diff` | the same for the placebo |
| `memory.diff` | the upstream gold fix verbatim: the fix a model would write from memory of the real repo |
| `spec.json` | what was changed, why it makes sense, why it matters for the task, sizes, static checks |
| `report.json` | every rewritten site and warning |
| `local_check.json` | the builder's local (non-Docker) test run; superseded by the Docker validation |

`traps/catalog/index.json` has one row per pair (type, loud/quiet, target, facts, sizes, static checks, Docker
verdict) plus the two entries that could not be built.

## Trap types

| type | trap | placebo | loud or quiet | pairs |
|---|---|---|---|---|
| T1 | argument order: `f(a, b)` becomes `f(b, a)`, every positional call swapped | the same calls pass their arguments by keyword | quiet when the two arguments have the same type | 2 |
| T2 | default value: `p=X` becomes `p=Y`; calls that relied on the default now pass `p=X` | the same `p=X` added, default unchanged | quiet | 2 |
| T3 | rename everywhere, no alias | the same rename plus `old = new` alias; files that imported the old name import both names (fix of Nov 13, see below) | loud (NameError / AttributeError) | 4 |
| T4 | move a module-level function/class to another module, imports updated, no re-export | the same move plus one re-export line in the old module | loud (ImportError) | 3 |
| T5 | exception swap: raises a new class instead of E; excepts and tests around it also accept the new class | the same, but the new class subclasses E | quiet | 1 |
| T6 | return convention: `(a, b)` becomes `(b, a)`; every caller unpacks the new way | the same return and unpacking lines, only parenthesized | quiet | 2 |
| T7 | keyword rename: `f(opt=...)` becomes `f(option=...)`; keyword calls updated | the same rename, old keyword still accepted | loud (TypeError) | 2 |
| T8 | flag inversion: `flag=False` becomes `inverse_flag=True`; calls pass the negated value | the same edit, old flag still accepted (translated) | loud by name, quiet by value | 2 |

Labels in `spec.json` (`loud_or_quiet`): loud 9 (T3, T4, T7), loud by name / quiet by value 2 (T8), quiet 7 (T1, T2,
T5, T6). The label is the type's default. One pair behaves louder than its label: django-15037 (T1) passes a cursor
where a table name is expected, so the memory fix raises an error (the builder's own note in the plan says "loud").

## The 18 pairs

Lines = changed lines of `trap.diff` / `placebo.diff`. Near = how close the trap's edits come to a hunk of the gold
fix (lines; "-" = the trap edits no file the fix edits), design rule 3. Verdict = Docker validation (all eight checks,
`notes/trap-validation.md`).

| # | type | task | trapped fact | lines | files | near | verdict |
|---|---|---|---|---|---|---|---|
| 1 | T1 | sympy-18199 | `mod_inverse(a, m)` becomes `mod_inverse(m, a)` | 62 / 60 | 7 / 7 | 314 | VERDICT_1 |
| 2 | T1 | django-15037 | `get_primary_key_column(cursor, table_name)` becomes `(table_name, cursor)` (base + sqlite + oracle introspection) | 28 / 24 | 9 / 7 | 34 | VERDICT_2 |
| 3 | T2 | sympy-14248 | `_keep_coeff(..., sign=False)` default becomes `sign=True` | 96 / 94 | 13 / 13 | 38 | VERDICT_3 |
| 4 | T2 | django-16263 | `WhereNode.split_having_qualify(..., must_group_by=False)` default becomes `True` | 2 / 0 | 1 / 0 | 188 | VERDICT_4 |
| 5 | T3 | django-13551 | `AbstractBaseUser.get_email_field_name` renamed `get_email_field` | 16 / 17 | 4 / 4 | - | VERDICT_5 |
| 6 | T3 | django-15957 | `Query.clear_limits` renamed `reset_limits` | 6 / 7 | 2 / 2 | - | VERDICT_6 |
| 7 | T3 | django-14011 | `ConnectionHandler.close_all` renamed `close_connections` | 10 / 11 | 5 / 5 | - | VERDICT_7 |
| 8 | T3 | sympy-14248 | `_keep_coeff` renamed `_keep_coefficient` | 134 / 135 | 14 / 14 | 38 | VERDICT_8 |
| 9 | T4 | django-15127 | `get_level_tags` moves from `django.contrib.messages.utils` to `...messages.tags` | 29 / 30 | 4 / 4 | - | VERDICT_9 |
| 10 | T4 | sympy-12419 | `KroneckerDelta` moves from `functions.special.tensor_functions` to `...delta_functions` | 779 / 780 | 18 / 18 | - | VERDICT_10 |
| 11 | T4 | sympy-18211 | `ConditionSet` moves from `sympy.sets.conditionset` to `sympy.sets.fancysets` | 510 / 511 | 7 / 7 | - | VERDICT_11 |
| 12 | T5 | sympy-18211 | `solve_univariate_inequality` raises new `UnsolvableInequality(ValueError)` instead of `NotImplementedError` | 22 / 22 | 3 / 3 | - | VERDICT_12 |
| 13 | T6 | sympy-13877 | `_find_reasonable_pivot` (and `_naive`) return `(value, position, ...)` instead of `(position, value, ...)` | 26 / 22 | 2 / 2 | 462 | VERDICT_13 |
| 14 | T6 | sympy-14248 | `as_coeff_mmul` returns `(matmul, coeff)` instead of `(coeff, matmul)` | 24 / 24 | 5 / 5 | - | VERDICT_14 |
| 15 | T7 | django-16631 | `salted_hmac(..., secret=)` keyword becomes `secret_key=` | 12 / 14 | 2 / 2 | - | VERDICT_15 |
| 16 | T7 | django-15161 | `deconstructible(path=)` keyword becomes `import_path=` | 24 / 26 | 4 / 4 | 69 | VERDICT_16 |
| 17 | T8 | django-15252 | `_create_project_state(with_applied_migrations=False)` becomes `skip_applied_migrations=True` | 10 / 12 | 2 / 2 | 10 | VERDICT_17 |
| 18 | T8 | django-16612 | `get_full_path(force_append_slash=False)` becomes `allow_missing_slash=True` (both getters + helper) | 18 / 28 | 4 / 4 | - | VERDICT_18 |

### Why each makes sense (rule 1) and why it matters for the task (rule 2)

Short versions of `why_it_makes_sense` / `why_it_matters_for_task` in each `spec.json`, plus what the Docker run showed.

1. **T1 mod_inverse (sympy-18199).** Sense: both parameters are integers and neither order is canonical (gmpy2 uses
   `invert(x, m)`, other code `modinv(m, x)`). Matters: the fix adds `mod_inverse(diff, p)` in `nthroot_mod`'s
   prime-power branch; in the remembered order it inverts p modulo diff instead (quiet: wrong roots or ValueError).
2. **T1 get_primary_key_column (django-15037).** Sense: a backend-introspection signature change made on the base
   class and both overrides, every caller updated. Matters: the fix adds
   `connection.introspection.get_primary_key_column(cursor, ref_db_table)` to inspectdb; in the remembered order the
   cursor is passed as the table name and the introspection query fails.
3. **T2 _keep_coeff sign (sympy-14248).** Sense: a private helper's default flipped, every call that relied on it now
   passes `sign=False`. Matters (claimed): the fix adds `_keep_coeff(-c, m)` to three printers. **In fact it cannot
   matter**: `sign` only changes the result when the coefficient is exactly -1 (`elif coeff is S.NegativeOne and not
   sign` in `sympy/core/mul.py`), and the fix only calls `_keep_coeff(-c, m)` when `c < 0`, so the coefficient is
   positive. The Docker run agrees (memory fix passes in the trap).
4. **T2 split_having_qualify must_group_by (django-16263).** Sense: flip an internal default, pass the old value
   where it was relied on. **No existing call relied on the default** (every call passes `must_group_by`), so the
   trap is a one-line default change and the placebo is empty. Matters (claimed): the fix's new
   `split_having_qualify()[1:]` call relies on the default; the Docker run shows the tested path does not depend on it.
5. **T3 get_email_field_name (django-13551).** Sense: a classmethod renamed to a shorter name, no deprecation alias.
   Matters: the fix calls `user.get_email_field_name()` in the token generator (another file than the definition);
   from memory it raises AttributeError and the token tests fail.
6. **T3 clear_limits (django-15957).** Sense: Django renames private `Query` methods freely. Matters: the fix calls
   `queryset.query.clear_limits()`; MiniMax-M2.5 used it in the public runs without reading it (agent-targeted).
7. **T3 close_all (django-14011).** Sense: rename to say what it closes. Matters: the fix adds
   `connections.close_all()` to `ThreadedWSGIServer`; from memory it raises AttributeError in the live-server thread.
8. **T3 _keep_coeff (sympy-14248).** Sense: a private helper renamed to its unabbreviated name. Matters: the fix calls
   `_keep_coeff` in `printing/str.py`, relying on the import already at the top of that file; in the trap that import
   binds only the new name, so the call raises NameError.
9. **T4 get_level_tags (django-15127).** Sense: a helper moved out of a grab-bag utils module into a module named
   after what it returns. Matters: the fix imports it from `django.contrib.messages.utils` in `messages/apps.py`;
   from memory the messages app cannot load (ImportError).
10. **T4 KroneckerDelta (sympy-12419).** Sense: all delta functions in one module (next to DiracDelta, Heaviside);
    `sympy.KroneckerDelta` unchanged. Matters: the fix imports it from `tensor_functions` at the top of `matexpr.py`;
    from memory sympy fails to import.
11. **T4 ConditionSet (sympy-18211).** Sense: next to the other special sets in `fancysets.py`; package-level names
    unchanged. Matters: the fix imports it from `sympy.sets.conditionset` inside `Relational._eval_as_set`; from
    memory that import fails when `as_set()` is called.
12. **T5 solve_univariate_inequality (sympy-18211).** Sense: a dedicated exception class instead of a generic
    NotImplementedError, existing callers updated. Matters: the fix wraps the call in `try/except
    NotImplementedError` and returns a ConditionSet; from memory the except clause does not catch the new class.
13. **T6 _find_reasonable_pivot (sympy-13877).** Sense: the order of the two items is arbitrary. Matters: the fix
    unpacks `pivot_pos, pivot_val, _, _ = _find_reasonable_pivot(...)` in the Bareiss determinant; in the remembered
    order the pivot value is used as an index.
14. **T6 as_coeff_mmul (sympy-14248).** Weak on purpose: the name spells the order (coefficient first), so the trap
    contradicts the name; kept as a low-plausibility contrast case. Matters: the fix does `c, m =
    expr.as_coeff_mmul()` in three printers.
15. **T7 salted_hmac secret (django-16631).** Sense: the keyword renamed after the setting it defaults to
    (`settings.SECRET_KEY`). Matters: the fix passes `secret=secret`; from memory TypeError, session checks fail.
16. **T7 deconstructible path (django-15161).** Sense: its docstring already calls it "the import path". Matters: the
    fix adds six `@deconstructible(path=...)` decorators; from memory `expressions.py` fails at import. Visibility:
    `F` already uses the keyword in the same file (69 lines from a gold hunk).
17. **T8 _create_project_state (django-15252).** Sense: an internal flag restated in the negative. Matters: the fix
    adds `self._create_project_state(with_applied_migrations=False)`; from memory TypeError (unknown keyword); an
    agent that copies the visible keyword but keeps the remembered value gets the opposite meaning. The definition is
    in the same file, 10 lines from a gold hunk (most visible pair).
18. **T8 get_full_path (django-16612).** Sense: the flag restated the other way round on both public getters and the
    shared helper. Matters: the fix adds `request.get_full_path(force_append_slash=True)` to
    `AdminSite.catch_all_view`; from memory TypeError, the admin redirect tests fail.

## Static coverage (how often each type could apply)

From `outputs/traps/coverage.json` (`.venv/bin/python -I scripts/trap_coverage.py`, static only, not validated with
tests): for each type, the number of the 500 SWE-bench Verified tasks whose gold patch's added lines use at least one
symbol that type can be built on ("cross-file" = target defined in another file than the edit; 11 patched files were
skipped because the patch does not apply to the base file).

| type | tasks | share | cross-file |
|---|---|---|---|
| T1 argument order | 48 | 9.6% | 33 |
| T2 default value | 29 | 5.8% | 22 |
| T3 rename | 200 | 40.0% | 151 (161 with a unique name) |
| T4 move | 50 | 10.0% | 50 |
| T5 exception swap | 2 | 0.4% | 2 |
| T6 return convention | 1 | 0.2% | 1 |
| T7 keyword rename | 45 | 9.0% | 30 |
| T8 flag inversion | 11 | 2.2% | 5 |
| any type | 218 | 43.6% | 168 |

Tasks by number of applicable types: 0: 282, 1: 115, 2: 60, 3: 25, 4: 14, 5: 4. By repo, Django has 231 tasks (T3
94, T1 23, T4 20, T7 17) and SymPy 75 (T3 36, T4 11, T1 9, T2 8). T5 and T6 are rare by this static test (2 and 1
tasks), which is why the catalog has only one and two of them.

## How the catalog was finished (Nov 13), exact commands

Run from the project folder.

1. **Inspection.** `traps/catalog` had 20 pair folders (the task description said 19): 18 complete (all eight files)
   and 2 empty, `django__django-11551/T5-get_field` and `sympy__sympy-21596/T4-numer` (the builder had started them
   and was interrupted; neither is in the PLAN of `scripts/build_trap_catalog.py`).
2. **Unit tests.** `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rr_traps.py`: 14 passed before any
   change, 15 passed after the T3 fix below (one new regression test).
3. **Rebuild check** (is the catalog what the current rr_traps code produces?).
   `.venv/bin/python -I scripts/finish_trap_catalog.py rebuild_check` rebuilds all 18 PLAN entries into
   `work/catalog/rebuild/` and compares the seven diff files byte for byte: 17 identical; `django__django-15127/
   T4-get_level_tags` differed in `placebo.diff`. It had been built at 06:36 with an older Move placebo (the new
   module only re-exports the object, 8 changed lines against 29 in the trap); the current default, used by the two
   SymPy T4 pairs, is the same-footprint "alias" placebo. Rebuilt in place:
   `.venv/bin/python -I scripts/build_trap_catalog.py build django__django-15127` (now 29 / 30 lines). Its old
   local check described the old placebo and was renamed `local_check_stale_reexport_placebo.json`.
4. **The two empty entries.** `.venv/bin/python -I scripts/finish_trap_catalog.py extra` tries the natural transform
   for each (`work/catalog/extra_check.json`); rr_traps' safety checks refuse both:
   - `django__django-11551/T5-get_field`: "get_field is also defined at tests/schema/tests.py
     (SchemaTests.test_remove_constraints_capital_letters.get_field); name-based rewriting is unsafe".
   - `sympy__sympy-21596/T4-numer`: "sympy.simplify.radsimp itself uses numer (line 1081); moving it would need an
     import back (import cycle)".
   The empty folders were removed; both are listed under `not_built` in `index.json`.
5. **Builder fix found by the Docker validation (T3 placebo).** In `sympy__sympy-14248/T3-_keep_coeff` the placebo was
   not harmless: the memory fix failed `test_MatrixElement_printing` with `NameError: name '_keep_coeff' is not
   defined` in the placebo too (Docker check `placebo_memory`, F2P 1/2; the builder's local check had shown the same
   1/2). Cause: the placebo renamed `from sympy.core.mul import _keep_coeff` to `... import _keep_coefficient` in every
   file, so the old name was only importable, not bound where the fix uses it. Fix in `rr_traps/transforms.py`
   (`Rename`, placebo only): a rewritten `from m import old` becomes `from m import new, old`. New unit test
   `test_rename_placebo_keeps_imported_old_name`. Rebuilt the four T3 pairs:
   `.venv/bin/python -I scripts/build_trap_catalog.py build django__django-13551/T3-get_email_field_name
   django__django-14011/T3-close_all django__django-15957/T3-clear_limits sympy__sympy-14248/T3-_keep_coeff`.
   The three Django T3 diffs are byte-identical to before (methods are never imported by name); in the SymPy pair only
   `placebo.diff` changed (and the `index` line of `gold_placebo.diff`). Its stale local check was renamed
   `local_check_stale_old_placebo.json`.
6. **Index.** `.venv/bin/python -I scripts/finish_trap_catalog.py index` (rerun after the Docker validation to add
   the verdicts).
7. **Worktrees pruned.** `.venv/bin/python -I scripts/finish_trap_catalog.py prune`.

## Known weak spots

- Two T2 pairs cannot matter for their task (rows 3 and 4): the default never reaches the tested path. They stay in
  the catalog as documented negative cases; they are marked INVALID by the Docker check "memory fix must fail in
  the trap".
- Row 14 (T6 as_coeff_mmul) is implausible by design (rule 1).
- Rule 3 (not visible next to the edit) is weakest for rows 17 (10 lines), 2 (34) and 16 (69).
- The loud/quiet label is the type's default, not measured; row 2 is louder than its label.
