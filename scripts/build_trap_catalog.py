"""Build the memory-trap catalog on SWE-bench Verified (static build + optional local test check).

The PLAN below lists every (task, trap) pair: which fact is trapped, how, and why it makes sense and
matters for the task. For each entry rr_traps.builder writes traps/catalog/<instance_id>/<trap_id>/
(trap.diff, placebo.diff, gold_*/tests_* carried diffs, memory.diff, spec.json, report.json).

Usage (from the project folder):
  .venv/bin/python -I scripts/build_trap_catalog.py build [ID ...]   # ID = instance_id or instance_id/trap_id
  .venv/bin/python -I scripts/build_trap_catalog.py check [ID ...]   # run the task tests locally (no Docker)
  .venv/bin/python -I scripts/build_trap_catalog.py index            # traps/catalog/index.json
  .venv/bin/python -I scripts/build_trap_catalog.py prune            # remove the git worktrees
Excluded tasks (used elsewhere): django__django-16100, django__django-15563, sympy__sympy-19040,
sympy__sympy-15345 (other session) and the pilot tasks django__django-11149, sympy__sympy-13615.
"""
import concurrent.futures
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # -I leaves the project folder off sys.path
from rr_traps import builder, localcheck  # noqa: E402

EXCLUDED = {'django__django-16100', 'django__django-15563', 'sympy__sympy-19040', 'sympy__sympy-15345',
            'django__django-11149', 'sympy__sympy-13615'}


def E(iid, trap_id, transform, sense, matters, **kw):
    return dict(instance_id=iid, trap_id=trap_id, transform=transform, why_it_makes_sense=sense,
                why_it_matters_for_task=matters, **kw)


PLAN = [
    # ------------------------------------------------------------------ T1 argument order
    E('sympy__sympy-18199', 'T1-mod_inverse',
      dict(type='T1', defs=[['sympy/core/numbers.py', 'mod_inverse']]),
      'Both parameters are integers and neither order is canonical (gmpy2 uses invert(x, m), other code '
      'modinv(m, x)); reordering a helper\'s parameters and every call is a routine signature change.',
      'The gold patch adds root = (root - (rootn - a) * mod_inverse(diff, p)) % ppow in nthroot_mod\'s '
      'prime-power branch (test_solve_modular). Written in the remembered order, the call computes the '
      'inverse of p modulo diff instead: quiet, wrong roots or a ValueError.',
      notes='quiet trap; mod_inverse is also exported as sympy.mod_inverse'),
    E('django__django-15037', 'T1-get_primary_key_column',
      dict(type='T1', defs=[['django/db/backends/base/introspection.py',
                             'BaseDatabaseIntrospection.get_primary_key_column'],
                            ['django/db/backends/sqlite3/introspection.py',
                             'DatabaseIntrospection.get_primary_key_column'],
                            ['django/db/backends/oracle/introspection.py',
                             'DatabaseIntrospection.get_primary_key_column']]),
      'A backend-introspection signature change applied to the base class and both backend overrides, '
      'with every caller updated; (table_name, cursor) is as natural as (cursor, table_name).',
      'The gold patch adds connection.introspection.get_primary_key_column(cursor, ref_db_table) in inspectdb '
      '(test_foreign_key_to_field). In the remembered order the cursor is passed as the table name: the '
      'introspection query fails.',
      notes='loud (the two arguments have different types); definition lives in another file'),
    # ------------------------------------------------------------------ T2 default value
    E('sympy__sympy-14248', 'T2-_keep_coeff-sign',
      dict(type='T2', defs=[['sympy/core/mul.py', '_keep_coeff']], init=dict(param='sign', new_default='True')),
      'A private helper\'s keyword default flipped (sign=False -> sign=True), with every call that relied on '
      'the old default now passing sign=False explicitly: behavior of the repo is unchanged.',
      'The gold patch adds expr = _keep_coeff(-c, m) to three printers and relies on sign=False. With the new '
      'default the minus sign is kept inside the coefficient, so the printed matrix expressions differ from '
      'what test_MatrixSymbol_printing expects.',
      notes='quiet trap'),
    E('django__django-16263', 'T2-split_having_qualify-must_group_by',
      dict(type='T2', defs=[['django/db/models/sql/where.py', 'WhereNode.split_having_qualify']],
           init=dict(param='must_group_by', new_default='True')),
      'Flipping an internal keyword default and passing the old value explicitly at every call keeps the '
      'ORM\'s behavior; projects make such changes when the common case changes.',
      'The gold patch adds any(self.where.split_having_qualify()[1:]) in get_aggregation and relies on '
      'must_group_by=False; with the new default the split treats filters differently.',
      notes='quiet; needs a test run to know whether the default matters on the tested path'),
    # ------------------------------------------------------------------ T3 rename
    E('django__django-13551', 'T3-get_email_field_name',
      dict(type='T3', defs=[['django/contrib/auth/base_user.py', 'AbstractBaseUser.get_email_field_name']],
           init=dict(new_name='get_email_field')),
      'A classmethod that returns the email field\'s name renamed to a shorter name, everywhere, without a '
      'deprecation alias.',
      'The gold patch adds email_field = user.get_email_field_name() to PasswordResetTokenGenerator. Written '
      'from memory the call raises AttributeError, so every token test fails.',
      notes='loud; definition in another file (base_user.py) than the edit (tokens.py)'),
    E('django__django-15957', 'T3-clear_limits',
      dict(type='T3', defs=[['django/db/models/sql/query.py', 'Query.clear_limits']],
           init=dict(new_name='reset_limits')),
      'Internal Query API rename (clear_limits -> reset_limits); Django renames private Query methods freely.',
      'The gold patch calls queryset.query.clear_limits() in the new prefetch window filter (memory tier for '
      'MiniMax-M2.5 in the public runs: used without being read). Written from memory it raises AttributeError.',
      notes='loud; agent-targeted (minimax-m2.5 no_use + strong)'),
    E('django__django-14011', 'T3-close_all',
      dict(type='T3', defs=[['django/db/utils.py', 'ConnectionHandler.close_all']],
           init=dict(new_name='close_connections')),
      'Rename of a handler method to say what it closes; every caller (servers, test framework) updated.',
      'The gold patch adds connections.close_all() to ThreadedWSGIServer; from memory this raises '
      'AttributeError in the live-server thread and the live-server tests fail.',
      notes='loud; Kimi and MiniMax relied on close_all without reading its definition (unseen, not memory tier)'),
    E('sympy__sympy-14248', 'T3-_keep_coeff',
      dict(type='T3', defs=[['sympy/core/mul.py', '_keep_coeff']], init=dict(new_name='_keep_coefficient')),
      'A private helper renamed to its unabbreviated name everywhere, no alias.',
      'The gold patch imports and calls _keep_coeff in three printers; from memory the import fails.',
      notes='loud'),
    # ------------------------------------------------------------------ T4 move
    E('django__django-15127', 'T4-get_level_tags',
      dict(type='T4', defs=[['django/contrib/messages/utils.py', 'get_level_tags']],
           init=dict(new_file='django/contrib/messages/tags.py',
                     docstring='Message level tags.')),
      'A small helper moved from a grab-bag utils module into a module named after what it returns; all '
      'imports updated, no re-export (as in a major-version cleanup).',
      'The gold patch adds from django.contrib.messages.utils import get_level_tags to messages/apps.py. From '
      'memory this import fails, so the messages app cannot load.',
      notes='loud (ImportError at app loading)'),
    E('sympy__sympy-12419', 'T4-KroneckerDelta',
      dict(type='T4', defs=[['sympy/functions/special/tensor_functions.py', 'KroneckerDelta']],
           init=dict(new_file='sympy/functions/special/delta_functions.py')),
      'KroneckerDelta moved next to DiracDelta and Heaviside in delta_functions.py ("all delta functions in '
      'one module"); the package-level name sympy.KroneckerDelta is unchanged.',
      'The gold patch adds from sympy.functions.special.tensor_functions import KroneckerDelta at the top of '
      'matexpr.py. From memory the import fails while sympy is being imported.',
      notes='loud (sympy fails to import); DeepSeek relied on a different import (Relational) here'),
    E('sympy__sympy-18211', 'T4-ConditionSet',
      dict(type='T4', defs=[['sympy/sets/conditionset.py', 'ConditionSet']],
           init=dict(new_file='sympy/sets/fancysets.py')),
      'ConditionSet moved into fancysets.py next to the other special sets (ImageSet, Range, ComplexRegion); '
      'the package-level names sympy.ConditionSet and sympy.sets.ConditionSet are unchanged.',
      'The gold patch adds from sympy.sets.conditionset import ConditionSet inside Relational._eval_as_set; '
      'from memory that import fails when as_set() is called (test_issue_18188).',
      notes='loud (ImportError at call time, not at import time)'),
    # ------------------------------------------------------------------ T5 exception swap
    E('sympy__sympy-18211', 'T5-solve_univariate_inequality',
      dict(type='T5', defs=[['sympy/solvers/inequalities.py', 'solve_univariate_inequality']],
           init=dict(old_exc='NotImplementedError', new_exc='UnsolvableInequality', new_base='ValueError',
                     doc='Raised by solve_univariate_inequality() when it cannot solve the inequality.')),
      'Replacing a generic NotImplementedError with a dedicated exception class is a common clean-up; the '
      'callers that caught NotImplementedError were updated to catch the new class too.',
      'The gold patch wraps solve_univariate_inequality(...) in try/except NotImplementedError and returns a '
      'ConditionSet (test_issue_18188). From memory the except clause does not catch the new exception.',
      notes='quiet'),
    # ------------------------------------------------------------------ T6 return convention
    E('sympy__sympy-13877', 'T6-_find_reasonable_pivot',
      dict(type='T6', defs=[['sympy/matrices/matrices.py', '_find_reasonable_pivot'],
                            ['sympy/matrices/matrices.py', '_find_reasonable_pivot_naive']],
           init=dict(size=4, i=0, j=1)),
      'Both private pivot helpers return (value, position, ...) instead of (position, value, ...); the order '
      'of the two items is arbitrary and every caller unpacks it the new way.',
      'The gold patch adds pivot_pos, pivot_val, _, _ = _find_reasonable_pivot(...) to the Bareiss determinant. '
      'Unpacked in the remembered order, the pivot value is used as an index.',
      notes='quiet; docstring prose also updated (prose edits listed in spec.transform.init)'),
    E('sympy__sympy-14248', 'T6-as_coeff_mmul',
      dict(type='T6', defs=[['sympy/matrices/expressions/matexpr.py', 'MatrixExpr.as_coeff_mmul'],
                            ['sympy/matrices/expressions/matmul.py', 'MatMul.as_coeff_mmul']],
           init=dict(size=2, i=0, j=1)),
      'Weak: the method name as_coeff_mmul spells the order (coefficient first), so returning (matmul, '
      'coefficient) contradicts the name. Kept as a plausibility-contrast case.',
      'The gold patch adds c, m = expr.as_coeff_mmul() to three printers and tests the sign of c; unpacked in '
      'the remembered order c is the matrix product.',
      notes='quiet; low plausibility (rule 1) by design'),
    # ------------------------------------------------------------------ T7 keyword rename
    E('django__django-16631', 'T7-salted_hmac-secret',
      dict(type='T7', defs=[['django/utils/crypto.py', 'salted_hmac']],
           init=dict(param='secret', new_param='secret_key')),
      'Keyword renamed to match the setting it defaults to (settings.SECRET_KEY); every caller updated.',
      'The gold patch passes secret=secret to salted_hmac in AbstractBaseUser._get_session_auth_hash; from '
      'memory this raises TypeError and session verification fails.',
      notes='loud'),
    E('django__django-15161', 'T7-deconstructible-path',
      dict(type='T7', defs=[['django/utils/deconstruct.py', 'deconstructible']],
           init=dict(param='path', new_param='import_path')),
      'The decorator\'s docstring already calls it "the import path"; the keyword is renamed to say so.',
      'The gold patch adds @deconstructible(path=\'django.db.models.Func\') and five more decorators; from '
      'memory each raises TypeError when expressions.py is imported.',
      notes='loud (import-time error); check visibility: F already uses the keyword in the same file'),
    # ------------------------------------------------------------------ T8 flag inversion
    E('django__django-15252', 'T8-_create_project_state',
      dict(type='T8', defs=[['django/db/migrations/executor.py', 'MigrationExecutor._create_project_state']],
           init=dict(param='with_applied_migrations', new_param='skip_applied_migrations')),
      'An internal flag restated in the negative (skip_applied_migrations=True instead of '
      'with_applied_migrations=False); callers pass the negated value.',
      'The gold patch adds return self._create_project_state(with_applied_migrations=False). From memory the '
      'keyword is unknown (TypeError); an agent that copies the visible keyword but keeps the remembered value '
      'gets the opposite meaning.',
      notes='loud by name, quiet by value; definition in the same file'),
    E('django__django-16612', 'T8-get_full_path',
      dict(type='T8', defs=[['django/http/request.py', 'HttpRequest.get_full_path'],
                            ['django/http/request.py', 'HttpRequest.get_full_path_info'],
                            ['django/http/request.py', 'HttpRequest._get_full_path']],
           init=dict(param='force_append_slash', new_param='allow_missing_slash')),
      'The flag is restated the other way round (allow_missing_slash=True instead of force_append_slash=False) '
      'on both public getters and the shared helper; callers pass the negated value.',
      'The gold patch adds request.get_full_path(force_append_slash=True) to AdminSite.catch_all_view; from '
      'memory this raises TypeError and the admin redirect tests fail.',
      notes='loud by name'),
]

# docstrings that spell out a return order are prose: edit them explicitly (trap mode only)
PROSE = {
    ('sympy__sympy-13877', 'T6-_find_reasonable_pivot'): [],
}


def plan_entries(ids):
    out = []
    for e in PLAN:
        if e['instance_id'] in EXCLUDED:
            raise SystemExit(f"{e['instance_id']} is excluded")
        key = f"{e['instance_id']}/{e['trap_id']}"
        if not ids or e['instance_id'] in ids or key in ids:
            e = dict(e)
            e['transform'] = dict(e['transform'], init=dict(e['transform'].get('init', {})))
            prose = PROSE.get((e['instance_id'], e['trap_id']))
            if prose:
                e['transform']['init']['prose_edits'] = prose
            out.append(e)
    return out


def cmd_build(ids):
    done = []
    for e in plan_entries(ids):
        try:
            spec = builder.build(e)
            done.append((e['instance_id'], e['trap_id'], 'ok', spec['problems']))
        except Exception as ex:  # report and go on with the next entry
            done.append((e['instance_id'], e['trap_id'], f'FAILED: {type(ex).__name__}: {str(ex)[:300]}', []))
            print(f"{e['instance_id']} {e['trap_id']}: FAILED {type(ex).__name__}: {str(ex)[:500]}", flush=True)
    print(json.dumps(done, indent=1))


def cmd_check(ids, workers=4):
    by_task = {}
    for e in plan_entries(ids):
        d = os.path.join(builder.OUT, e['instance_id'], e['trap_id'])
        if os.path.exists(os.path.join(d, 'spec.json')):
            by_task.setdefault(e['instance_id'], []).append(d)

    def run_task(dirs):  # entries of one task share a worktree: run them one after another
        out = []
        for d in dirs:
            t0 = time.time()
            try:
                res = localcheck.check_entry(d)
                out.append((d, res['summary'], round(time.time() - t0)))
            except Exception as ex:
                out.append((d, f'FAILED {type(ex).__name__}: {str(ex)[:300]}', round(time.time() - t0)))
            print(out[-1], flush=True)
        return out
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for res in pool.map(run_task, by_task.values()):
            pass


def cmd_index():
    rows = []
    for iid in sorted(os.listdir(builder.OUT)):
        p = os.path.join(builder.OUT, iid)
        if not os.path.isdir(p):
            continue
        for tid in sorted(os.listdir(p)):
            sp = os.path.join(p, tid, 'spec.json')
            if not os.path.exists(sp):
                continue
            s = json.load(open(sp))
            lc = os.path.join(p, tid, 'local_check.json')
            local = json.load(open(lc))['summary'] if os.path.exists(lc) else None
            rows.append(dict(instance_id=iid, trap_id=tid, path=os.path.relpath(os.path.join(p, tid), ROOT),
                             type=s['type'], type_title=s['type_title'], loud_or_quiet=s['loud_or_quiet'],
                             target=s['target_qualname'], target_file=s['target']['file'],
                             original_fact=s['original_fact'], trapped_fact=s['trapped_fact'],
                             lines_changed_trap=s['lines_changed_trap'],
                             lines_changed_placebo=s['lines_changed_placebo'],
                             files_changed_trap=len(s['files_changed']),
                             files_changed_placebo=len(s['files_changed_placebo']),
                             memory_applies_trap=s['checks']['trap']['memory_applies'],
                             memory_applies_placebo=s['checks']['placebo']['memory_applies'],
                             gold_lines_changed_by_trap=s['gold_lines_changed_by_trap'],
                             nearest_trap_line_to_gold_hunk=s['visibility']['nearest_trap_line_to_gold_hunk'],
                             problems=s['problems'], local_check=local))
    out = dict(description='Memory-trap catalog on SWE-bench Verified (one row per task x trap). Built by '
                           'scripts/build_trap_catalog.py with rr_traps; local_check = local (non-Docker) test '
                           'run summary, null if not run. The Docker validator is the authority.',
               n_entries=len(rows), n_tasks=len({r['instance_id'] for r in rows}),
               by_type={t: sum(r['type'] == t for r in rows) for t in sorted({r['type'] for r in rows})},
               entries=rows)
    json.dump(out, open(os.path.join(builder.OUT, 'index.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'entries'}, indent=1))


def cmd_prune():
    for iid in sorted(os.listdir(builder.WORK)):
        if iid.startswith(('django__', 'sympy__', 'sphinx', 'astropy', 'scikit', 'matplotlib', 'pydata', 'pytest')):
            builder.remove_worktree(builder.task(iid))
            print('pruned', iid)


if __name__ == '__main__':
    cmd, ids = sys.argv[1], sys.argv[2:]
    {'build': lambda: cmd_build(ids), 'check': lambda: cmd_check(ids), 'index': cmd_index,
     'prune': cmd_prune}[cmd]()
