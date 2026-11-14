"""Finish the memory-trap catalog after the interrupted build (helper next to build_trap_catalog.py).

Subcommands (run from the project folder):
  .venv/bin/python -I scripts/finish_trap_catalog.py rebuild_check   # rebuild every PLAN entry into
        work/catalog/rebuild/ and compare its files byte for byte with traps/catalog/ (is the catalog
        what the current rr_traps code produces?) -> work/catalog/rebuild_check.json
  .venv/bin/python -I scripts/finish_trap_catalog.py extra           # try the two entries the builder
        started but never finished (empty folders) -> work/catalog/extra_check.json
  .venv/bin/python -I scripts/finish_trap_catalog.py index           # traps/catalog/index.json
        (same rows as build_trap_catalog.py index, plus the Docker verdict when
        outputs/traps/validation.json exists, plus the entries that could not be built)
  .venv/bin/python -I scripts/finish_trap_catalog.py prune           # remove every git worktree in work/catalog
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # -I leaves the project folder off sys.path
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from rr_traps import builder  # noqa: E402
from rr_traps.transforms import TrapError, make  # noqa: E402
import build_trap_catalog as btc  # noqa: E402  (the PLAN of the 18 built entries lives there)

# files every catalog entry must have (report.json and local_check.json are extras)
NEEDED = ['spec.json', 'trap.diff', 'placebo.diff', 'gold_trap.diff', 'tests_trap.diff', 'gold_placebo.diff',
          'tests_placebo.diff', 'memory.diff']
DIFFS = NEEDED[1:]

# The two entries the builder had started (their folders existed but were empty). The folder names give
# the target; the transform settings below are the natural ones for that target. Both are refused by
# rr_traps' safety checks (see `extra`), so they stay out of the catalog.
EXTRA = [
    dict(instance_id='django__django-11551', trap_id='T5-get_field',
         transform=dict(type='T5', defs=[['django/db/models/options.py', 'Options.get_field']],
                        init=dict(old_exc='FieldDoesNotExist', new_exc='UnknownFieldError', new_base='LookupError',
                                  exc_file='django/core/exceptions.py'))),
    dict(instance_id='sympy__sympy-21596', trap_id='T4-numer',
         transform=dict(type='T4', defs=[['sympy/simplify/radsimp.py', 'numer']],
                        init=dict(new_file='sympy/simplify/fractions.py', placebo_style='reexport'))),
]
EXTRA_OUT = os.path.join(builder.WORK, 'extra_check.json')
REBUILD_DIR = os.path.join(builder.WORK, 'rebuild')
REBUILD_OUT = os.path.join(builder.WORK, 'rebuild_check.json')
VALIDATION = os.path.join(ROOT, 'outputs/traps/validation.json')


def read_or_none(path):
    return open(path, encoding='utf-8', errors='surrogateescape').read() if os.path.exists(path) else None


def cmd_rebuild_check():
    """Rebuild each PLAN entry into work/catalog/rebuild/ and compare with traps/catalog/."""
    rows = []
    for e in btc.plan_entries([]):
        key = f"{e['instance_id']}/{e['trap_id']}"
        try:
            builder.build(e, out_root=REBUILD_DIR, keep_worktree=True, log=lambda *a: None)
        except Exception as ex:  # a failed rebuild is a result too
            rows.append(dict(entry=key, rebuilt=False, error=f'{type(ex).__name__}: {str(ex)[:300]}'))
            print(rows[-1], flush=True)
            continue
        old_dir = os.path.join(builder.OUT, e['instance_id'], e['trap_id'])
        new_dir = os.path.join(REBUILD_DIR, e['instance_id'], e['trap_id'])
        differ = [f for f in DIFFS if read_or_none(os.path.join(old_dir, f)) != read_or_none(os.path.join(new_dir, f))]
        rows.append(dict(entry=key, rebuilt=True, identical=not differ, files_that_differ=differ))
        print(rows[-1], flush=True)
    out = dict(command='.venv/bin/python -I scripts/finish_trap_catalog.py rebuild_check',
               n_entries=len(rows), n_identical=sum(1 for r in rows if r.get('identical')), rows=rows)
    json.dump(out, open(REBUILD_OUT, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


def cmd_extra():
    """Try the trap edit of the two unfinished entries; record why rr_traps refuses them."""
    rows = []
    for e in EXTRA:
        t = builder.task(e['instance_id'])
        wt = builder.worktree(t)
        builder.reset(wt, t['base_commit'])
        try:
            rep = make(e['transform']).apply(wt, 'trap')
            rows.append(dict(entry=f"{e['instance_id']}/{e['trap_id']}", built=True, files=sorted(rep.files)))
        except TrapError as ex:
            rows.append(dict(entry=f"{e['instance_id']}/{e['trap_id']}", built=False,
                             refused_by_rr_traps=str(ex), transform=e['transform']))
        builder.reset(wt, t['base_commit'])
        print(rows[-1], flush=True)
    json.dump(dict(command='.venv/bin/python -I scripts/finish_trap_catalog.py extra', rows=rows),
              open(EXTRA_OUT, 'w'), indent=1)


def cmd_index():
    """traps/catalog/index.json: one row per complete entry (+ Docker verdict if validated)."""
    btc.cmd_index()  # the builder's own index (rows from spec.json and local_check.json)
    path = os.path.join(builder.OUT, 'index.json')
    idx = json.load(open(path))
    incomplete = []
    for iid in sorted(os.listdir(builder.OUT)):
        d = os.path.join(builder.OUT, iid)
        if not os.path.isdir(d):
            continue
        for tid in sorted(os.listdir(d)):
            missing = [f for f in NEEDED if not os.path.exists(os.path.join(d, tid, f))]
            if missing:
                incomplete.append(dict(entry=f'{iid}/{tid}', missing=missing))
    idx['incomplete_entries'] = incomplete
    if os.path.exists(EXTRA_OUT):
        idx['not_built'] = [dict(entry=r['entry'], reason=r.get('refused_by_rr_traps'))
                            for r in json.load(open(EXTRA_OUT))['rows'] if not r['built']]
    if os.path.exists(VALIDATION):
        val = json.load(open(VALIDATION))
        verdicts = {f"{p['instance_id']}/{p['trap_id']}": p for p in val['pairs']}
        for row in idx['entries']:
            p = verdicts.get(f"{row['instance_id']}/{row['trap_id']}")
            row['docker_verdict'] = p['verdict'] if p else None
            row['docker_failed_checks'] = p['failed_checks'] if p else None
        idx['n_valid_docker'] = sum(1 for r in idx['entries'] if r.get('docker_verdict') == 'VALID')
        idx['description'] += (' docker_verdict = VALID / INVALID from scripts/validate_trap_catalog.py (official '
                               'SWE-bench images, see outputs/traps/validation.json).')
    json.dump(idx, open(path, 'w'), indent=1)
    print(json.dumps({k: v for k, v in idx.items() if k != 'entries'}, indent=1))


def cmd_prune():
    """Remove every git worktree under work/catalog (the build/check worktrees and the rebuild ones)."""
    for iid in sorted(os.listdir(builder.WORK)):
        if os.path.exists(os.path.join(builder.WORK, iid, 'wt')):
            builder.remove_worktree(builder.task(iid))
            print('pruned', iid, flush=True)


if __name__ == '__main__':
    {'rebuild_check': cmd_rebuild_check, 'extra': cmd_extra, 'index': cmd_index, 'prune': cmd_prune}[sys.argv[1]]()
