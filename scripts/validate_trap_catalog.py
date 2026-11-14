"""Validate the memory-trap catalog in the official SWE-bench Docker images (no new images are built).

For every complete catalog pair traps/catalog/<instance_id>/<trap_id>/ and every check below we start a
FRESH container from swebench/sweb.eval.x86_64.<instance_id with __ -> _1776_>:latest
(--platform linux/amd64 --network none), apply the listed diffs at /testbed with `git apply -v`, run the
task's official test command and grade FAIL_TO_PASS / PASS_TO_PASS like the SWE-bench harness.

Where the official pieces come from (swebench 4.1.0, the last release that ships the per-repo specs):
  test command   MAP_REPO_VERSION_TO_SPECS[repo][version]['test_cmd'] + get_test_directives(instance),
                 run after the spec's eval_commands, between the harness's start/end markers
  parsing        MAP_REPO_TO_PARSER[repo] (parse_log_django / parse_log_sympy)
  grading        get_eval_tests_report: a test passes if its status is PASSED or XFAIL; a test that is
                 missing from the log or FAILED/ERROR fails
Two differences from the harness's eval.sh, both on purpose:
  * no `git checkout <base> <test files>` before the test patch: it would undo trap edits in test files;
    every container starts clean from the image, so it is not needed
  * the spec's install step (`python -m pip install -e .`) is skipped: the package is already installed in
    editable mode in the image (the script prints the imported package path to show it), and containers
    run with --network none

Checks for one pair (the two original_* checks are shared by all pairs of a task):
  original_nofix   test_patch                                expect: no FAIL_TO_PASS test passes
  original_gold    test_patch + gold patch                   expect: all F2P and all P2P pass
  trap_nofix       trap + tests_trap                         expect: no F2P test passes
  trap_gold        trap + tests_trap + gold_trap             expect: all F2P and P2P pass (still solvable)
  trap_memory      trap + tests_trap + memory (upstream fix) expect: at least one F2P test FAILS (trap matters)
  placebo_nofix    placebo + tests_placebo                   expect: no F2P test passes
  placebo_gold     placebo + tests_placebo + gold_placebo    expect: all F2P and P2P pass
  placebo_memory   placebo + tests_placebo + memory          expect: all F2P and P2P pass (placebo harmless)
A pair is VALID only if all eight checks meet their expectation.

Usage (from the project folder; one container at a time):
  .venv/bin/python -I scripts/validate_trap_catalog.py run [instance_id/trap_id ...]
  .venv/bin/python -I scripts/validate_trap_catalog.py pilots        # the two pilot traps, in their rr/ images
  .venv/bin/python -I scripts/validate_trap_catalog.py report        # outputs/traps/validation.{json,csv}
Results of each check are cached in work/catalog/validation/cache (keyed by image id + diffs + command), so a
rerun skips checks that are done. Full logs: work/catalog/validation/logs.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from swebench.harness.constants import END_TEST_OUTPUT, FAIL_TO_PASS, MAP_REPO_VERSION_TO_SPECS, PASS_TO_PASS, \
    START_TEST_OUTPUT
from swebench.harness.grading import get_eval_tests_report
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(ROOT, 'traps/catalog')
PARQUET = os.path.join(ROOT, 'data/meta/swebv/data/test-00000-of-00001.parquet')
VAL = os.path.join(ROOT, 'work/catalog/validation')
CACHE, LOGS, TMP = (os.path.join(VAL, d) for d in ('cache', 'logs', 'tmp'))
START_IMAGES = os.path.join(VAL, 'images_at_start.txt')   # written before the first pull
PULL_LOG = os.path.join(VAL, 'pulls.jsonl')
OUT_JSON = os.path.join(ROOT, 'outputs/traps/validation.json')
OUT_CSV = os.path.join(ROOT, 'outputs/traps/validation.csv')
PILOT_WORK = os.path.join(ROOT, 'work')

GB = 1e9
MIN_FREE = 15 * GB        # stop pulling below this much free disk (Mac and colima VM)
PULL_BUDGET = 20 * GB     # at most this much new image data
TIMEOUT = 3600            # seconds per test run
NEEDED = ['spec.json', 'trap.diff', 'placebo.diff', 'gold_trap.diff', 'tests_trap.diff', 'gold_placebo.diff',
          'tests_placebo.diff', 'memory.diff']
PKG = {'django/django': 'django', 'sympy/sympy': 'sympy'}

# check name -> (version diff, tests diff, fix diff, expectation); 'TEST'/'GOLD' = the task's own patches
CHECKS = {
    'original_nofix': (None, 'TEST', None, 'no_f2p_passes'),
    'original_gold': (None, 'TEST', 'GOLD', 'all_pass'),
    'trap_nofix': ('trap.diff', 'tests_trap.diff', None, 'no_f2p_passes'),
    'trap_gold': ('trap.diff', 'tests_trap.diff', 'gold_trap.diff', 'all_pass'),
    'trap_memory': ('trap.diff', 'tests_trap.diff', 'memory.diff', 'some_f2p_fails'),
    'placebo_nofix': ('placebo.diff', 'tests_placebo.diff', None, 'no_f2p_passes'),
    'placebo_gold': ('placebo.diff', 'tests_placebo.diff', 'gold_placebo.diff', 'all_pass'),
    'placebo_memory': ('placebo.diff', 'tests_placebo.diff', 'memory.diff', 'all_pass'),
}
EXPECT_TEXT = {'no_f2p_passes': 'no F2P test passes', 'all_pass': 'all F2P and P2P pass',
               'some_f2p_fails': 'at least one F2P test fails'}


def sh(*cmd, check=True, timeout=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f'{" ".join(cmd[:6])} failed: {r.stderr[-1000:]}')
    return r


_tasks = None


def task(iid):
    """One SWE-bench Verified row as a dict (FAIL_TO_PASS / PASS_TO_PASS as lists)."""
    global _tasks
    if _tasks is None:
        import pyarrow.parquet as pq
        cols = ['instance_id', 'repo', 'version', 'base_commit', 'patch', 'test_patch', 'FAIL_TO_PASS',
                'PASS_TO_PASS', 'environment_setup_commit']
        _tasks = pq.read_table(PARQUET, columns=cols).to_pandas().set_index('instance_id')
    row = dict(_tasks.loc[iid], instance_id=iid)
    row['F2P'], row['P2P'] = json.loads(row['FAIL_TO_PASS']), json.loads(row['PASS_TO_PASS'])
    return row


def official_image(iid):
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()


def test_command(t):
    """The official test command for this task: spec test_cmd + test directives from the test patch."""
    spec = MAP_REPO_VERSION_TO_SPECS[t['repo']][t['version']]
    return spec, ' '.join([spec['test_cmd'], *get_test_directives(t)])


def eval_script(t, diff_names):
    """Bash script run inside the container (same shape as the harness's eval.sh, see the module doc)."""
    spec, cmd = test_command(t)
    lines = ['#!/bin/bash', 'set -uxo pipefail', 'source /opt/miniconda3/bin/activate', 'conda activate testbed',
             'cd /testbed', *spec.get('eval_commands', []), 'git config --global --add safe.directory /testbed',
             'cd /testbed', 'git status --porcelain | head -20']
    for name in diff_names:
        lines.append(f"git apply -v /tmp/rr/{name} || {{ echo '>>>>> Patch Apply Failed: {name}'; exit 97; }}")
    pkg = PKG[t['repo']]
    lines += [f"python -c 'import {pkg}; print(\"PACKAGE_FILE\", {pkg}.__file__)'",
              f": '{START_TEST_OUTPUT}'", cmd, f": '{END_TEST_OUTPUT}'"]
    return '\n'.join(lines) + '\n'


def image_id(image):
    r = sh('docker', 'image', 'inspect', '-f', '{{.Id}}', image, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def grade(t, log_path):
    """Parse a test log like swebench.harness.grading.get_logs_eval + get_eval_tests_report."""
    content = open(log_path, encoding='utf-8', errors='replace').read()
    if '>>>>> Patch Apply Failed' in content:
        bad = re.search(r'>>>>> Patch Apply Failed: (\S+)', content)
        return dict(status='patch_failed', patch=bad.group(1) if bad else None)
    if START_TEST_OUTPUT not in content or END_TEST_OUTPUT not in content:
        return dict(status='no_test_output')
    test_part = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    parser = MAP_REPO_TO_PARSER[t['repo']]
    sm = parser(test_part, None) or parser(content, None)  # the harness falls back to the whole log
    rep = get_eval_tests_report(sm, {FAIL_TO_PASS: t['F2P'], PASS_TO_PASS: t['P2P']})
    pkg = re.search(r'PACKAGE_FILE (\S+)', content)
    return dict(status='ran', n_parsed=len(sm),
                f2p_passed=len(rep[FAIL_TO_PASS]['success']), f2p_failed=rep[FAIL_TO_PASS]['failure'],
                p2p_passed=len(rep[PASS_TO_PASS]['success']), p2p_failed=rep[PASS_TO_PASS]['failure'],
                package_file=pkg.group(1) if pkg else None)


def meets(expect, res, t):
    """Does a graded run meet the check's expectation?"""
    if res['status'] != 'ran':
        return False
    if expect == 'no_f2p_passes':
        return res['f2p_passed'] == 0
    if expect == 'some_f2p_fails':
        return len(res['f2p_failed']) > 0
    return not res['f2p_failed'] and not res['p2p_failed']  # all_pass


def cache_key(t, image, diffs):
    """What a check depends on: image name, test command and the non-empty diffs it applies."""
    used = [(n, d) for n, d in diffs if d and d.strip()]  # an empty diff (nothing to apply) is skipped
    return used, hashlib.sha1(json.dumps([image, test_command(t)[1], used]).encode('utf-8', 'surrogateescape')
                              ).hexdigest()


def cached(t, image, diffs, log_rel):
    """The cached result of this check if it was already run with exactly these inputs, else None."""
    cache_file = os.path.join(CACHE, log_rel + '.json')
    if os.path.exists(cache_file):
        old = json.load(open(cache_file))
        if old.get('key') == cache_key(t, image, diffs)[1]:
            return old
    return None


def run_check(t, image, label, diffs, expect, log_rel):
    """diffs: [(file name, text)] applied in order. Returns the result dict (cached by content key)."""
    _, cmd = test_command(t)
    used, key = cache_key(t, image, diffs)
    cache_file = os.path.join(CACHE, log_rel + '.json')
    old = cached(t, image, diffs, log_rel)
    if old:
        return old
    os.makedirs(TMP, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=TMP)
    names = []
    for k, (n, d) in enumerate(used):
        name = f'{k:02d}_{n}'
        with open(os.path.join(tmp, name), 'w', encoding='utf-8', errors='surrogateescape', newline='') as f:
            f.write(d if d.endswith('\n') else d + '\n')
        names.append(name)
    with open(os.path.join(tmp, 'eval.sh'), 'w') as f:
        f.write(eval_script(t, names))
    log_path = os.path.join(LOGS, log_rel + '.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cname = 'rrval-' + hashlib.sha1(log_rel.encode()).hexdigest()[:10]
    sh('docker', 'rm', '-f', cname, check=False)
    sh('docker', 'run', '-d', '--name', cname, '--platform', 'linux/amd64', '--network', 'none', '--memory', '4g',
       image, 'sleep', 'infinity')
    t0 = time.time()
    timed_out = False
    try:
        sh('docker', 'cp', tmp + '/.', f'{cname}:/tmp/rr')
        with open(log_path, 'w') as log:
            try:
                subprocess.run(['docker', 'exec', cname, 'bash', '/tmp/rr/eval.sh'], stdout=log,
                               stderr=subprocess.STDOUT, timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        sh('docker', 'rm', '-f', cname, check=False)
        shutil.rmtree(tmp, ignore_errors=True)
    runtime = round(time.time() - t0, 1)
    res = grade(t, log_path) if not timed_out else dict(status='timeout')
    res.update(check=label, expect=expect, expect_text=EXPECT_TEXT[expect], ok=meets(expect, res, t),
               runtime_s=runtime, image=image, image_id=image_id(image), applied=[n for n, _ in used],
               log=os.path.relpath(log_path, ROOT), key=key, test_command=cmd)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    json.dump(res, open(cache_file, 'w'), indent=1)
    return res


def short(res):
    if res['status'] != 'ran':
        return res['status']
    return (f"F2P {res['f2p_passed']}/{res['f2p_passed'] + len(res['f2p_failed'])} "
            f"P2P {res['p2p_passed']}/{res['p2p_passed'] + len(res['p2p_failed'])}")


# ------------------------------------------------------------------ images and disk

def start_images():
    return {line.split()[0] for line in open(START_IMAGES) if line.strip()}


def free_bytes():
    """Free bytes on the Mac data volume and in the colima VM's docker disk."""
    mac = shutil.disk_usage('/System/Volumes/Data').free
    r = sh('colima', 'ssh', '--', 'df', '-B1', '--output=avail', '/var/lib/docker', check=False, timeout=60)
    nums = [int(x) for x in r.stdout.split() if x.isdigit()]
    return mac, (nums[-1] if nums else None)


def vm_used():
    r = sh('colima', 'ssh', '--', 'df', '-B1', '--output=used', '/var/lib/docker', check=False, timeout=60)
    nums = [int(x) for x in r.stdout.split() if x.isdigit()]
    return nums[-1] if nums else None


def pulled_so_far():
    if not os.path.exists(PULL_LOG):
        return 0
    return sum(max(0, json.loads(line).get('vm_bytes_added') or 0) for line in open(PULL_LOG) if line.strip())


def ensure_image(image, log=print):
    """Pull the image if it is missing; refuse when disk or the pull budget would be exceeded."""
    if image_id(image):
        return True
    df_mac = sh('df', '-h', '/System/Volumes/Data').stdout.strip().split('\n')[-1]
    mac, vm = free_bytes()
    log(f'df before pull of {image}: {df_mac} | colima VM free {vm / GB if vm else -1:.1f} GB')
    if mac < MIN_FREE or (vm is not None and vm < MIN_FREE):
        log(f'NOT pulling {image}: free disk below 15 GB (Mac {mac / GB:.1f} GB, VM {vm / GB if vm else -1:.1f} GB)')
        return False
    if pulled_so_far() > PULL_BUDGET - 2 * GB:
        log(f'NOT pulling {image}: pull budget used ({pulled_so_far() / GB:.1f} GB of 20 GB)')
        return False
    before, t0 = vm_used(), time.time()
    r = sh('docker', 'pull', '--platform', 'linux/amd64', image, check=False, timeout=7200)
    after = vm_used()
    rec = dict(image=image, ok=r.returncode == 0, seconds=round(time.time() - t0),
               vm_bytes_added=(after - before) if before is not None and after is not None else None,
               mac_free_before=mac, vm_free_before=vm, df_h_before=df_mac, when=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(PULL_LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    log(f"pulled {image}: ok={rec['ok']} in {rec['seconds']} s, VM disk +{(rec['vm_bytes_added'] or 0) / GB:.2f} GB"
        f" (total pulled {pulled_so_far() / GB:.2f} GB)")
    return rec['ok']


def remove_image(image, log=print):
    """Remove an official image after its pairs are done, only if it was not there when we started."""
    if image in start_images() or not image.startswith('swebench/sweb.eval.'):
        return
    r = sh('docker', 'rmi', image, check=False)
    log(f'removed {image}: {"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')


# ------------------------------------------------------------------ catalog pairs

def catalog_pairs():
    """Complete pairs (all eight files present), as (instance_id, trap_id) sorted."""
    out = []
    for iid in sorted(os.listdir(CATALOG)):
        d = os.path.join(CATALOG, iid)
        if not os.path.isdir(d):
            continue
        for tid in sorted(os.listdir(d)):
            if all(os.path.exists(os.path.join(d, tid, f)) for f in NEEDED):
                out.append((iid, tid))
    return out


def read(path):
    return open(path, encoding='utf-8', errors='surrogateescape', newline='').read()


def task_order(pairs):
    """Tasks whose environment layers are already on disk first; tasks sharing an environment together."""
    present_envs = set()
    for img in start_images():
        m = re.match(r'swebench/sweb\.eval\.x86_64\.(.+):latest$', img)
        if m:
            iid = m.group(1).replace('_1776_', '__')
            try:
                t = task(iid)
                present_envs.add((t['repo'], t['version'], t['environment_setup_commit']))
            except KeyError:
                pass
    iids = sorted({p[0] for p in pairs})

    def env(i):
        t = task(i)
        return t['repo'], t['version'], t['environment_setup_commit']
    return sorted(iids, key=lambda i: (env(i) not in present_envs, env(i), i)), env


def task_checks(t, pairs):
    """[(label, diffs, expectation, log/cache name)] for one task: the two shared original checks, then the
    six trap/placebo checks of each of its pairs."""
    iid = t['instance_id']
    out = []
    for label in ('original_nofix', 'original_gold'):
        _, tests, fix, expect = CHECKS[label]
        diffs = [('test_patch.diff', t['test_patch'])] + ([('gold.diff', t['patch'])] if fix else [])
        out.append((label, diffs, expect, f'{iid}/_task/{label}'))
    for _, tid in [p for p in pairs if p[0] == iid]:
        d = os.path.join(CATALOG, iid, tid)
        for label, (ver, tests, fix, expect) in CHECKS.items():
            if label.startswith('original'):
                continue
            diffs = [(ver, read(os.path.join(d, ver))), (tests, read(os.path.join(d, tests)))]
            if fix:
                diffs.append((fix, read(os.path.join(d, fix))))
            out.append((label, diffs, expect, f'{iid}/{tid}/{label}'))
    return out


def cmd_run(only):
    pairs = [p for p in catalog_pairs() if not only or f'{p[0]}/{p[1]}' in only or p[0] in only]
    order, env = task_order(pairs)
    held = None  # (image, env) of the previous task, kept until the next image of the same env is pulled
    for iid in order:
        t = task(iid)
        image = official_image(iid)
        checks = task_checks(t, pairs)
        if all(cached(t, image, diffs, rel) for _, diffs, _, rel in checks):
            print(f'{iid}: all {len(checks)} checks already done (cache)', flush=True)
            continue
        if held and held[1] != env(iid):
            remove_image(held[0])
            held = None
        if not ensure_image(image):
            print(f'SKIPPED {iid}: image not available', flush=True)
            continue
        if held:
            remove_image(held[0])
            held = None
        for label, diffs, expect, rel in checks:
            res = run_check(t, image, label, diffs, expect, rel)
            print(f"{rel}: {short(res)} ok={res['ok']} ({res['runtime_s']} s)", flush=True)
        if image not in start_images():
            held = (image, env(iid))
    if held:
        remove_image(held[0])


# ------------------------------------------------------------------ pilots (their own rr/ images)

PILOTS = {
    # instance -> (image repository, folder with the diffs, tests diff per mode or None = the task's test_patch)
    'django__django-11149': ('rr/django-11149', 'django__django-11149', None),
    'sympy__sympy-13615': ('rr/sympy_1776_sympy-13615', 'sympy__sympy-13615', '{mode}.tests.diff'),
}


def pilot_checks(iid):
    """(label, image, diffs, expectation) for the pilot: no fix / carried gold / memory fix per version.
    The rr/ images already contain the trap or placebo edit, so only tests + fix are applied."""
    repo_img, folder, tests_name = PILOTS[iid]
    t = task(iid)
    w = os.path.join(PILOT_WORK, folder)
    out = []
    for mode in ('original', 'trap', 'placebo'):
        tests = ('test_patch.diff', t['test_patch']) if tests_name is None else \
            (tests_name.format(mode=mode), read(os.path.join(w, tests_name.format(mode=mode))))
        gold = (f'{mode}.gold.diff', read(os.path.join(w, f'{mode}.gold.diff')))
        memory = (f'{mode}.memory.diff', read(os.path.join(w, f'{mode}.memory.diff')))
        image = f'{repo_img}:{mode}'
        out.append((f'{mode}_nofix', image, [tests], 'no_f2p_passes'))
        out.append((f'{mode}_gold', image, [tests, gold], 'all_pass'))
        if memory[1] != gold[1]:  # in the original the memory fix is the gold patch itself
            out.append((f'{mode}_memory', image, [tests, memory], 'some_f2p_fails' if mode == 'trap' else 'all_pass'))
    return out


def cmd_pilots():
    for iid in PILOTS:
        t = task(iid)
        for label, image, diffs, expect in pilot_checks(iid):
            if not image_id(image):
                print(f'{iid} {label}: image {image} missing', flush=True)
                continue
            res = run_check(t, image, label, diffs, expect, f'pilots/{iid}/{label}')
            print(f"{iid} {label}: {short(res)} ok={res['ok']} ({res['runtime_s']} s)", flush=True)


# ------------------------------------------------------------------ report

def load(rel):
    p = os.path.join(CACHE, rel + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def slim(res):
    """The fields of a check result that go into validation.json."""
    if res is None:
        return dict(status='not_run', ok=False)
    keep = ['check', 'expect_text', 'ok', 'status', 'f2p_passed', 'f2p_failed', 'p2p_passed', 'p2p_failed',
            'runtime_s', 'image', 'applied', 'log', 'package_file', 'patch']
    return {k: res[k] for k in keep if k in res}


def cmd_report():
    pairs_out, rows = [], []
    for iid, tid in catalog_pairs():
        t = task(iid)
        spec = json.load(open(os.path.join(CATALOG, iid, tid, 'spec.json')))
        checks = {}
        for label in CHECKS:
            rel = f'{iid}/_task/{label}' if label.startswith('original') else f'{iid}/{tid}/{label}'
            checks[label] = slim(load(rel))
        failed = [k for k, v in checks.items() if not v.get('ok')]
        not_run = [k for k, v in checks.items() if v.get('status') == 'not_run']
        verdict = 'NOT_RUN' if len(not_run) == len(checks) else 'VALID' if not failed else \
            'INCOMPLETE' if not_run else 'INVALID'
        pairs_out.append(dict(instance_id=iid, trap_id=tid, type=spec['type'], loud_or_quiet=spec['loud_or_quiet'],
                              image=official_image(iid), test_command=test_command(t)[1], n_F2P=len(t['F2P']),
                              n_P2P=len(t['P2P']), verdict=verdict, failed_checks=failed, checks=checks))
        for label, v in checks.items():
            rows.append(csv_row(iid, tid, label, v, shared=label.startswith('original')))
    pilots_out = []
    for iid in PILOTS:
        t = task(iid)
        checks = {label: slim(load(f'pilots/{iid}/{label}')) for label, *_ in pilot_checks(iid)}
        failed = [k for k, v in checks.items() if not v.get('ok')]
        not_run = [k for k, v in checks.items() if v.get('status') == 'not_run']
        verdict = 'NOT_RUN' if len(not_run) == len(checks) else 'VALID' if not failed else \
            'INCOMPLETE' if not_run else 'INVALID'
        pilots_out.append(dict(instance_id=iid, images=PILOTS[iid][0] + ':{original,trap,placebo}',
                               test_command=test_command(t)[1], n_F2P=len(t['F2P']), n_P2P=len(t['P2P']),
                               verdict=verdict, failed_checks=failed, checks=checks))
        for label, v in checks.items():
            rows.append(csv_row(iid, 'pilot', label, v, shared=False))
    pulls = [json.loads(line) for line in open(PULL_LOG)] if os.path.exists(PULL_LOG) else []
    import swebench
    out = dict(
        description='Docker validation of the memory-trap catalog (traps/catalog) in the official SWE-bench images, '
                    'one fresh container per check, --platform linux/amd64 --network none (colima, Apple silicon, '
                    'Rosetta). See scripts/validate_trap_catalog.py for the checks and their expectations.',
        command='.venv/bin/python -I scripts/validate_trap_catalog.py run; ... pilots; ... report',
        swebench_version=swebench.__version__,
        checks={k: dict(version=v[0], tests=v[1], fix=v[2], expect=EXPECT_TEXT[v[3]]) for k, v in CHECKS.items()},
        n_pairs=len(pairs_out), n_valid=sum(p['verdict'] == 'VALID' for p in pairs_out),
        verdicts={p['instance_id'] + '/' + p['trap_id']: p['verdict'] for p in pairs_out},
        pairs=pairs_out, pilots=pilots_out,
        image_pulls=pulls, pulled_bytes_vm=sum(max(0, p.get('vm_bytes_added') or 0) for p in pulls),
    )
    json.dump(out, open(OUT_JSON, 'w'), indent=1)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: out[k] for k in ('n_pairs', 'n_valid', 'verdicts')}, indent=1))
    for p in pilots_out:
        print(p['instance_id'], p['verdict'], p['failed_checks'])


def csv_row(iid, tid, label, v, shared):
    f2p_failed, p2p_failed = v.get('f2p_failed') or [], v.get('p2p_failed') or []
    failing = f2p_failed + p2p_failed
    return dict(instance_id=iid, trap_id=tid, check=label, shared_task_check=shared, expected=v.get('expect_text'),
                result='pass' if v.get('ok') else 'not_run' if v.get('status') == 'not_run' else 'fail',
                status=v.get('status'),
                f2p_passed=v.get('f2p_passed'), f2p_total=(v.get('f2p_passed') or 0) + len(f2p_failed)
                if v.get('status') == 'ran' else None,
                p2p_passed=v.get('p2p_passed'), p2p_total=(v.get('p2p_passed') or 0) + len(p2p_failed)
                if v.get('status') == 'ran' else None,
                n_failing_tests=len(failing) if v.get('status') == 'ran' else None,
                failing_tests='; '.join(failing[:25]) + (f'; ... (+{len(failing) - 25} more)' if len(failing) > 25 else ''),
                runtime_s=v.get('runtime_s'), log=v.get('log'))


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'run': lambda: cmd_run(sys.argv[2:]), 'pilots': cmd_pilots, 'report': cmd_report}[cmd]()
