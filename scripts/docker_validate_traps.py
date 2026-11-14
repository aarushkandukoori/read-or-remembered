"""Docker validation of the memory-trap catalog: a fresh run in priority order (Nov 14, 2026).

The Nov 13 validator (scripts/validate_trap_catalog.py) was interrupted by a memory crash. This script REUSES
it for the official pieces (test command, eval script, image pull/remove with the disk checks) and adds:
  * its own folder work/trap_validation/ (fresh logs; nothing from the Nov 13 cache is reused)
  * the task's priority order: one pair per trap type T1..T8 first, then the django-11149 pilot in its
    rr/django-11149:{original,trap,placebo} images, then the remaining pairs; no new pair starts after the
    wall-time deadline (argument --deadline HH:MM, local time)
  * a memory check before every container: wait while memguard says danger/emergency; restart Colima
    (only when memory is ok/tight) if it was stopped
  * exact SWE-bench grading of the test output (harness parser, no fallback to the whole log)
  * the verdict defined by the task (letters a-d below) and the report files

Every check runs in a FRESH container (--platform linux/amd64 --network none), the diffs are applied at
/testbed with `git apply -v`, then the task's official test command runs.

  check            diffs applied                              expectation                     task letter
  original_nofix   test_patch                                 no FAIL_TO_PASS test passes     extra
  original_gold    test_patch + gold patch                    all F2P and all P2P pass        (d)
  trap_nofix       trap + tests_trap                          no F2P test passes              (d)
  trap_gold        trap + tests_trap + gold_trap              all F2P and all P2P pass        (a)
  trap_memory      trap + tests_trap + memory (upstream fix)  at least one F2P test fails     (b)
  placebo_nofix    placebo + tests_placebo                    no F2P test passes              extra
  placebo_gold     placebo + tests_placebo + gold_placebo     all F2P and all P2P pass        (d)
  placebo_memory   placebo + tests_placebo + memory           all F2P pass                    (c)

VALID = (a), (b), (c) meet their expectation and so does every (d) check that ran. The two "extra" checks
are sanity checks of the task itself; they are reported but not part of the verdict.

Usage (from the project folder):
  .venv/bin/python -I scripts/docker_validate_traps.py run --deadline 03:03
  .venv/bin/python -I scripts/docker_validate_traps.py report
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_trap_catalog as vtc  # noqa: E402  (the Nov 13 validator, reused)
from swebench.harness.constants import END_TEST_OUTPUT, FAIL_TO_PASS, PASS_TO_PASS, START_TEST_OUTPUT  # noqa: E402
from swebench.harness.grading import get_eval_tests_report  # noqa: E402
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER  # noqa: E402

ROOT = vtc.ROOT
WORK = os.path.join(ROOT, 'work/trap_validation')
CACHE, LOGS, TMP = (os.path.join(WORK, d) for d in ('cache', 'logs', 'tmp'))
RUN_STATE = os.path.join(WORK, 'run_state.json')   # deadline, not-run pairs, removed images
OUT_JSON = os.path.join(ROOT, 'outputs/traps/validation.json')
OUT_CSV = os.path.join(ROOT, 'outputs/traps/validation.csv')

# Point the reused helpers at this run's folder (start image list, pull log with the 20 GB budget).
vtc.VAL, vtc.CACHE, vtc.LOGS, vtc.TMP = WORK, CACHE, LOGS, TMP
vtc.START_IMAGES = os.path.join(WORK, 'images_at_start.txt')   # written before the first pull
vtc.PULL_LOG = os.path.join(WORK, 'pulls.jsonl')

TIMEOUT = 1800   # seconds for one test run
EXPECT_TEXT = dict(vtc.EXPECT_TEXT, all_f2p_pass='all F2P pass')

# check -> (task letter, version diff, tests diff, fix diff, expectation); TEST/GOLD = the task's own patches
CHECKS = {
    'original_nofix': ('extra', None, 'TEST', None, 'no_f2p_passes'),
    'original_gold': ('d', None, 'TEST', 'GOLD', 'all_pass'),
    'trap_nofix': ('d', 'trap.diff', 'tests_trap.diff', None, 'no_f2p_passes'),
    'trap_gold': ('a', 'trap.diff', 'tests_trap.diff', 'gold_trap.diff', 'all_pass'),
    'trap_memory': ('b', 'trap.diff', 'tests_trap.diff', 'memory.diff', 'some_f2p_fails'),
    'placebo_nofix': ('extra', 'placebo.diff', 'tests_placebo.diff', None, 'no_f2p_passes'),
    'placebo_gold': ('d', 'placebo.diff', 'tests_placebo.diff', 'gold_placebo.diff', 'all_pass'),
    'placebo_memory': ('c', 'placebo.diff', 'tests_placebo.diff', 'memory.diff', 'all_f2p_pass'),
}

# Priority 1: one pair per trap type, the cleanest of each type (reasons in notes/trap-validation.md).
# Fast Django tasks first so most types are done early; sympy-18199 (about 100 s per check) last.
PRIORITY1 = {
    'T3': 'django__django-15957/T3-clear_limits',
    'T4': 'django__django-15127/T4-get_level_tags',
    'T7': 'django__django-16631/T7-salted_hmac-secret',
    'T8': 'django__django-16612/T8-get_full_path',
    'T5': 'sympy__sympy-18211/T5-solve_univariate_inequality',
    'T6': 'sympy__sympy-13877/T6-_find_reasonable_pivot',
    'T2': 'sympy__sympy-14248/T2-_keep_coeff-sign',
    'T1': 'sympy__sympy-18199/T1-mod_inverse',
}
PILOT = 'django__django-11149'      # priority 2: the pilot trap, in its own rr/ images
PILOT_DIFFS = os.path.join(ROOT, 'work', PILOT)


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def sh(*cmd, timeout=120):
    """Run a command, never raise; returns CompletedProcess (returncode 124 on timeout)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, '', 'timeout')


def read(path):
    return open(path, encoding='utf-8', errors='surrogateescape', newline='').read()


# ------------------------------------------------------------------ memory and Docker health

def mem_stage():
    """memguard's stage: ok / tight / danger / emergency (unknown if memguard does not answer)."""
    r = sh('python3', os.path.expanduser('~/Projects/memguard/memguard.py'), 'status', timeout=60)
    m = re.search(r'stage (\w+)', r.stdout)
    return m.group(1) if m else 'unknown'


def wait_for_memory():
    """Block while the Mac is short of memory (the task: proceed only when ok or tight)."""
    while True:
        stage = mem_stage()
        if stage not in ('danger', 'emergency'):
            return stage
        log(f'memguard stage {stage}: waiting 30 s before starting a container')
        time.sleep(30)


def ensure_docker():
    """Docker must answer; memguard stops Colima when it is idle, so start it again if needed."""
    for _ in range(3):
        if sh('docker', 'info', '--format', '{{.ServerVersion}}', timeout=60).returncode == 0:
            return True
        wait_for_memory()
        log('docker does not answer (Colima stopped?): running colima start')
        r = sh('colima', 'start', timeout=900)
        log(f'colima start exit {r.returncode}')
    return sh('docker', 'info', '--format', '{{.ServerVersion}}', timeout=60).returncode == 0


# ------------------------------------------------------------------ one check = one fresh container

def grade(t, log_path):
    """Grade a test log exactly like swebench.harness.grading (get_logs_eval + get_eval_tests_report)."""
    content = read(log_path)
    bad = re.search(r'>>>>> Patch Apply Failed: (\S+)', content)
    if bad:
        return dict(status='patch_failed', patch=bad.group(1))
    if START_TEST_OUTPUT not in content or END_TEST_OUTPUT not in content:
        return dict(status='no_test_output')
    test_part = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    status_map = MAP_REPO_TO_PARSER[t['repo']](test_part, None)
    rep = get_eval_tests_report(status_map, {FAIL_TO_PASS: t['F2P'], PASS_TO_PASS: t['P2P']})
    pkg = re.search(r'PACKAGE_FILE (\S+)', content)
    return dict(status='ran', n_parsed=len(status_map),
                f2p_passed=len(rep[FAIL_TO_PASS]['success']), f2p_failed=rep[FAIL_TO_PASS]['failure'],
                p2p_passed=len(rep[PASS_TO_PASS]['success']), p2p_failed=rep[PASS_TO_PASS]['failure'],
                package_file=pkg.group(1) if pkg else None)


def meets(expect, res):
    if res['status'] != 'ran':
        return False
    if expect == 'all_f2p_pass':
        return not res['f2p_failed']
    return vtc.meets(expect, res, None)


def container_run(t, image, rel, used):
    """Start a fresh container, copy the diffs + eval script in, run it, remove the container."""
    os.makedirs(TMP, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=TMP)
    names = []
    for k, (n, d) in enumerate(used):
        name = f'{k:02d}_{n}'
        with open(os.path.join(tmp, name), 'w', encoding='utf-8', errors='surrogateescape', newline='') as f:
            f.write(d if d.endswith('\n') else d + '\n')
        names.append(name)
    with open(os.path.join(tmp, 'eval.sh'), 'w') as f:
        f.write(vtc.eval_script(t, names))   # official eval_commands + git apply + test command
    log_path = os.path.join(LOGS, rel + '.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cname = 'rrval2-' + hashlib.sha1(rel.encode()).hexdigest()[:10]
    sh('docker', 'rm', '-f', cname)
    t0 = time.time()
    try:
        r = sh('docker', 'run', '-d', '--name', cname, '--platform', 'linux/amd64', '--network', 'none',
               '--memory', '4g', image, 'sleep', 'infinity')
        if r.returncode:
            return dict(status='infra_error', error='docker run: ' + r.stderr[-300:]), log_path, 0.0
        r = sh('docker', 'cp', tmp + '/.', f'{cname}:/tmp/rr')
        if r.returncode:
            return dict(status='infra_error', error='docker cp: ' + r.stderr[-300:]), log_path, 0.0
        with open(log_path, 'w') as out:
            try:
                p = subprocess.run(['docker', 'exec', cname, 'bash', '/tmp/rr/eval.sh'], stdout=out,
                                   stderr=subprocess.STDOUT, timeout=TIMEOUT)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
    finally:
        sh('docker', 'rm', '-f', cname)
        shutil.rmtree(tmp, ignore_errors=True)
    runtime = round(time.time() - t0, 1)
    if rc == 'timeout':
        return dict(status='timeout'), log_path, runtime
    res = grade(t, log_path)
    text = read(log_path)
    docker_trouble = any(s in text for s in ('Cannot connect to the Docker daemon', 'Error response from daemon',
                                             'No such container'))
    if res['status'] == 'no_test_output' and (docker_trouble or rc in (125, 126, 127, 137, 255)):
        res = dict(status='infra_error', error=f'exec exit {rc}: ' + text[-300:])
    res['exec_exit'] = rc
    return res, log_path, runtime


def run_check(t, image, label, diffs, expect, rel):
    """One check (cached by image id + test command + diffs, so a restarted run skips finished checks)."""
    used = [(n, d) for n, d in diffs if d and d.strip()]   # an empty diff has nothing to apply
    img_id = vtc.image_id(image)
    cmd = vtc.test_command(t)[1]
    key = hashlib.sha1(json.dumps([image, img_id, cmd, used]).encode('utf-8', 'surrogateescape')).hexdigest()
    cache_file = os.path.join(CACHE, rel + '.json')
    if os.path.exists(cache_file):
        old = json.load(open(cache_file))
        if old.get('key') == key and old.get('status') != 'infra_error':
            return old
    for attempt in (1, 2, 3):
        wait_for_memory()
        ensure_docker()
        res, log_path, runtime = container_run(t, image, rel, used)
        if res['status'] != 'infra_error':
            break
        log(f"{rel}: Docker problem, attempt {attempt}: {res['error'][:200]}")
        time.sleep(15)
    res.update(check=label, letter=CHECKS[label][0], expect=expect, expect_text=EXPECT_TEXT[expect],
               ok=meets(expect, res), runtime_s=runtime, image=image, image_id=img_id,
               applied=[n for n, _ in used], log=os.path.relpath(log_path, ROOT), key=key, test_command=cmd,
               finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    json.dump(res, open(cache_file, 'w'), indent=1)
    return res


def short(res):
    if res['status'] != 'ran':
        return res['status'] + (f" ({res['patch']})" if res.get('patch') else '')
    return (f"F2P {res['f2p_passed']}/{res['f2p_passed'] + len(res['f2p_failed'])} "
            f"P2P {res['p2p_passed']}/{res['p2p_passed'] + len(res['p2p_failed'])}")


# ------------------------------------------------------------------ what to run

def pair_checks(t, tid, with_original):
    """[(label, image, diffs, expectation, cache/log name)] for one catalog pair."""
    iid = t['instance_id']
    image = vtc.official_image(iid)
    d = os.path.join(vtc.CATALOG, iid, tid)
    out = []
    for label, (_, ver, tests, fix, expect) in CHECKS.items():
        if label.startswith('original'):
            if with_original:   # shared by all pairs of a task, run once with its first pair
                diffs = [('test_patch.diff', t['test_patch'])] + ([('gold.diff', t['patch'])] if fix else [])
                out.append((label, image, diffs, expect, f'{iid}/_task/{label}'))
            continue
        diffs = [(ver, read(os.path.join(d, ver))), (tests, read(os.path.join(d, tests)))]
        if fix:
            diffs.append((fix, read(os.path.join(d, fix))))
        out.append((label, image, diffs, expect, f'{iid}/{tid}/{label}'))
    return out


def pilot_checks():
    """The pilot: the rr/ images already contain the edit, so only tests + fix are applied. Its test patch
    touches tests/admin_inlines only, which neither edit changes, so the task's own test patch is used."""
    t = vtc.task(PILOT)
    tests = ('test_patch.diff', t['test_patch'])
    out = []
    for mode in ('original', 'trap', 'placebo'):
        image = f'rr/django-11149:{mode}'
        gold = (f'{mode}.gold.diff', read(os.path.join(PILOT_DIFFS, f'{mode}.gold.diff')))
        memory = (f'{mode}.memory.diff', read(os.path.join(PILOT_DIFFS, f'{mode}.memory.diff')))
        out.append((f'{mode}_nofix', image, [tests], CHECKS[f'{mode}_nofix'][4], f'pilots/{PILOT}/{mode}_nofix'))
        out.append((f'{mode}_gold', image, [tests, gold], CHECKS[f'{mode}_gold'][4], f'pilots/{PILOT}/{mode}_gold'))
        if mode != 'original':   # in the original the memory fix is the gold patch itself
            out.append((f'{mode}_memory', image, [tests, memory], CHECKS[f'{mode}_memory'][4],
                        f'pilots/{PILOT}/{mode}_memory'))
    return out


def plan():
    """Ordered jobs: ('pair', 'iid/tid', priority) or ('pilot', PILOT, 2)."""
    pairs = [f'{i}/{t}' for i, t in vtc.catalog_pairs()]
    p1 = list(PRIORITY1.values())
    rest = sorted((p for p in pairs if p not in p1), key=lambda p: (not p.startswith('django'), p))
    return [('pair', p, 1) for p in p1] + [('pilot', PILOT, 2)] + [('pair', p, 3) for p in rest]


def save_state(state):
    json.dump(state, open(RUN_STATE, 'w'), indent=1)


def cmd_run(deadline_hhmm):
    os.makedirs(WORK, exist_ok=True)
    assert os.path.exists(vtc.START_IMAGES), 'record the starting image list first'
    h, m = map(int, deadline_hhmm.split(':'))
    now = time.localtime()
    deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
    jobs = plan()
    state = dict(deadline=time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline)),
                 started=time.strftime('%Y-%m-%d %H:%M:%S'), plan=[j[1] for j in jobs], not_run=[], done=[],
                 removed_images=[], skipped=[])
    save_state(state)
    original_done = set()
    for k, (kind, name, prio) in enumerate(jobs):
        if time.time() > deadline:
            state['not_run'].append(dict(job=name, priority=prio, reason='wall-time budget reached'))
            save_state(state)
            log(f'NOT RUN (time budget): {name}')
            continue
        if kind == 'pilot':
            t = vtc.task(PILOT)
            checks = pilot_checks()
        else:
            iid, tid = name.split('/')
            t = vtc.task(iid)
            image = vtc.official_image(iid)
            if not vtc.ensure_image(image, log=log):   # df -h, 15 GB free and 20 GB pull budget checks
                state['skipped'].append(dict(job=name, priority=prio, reason=f'image {image} not available'))
                save_state(state)
                log(f'SKIPPED {name}: image not available')
                continue
            checks = pair_checks(t, tid, with_original=iid not in original_done)
        log(f'--- {name} (priority {prio}): {len(checks)} checks')
        for label, image, diffs, expect, rel in checks:
            res = run_check(t, image, label, diffs, expect, rel)
            log(f"{rel}: {short(res)} expect '{EXPECT_TEXT[expect]}' -> {'ok' if res['ok'] else 'NOT MET'} "
                f"({res['runtime_s']} s)")
        state['done'].append(name)
        if kind == 'pair':
            original_done.add(iid)
            # remove the official image once no later job needs it (never one present at the start)
            later = [j[1].split('/')[0] for j in jobs[k + 1:] if j[0] == 'pair']
            if iid not in later and image not in vtc.start_images():
                vtc.remove_image(image, log=log)
                state['removed_images'].append(image)
        save_state(state)
    state['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_state(state)
    log('run finished')


# ------------------------------------------------------------------ report

def load(rel):
    p = os.path.join(CACHE, rel + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def slim(res):
    if res is None:
        return dict(status='not_run', ok=None)
    keep = ['check', 'letter', 'expect_text', 'ok', 'status', 'f2p_passed', 'f2p_failed', 'p2p_passed',
            'p2p_failed', 'runtime_s', 'image', 'image_id', 'applied', 'log', 'package_file', 'patch', 'exec_exit',
            'finished']
    return {k: res[k] for k in keep if k in res}


def verdict_of(checks):
    """VALID = (a), (b), (c) ok and every (d) that ran ok; INCOMPLETE if one of (a)-(c) did not run."""
    ran = {k: v for k, v in checks.items() if v['status'] != 'not_run'}
    if not ran:
        return 'NOT_RUN', []
    failed = [k for k, v in ran.items() if CHECKS[k][0] != 'extra' and not v['ok']]
    required_missing = [k for k in checks if CHECKS[k][0] in 'abc' and k not in ran]
    if failed:
        return 'INVALID', failed
    return ('INCOMPLETE' if required_missing else 'VALID'), failed


def csv_row(iid, tid, ptype, label, v):
    f2p_failed, p2p_failed = v.get('f2p_failed') or [], v.get('p2p_failed') or []
    ran = v['status'] == 'ran'
    failing = f2p_failed + p2p_failed
    return dict(instance_id=iid, trap_id=tid, type=ptype, check=label, task_letter=CHECKS[label][0],
                expected=EXPECT_TEXT[CHECKS[label][4]],
                result='not_run' if v['status'] == 'not_run' else 'pass' if v.get('ok') else 'fail',
                status=v['status'], f2p_passed=v.get('f2p_passed'),
                f2p_total=(v.get('f2p_passed') or 0) + len(f2p_failed) if ran else None,
                p2p_passed=v.get('p2p_passed'),
                p2p_total=(v.get('p2p_passed') or 0) + len(p2p_failed) if ran else None,
                n_failing_tests=len(failing) if ran else None,
                failing_tests='; '.join(failing[:25]) + (f'; ... (+{len(failing) - 25} more)' if len(failing) > 25
                                                          else ''),
                runtime_s=v.get('runtime_s'), log=v.get('log'))


def cmd_report():
    state = json.load(open(RUN_STATE)) if os.path.exists(RUN_STATE) else {}
    prio = {name: p for _, name, p in plan()}
    pairs_out, rows = [], []
    for iid, tid in vtc.catalog_pairs():
        t = vtc.task(iid)
        spec = json.load(open(os.path.join(vtc.CATALOG, iid, tid, 'spec.json')))
        checks = {}
        for label in CHECKS:
            rel = f'{iid}/_task/{label}' if label.startswith('original') else f'{iid}/{tid}/{label}'
            checks[label] = slim(load(rel))
        verdict, failed = verdict_of(checks)
        extra_failed = [k for k, v in checks.items() if CHECKS[k][0] == 'extra' and v['status'] != 'not_run'
                        and not v['ok']]
        pairs_out.append(dict(instance_id=iid, trap_id=tid, type=spec['type'], loud_or_quiet=spec['loud_or_quiet'],
                              priority=prio.get(f'{iid}/{tid}'), image=vtc.official_image(iid),
                              test_command=vtc.test_command(t)[1], n_F2P=len(t['F2P']), n_P2P=len(t['P2P']),
                              verdict=verdict, failed_checks=failed, extra_checks_failed=extra_failed,
                              checks=checks))
        rows += [csv_row(iid, tid, spec['type'], label, v) for label, v in checks.items()]
    t = vtc.task(PILOT)
    checks = {label: slim(load(rel)) for label, _, _, _, rel in pilot_checks()}
    verdict, failed = verdict_of(checks)
    pilot = dict(instance_id=PILOT, trap='T1 argument order: get_permission_codename(action, opts) -> (opts, action)',
                 images='rr/django-11149:{original,trap,placebo}', test_command=vtc.test_command(t)[1],
                 n_F2P=len(t['F2P']), n_P2P=len(t['P2P']), verdict=verdict, failed_checks=failed,
                 note='placebo.memory.diff is byte-identical to placebo.gold.diff and original.memory.diff to '
                      'original.gold.diff, so original_memory is not a separate check',
                 checks=checks)
    rows += [csv_row(PILOT, 'pilot-T1-get_permission_codename', 'T1', label, v) for label, v in checks.items()]
    pulls = [json.loads(line) for line in open(vtc.PULL_LOG)] if os.path.exists(vtc.PULL_LOG) else []
    import swebench
    out = dict(
        description='Docker validation of the memory-trap catalog (traps/catalog) and the django-11149 pilot in the '
                    'official SWE-bench images (pilot: its rr/ images), one fresh container per check, '
                    '--platform linux/amd64 --network none, colima on Apple silicon. Checks and verdict rule: '
                    'scripts/docker_validate_traps.py.',
        commands=['.venv/bin/python -I scripts/docker_validate_traps.py run --deadline 03:03',
                  '.venv/bin/python -I scripts/docker_validate_traps.py report'],
        swebench_version=swebench.__version__,
        verdict_rule='VALID = (a) trap_gold, (b) trap_memory, (c) placebo_memory meet their expectation and every '
                     '(d) check that ran (original_gold, trap_nofix, placebo_gold) does too; original_nofix and '
                     'placebo_nofix are extra sanity checks, reported but not part of the verdict',
        checks={k: dict(task_letter=v[0], version=v[1], tests=v[2], fix=v[3], expect=EXPECT_TEXT[v[4]])
                for k, v in CHECKS.items()},
        run=state,
        n_pairs=len(pairs_out),
        verdict_counts={v: sum(p['verdict'] == v for p in pairs_out)
                        for v in ('VALID', 'INVALID', 'INCOMPLETE', 'NOT_RUN')},
        verdicts={p['instance_id'] + '/' + p['trap_id']: p['verdict'] for p in pairs_out},
        pairs=pairs_out, pilot=pilot,
        image_pulls=pulls, pulled_bytes_vm=sum(max(0, p.get('vm_bytes_added') or 0) for p in pulls),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, 'w'), indent=1)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: out[k] for k in ('n_pairs', 'verdict_counts', 'verdicts')}, indent=1))
    print('pilot', pilot['verdict'], pilot['failed_checks'])


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        cmd_run(sys.argv[sys.argv.index('--deadline') + 1] if '--deadline' in sys.argv else '03:03')
    elif sys.argv[1] == 'report':
        cmd_report()
