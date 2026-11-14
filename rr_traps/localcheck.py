"""Run a catalog entry's task tests locally (no Docker), as a quick check before the Docker validator.

Configurations (FAIL_TO_PASS = the task's target tests):
  original + tests + gold     environment sanity: target tests should pass (if not, the local Python
                              environment is not good enough for this task and the rest is unreliable)
  trap     + tests' + gold'   the carried fix still solves the task in the trap version
  trap     + tests' + memory  the verbatim upstream fix in the trap version: should FAIL (the trap bites)
  placebo  + tests' + gold'   should pass
  placebo  + tests' + memory  should pass (the placebo does not bite)
Only the test modules/files of the task are run, so "P2P" counts only PASS_TO_PASS tests in those.

Local runs use plain virtualenvs (work/catalog/venvs), not the official SWE-bench images, so optional
dependencies may be missing; the Docker validator is the authority.
"""
import json
import os
import re
import subprocess

from . import builder as B
from . import common as C

VENVS = os.path.join(B.WORK, 'venvs')


def python_for(t):
    v = str(t['version'])
    major, minor = (int(x) for x in v.split('.')[:2])
    if t['repo'] == 'django/django':
        return os.path.join(VENVS, 'py311' if (major, minor) >= (5, 0) else 'py38', 'bin/python')
    if t['repo'] == 'sympy/sympy':
        return os.path.join(VENVS, 'py39' if (major, minor) >= (1, 7) else 'py38', 'bin/python')
    return None


def django_labels(t):
    labels = set()
    for name in json.loads(t['FAIL_TO_PASS']) + json.loads(t['PASS_TO_PASS']):
        m = re.search(r'\(([\w.]+)\)\s*$', name)
        if m:
            parts = m.group(1).split('.')
            mod = []
            for p in parts:
                if p[:1].isupper():
                    break
                mod.append(p)
            if mod:
                labels.add('.'.join(mod))
    for f in C.diff_files(t['test_patch']):
        if f.startswith('tests/') and f.endswith('.py') and not f.endswith('__init__.py'):
            labels.add(f[len('tests/'):-3].replace('/', '.'))
    # a label that is a prefix of another (package vs module) runs the module twice: keep the shortest
    out = sorted(labels)
    return [x for x in out if not any(x != y and x.startswith(y + '.') for y in out)]


def run_tests(t, wt, timeout=1500):
    """Run the task's tests in worktree wt. Returns ({test name: status}, tail of the output)."""
    py = python_for(t)
    env = dict(os.environ, PYTHONPATH=wt, PYTHONDONTWRITEBYTECODE='1', LANG='en_US.UTF-8', LC_ALL='en_US.UTF-8')
    if t['repo'] == 'django/django':
        cmd = [py, 'runtests.py', '--settings=test_sqlite', '--parallel', '1', '-v', '2'] + django_labels(t)
        cwd = os.path.join(wt, 'tests')
    else:
        files = [f for f in C.diff_files(t['test_patch']) if f.endswith('.py')]
        cmd = [py, 'bin/test', '-C', '--verbose'] + files
        cwd = wt
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        text = r.stdout + '\n' + r.stderr
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
        text += '\nTIMEOUT'
    res = {}
    lines = text.split('\n')
    for k, line in enumerate(lines):
        if t['repo'] == 'django/django':
            if ' ... ' in line:
                name, _, status = line.partition(' ... ')
                status = status.strip()
                res[name.strip()] = 'ok' if status.startswith('ok') or status.startswith('expected failure') \
                    or status.startswith('skipped') else status
                # Django 3.x prints a docstring test as "name (module.Class)\nFirst docstring line ... ok"
                if k and re.search(r'\([\w.]+\)\s*$', lines[k - 1]) and ' ... ' not in lines[k - 1]:
                    res[lines[k - 1].strip()] = res[name.strip()]
        else:
            m = re.match(r'^(test_\w+)\s+(ok|F|E|f|X|w|Skipped.*)\s*$', line.strip())
            if m:
                res[m.group(1)] = 'ok' if m.group(2) in ('ok', 'f') or m.group(2).startswith('Skipped') else m.group(2)
    return res, text[-3000:]


def score(t, res):
    f2p, p2p = json.loads(t['FAIL_TO_PASS']), json.loads(t['PASS_TO_PASS'])
    found_p2p = [n for n in p2p if n in res]
    return dict(F2P=f"{sum(res.get(n) == 'ok' for n in f2p)}/{len(f2p)}",
                F2P_ok=all(res.get(n) == 'ok' for n in f2p),
                P2P=f"{sum(res.get(n) == 'ok' for n in found_p2p)}/{len(found_p2p)} (of {len(p2p)}; only the modules run)",
                P2P_failing=[n for n in found_p2p if res.get(n) != 'ok'][:10],
                n_tests_seen=len(res), n_failing=sum(1 for s in res.values() if s != 'ok'))


def check_entry(entry_dir, log=print):
    """Run the five configurations for one catalog entry; writes local_check.json next to spec.json."""
    spec = json.load(open(os.path.join(entry_dir, 'spec.json')))
    t = B.task(spec['instance_id'])
    wt = B.worktree(t)
    base = t['base_commit']
    rd = lambda n: C.read(entry_dir, n)
    out = dict(python=python_for(t), configs={})
    plan = [('original + tests + gold', None, [t['test_patch'], t['patch']]),
            ("trap + tests' + gold'", 'trap', [rd('tests_trap.diff'), rd('gold_trap.diff')]),
            ("trap + tests' + memory", 'trap', [rd('tests_trap.diff'), rd('memory.diff')]),
            ("placebo + tests' + gold'", 'placebo', [rd('tests_placebo.diff'), rd('gold_placebo.diff')]),
            ("placebo + tests' + memory", 'placebo', [rd('tests_placebo.diff'), rd('memory.diff')])]
    for label, mode, patches in plan:
        B.reset(wt, base)
        ok = True
        if mode and rd(f'{mode}.diff').strip():
            ok, msg = B.apply_patch(wt, rd(f'{mode}.diff'))
        for p in patches:
            if ok and p.strip():
                ok, msg = B.apply_patch(wt, p)
        if not ok:
            out['configs'][label] = dict(result='patch did not apply', error=msg)
            log(f"  {spec['instance_id']} {spec['trap_id']} {label}: patch did not apply")
            continue
        res, tail = run_tests(t, wt)
        sc = score(t, res)
        if not res:
            sc['output_tail'] = tail[-1500:]
        out['configs'][label] = sc
        log(f"  {spec['instance_id']} {spec['trap_id']} {label}: F2P {sc['F2P']}, P2P {sc['P2P']}")
    B.reset(wt, base)
    c = out['configs']
    get = lambda k: c.get(k, {}).get('F2P_ok')
    out['summary'] = dict(
        env_ok=get('original + tests + gold'),
        solvable_in_trap=get("trap + tests' + gold'"),
        trap_bites=(get("trap + tests' + memory") is False) or
        c.get("trap + tests' + memory", {}).get('result') == 'patch did not apply',
        memory_fails_in_trap_by=('patch does not apply' if c.get("trap + tests' + memory", {}).get('result')
                                 else 'tests' if get("trap + tests' + memory") is False else None),
        solvable_in_placebo=get("placebo + tests' + gold'"),
        placebo_harmless=get("placebo + tests' + memory"),
    )
    s = out['summary']
    s['all_good'] = bool(s['env_ok'] and s['solvable_in_trap'] and s['trap_bites'] and s['solvable_in_placebo']
                         and s['placebo_harmless'] and s['memory_fails_in_trap_by'] == 'tests')
    json.dump(out, open(os.path.join(entry_dir, 'local_check.json'), 'w'), indent=1)
    return out
