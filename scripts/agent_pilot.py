"""Agent pilot on one memory trap: run mini-swe-agent (DeepSeek) on the original, trap and placebo
versions of django__django-11149 inside Docker, then grade each run with the task's tests.

  build                 make images rr/django-11149:{original,trap,placebo}. Each starts from the
                        official SWE-bench image, applies the edit, and replaces git history with a
                        single commit, so `git log`/`git diff` cannot reveal the original code.
  run COND K MODEL      one agent run -> outputs/agent_runs/COND-K.traj.json
  grade                 apply each run's patch + the task's tests (carried into that version) and run them
Usage: .venv/bin/python -I scripts/agent_pilot.py build | run trap 1 deepseek/deepseek-chat | grade
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IID = 'django__django-11149'
BASE_IMAGE = 'swebench/sweb.eval.x86_64.django_1776_django-11149:latest'
WORK = ROOT / 'work' / IID
RUNS = ROOT / 'outputs' / os.environ.get('RR_RUNS', 'agent_runs')
PLAT = ['--platform', 'linux/amd64']


def sh(*cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode:
        raise RuntimeError(f'{cmd[:6]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return r


def task():
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / 'data/meta/swebv/data/test-00000-of-00001.parquet').to_pandas().set_index('instance_id')
    return t.loc[IID]


def build():
    for cond in ('original', 'trap', 'placebo'):
        cid = sh('docker', 'run', '-d', *PLAT, BASE_IMAGE, 'sleep', '3600').stdout.strip()
        try:
            if cond != 'original':
                sh('docker', 'cp', str(WORK / f'{cond}.edit.diff'), f'{cid}:/tmp/edit.diff')
                sh('docker', 'exec', cid, 'bash', '-c', 'cd /testbed && git apply /tmp/edit.diff && rm /tmp/edit.diff')
            sh('docker', 'exec', cid, 'bash', '-c',
               'cd /testbed && rm -rf .git && git init -q && git add -A && '
               'git -c user.name=dev -c user.email=dev@example.com commit -qm "Initial commit"')
            sh('docker', 'commit', cid, f'rr/django-11149:{cond}')
            print('built', cond)
        finally:
            sh('docker', 'rm', '-f', cid, check=False)


def carried_tests(cond):
    """The task's test changes as they look in this version (same edit applied on top)."""
    t = task()
    repo = ROOT / 'data/repos/django_django'
    d = WORK / f'{cond}-tests'
    if d.exists():
        sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
    sh('git', '-C', str(repo), 'worktree', 'add', '--detach', str(d), t.base_commit)
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
        f.write(t.test_patch)
    sh('git', 'apply', f.name, cwd=d)
    if cond != 'original':
        sh(sys.executable, '-I', str(ROOT / 'scripts/make_argswap_trap.py'), str(d),
           'django/contrib/auth/__init__.py', 'get_permission_codename', cond)
    sh('git', 'add', '-A', cwd=d)
    vc = sh('git', 'rev-parse', 'HEAD', cwd=WORK / cond).stdout.strip()
    out = sh('git', 'diff', '--cached', vc, cwd=d).stdout
    sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
    return out


def run(cond, k, model_name):
    import yaml
    from minisweagent.agents import get_agent
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    cfg = yaml.safe_load(open(builtin_config_dir / 'benchmarks' / 'swebench.yaml'))
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f'{cond}-{k}.traj.json'
    env_cfg = dict(cfg['environment'], environment_class='docker', image=f'rr/django-11149:{cond}',
                   run_args=['--rm', *PLAT], timeout=180)
    env = get_environment(env_cfg)
    mcfg = dict(cfg['model'], model_name=model_name, cost_tracking='default',
                litellm_model_registry=str(ROOT / 'scripts' / 'model_registry.json'))
    mcfg['model_kwargs'] = {k2: v for k2, v in mcfg.get('model_kwargs', {}).items() if k2 != 'parallel_tool_calls'}
    agent_cfg = dict(cfg['agent'], agent_class='default', cost_limit=0.35, step_limit=120, output_path=str(out))
    agent = get_agent(get_model(config=mcfg), env, agent_cfg)
    try:
        result = agent.run(task().problem_statement)
    finally:
        print(json.dumps({'cond': cond, 'k': k, 'cost': round(agent.cost, 4), 'calls': agent.n_calls}))
    return result


def submission(traj):
    info = json.load(open(traj)).get('info', {})
    return info.get('submission') or '', info.get('exit_status')


def grade():
    t = task()
    f2p, p2p = json.loads(t.FAIL_TO_PASS), json.loads(t.PASS_TO_PASS)
    tests = {c: carried_tests(c) for c in ('original', 'trap', 'placebo')}
    rows = []
    for traj in sorted(RUNS.glob('*.traj.json')):
        cond = traj.name.split('-')[0]
        patch, status = submission(traj)
        row = {'run': traj.name.replace('.traj.json', ''), 'exit': status}
        if not patch.strip():
            row['result'] = 'no patch'
            rows.append(row)
            continue
        cid = sh('docker', 'run', '-d', *PLAT, f'rr/django-11149:{cond}', 'sleep', '1800').stdout.strip()
        try:
            for name, p in (('tests', tests[cond]), ('fix', patch)):
                with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
                    f.write(p if p.endswith('\n') else p + '\n')
                sh('docker', 'cp', f.name, f'{cid}:/tmp/{name}.diff')
            r = sh('docker', 'exec', cid, 'bash', '-c',
                   'cd /testbed && git apply /tmp/tests.diff && (git apply /tmp/fix.diff || patch -p1 < /tmp/fix.diff)',
                   check=False)
            if r.returncode:
                row['result'] = 'patch did not apply'
                rows.append(row)
                continue
            r = sh('docker', 'exec', cid, 'bash', '-c',
                   'source /opt/miniconda3/bin/activate testbed 2>/dev/null; cd /testbed/tests && '
                   'python runtests.py admin_inlines --parallel 1 -v 2', check=False)
            res = {}
            for line in (r.stdout + r.stderr).split('\n'):
                if ' ... ' in line:
                    a, _, b = line.partition(' ... ')
                    res[a.strip()] = b.strip()
            ok = lambda names: sum(1 for n in names if res.get(n) == 'ok')
            fails = [n for n, s in res.items() if s in ('FAIL', 'ERROR')]
            row.update(F2P=f'{ok(f2p)}/{len(f2p)}', failing_tests=len(fails),
                       resolved=ok(f2p) == len(f2p) and not fails)
        finally:
            sh('docker', 'rm', '-f', cid, check=False)
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(RUNS / 'grades.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        build()
    elif cmd == 'run':
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == 'grade':
        grade()
