"""Memory-trap pipeline for any configured SWE-bench task (argument-order traps), all in Docker.

  prepare IID            worktrees for original/trap/placebo, carried gold/test/memory patches,
                         Docker images rr/<iid>:<cond> (edit applied, git history squashed)
  validate IID           run the task's tests in each version: no fix / carried gold / memory fix
  run IID COND K MODEL   one mini-swe-agent run -> outputs/runs/<iid>/<cond>-<k>.traj.json
  grade IID              grade every run of that task
  analyze IID            what each run did around the trapped function
Usage: .venv/bin/python -I scripts/trap_pipeline.py prepare sympy__sympy-13615
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAT = ['--platform', 'linux/amd64']
ACT = 'source /opt/miniconda3/bin/activate testbed 2>/dev/null; '
TASKS = {
    'django__django-11149': dict(def_file='django/contrib/auth/__init__.py', func='get_permission_codename',
                                 tests='cd /testbed/tests && python runtests.py admin_inlines --parallel 1 -v 2',
                                 parser='django'),
    'sympy__sympy-13615': dict(def_file='sympy/utilities/iterables.py', func='sift',
                               tests='cd /testbed && bin/test -C --verbose sympy/sets/tests/test_sets.py',
                               parser='sympy'),
}
G = ['git', '-c', 'user.name=dev', '-c', 'user.email=dev@example.com']


def sh(*cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode:
        raise RuntimeError(f'{cmd[:6]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return r


def task(iid):
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / 'data/meta/swebv/data/test-00000-of-00001.parquet').to_pandas().set_index('instance_id')
    return t.loc[iid]


def paths(iid):
    t = task(iid)
    return (t, TASKS[iid], ROOT / 'data/repos' / t.repo.replace('/', '_'), ROOT / 'work' / iid,
            ROOT / 'outputs' / 'runs' / (iid + os.environ.get('RR_TAG', '')), 'rr/' + iid.replace('__', '_1776_').lower())


def apply_patch(d, patch):
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
        f.write(patch if patch.endswith('\n') else patch + '\n')
    sh('git', 'apply', '--whitespace=nowarn', f.name, cwd=d)


def prepare(iid):
    t, cfg, repo, work, runs, image = paths(iid)
    work.mkdir(parents=True, exist_ok=True)

    def fresh(name):
        d = work / name
        if d.exists():
            sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
        sh('git', '-C', str(repo), 'worktree', 'add', '--detach', str(d), t.base_commit)
        return d

    def edit(d, mode):
        if mode != 'original':
            return sh(sys.executable, '-I', str(ROOT / 'scripts/make_argswap_trap.py'), str(d),
                      cfg['def_file'], cfg['func'], mode).stdout.strip()
        return 'no edit'

    pat = re.compile(rf"({cfg['func']}\()\s*([^,()]+?)\s*,\s*([^,()]+?)\s*([,)])")
    for mode in ('original', 'trap', 'placebo'):
        base = fresh(mode)
        note = edit(base, mode)
        sh(*G, 'add', '-A', cwd=base)
        sh(*G, 'commit', '-q', '--allow-empty', '-m', mode, cwd=base)
        vc = sh('git', 'rev-parse', 'HEAD', cwd=base).stdout.strip()
        (work / f'{mode}.edit.diff').write_text(sh('git', 'diff', 'HEAD~1', 'HEAD', cwd=base).stdout)
        for name, patch in (('gold', t.patch), ('tests', t.test_patch)):
            d = fresh(f'{mode}-carry')
            apply_patch(d, patch)
            edit(d, mode)
            sh(*G, 'add', '-A', cwd=d)
            (work / f'{mode}.{name}.diff').write_text(sh('git', 'diff', '--cached', vc, cwd=d).stdout)
            sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
        gold = (work / f'{mode}.gold.diff').read_text()
        memory = '\n'.join(pat.sub(r'\1\3, \2\4', l) if (mode == 'trap' and l.startswith('+')
                           and not l.startswith('+++')) else l for l in gold.split('\n'))
        (work / f'{mode}.memory.diff').write_text(memory)
        # Docker image: official SWE-bench image + edit + one-commit history
        base_image = f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_').lower()}:latest"
        cid = sh('docker', 'run', '-d', *PLAT, base_image, 'sleep', '3600').stdout.strip()
        try:
            if mode != 'original':
                sh('docker', 'cp', str(work / f'{mode}.edit.diff'), f'{cid}:/tmp/edit.diff')
                sh('docker', 'exec', cid, 'bash', '-c', 'cd /testbed && git apply /tmp/edit.diff && rm /tmp/edit.diff')
            sh('docker', 'exec', cid, 'bash', '-c', 'cd /testbed && rm -rf .git && git init -q && git add -A && '
               'git -c user.name=dev -c user.email=dev@example.com commit -qm "Initial commit"')
            sh('docker', 'commit', cid, f'{image}:{mode}')
        finally:
            sh('docker', 'rm', '-f', cid, check=False)
        changed = sum(1 for l in (work / f'{mode}.edit.diff').read_text().split('\n')
                      if l[:1] in ('+', '-') and l[:3] not in ('+++', '---'))
        print(json.dumps({'mode': mode, 'edit': note, 'changed_lines': changed}), flush=True)


def run_tests(iid, cond, patches):
    t, cfg, repo, work, runs, image = paths(iid)
    cid = sh('docker', 'run', '-d', *PLAT, f'{image}:{cond}', 'sleep', '1800').stdout.strip()
    try:
        for i, p in enumerate(patches):
            with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
                f.write(p if p.endswith('\n') else p + '\n')
            sh('docker', 'cp', f.name, f'{cid}:/tmp/p{i}.diff')
            r = sh('docker', 'exec', cid, 'bash', '-c',
                   f'cd /testbed && (git apply /tmp/p{i}.diff || patch -p1 --forward < /tmp/p{i}.diff)', check=False)
            if r.returncode:
                return None
        r = sh('docker', 'exec', cid, 'bash', '-c', ACT + cfg['tests'], check=False)
    finally:
        sh('docker', 'rm', '-f', cid, check=False)
    res = {}
    for line in (r.stdout + r.stderr).split('\n'):
        if cfg['parser'] == 'django' and ' ... ' in line:
            a, _, b = line.partition(' ... ')
            res[a.strip()] = b.strip()
        elif cfg['parser'] == 'sympy':
            m = re.match(r'^(test_\w+)\s+(ok|F|E|f|X|Skipped.*)\s*$', line.strip())
            if m:
                res[m.group(1)] = 'ok' if m.group(2) in ('ok', 'f') else m.group(2)
    return res


def score(iid, res):
    t = task(iid)
    f2p, p2p = json.loads(t.FAIL_TO_PASS), json.loads(t.PASS_TO_PASS)
    if res is None:
        return {'result': 'patch did not apply'}
    ok = lambda names: sum(1 for n in names if res.get(n) == 'ok')
    fails = [n for n, s in res.items() if s in ('FAIL', 'ERROR', 'F', 'E')]
    return {'F2P': f'{ok(f2p)}/{len(f2p)}', 'P2P': f'{ok(p2p)}/{len(p2p)}', 'failing': len(fails),
            'resolved': ok(f2p) == len(f2p) and ok(p2p) == len(p2p)}


def validate(iid):
    t, cfg, repo, work, runs, image = paths(iid)
    out = {}
    for cond in ('original', 'trap', 'placebo'):
        rd = lambda n: (work / f'{cond}.{n}.diff').read_text()
        out[cond] = {label: score(iid, run_tests(iid, cond, ps)) for label, ps in
                     (('no fix', [rd('tests')]), ("gold'", [rd('tests'), rd('gold')]),
                      ('memory fix', [rd('tests'), rd('memory')]))}
        print(cond, json.dumps(out[cond]), flush=True)
    json.dump(out, open(work / 'validation.json', 'w'), indent=1)


def run(iid, cond, k, model_name):
    import yaml
    from minisweagent.agents import get_agent
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    t, cfg, repo, work, runs, image = paths(iid)
    c = yaml.safe_load(open(builtin_config_dir / 'benchmarks' / 'swebench.yaml'))
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / f'{cond}-{k}.traj.json'
    env = get_environment(dict(c['environment'], environment_class='docker', image=f'{image}:{cond}',
                               run_args=['--rm', *PLAT, *(['--network', 'none'] if os.environ.get('RR_NONET') else [])],
                               timeout=180))
    mcfg = dict(c['model'], model_name=model_name,
                litellm_model_registry=str(ROOT / 'scripts' / 'model_registry.json'))
    mcfg['model_kwargs'] = {a: b for a, b in mcfg.get('model_kwargs', {}).items() if a != 'parallel_tool_calls'}
    agent = get_agent(get_model(config=mcfg), env,
                      dict(c['agent'], agent_class='default', cost_limit=0.35, step_limit=120, output_path=str(out)))
    try:
        agent.run(t.problem_statement)
    finally:
        print(json.dumps({'iid': iid, 'cond': cond, 'k': k, 'cost': round(agent.cost, 4), 'calls': agent.n_calls}))


def grade(iid):
    t, cfg, repo, work, runs, image = paths(iid)
    rows = []
    for traj in sorted(runs.glob('*.traj.json')):
        cond = traj.name.split('-')[0]
        info = json.load(open(traj)).get('info', {})
        patch = info.get('submission') or ''
        row = {'run': traj.name.replace('.traj.json', ''), 'exit': info.get('exit_status')}
        row.update(score(iid, run_tests(iid, cond, [(work / f'{cond}.tests.diff').read_text(), patch]))
                   if patch.strip() else {'result': 'no patch'})
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(runs / 'grades.json', 'w'), indent=1)


def first_arg(text):
    depth, j = 0, 0
    while j < len(text):
        c = text[j]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth == 0:
                break
            depth -= 1
        elif c == ',' and depth == 0:
            break
        j += 1
    a = text[:j].strip()
    return 'keywords' if re.match(r'^\w+\s*=(?!=)', a) else f'first arg: {a[:30]}'


def analyze(iid):
    t, cfg, repo, work, runs, image = paths(iid)
    func = cfg['func']
    call = re.compile(func + r'\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*[,)]')
    mem = re.compile(r'(from memory|recall|remember|upstream|modern (?:Django|SymPy|sympy)|newer version)', re.I)
    p0 = re.search(rf'def {func}\((\w+)', (work / 'original' / cfg['def_file']).read_text()).group(1)
    rows = []
    for f in sorted(runs.glob('*.traj.json')):
        tr = json.load(open(f))
        msgs = tr['messages']
        steps, saw_def, saw_call, first_use = 0, None, None, None
        hid = vis = 0
        for i, m in enumerate(msgs):
            if m.get('role') != 'assistant':
                continue
            steps += 1
            obs = '\n'.join(str(n.get('content') or '') for n in msgs[i + 1:i + 6] if n.get('role') in ('tool', 'user'))
            if saw_def is None and f'def {func}(' in obs:
                saw_def = steps
            if saw_call is None and call.search(obs):
                saw_call = steps
            cmds = ' '.join((tc.get('function') or {}).get('arguments', '') for tc in (m.get('tool_calls') or []))
            if first_use is None and f'{func}(' in cmds:
                first_use = steps
            hid += bool(mem.search(m.get('reasoning_content') or ''))
            vis += bool(mem.search(m.get('content') or ''))
        patch = tr.get('info', {}).get('submission') or ''
        added = '\n'.join(l for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
        orders = sorted({first_arg(added[m.end():]) for m in re.finditer(rf'\b{func}\(', added)})
        rows.append(dict(run=f.name.replace('.traj.json', ''), steps=steps, saw_definition_at=saw_def,
                         saw_other_call_at=saw_call, first_wrote_call_at=first_use,
                         patch_calls=orders or ['no call in patch'], memory_steps_hidden=hid, memory_steps_visible=vis))
        print(json.dumps(rows[-1]))
    json.dump(rows, open(runs / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd, iid = sys.argv[1], sys.argv[2]
    {'prepare': lambda: prepare(iid), 'validate': lambda: validate(iid), 'grade': lambda: grade(iid),
     'analyze': lambda: analyze(iid), 'run': lambda: run(iid, *sys.argv[3:6])}[cmd]()
