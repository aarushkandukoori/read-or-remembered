"""Agent-targeted memory traps, network off (all agent containers run with --network none).

  image IID COND            build rr-targeted/<iid>:<cond> (cond=original, or trap/placebo after prepare)
  run IID COND K            one DeepSeek (deepseek-flash) mini-swe-agent run -> runs/<iid>/<cond>-<k>.traj.json
  baseline IID              what the original runs called in their own patches, and what they had seen
  prepare IID KIND FUNC DEF_FILE [PARAM NEWNAME]
                            KIND=argswap or kwrename; builds trap + placebo worktrees, carried gold/test
                            patches, the "own patch" (the agent's baseline patch) and images
  validate IID              tests: no fix / carried gold / agent's own baseline patch, in each version
  grade IID                 grade all runs for the task
Data from ~/Projects/read-or-remembered is only read, never written.
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RR = Path.home() / 'Projects' / 'read-or-remembered'
PLAT = ['--platform', 'linux/amd64']
ACT = 'source /opt/miniconda3/bin/activate testbed 2>/dev/null; '
G = ['git', '-c', 'user.name=dev', '-c', 'user.email=dev@example.com']
MODEL = 'deepseek/deepseek-flash'


def sh(*cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode:
        raise RuntimeError(f'{cmd[:6]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return r


def task(iid):
    import pyarrow.parquet as pq
    t = pq.read_table(RR / 'data/meta/swebv/data/test-00000-of-00001.parquet').to_pandas().set_index('instance_id')
    return t.loc[iid]


def test_cmd(t):
    files = [l[6:] for l in t.test_patch.split('\n') if l.startswith('+++ b/')]
    if t.repo == 'django/django':
        labels = sorted({f[len('tests/'):-3].replace('/', '.') for f in files if f.startswith('tests/') and f.endswith('.py')})
        return 'cd /testbed/tests && python runtests.py ' + ' '.join(labels) + ' --parallel 1 -v 2', 'django'
    return 'cd /testbed && bin/test -C --verbose ' + ' '.join(f for f in files if f.endswith('.py')), 'sympy'


def base_image(iid):
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_').lower()}:latest"


def my_image(iid, cond):
    return f"rr-targeted/{iid.replace('__', '_1776_').lower()}:{cond}"


def work(iid):
    return HERE / 'work' / iid


def build_image(iid, cond):
    cid = sh('docker', 'run', '-d', *PLAT, base_image(iid), 'sleep', '3600').stdout.strip()
    try:
        if cond != 'original':
            sh('docker', 'cp', str(work(iid) / f'{cond}.edit.diff'), f'{cid}:/tmp/edit.diff')
            sh('docker', 'exec', cid, 'bash', '-c', 'cd /testbed && git apply /tmp/edit.diff && rm /tmp/edit.diff')
        sh('docker', 'exec', cid, 'bash', '-c', 'cd /testbed && rm -rf .git && git init -q && git add -A && '
           'git -c user.name=dev -c user.email=dev@example.com commit -qm "Initial commit"')
        sh('docker', 'commit', cid, my_image(iid, cond))
    finally:
        sh('docker', 'rm', '-f', cid, check=False)
    print('built', my_image(iid, cond))


def run(iid, cond, k):
    import yaml
    from minisweagent.agents import get_agent
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    t = task(iid)
    c = yaml.safe_load(open(builtin_config_dir / 'benchmarks' / 'swebench.yaml'))
    out_dir = HERE / 'runs' / iid
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{cond}-{k}.traj.json'
    env = get_environment(dict(c['environment'], environment_class='docker', image=my_image(iid, cond),
                               run_args=['--rm', *PLAT, '--network', 'none'], timeout=180))
    mcfg = dict(c['model'], model_name=MODEL, litellm_model_registry=str(HERE / 'scripts' / 'model_registry.json'))
    mcfg['model_kwargs'] = {a: b for a, b in mcfg.get('model_kwargs', {}).items() if a != 'parallel_tool_calls'}
    agent = get_agent(get_model(config=mcfg), env,
                      dict(c['agent'], agent_class='default', cost_limit=0.35, step_limit=120, output_path=str(out)))
    try:
        agent.run(t.problem_statement)
    finally:
        print(json.dumps({'iid': iid, 'cond': cond, 'k': k, 'cost': round(agent.cost, 4), 'calls': agent.n_calls}))


def run_tests(iid, cond, patches):
    t = task(iid)
    cmd, parser = test_cmd(t)
    cid = sh('docker', 'run', '-d', *PLAT, '--network', 'none', my_image(iid, cond), 'sleep', '1800').stdout.strip()
    try:
        for i, p in enumerate(patches):
            with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
                f.write(p if p.endswith('\n') else p + '\n')
            sh('docker', 'cp', f.name, f'{cid}:/tmp/p{i}.diff')
            r = sh('docker', 'exec', cid, 'bash', '-c',
                   f'cd /testbed && (git apply /tmp/p{i}.diff || patch -p1 --forward < /tmp/p{i}.diff)', check=False)
            if r.returncode:
                return None
        r = sh('docker', 'exec', cid, 'bash', '-c', ACT + cmd, check=False)
    finally:
        sh('docker', 'rm', '-f', cid, check=False)
    res = {}
    for line in (r.stdout + r.stderr).split('\n'):
        if parser == 'django' and ' ... ' in line:
            a, _, b = line.partition(' ... ')
            res[a.strip()] = b.strip()
        elif parser == 'sympy':
            m = re.match(r'^(test_\w+)\s+(ok|F|E|f|X|Skipped.*)\s*$', line.strip())
            if m:
                res[m.group(1)] = 'ok' if m.group(2) in ('ok', 'f') else m.group(2)
    return res


def score(iid, res):
    t = task(iid)
    f2p, p2p = json.loads(t.FAIL_TO_PASS), json.loads(t.PASS_TO_PASS)
    if res is None:
        return {'result': 'patch did not apply', 'resolved': False}
    ok = lambda names: sum(1 for n in names if res.get(n) == 'ok')
    fails = [n for n, s in res.items() if s in ('FAIL', 'ERROR', 'F', 'E')]
    return {'F2P': f'{ok(f2p)}/{len(f2p)}', 'P2P': f'{ok(p2p)}/{len(p2p)}', 'failing': len(fails),
            'resolved': ok(f2p) == len(f2p) and ok(p2p) == len(p2p)}


def submission(path):
    info = json.load(open(path)).get('info', {})
    return info.get('submission') or ''


def grade(iid):
    rows = []
    for traj in sorted((HERE / 'runs' / iid).glob('*.traj.json')):
        cond = traj.name.split('-')[0]
        patch = submission(traj)
        tests = (work(iid) / f'{cond}.tests.diff').read_text() if cond != 'original' else task(iid).test_patch
        row = {'run': traj.name.replace('.traj.json', '')}
        row.update(score(iid, run_tests(iid, cond, [tests, patch])) if patch.strip() else {'result': 'no patch', 'resolved': False})
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(HERE / 'runs' / iid / 'grades.json', 'w'), indent=1)


def calls_in(text, func):
    return len(re.findall(rf'(?<![\w]){re.escape(func)}\(', text))


def baseline(iid, funcs):
    """For each original run: does its patch call FUNC, had it seen the def or other call sites first?"""
    for traj in sorted((HERE / 'runs' / iid).glob('original-*.traj.json')):
        tr = json.load(open(traj))
        patch = submission(traj)
        added = '\n'.join(l for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
        obs = '\n'.join(str(m.get('content') or '') for m in tr['messages'] if m.get('role') in ('tool', 'user'))
        for f in funcs:
            print(json.dumps({'run': traj.name, 'func': f, 'patch_calls': calls_in(added, f),
                              'saw_def': f'def {f}(' in obs, 'call_sites_seen': calls_in(obs, f),
                              'lines': [l.strip()[:110] for l in added.split('\n') if f + '(' in l][:3]}))


def prepare(iid, kind, func, def_file, param=None, newname=None):
    t = task(iid)
    w = work(iid)
    w.mkdir(parents=True, exist_ok=True)
    repo = HERE / 'repos' / t.repo.replace('/', '_')

    def fresh(name):
        d = w / name
        if d.exists():
            sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
        sh('git', '-C', str(repo), 'worktree', 'add', '--detach', str(d), t.base_commit)
        return d

    def edit(d, mode):
        if kind == 'argswap':
            return sh(sys.executable, '-I', str(HERE / 'scripts/make_argswap_trap.py'), str(d), def_file, func, mode).stdout.strip()
        return sh(sys.executable, '-I', str(HERE / 'scripts/make_kwrename_trap.py'), str(d), def_file, func, param, newname, mode).stdout.strip()

    for mode in ('trap', 'placebo'):
        base = fresh(mode)
        note = edit(base, mode)
        sh(*G, 'add', '-A', cwd=base)
        sh(*G, 'commit', '-q', '-m', mode, cwd=base)
        vc = sh('git', 'rev-parse', 'HEAD', cwd=base).stdout.strip()
        (w / f'{mode}.edit.diff').write_text(sh('git', 'diff', 'HEAD~1', 'HEAD', cwd=base).stdout)
        for name, patch in (('gold', t.patch), ('tests', t.test_patch)):
            d = fresh(f'{mode}-carry')
            with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
                f.write(patch)
            sh('git', 'apply', f.name, cwd=d)
            edit(d, mode)
            sh(*G, 'add', '-A', cwd=d)
            (w / f'{mode}.{name}.diff').write_text(sh('git', 'diff', '--cached', vc, cwd=d).stdout)
            sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
        changed = sum(1 for l in (w / f'{mode}.edit.diff').read_text().split('\n') if l[:1] in '+-' and l[:3] not in ('+++', '---'))
        print(json.dumps({'mode': mode, 'edit': note, 'changed_lines': changed}), flush=True)
        build_image(iid, mode)
    json.dump(dict(kind=kind, func=func, def_file=def_file, param=param, newname=newname), open(w / 'trap.json', 'w'))


def carried_own(iid, cond, own_patch):
    """The agent's own baseline patch carried into this version (same edit applied on top), and the
    'memory' variant: that carried patch with the agent's own new calls put back in the original form."""
    t = task(iid)
    w = work(iid)
    cfg = json.load(open(w / 'trap.json'))
    repo = HERE / 'repos' / t.repo.replace('/', '_')
    d = w / f'{cond}-own'
    if d.exists():
        sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
    sh('git', '-C', str(repo), 'worktree', 'add', '--detach', str(d), t.base_commit)
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
        f.write(own_patch if own_patch.endswith('\n') else own_patch + '\n')
    sh('git', 'apply', f.name, cwd=d)
    script = 'make_argswap_trap.py' if cfg['kind'] == 'argswap' else 'make_kwrename_trap.py'
    extra = [] if cfg['kind'] == 'argswap' else [cfg['param'], cfg['newname']]
    sh(sys.executable, '-I', str(HERE / 'scripts' / script), str(d), cfg['def_file'], cfg['func'], *extra, cond)
    sh(*G, 'add', '-A', cwd=d)
    vc = sh('git', 'rev-parse', 'HEAD', cwd=w / cond).stdout.strip()
    carried = sh('git', 'diff', '--cached', vc, cwd=d).stdout
    sh('git', '-C', str(repo), 'worktree', 'remove', '--force', str(d))
    func = cfg['func']
    if cfg['kind'] == 'argswap':
        pat = re.compile(rf'({func}\()\s*([^,()]+(?:\([^()]*\))?[^,()]*?)\s*,\s*([^,()]+(?:\[[^\]]*\])?(?:\([^()]*\))?)\s*([,)])')
        undo = lambda l: pat.sub(r'\1\3, \2\4', l)
    else:
        undo = lambda l: re.sub(rf'({func}\([^\n]*?)\b{cfg["newname"]}=', rf'\1{cfg["param"]}=', l)
    memory = '\n'.join(undo(l) if cond == 'trap' and l.startswith('+') and not l.startswith('+++') else l
                       for l in carried.split('\n'))
    return carried, memory


def validate(iid):
    t = task(iid)
    w = work(iid)
    own = sorted((HERE / 'runs' / iid).glob('original-*.traj.json'))
    own_patch = submission(own[0]) if own else ''
    out = {}
    for cond in ('original', 'trap', 'placebo'):
        tests = t.test_patch if cond == 'original' else (w / f'{cond}.tests.diff').read_text()
        gold = t.patch if cond == 'original' else (w / f'{cond}.gold.diff').read_text()
        runs_ = [('no fix', [tests]), ("gold'", [tests, gold])]
        if cond == 'original':
            runs_.append(("agent's own baseline patch", [tests, own_patch]))
        else:
            carried, memory = carried_own(iid, cond, own_patch)
            (w / f'{cond}.own_carried.diff').write_text(carried)
            (w / f'{cond}.own_memory.diff').write_text(memory)
            runs_ += [("agent's patch, carried over", [tests, carried]),
                      ("agent's patch as written from memory", [tests, memory])]
        out[cond] = {label: score(iid, run_tests(iid, cond, ps)) for label, ps in runs_}
        print(cond, json.dumps(out[cond]), flush=True)
    json.dump(out, open(w / 'validation.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd, iid, *rest = sys.argv[1:]
    if cmd == 'image':
        build_image(iid, rest[0])
    elif cmd == 'run':
        run(iid, rest[0], rest[1])
    elif cmd == 'baseline':
        baseline(iid, rest)
    elif cmd == 'prepare':
        prepare(iid, *rest)
    elif cmd == 'validate':
        validate(iid)
    elif cmd == 'grade':
        grade(iid)
