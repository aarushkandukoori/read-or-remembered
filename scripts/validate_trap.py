"""Build trap + placebo versions of one SWE-bench task and check them with the task's own tests.

For each version (original, trap, placebo):
  base     the repo at the task's base commit with the edit applied (no edit for original)
  gold'    the gold fix carried into that version: edit(base + gold) minus edit(base)
  tests'   the task's test changes, carried over the same way
  memory   gold' with the fix's own new calls written in the ORIGINAL argument order, i.e. the fix
           as someone would write it from memory of the original repo
Runs: no fix (FAIL_TO_PASS should fail), gold' (all should pass), memory (should pass in the
placebo and fail in the trap: the trap "bites").
Usage: python -I validate_trap.py INSTANCE_ID DEF_FILE FUNC PYTHON TEST_LABEL
"""
import json, os, re, subprocess, sys, tempfile
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
iid, def_file, func, python, test_label = sys.argv[1:6]
t = pq.read_table(os.path.join(ROOT, 'data/meta/swebv/data/test-00000-of-00001.parquet')).to_pandas().set_index('instance_id')
task = t.loc[iid]
repo = os.path.join(ROOT, 'data/repos', task.repo.replace('/', '_'))
work = os.path.join(ROOT, 'work', iid)
os.makedirs(work, exist_ok=True)
G = ['git', '-c', 'user.name=trap', '-c', 'user.email=trap@example.com']


def sh(*cmd, cwd=None, check=True, **kw):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)
    if check and r.returncode:
        raise RuntimeError(f'{cmd} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return r


def fresh(name, commit):
    d = os.path.join(work, name)
    if os.path.exists(d):
        sh('git', '-C', repo, 'worktree', 'remove', '--force', d)
    sh('git', '-C', repo, 'worktree', 'add', '--detach', d, commit)
    return d


def apply_patch(d, patch):
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
        f.write(patch)
    sh('git', 'apply', '--whitespace=nowarn', f.name, cwd=d)


def edit(d, mode):
    if mode == 'original':
        return 'no edit'
    return sh(sys.executable, '-I', os.path.join(ROOT, 'scripts/make_argswap_trap.py'), d, def_file, func, mode).stdout.strip()


def carried(mode, version_commit, patch):
    d = fresh(f'{mode}-carry', task.base_commit)
    apply_patch(d, patch)
    edit(d, mode)
    sh(*G, 'add', '-A', cwd=d)
    return sh('git', 'diff', '--cached', version_commit, cwd=d).stdout


def from_memory(gold):
    """Swap the first two arguments back on the fix's added lines (original order)."""
    pat = re.compile(rf'({func}\()\s*([^,()]+?)\s*,\s*([^,()]+?)\s*([,)])')
    return '\n'.join(pat.sub(r'\1\3, \2\4', l) if l.startswith('+') and not l.startswith('+++') else l
                     for l in gold.split('\n'))


def run_tests(d):
    env = dict(os.environ, PYTHONPATH=d)
    r = sh(python, 'runtests.py', test_label, '--parallel', '1', '-v', '2', cwd=os.path.join(d, 'tests'),
           check=False, env=env)
    res = {}
    for line in (r.stderr + r.stdout).split('\n'):
        if ' ... ' in line:  # tests with a docstring print the docstring instead of the name
            name, _, status = line.partition(' ... ')
            res[name.strip()] = status.strip()
    return res


f2p, p2p = json.loads(task.FAIL_TO_PASS), json.loads(task.PASS_TO_PASS)
summary = {}
for mode in ('original', 'trap', 'placebo'):
    base = fresh(mode, task.base_commit)
    note = edit(base, mode)
    sh(*G, 'add', '-A', cwd=base)
    sh(*G, 'commit', '-q', '--allow-empty', '-m', mode, cwd=base)
    vc = sh('git', 'rev-parse', 'HEAD', cwd=base).stdout.strip()
    edit_diff = sh('git', 'diff', 'HEAD~1', 'HEAD', cwd=base).stdout
    gold = carried(mode, vc, task.patch)
    tests = carried(mode, vc, task.test_patch)
    memory = from_memory(gold) if mode == 'trap' else gold
    open(os.path.join(work, f'{mode}.gold.diff'), 'w').write(gold)
    open(os.path.join(work, f'{mode}.memory.diff'), 'w').write(memory)
    rows = {}
    for label, patches in (('no fix', [tests]), ("gold'", [tests, gold]), ('memory fix', [tests, memory])):
        d = fresh(f'{mode}-run', vc)
        try:
            for p in patches:
                apply_patch(d, p)
        except RuntimeError:
            rows[label] = 'patch did not apply'
            continue
        res = run_tests(d)
        ok = lambda names: sum(1 for n in names if res.get(n) == 'ok')
        rows[label] = f'F2P {ok(f2p)}/{len(f2p)}, P2P {ok(p2p)}/{len(p2p)}'
    changed = sum(1 for l in edit_diff.split('\n') if l[:1] in ('+', '-') and l[:3] not in ('+++', '---'))
    summary[mode] = dict(edit=note, changed_lines=changed, **rows)
    print(mode, json.dumps(summary[mode]), flush=True)
json.dump(summary, open(os.path.join(ROOT, 'outputs', f'trap_validation_{iid}.json'), 'w'), indent=1)
