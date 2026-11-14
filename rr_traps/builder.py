"""Build one catalog entry: a (SWE-bench task, trap) pair.

For a task and a transform, in a git worktree of the task's repo at the base commit
(work/catalog/<instance_id>/wt), we write to traps/catalog/<instance_id>/<trap_id>/:

  trap.diff, placebo.diff      the trap edit and the placebo edit of the repo (vs the base commit)
  gold_trap.diff, tests_trap.diff, gold_placebo.diff, tests_placebo.diff
                               the task's gold patch and test patch carried through the same edit:
                               edit(base + patch) minus edit(base). They apply on top of trap.diff
                               (or placebo.diff).
  memory.diff                  the upstream gold patch, verbatim: the fix a model would write from memory
  spec.json                    what was changed, why, sizes, and static checks
  report.json                  every rewritten site and every warning (for review)

Static checks recorded in spec.json (no tests are run here; see localcheck.py and the Docker validator):
  memory_applies     the verbatim gold patch applies on the trap / placebo version (git apply --check)
  carried_applies    tests + carried gold apply on their version
  syntax_ok          every .py file the edit touched still compiles
  gold_lines_changed_by_trap   added lines of the gold patch that the trap rewrote (0 = the trap does not
                               touch the fix, so it cannot matter for the task)
  visibility         how close the trap's edits come to the lines the gold patch edits
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

from . import common as C
from .transforms import make

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, 'data/meta/swebv/data/test-00000-of-00001.parquet')
REPOS = os.path.join(ROOT, 'data/repos')
WORK = os.path.join(ROOT, 'work/catalog')
OUT = os.path.join(ROOT, 'traps/catalog')
GIT = ['git', '-c', 'user.name=trap', '-c', 'user.email=trap@example.com', '-c', 'core.hooksPath=/dev/null',
       '-c', 'advice.detachedHead=false']

_tasks = None


def task(iid):
    """One SWE-bench Verified row (as a dict)."""
    global _tasks
    if _tasks is None:
        import pyarrow.parquet as pq
        _tasks = pq.read_table(PARQUET).to_pandas().set_index('instance_id')
    return dict(_tasks.loc[iid], instance_id=iid)


def sh(*cmd, cwd=None, check=True, input=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, input=input)
    if check and r.returncode:
        raise RuntimeError(f'{" ".join(cmd[:8])} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    return r


def repo_dir(t):
    return os.path.join(REPOS, t['repo'].replace('/', '_'))


def worktree(t):
    """A worktree of the task's repo at its base commit (created once, reused)."""
    wt = os.path.join(WORK, t['instance_id'], 'wt')
    if not os.path.exists(os.path.join(wt, '.git')):
        os.makedirs(os.path.dirname(wt), exist_ok=True)
        sh('git', '-C', repo_dir(t), 'worktree', 'add', '-f', '--detach', wt, t['base_commit'])
    return wt


def remove_worktree(t):
    wt = os.path.join(WORK, t['instance_id'], 'wt')
    if os.path.exists(wt):
        sh('git', '-C', repo_dir(t), 'worktree', 'remove', '--force', wt, check=False)
    sh('git', '-C', repo_dir(t), 'worktree', 'prune', check=False)


def reset(wt, commit):
    """Throw away every change: the worktree is exactly `commit` again."""
    sh(*GIT, 'checkout', '-q', '-f', '--detach', commit, cwd=wt)
    sh(*GIT, 'clean', '-qfdx', cwd=wt)


def apply_patch(wt, patch, check_only=False):
    """git apply a patch text; returns (ok, message)."""
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as f:
        f.write(patch if patch.endswith('\n') else patch + '\n')
        name = f.name
    args = ['apply', '--whitespace=nowarn'] + (['--check'] if check_only else []) + [name]
    r = sh(*GIT, *args, cwd=wt, check=False)
    os.unlink(name)
    return r.returncode == 0, (r.stderr or r.stdout).strip()[-500:]


def syntax_errors(wt, files):
    """Python files (of `files`) that no longer compile."""
    bad = []
    for rel in files:
        if rel.endswith('.py') and os.path.exists(os.path.join(wt, rel)):
            try:
                compile(C.read(wt, rel), rel, 'exec', dont_inherit=True)
            except SyntaxError as e:
                bad.append(f'{rel}:{e.lineno}: {e.msg}')
    return bad


def added_lines(diff_text):
    return [line[1:] for line in diff_text.split('\n') if line.startswith('+') and not line.startswith('+++')]


def hunk_ranges(diff_text):
    """{file: [(start, end)]} old-side line ranges of each hunk of a unified diff."""
    out, cur = {}, None
    for line in diff_text.split('\n'):
        if line.startswith('+++ '):
            p = line[4:].strip()
            cur = p[2:] if p.startswith('b/') else p
            out.setdefault(cur, [])
        elif line.startswith('@@') and cur:
            m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if m:
                s, n = int(m.group(1)), int(m.group(2) or 1)
                out[cur].append((s, s + max(n, 1) - 1))
    return out


def changed_line_numbers(diff_text):
    """{file: [old-side line numbers of removed lines (or insertion points)]} of a unified diff."""
    out, cur, pos = {}, None, 0
    for line in diff_text.split('\n'):
        if line.startswith('+++ '):
            p = line[4:].strip()
            cur = p[2:] if p.startswith('b/') else p
            out.setdefault(cur, [])
        elif line.startswith('@@') and cur:
            m = re.match(r'@@ -(\d+)', line)
            pos = int(m.group(1)) if m else 0
        elif cur and line.startswith('-') and not line.startswith('---'):
            out[cur].append(pos)
            pos += 1
        elif cur and line.startswith('+') and not line.startswith('+++'):
            out[cur].append(pos)
        elif cur and line.startswith(' '):
            pos += 1
    return out


def visibility(trap_diff, gold_patch):
    """How close the trap's edits come to the code the gold patch edits (rule 3 of the design: the new
    fact should not sit right next to the edit, or the agent just copies it)."""
    gold = hunk_ranges(gold_patch)
    trap = changed_line_numbers(trap_diff)
    in_files, nearest = 0, None
    for f, ranges in gold.items():
        lines = trap.get(f, [])
        in_files += len(lines)
        for ln in lines:
            for a, b in ranges:
                d = 0 if a <= ln <= b else min(abs(ln - a), abs(ln - b))
                nearest = d if nearest is None else min(nearest, d)
    return dict(trap_lines_in_gold_files=in_files, nearest_trap_line_to_gold_hunk=nearest,
                gold_files=sorted(gold))


def build(entry, out_root=OUT, keep_worktree=True, log=print):
    """Build one (task, trap) entry. `entry` keys: instance_id, trap_id, transform (a spec dict for
    transforms.make), why_it_makes_sense, why_it_matters_for_task, optional notes."""
    t = task(entry['instance_id'])
    base = t['base_commit']
    out = os.path.join(out_root, entry['instance_id'], entry['trap_id'])
    os.makedirs(out, exist_ok=True)
    wt = worktree(t)
    started = time.time()
    versions, reports, diffs, transforms = {}, {}, {}, {}
    problems = []
    for mode in ('trap', 'placebo'):
        reset(wt, base)
        tr = make(entry['transform'])
        rep = tr.apply(wt, mode)
        transforms[mode] = tr
        bad = syntax_errors(wt, rep.files)
        if bad:
            problems.append(f'{mode}: syntax errors {bad[:5]}')
        sh(*GIT, 'add', '-A', cwd=wt)
        sh(*GIT, 'commit', '-q', '--allow-empty', '-m', f'{mode} edit', cwd=wt)
        versions[mode] = sh('git', 'rev-parse', 'HEAD', cwd=wt).stdout.strip()
        diffs[mode] = sh('git', 'diff', base, versions[mode], cwd=wt).stdout
        C.write(out, f'{mode}.diff', diffs[mode])
        reports[mode] = rep.as_dict()
        reports[mode]['describe'] = tr.describe()
        reports[mode]['spec'] = tr.spec()
        for name, patch in (('gold', t['patch']), ('tests', t['test_patch'])):
            reset(wt, base)
            ok, msg = apply_patch(wt, patch)
            if not ok:
                problems.append(f'{name} patch does not apply at the base commit: {msg}')
                C.write(out, f'{name}_{mode}.diff', '')
                continue
            tr2 = make(entry['transform'])
            rep2 = tr2.apply(wt, mode)
            bad = syntax_errors(wt, rep2.files)
            if bad:
                problems.append(f'{name}_{mode}: syntax errors {bad[:5]}')
            sh(*GIT, 'add', '-A', cwd=wt)
            d = sh('git', 'diff', '--cached', versions[mode], cwd=wt).stdout
            diffs[f'{name}_{mode}'] = d
            C.write(out, f'{name}_{mode}.diff', d)
            reports[mode][f'{name}_warnings'] = [w for w in rep2.warnings if w not in rep.warnings]
    C.write(out, 'memory.diff', t['patch'] if t['patch'].endswith('\n') else t['patch'] + '\n')
    # static checks on each version
    checks = {}
    for mode in ('trap', 'placebo'):
        reset(wt, versions[mode])
        mem_ok, mem_msg = apply_patch(wt, t['patch'], check_only=True)
        ok_t, msg_t = apply_patch(wt, diffs.get(f'tests_{mode}', ''), check_only=False)
        ok_g, msg_g = apply_patch(wt, diffs.get(f'gold_{mode}', ''), check_only=True) if ok_t else (False, 'tests failed')
        checks[mode] = dict(memory_applies=mem_ok, memory_apply_error=None if mem_ok else mem_msg,
                            carried_applies=ok_t and ok_g, carried_apply_error=None if ok_t and ok_g else (msg_t or msg_g))
        if mem_ok:  # the memory patch on this version: does everything still compile?
            reset(wt, versions[mode])
            apply_patch(wt, t['patch'])
            checks[mode]['memory_syntax_errors'] = syntax_errors(wt, C.diff_files(t['patch']))
    reset(wt, base)
    tr = transforms['trap']
    desc = reports['trap']['describe']
    mem_added = added_lines(t['patch'])
    gold_trap_added = set(added_lines(diffs.get('gold_trap', '')))
    gold_placebo_added = set(added_lines(diffs.get('gold_placebo', '')))
    spec = dict(
        instance_id=entry['instance_id'], trap_id=entry['trap_id'], type=tr.type_id, type_title=tr.title,
        loud_or_quiet=entry.get('loud_or_quiet', tr.loud_or_quiet),
        target=dict(file=tr.def_file, qualname=tr.qual, all_definitions=[list(d) for d in tr.defs]),
        target_qualname=tr.qual,
        original_fact=desc['original_fact'], trapped_fact=desc['trapped_fact'],
        placebo_kind=tr.placebo_kind,
        files_changed=sorted(C.diff_files(diffs['trap'])),
        files_changed_placebo=sorted(C.diff_files(diffs['placebo'])),
        lines_changed_trap=C.count_changed_lines(diffs['trap']),
        lines_changed_placebo=C.count_changed_lines(diffs['placebo']),
        sites_trap=reports['trap']['sites_by_kind'], sites_placebo=reports['placebo']['sites_by_kind'],
        why_it_makes_sense=entry['why_it_makes_sense'],
        why_it_matters_for_task=entry['why_it_matters_for_task'],
        gold_lines_changed_by_trap=sum(1 for line in mem_added if line not in gold_trap_added),
        gold_lines_changed_by_placebo=sum(1 for line in mem_added if line not in gold_placebo_added),
        checks=checks,
        visibility=visibility(diffs['trap'], t['patch']),
        warnings=dict(trap=reports['trap']['warnings'][:50], placebo=reports['placebo']['warnings'][:50]),
        problems=problems,
        transform=tr.spec(),
        task=dict(repo=t['repo'], base_commit=base, version=t['version'],
                  FAIL_TO_PASS=json.loads(t['FAIL_TO_PASS']), n_PASS_TO_PASS=len(json.loads(t['PASS_TO_PASS']))),
        notes=entry.get('notes', ''),
        built_with='.venv/bin/python -I scripts/build_trap_catalog.py build ' + entry['instance_id'],
        build_seconds=round(time.time() - started, 1),
    )
    json.dump(spec, open(os.path.join(out, 'spec.json'), 'w'), indent=1)
    json.dump(dict(trap=reports['trap'], placebo=reports['placebo']), open(os.path.join(out, 'report.json'), 'w'),
              indent=1)
    if not keep_worktree:
        remove_worktree(t)
    log(f"{entry['instance_id']} {entry['trap_id']}: trap {spec['lines_changed_trap']} lines / "
        f"placebo {spec['lines_changed_placebo']} lines; memory applies "
        f"{checks['trap']['memory_applies']}/{checks['placebo']['memory_applies']}; gold lines changed by trap "
        f"{spec['gold_lines_changed_by_trap']}; problems {problems}")
    return spec


if __name__ == '__main__':
    sys.exit('use scripts/build_trap_catalog.py')
