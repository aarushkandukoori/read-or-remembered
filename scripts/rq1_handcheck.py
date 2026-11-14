"""Hand-check helper for RQ1: draw random named-before-seen (NBS) mentions and show, for each,
the trajectory text around the mention plus independent checks, so a reader can judge whether
  (1) it really is a statement/use of a repo fact,
  (2) it really was not visible earlier in the run (task text or earlier tool outputs),
  (3) the truth label (real / stale / future / never / match / ...) is right.

Usage:
  .venv/bin/python -I scripts/rq1_handcheck.py sample [N_PER_BENCH] [SEED] [wrong]   # prints the cases
  (add 'wrong' to draw only from named-before-seen facts labeled wrong)
The printout goes to the terminal only (DeepSWE text must not be copied into outputs/ or notes/).
Judgments are written by hand to data/repos_deepswe/_rq1_cache/handcheck_judgments.json and
copied into outputs/rq1/facts_summary.json by "rq1_fact_provenance.py summarize".
"""
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq1_fact_provenance as M  # noqa: E402


def load_rows(bench):
    with open(os.path.join(M.OUT, f'facts_{bench}.jsonl')) as f:
        return [json.loads(l) for l in f]


def window(text, needle, width=260):
    i = text.find(needle)
    if i < 0:
        return None
    return text[max(0, i - width):i + len(needle) + width]


def independent_seen(task, steps, upto, terms):
    """Where (if anywhere) do the terms appear (as plain substrings) before step `upto`:
    in the task text or in earlier tool outputs?"""
    hits = []
    for t in terms:
        if not t:
            continue
        if t in task:
            hits.append((t, 'task'))
        for st in steps[:upto - 1]:
            if t in st['obs']:
                hits.append((t, f"output of step {st['step']}"))
                break
    return hits


def git_check(repo_dir, commit, row):
    """A direct git look-up, independent of the parse caches."""
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
    ft, fact = row['fact_type'], row['fact']
    if ft == 'path':
        out = subprocess.run(['git', '-C', repo_dir, 'ls-tree', '-r', '--name-only', commit],
                             capture_output=True, text=True, env=env).stdout.split('\n')
        return [p for p in out if p == fact or p.endswith('/' + fact)][:3]
    name = re.split(r'[(.]', fact.split('(')[0])[-1] if ft != 'kwarg' else fact.split('(')[0]
    if ft in ('module', 'symbol'):
        name = fact.rsplit('.', 1)[-1]
    pat = (r'(def|class|func|function|fn|type|const|let|var|struct|enum|interface)\s+(\([^)]*\)\s*)?' + re.escape(name)
           + r'\b|^\s*' + re.escape(name) + r'\s*[:=]')
    r = subprocess.run(['git', '-C', repo_dir, 'grep', '-n', '-P', pat, commit, '--', '*.py', '*.go', '*.ts', '*.tsx',
                        '*.js', '*.rs'], capture_output=True, text=True, env=env)  # -P: Perl regex (\s, \b)
    return r.stdout.split('\n')[:4]


def main():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261113
    rng = random.Random(seed)
    jobs = {('swebv', j['trial']): j for j in M.swebv_jobs()}
    jobs.update({('deepswe', j['trial']): j for j in M.deepswe_jobs()})
    k = 0
    only_wrong = len(sys.argv) > 4 and sys.argv[4] == 'wrong'
    for bench in ('swebv', 'deepswe'):
        # judged named-before-seen rows (external library names and 'planned' mentions are not judged)
        nbs = [r for r in load_rows(bench) if r['nbs'] and r.get('wrong') is not None
               and (r['wrong'] or not only_wrong)]
        for row in rng.sample(nbs, n):
            k += 1
            j = jobs[(bench, row['trial'])]
            if bench == 'swebv':
                task, steps, patch = M.read_minisweagent(j['path'])
            else:
                task, steps, patch, _ = M.read_atif(j['path'])
            st = steps[row['step'] - 1]
            text = {'reasoning': st['reasoning'], 'visible': st['visible'],
                    'command': '\n'.join(st['commands'])}[row['field']]
            key = row['fact'].split('(')[0] if row['fact_type'] != 'path' else row['fact']
            last = key.rsplit('.', 1)[-1] if row['fact_type'] in ('module', 'symbol') else key
            if row['fact_type'] == 'kwarg':
                last = row['fact'].split('(')[1].rstrip('=)')
            print('=' * 100)
            print(f"[{k}] {bench} {row['model']} {row['trial']} step {row['step']}/{len(steps)} field={row['field']}")
            print(f"    fact_type={row['fact_type']} how={row.get('how')} fact={row['fact']!r} seen={row['seen']} truth={row['truth']} "
                  f"contradicted={row.get('contradicted_step')} changed_course={row.get('changed_course')}")
            w = window(text, last) or window(text, key) or text[:600]
            print('--- text around the mention ---')
            print(w)
            print('--- independent search before this step ---')
            terms = {key, last}
            if row['fact_type'] == 'path':  # file name, and the path without its extension
                terms |= {key.rsplit('/', 1)[-1], '/'.join(key.rsplit('.', 1)[0].split('/')[-2:])}
            print(independent_seen(task, steps, row['step'], sorted(terms)))
            print('--- git at base commit ---')
            print(git_check(j['repo_dir'], j['base_commit'], row))


if __name__ == '__main__':
    main()
