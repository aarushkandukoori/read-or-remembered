"""Make blobless clones of the repos behind the DeepSWE sample (for RQ1 fact provenance).

A blobless clone (git clone --filter=blob:none) has every commit and every folder listing of the
default branch, but no file contents. That is enough to check whether a file path existed at a
commit (git ls-tree) and when paths were added (git log). File contents are downloaded later, in
one batch per commit, only for the source files we need (see rq1_repo_index.py).

Rules:
  - one folder per GitHub repo: data/repos_deepswe/<org>_<repo>
  - repos used by more trials are cloned first, so a cap hits the least-used repos
  - a clone that grows past PER_REPO_MB is stopped and deleted (logged as "giant")
  - nothing new is started once the folder holds TOTAL_MB
  - if a task's base commit is not on the default branch, that one commit is fetched too

Usage:
  python3 -I scripts/rq1_clone_deepswe.py data/deepswe data/repos_deepswe
Writes data/repos_deepswe/clone_log.jsonl (one line per repo: size, status, reason).
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

PER_REPO_MB = 900      # stop a single clone above this size
TOTAL_MB = 5000        # the task's cap for data/repos_deepswe (5 GB)


def folder_mb(path):
    """Size of a folder in MB (du -sk)."""
    out = subprocess.run(['du', '-sk', path], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1024 if out else 0.0


def git(repo, *args, timeout=600):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def needed_repos(data_dir):
    """{'org/repo': {'trials': n, 'commits': {sha, ...}}} for every trial in the sample."""
    with open(os.path.join(data_dir, 'derived', 'base_commit_dates.csv')) as f:
        full_sha = {r['task_id']: (r['repository'], r['full_sha'] or r['base_commit']) for r in csv.DictReader(f)}
    with open(os.path.join(data_dir, 'sample_manifest.csv')) as f:
        manifest = list(csv.DictReader(f))
    repos = {}
    for m in manifest:
        if m['source'] == 'deep-swe':
            repo, sha = full_sha[m['task_name']]
        else:
            # SWE-bench Pro names look like instance_<org>__<repo>-<40-char commit>[-v...]
            hit = re.match(r'instance_(.+?)__(.+?)-([0-9a-f]{40})', m['task_name'])
            repo, sha = f'{hit.group(1)}/{hit.group(2)}', hit.group(3)
        entry = repos.setdefault(repo, {'trials': 0, 'commits': set()})
        entry['trials'] += 1
        entry['commits'].add(sha)
    return repos


def clone(repo, dest, budget_mb):
    """Blobless, single-branch, no-tags clone with a size watchdog. Returns (status, reason)."""
    url = f'https://github.com/{repo}.git'
    proc = subprocess.Popen(['git', 'clone', '--quiet', '--filter=blob:none', '--no-checkout', '--single-branch',
                             '--no-tags', url, dest], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    limit = min(PER_REPO_MB, budget_mb)
    while proc.poll() is None:
        time.sleep(3)
        if os.path.isdir(dest) and folder_mb(dest) > limit:
            proc.kill()
            proc.wait()
            shutil.rmtree(dest, ignore_errors=True)
            return 'skipped', f'giant: blobless clone passed {limit:.0f} MB before finishing'
    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        return 'failed', (proc.stderr.read() or '')[-300:]
    return 'ok', ''


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # keep these clones out of the project's git repo (data/repos_deepswe is not in the top .gitignore)
    with open(os.path.join(out_dir, '.gitignore'), 'w') as f:
        f.write('*\n')
    log_path = os.path.join(out_dir, 'clone_log.jsonl')
    done = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            done = {json.loads(l)['repo'] for l in f if l.strip()}
    repos = needed_repos(data_dir)
    order = sorted(repos, key=lambda r: (-repos[r]['trials'], r))
    for repo in order:
        if repo in done:
            continue
        dest = os.path.join(out_dir, repo.replace('/', '_'))
        used = folder_mb(out_dir)
        row = {'repo': repo, 'trials': repos[repo]['trials'], 'n_base_commits': len(repos[repo]['commits'])}
        if used > TOTAL_MB - 50:
            row.update(status='skipped', reason=f'total cap: folder already {used:.0f} MB')
        else:
            t0 = time.time()
            status, reason = ('ok', 'already cloned') if os.path.isdir(dest) else clone(repo, dest, TOTAL_MB - used)
            row.update(status=status, reason=reason, seconds=round(time.time() - t0))
            if status == 'ok':
                # make sure every base commit is present; fetch the ones that are not on the default branch
                missing = []
                for sha in sorted(repos[repo]['commits']):
                    if git(dest, 'cat-file', '-e', sha + '^{commit}')[0] != 0:
                        git(dest, '-c', 'gc.auto=0', 'fetch', '--quiet', '--filter=blob:none', 'origin', sha)
                        if git(dest, 'cat-file', '-e', sha + '^{commit}')[0] != 0:
                            missing.append(sha)
                row['missing_base_commits'] = missing
                row['mb'] = round(folder_mb(dest), 1)
        print(json.dumps(row), flush=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
    print(f'done; folder size {folder_mb(out_dir):.0f} MB')


if __name__ == '__main__':
    main(*sys.argv[1:3])
