"""Find the date of every DeepSWE task's base commit (the repo snapshot the agent starts from).

For each task we fetch only the commit object (no files, no history) from GitHub:
    git fetch --depth=1 --filter=tree:0 <repo_url> <base_commit>
then read its dates with git log. One small bare repo per GitHub project goes in
the cache folder (data/deepswe/base_commits/); it can be deleted once the CSV exists.

A few tasks give a shortened hash (e.g. "7cef35b"), which git fetch can't use.
For those we ask the public GitHub API, which accepts short hashes
(at most a handful of calls, well under the 60/hour anonymous limit).

Usage:
  python3 -I scripts/deepswe_base_commit_dates.py data/deepswe/meta/v1.1/tasks.json \
      data/deepswe/base_commits data/deepswe/derived/base_commit_dates.csv
"""
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request


def git(args, cwd):
    """Run a git command and return its output text ('' if it failed)."""
    r = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, timeout=300)
    return r.stdout.strip() if r.returncode == 0 else ''


def github_api_commit(repo, sha):
    """Ask GitHub's public API for a commit (works with short hashes).
    Returns (full_sha, committer_date, author_date) or ('', '', '')."""
    req = urllib.request.Request(f'https://api.github.com/repos/{repo}/commits/{sha}',
                                 headers={'Accept': 'application/vnd.github+json',
                                          'User-Agent': 'research-fetcher'})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as err:
        print('  GitHub API failed for', repo, sha, err)
        return '', '', ''
    time.sleep(1)
    return d['sha'], d['commit']['committer']['date'], d['commit']['author']['date']


def main(tasks_json, repos_dir, out_csv):
    with open(tasks_json) as f:
        tasks = json.load(f)['rows']
    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for t in sorted(tasks, key=lambda t: t['id']):
        bare = os.path.join(repos_dir, t['repository'].replace('/', '__') + '.git')
        if not os.path.isdir(bare):
            subprocess.run(['git', 'init', '--bare', '--quiet', bare], check=True)
        sha = t['base_commit_hash']
        # only fetch if we don't already have this commit
        if not git(['cat-file', '-t', sha], bare):
            git(['fetch', '--quiet', '--depth=1', '--filter=tree:0', t['repository_url'], sha], bare)
            time.sleep(0.3)  # be polite to GitHub
        line = git(['log', '-1', '--format=%H|%cI|%aI', sha], bare)
        full_sha, committed, authored = line.split('|') if line else ('', '', '')
        how = 'git'
        if not line:  # short hash: fall back to the GitHub API
            full_sha, committed, authored = github_api_commit(t['repository'], sha)
            how = 'github_api'
        rows.append({'task_id': t['id'], 'repository': t['repository'], 'language': t['language'],
                     'base_commit': sha, 'full_sha': full_sha, 'commit_date': committed,
                     'author_date': authored, 'looked_up_with': how})
        print(t['id'], committed or 'NOT FOUND', flush=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('wrote', out_csv, len(rows), 'rows')


if __name__ == '__main__':
    main(*sys.argv[1:4])
