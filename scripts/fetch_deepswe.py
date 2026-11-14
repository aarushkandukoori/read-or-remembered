"""Download DeepSWE (Datacurve) trial data: the index files, then a sample of trials.

Where the data lives (found by reading the site's own JavaScript, Nov 2026):
  index files : https://deepswe.datacurve.ai/artifacts/<version>/{release,trials,tasks}.json
  trial files : https://deepswe-downloads.datacurve.ai/<path pattern from release.json>
                e.g. v1.1/trial-artifacts/<trial_name>/agent/trajectory.json

Usage (run from the project folder):
  python3 -I scripts/fetch_deepswe.py meta    data/deepswe
  python3 -I scripts/fetch_deepswe.py sample  data/deepswe --cap-gb 1.5
  python3 -I scripts/fetch_deepswe.py loghead data/deepswe   # 4 KB run-settings header per config

What 'sample' picks (same picks every time, so a rerun resumes instead of re-picking):
  A "group" is one model + effort setting (a "config") inside one slice of the data:
    - v1.1 full benchmark: every config that has trajectories
    - v1 full benchmark:   only configs of models that are NOT in v1.1 (older models)
    - v1 cross-bench:      9 configs x {DeepSWE, SWE-bench Pro}, 30 tasks each, that
                           Datacurve also graded with an LLM judge (the "critique" field)
  Each group gets --per-group trials (default 12), but every model gets at least
  --min-per-model trials (default 20), so a model with a single config gets 20.
  Cross-bench groups get all 30 tasks, so the two benchmarks can be compared task by task.
  Inside a slice we shuffle the task list with a fixed seed and take one trial per task
  (the attempt with the smallest hash), so all groups in a slice share the same tasks.
  Downloads go round-robin (trial 1 of every group, then trial 2 of every group, ...)
  so if the size cap stops us early, every model still has even coverage.
  A group whose files the server refuses (HTTP 403, e.g. gemini-3-8-flash in Nov 2026)
  is skipped after its first trial.

Per trial we save: trajectory.json, model.patch, and the verifier results
(reward.json, ctrf.json, test-stdout.txt). The mini-swe-agent.txt log is skipped
because it repeats the trajectory. Any single file over --max-file-mb (default 5 MB;
20 MB for trajectories) is skipped and noted in the manifest: a few agents committed
binaries, giving 20-60 MB patches that would eat the size budget.

Resumable: a file that already exists (non-empty) is skipped.
Polite: one request at a time over one reused connection, with a short pause.
"""
import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import sys
import time
import urllib.parse
from collections import Counter

SITE = 'https://deepswe.datacurve.ai/artifacts'
VERSIONS = ['v1.1', 'v1']
INDEX_FILES = ['release.json', 'tasks.json', 'trials.json']
HEADERS = {'User-Agent': 'research-trajectory-fetcher/0.1 (polite, single-threaded)'}

_connections = {}  # one open connection per host, reused for every request (keep-alive)


class BudgetFull(Exception):
    """Raised when the next file would push the folder past the size cap."""


class TooBig(Exception):
    """Raised when one file is over the per-file limit (e.g. a patch full of binaries)."""


def request(method, url, budget_left=None, max_bytes=None):
    """Send one HTTP request over a reused connection. Returns (status, headers, body).

    Before reading the body we look at its size: over max_bytes raises TooBig,
    over budget_left raises BudgetFull. Broken connections are reopened and
    retried 3 times.
    """
    parts = urllib.parse.urlsplit(url)
    for attempt in range(3):
        conn = _connections.get(parts.netloc)
        if conn is None:
            conn = http.client.HTTPSConnection(parts.netloc, timeout=120)
            _connections[parts.netloc] = conn
        try:
            conn.request(method, parts.path, headers=HEADERS)
            resp = conn.getresponse()
            size = int(resp.headers.get('Content-Length') or 0)
            too_big = max_bytes is not None and size > max_bytes
            over_budget = budget_left is not None and size > budget_left
            if too_big or over_budget:
                conn.close()  # don't read the body; drop the connection instead
                _connections.pop(parts.netloc, None)
                raise TooBig() if too_big else BudgetFull()
            body = resp.read()
            return resp.status, resp.headers, body
        except (http.client.HTTPException, OSError) as err:
            conn.close()
            _connections.pop(parts.netloc, None)
            print('  retrying after error:', err, flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError('gave up after 3 tries: ' + url)


def download(url, out, budget_left=None, max_bytes=None):
    """Save url to the file 'out'.

    Returns the number of bytes written, 0 if the file was already there,
    None if the server says the file is not available (403/404),
    or -1 if the file is over max_bytes (skipped, not saved).
    We write to out.part first and rename at the end, so a half-finished
    file never looks finished.
    """
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 0
    try:
        status, _, body = request('GET', url, budget_left, max_bytes)
    except TooBig:
        return -1
    if status in (403, 404):
        return None
    if status != 200:
        raise RuntimeError(f'HTTP {status} for {url}')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.part', 'wb') as f:
        f.write(body)
    os.replace(out + '.part', out)
    return len(body)


def folder_size(path):
    """Total size in bytes of every file under 'path'."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def cmd_meta(out_dir):
    """Download release.json, tasks.json and trials.json for every version.
    Re-downloads a file only when its size on the server changed."""
    for v in VERSIONS:
        for name in INDEX_FILES:
            url = f'{SITE}/{v}/{name}'
            out = os.path.join(out_dir, 'meta', v, name)
            _, headers, _ = request('HEAD', url)
            remote = int(headers.get('Content-Length', -1))
            if os.path.exists(out) and os.path.getsize(out) == remote:
                print('ok  ', out, remote)
                continue
            if os.path.exists(out):
                os.remove(out)
            print('got ', out, download(url, out))
            time.sleep(0.5)


def stable_hash(text, seed):
    """A repeatable pseudo-random value for a string (same input gives same value)."""
    return hashlib.sha1(f'{seed}:{text}'.encode()).hexdigest()


def load_rows(out_dir, version):
    with open(os.path.join(out_dir, 'meta', version, 'trials.json')) as f:
        return json.load(f)['rows']


def pick_groups(out_dir, per_group, min_per_model, seed):
    """Return {(version, scope, source, config): [chosen trial rows]}."""
    rows11 = load_rows(out_dir, 'v1.1')
    rows1 = load_rows(out_dir, 'v1')
    models11 = {r['model'] for r in rows11}

    def usable(r):
        # needs a trajectory, and skip trials that died with an infrastructure error
        return r.get('has_trajectory') and r.get('included_in_score', True)

    candidates = {}  # group key -> list of rows
    for r in rows11:
        if usable(r):
            candidates.setdefault(('v1.1', 'full', r['source'], r['config']), []).append(r)
    for r in rows1:
        if not usable(r):
            continue
        if r['eval_scope'] == 'full' and r['model'] not in models11:
            candidates.setdefault(('v1', 'full', r['source'], r['config']), []).append(r)
        elif r['eval_scope'] == 'cross-bench':
            candidates.setdefault(('v1', 'cross-bench', r['source'], r['config']), []).append(r)

    # how many configs each model has inside its slice (version, scope, source)
    model_of = {key: rs[0]['model'] for key, rs in candidates.items()}
    n_configs = Counter(key[:3] + (model_of[key],) for key in candidates)

    groups = {}
    for key, rs in sorted(candidates.items()):
        tasks = sorted({r['task_name'] for r in rs}, key=lambda t: stable_hash(t, seed))
        if key[1] == 'cross-bench':
            want = len(tasks)  # small judged sample: take every task
        else:
            want = max(per_group, math.ceil(min_per_model / n_configs[key[:3] + (model_of[key],)]))
        chosen = []
        for t in tasks[:want]:
            attempts = [r for r in rs if r['task_name'] == t]
            chosen.append(min(attempts, key=lambda r: stable_hash(r['trial_name'], seed)))
        groups[key] = chosen
    return groups


def trial_urls(release, row, version):
    """List (url, local file name) pairs for one trial, using the release's path patterns."""
    base = release['artifact_base_url'].rstrip('/')
    pat = release['artifact_patterns']
    name = urllib.parse.quote(row['trial_name'])
    out = []
    if row.get('has_trajectory'):
        out.append((f"{base}/{pat['trajectory'].replace('{trial_name}', name)}", 'trajectory.json'))
    if row.get('has_model_patch'):
        out.append((f"{base}/{pat['model_patch'].replace('{trial_name}', name)}", 'model.patch'))
    if version == 'v1.1':
        for f in ['reward.json', 'ctrf.json', 'test-stdout.txt']:
            if f in (row.get('verifier_files') or []):
                path = pat['verifier_file'].replace('{trial_name}', name).replace('{file}', f)
                out.append((f'{base}/{path}', f))
    elif row.get('has_verifier_output'):
        out.append((f"{base}/{pat['verifier_output'].replace('{trial_name}', name)}",
                    'test-stdout.txt'))
    return out


def cmd_sample(out_dir, per_group, min_per_model, cap_gb, max_file_mb, delay, seed):
    cap = int(cap_gb * 1e9)
    max_file = int(max_file_mb * 1e6)
    releases = {}
    for v in VERSIONS:
        with open(os.path.join(out_dir, 'meta', v, 'release.json')) as f:
            releases[v] = json.load(f)
    groups = pick_groups(out_dir, per_group, min_per_model, seed)
    print(len(groups), 'groups,', sum(len(g) for g in groups.values()), 'trials planned', flush=True)

    # round-robin order: rank 0 of every group, then rank 1, ...
    longest = max(len(g) for g in groups.values())
    order = [(key, groups[key][rank]) for rank in range(longest)
             for key in sorted(groups) if rank < len(groups[key])]

    used = folder_size(out_dir)
    dead_groups = set()  # groups whose files the server refuses
    rows_out = []
    try:
        for key, row in order:
            if key in dead_groups:
                continue
            version = key[0]
            folder = os.path.join(out_dir, 'trials', version, row['trial_name'])
            got_files, got_bytes, refused, too_big = [], 0, 0, []
            for url, fname in trial_urls(releases[version], row, version):
                path = os.path.join(folder, fname)
                # trajectories are the main data, so they get 4x the per-file limit
                limit = 4 * max_file if fname == 'trajectory.json' else max_file
                n = download(url, path, budget_left=cap - used, max_bytes=limit)
                if n is None:
                    refused += 1
                    continue
                if n == -1:
                    too_big.append(fname)
                    continue
                used += n
                if n:
                    time.sleep(delay)
                got_files.append(fname)
                got_bytes += os.path.getsize(path)
            if not got_files and refused:
                print('server refuses files for', key, '- skipping this group', flush=True)
                dead_groups.add(key)
                continue
            rows_out.append({'version': version, 'scope': key[1], 'source': key[2],
                             'config': key[3], 'model': row['model'],
                             'reasoning_effort': row.get('reasoning_effort'),
                             'task_name': row['task_name'], 'trial_name': row['trial_name'],
                             'passed': row.get('passed'), 'errored': row.get('errored'),
                             'n_agent_steps': row.get('n_agent_steps'),
                             'files': ';'.join(got_files), 'bytes': got_bytes,
                             'skipped_too_big': ';'.join(too_big)})
            if len(rows_out) % 100 == 0:
                print(f'{len(rows_out)} trials done, folder now {used / 1e9:.3f} GB', flush=True)
    except BudgetFull:
        print('next file would pass the size cap, stopping', flush=True)
    finally:
        # the manifest lists every sampled trial that is on disk
        manifest = os.path.join(out_dir, 'sample_manifest.csv')
        with open(manifest, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]) if rows_out else ['trial_name'])
            w.writeheader()
            w.writerows(rows_out)
        print(f'done: {len(rows_out)} trials, folder {used / 1e9:.3f} GB, manifest {manifest}')


def cmd_loghead(out_dir, delay):
    """For one sampled trial per config, save the first 4 KB of the mini-swe-agent.txt log.
    That header shows the exact model settings of the run, e.g. whether thinking text
    was requested or hidden ('model.model_kwargs.thinking=...'). Uses an HTTP Range request,
    so only 4 KB per file is downloaded."""
    with open(os.path.join(out_dir, 'sample_manifest.csv')) as f:
        manifest = list(csv.DictReader(f))
    first = {}  # (version, config) -> first sampled trial
    for m in manifest:
        first.setdefault((m['version'], m['config']), m)
    for (version, config), m in sorted(first.items()):
        with open(os.path.join(out_dir, 'meta', version, 'release.json')) as f:
            release = json.load(f)
        path = release['artifact_patterns']['agent_log'].replace(
            '{trial_name}', urllib.parse.quote(m['trial_name']))
        out = os.path.join(out_dir, 'agent_log_heads', version, config + '.txt')
        if os.path.exists(out):
            continue
        parts = urllib.parse.urlsplit(release['artifact_base_url'])
        conn = http.client.HTTPSConnection(parts.netloc, timeout=60)
        conn.request('GET', '/' + path, headers=dict(HEADERS, Range='bytes=0-4095'))
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if resp.status in (200, 206):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f:
                f.write(body)
        print(resp.status, version, config, m['trial_name'])
        time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['meta', 'sample', 'loghead'])
    ap.add_argument('out_dir', help='e.g. data/deepswe')
    ap.add_argument('--per-group', type=int, default=12, help='trials per model+effort config')
    ap.add_argument('--min-per-model', type=int, default=20, help='at least this many per model')
    ap.add_argument('--cap-gb', type=float, default=1.5, help='never let the folder pass this size')
    ap.add_argument('--max-file-mb', type=float, default=5,
                    help='skip any single file bigger than this (a few patches are 20-60 MB of binaries)')
    ap.add_argument('--delay', type=float, default=0.1, help='seconds to wait between downloads')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.command == 'meta':
        cmd_meta(a.out_dir)
    elif a.command == 'loghead':
        cmd_loghead(a.out_dir, a.delay)
    else:
        cmd_sample(a.out_dir, a.per_group, a.min_per_model, a.cap_gb, a.max_file_mb, a.delay, a.seed)


if __name__ == '__main__':
    sys.exit(main())
