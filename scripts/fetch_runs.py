"""Download public mini-swe-agent trajectories (+ per-task report.json) from the SWE-bench
submissions bucket. Usage: python3 -I scripts/fetch_runs.py data/raw RUN [RUN ...]"""
import os, re, sys, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

BUCKET = 'https://swe-bench-submissions.s3.amazonaws.com/'

def list_keys(prefix):
    keys, tok = [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if tok:
            q['continuation-token'] = tok
        x = urllib.request.urlopen(BUCKET + '?' + urllib.parse.urlencode(q), timeout=60).read().decode()
        keys += re.findall(r'<Key>([^<]+)</Key>', x)
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', x)
        if not m:
            return keys
        tok = m.group(1)

def get(key, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = urllib.request.urlopen(BUCKET + urllib.parse.quote(key), timeout=300).read()
    with open(out + '.part', 'wb') as f:
        f.write(data)
    os.replace(out + '.part', out)

root, runs = sys.argv[1], sys.argv[2:]
for run in runs:
    keys = [k for k in list_keys(f'bash-only/{run}/')
            if k.endswith('.traj.json') or k.endswith('/report.json')]
    jobs = [(k, os.path.join(root, run, k.split(f'{run}/', 1)[1])) for k in keys]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: get(*j), jobs))
    print(run, len(jobs), 'files', flush=True)
