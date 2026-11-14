"""Download, in one batch per repo, the file contents (git "blobs") we need from the blobless clones.

Why: data/repos/* are blobless clones (git clone --filter=blob:none). They have every commit and
folder listing, but a file's contents are only downloaded the first time git needs them, one file
per network request. Reading thousands of files that way takes hours. Instead we list the files we
need at each task's base commit, find the ones not downloaded yet, and ask GitHub for all of them
in a single request (the same request git itself makes for a missing file, just batched).

Usage:
  python3 -I scripts/prefetch_blobs.py SWEBV_PARQUET REPOS_DIR KIND [INSTANCE_ID_FILE]
KIND
  src    .py files outside test folders (needed to index where functions/classes are defined)
  all    every .py file plus .rst/.txt docs (needed to count call sites with git grep)
INSTANCE_ID_FILE  optional: one instance id per line; only those tasks' base commits are used
"""
import os, subprocess, sys
import pyarrow.parquet as pq

# never let git download a missing file on its own while we are only checking what is missing
NO_FETCH = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def is_test_path(path):
    """True for files in test folders, test_*.py files and conftest.py."""
    parts = path.split('/')
    return (any(p in ('tests', 'test', 'testing') for p in parts[:-1])
            or parts[-1].startswith('test_') or parts[-1] == 'conftest.py')


def wanted(path, kind):
    if kind == 'src':
        return path.endswith('.py') and not is_test_path(path)
    return path.endswith(('.py', '.rst', '.txt'))


def main():
    parquet, repos_dir, kind = sys.argv[1:4]
    only = set(open(sys.argv[4]).read().split()) if len(sys.argv) > 4 else None
    tasks = pq.read_table(parquet).to_pandas()
    if only:
        tasks = tasks[tasks.instance_id.isin(only)]
    for repo, group in tasks.groupby('repo'):
        rp = os.path.join(repos_dir, repo.replace('/', '_'))
        blob_ids = set()
        for commit in group.base_commit.unique():
            listing = subprocess.run(['git', '-C', rp, 'ls-tree', '-r', commit],
                                     capture_output=True, text=True).stdout
            for line in listing.splitlines():
                meta, path = line.split('\t', 1)
                mode, typ, oid = meta.split()
                if typ == 'blob' and wanted(path, kind):
                    blob_ids.add(oid)
        # which of these are not downloaded yet? (cat-file prints "<id> missing" for those)
        check = subprocess.run(['git', '-C', rp, 'cat-file', '--batch-check'], input='\n'.join(blob_ids) + '\n',
                               capture_output=True, text=True, env=NO_FETCH).stdout
        missing = [l.split()[0] for l in check.splitlines() if l.endswith(' missing')]
        if missing:
            # one batched download; same flags git uses internally for a lazy fetch
            r = subprocess.run(['git', '-C', rp, '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', 'origin',
                                '--no-tags', '--no-write-fetch-head', '--recurse-submodules=no',
                                '--filter=blob:none', '--stdin'],
                               input='\n'.join(missing) + '\n', capture_output=True, text=True)
            status = 'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr[-300:]
        else:
            status = 'nothing to fetch'
        print(f'{repo}: {len(blob_ids)} files wanted, {len(missing)} downloaded now ({status})', flush=True)


if __name__ == '__main__':
    main()
