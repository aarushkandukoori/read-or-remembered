"""Download side of RQ1 on You Wang et al.'s SWE-bench Pro claims: repo clones and the file contents we need.

Everything that touches the network lives here. The analysis script (rq1_claim_provenance.py) never
downloads anything: it runs git with GIT_NO_LAZY_FETCH=1, so a missing file shows up as an error.

Steps (run from the project root, in this order):
  python3 -I scripts/rq1_pro_repos.py clone       blobless clones of the 11 repos (+ openlibrary's
                                                  vendor/infogami submodule), base commits checked
  python3 -I scripts/rq1_pro_repos.py evidence    file versions cited as "repository" evidence
  python3 -I scripts/rq1_pro_repos.py snapshots   pick the history snapshots (writes snapshots.json)
  python3 -I scripts/rq1_pro_repos.py trees base  all searchable files at the 100 base commits
  python3 -I scripts/rq1_pro_repos.py trees snap  all searchable files at the snapshot commits
  python3 -I scripts/rq1_pro_repos.py trees manifest  go.mod / package.json / requirements files at base
  python3 -I scripts/rq1_pro_repos.py size        disk use per clone

What "blobless" means: git clone --filter=blob:none downloads every commit and every folder listing,
but no file contents. We then ask GitHub for exactly the file versions (blobs) we need, in one batched
request per repo (the same request git makes for a single missing file, just with many ids).
--single-branch keeps only the default branch; base commits that are not on it are fetched by id.
"""
import datetime
import json
import os
import re
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(PROJECT, "data", "faultyclaims", "reasoning_replication")
REPOS_DIR = os.path.join(PROJECT, "data", "repos_pro")
SNAPSHOTS_JSON = os.path.join(REPOS_DIR, "snapshots.json")
CAP_BYTES = 3 * 1024 ** 3  # the task's disk cap for all clones together

# Repo name as written in the SWE-bench Pro instance ids ("instance_<org>__<repo>-<sha>...") -> GitHub URL.
REPOS = {
    "NodeBB__NodeBB": "https://github.com/NodeBB/NodeBB.git",
    "ansible__ansible": "https://github.com/ansible/ansible.git",
    "element-hq__element-web": "https://github.com/element-hq/element-web.git",
    "flipt-io__flipt": "https://github.com/flipt-io/flipt.git",
    "future-architect__vuls": "https://github.com/future-architect/vuls.git",
    "gravitational__teleport": "https://github.com/gravitational/teleport.git",
    "internetarchive__openlibrary": "https://github.com/internetarchive/openlibrary.git",
    "navidrome__navidrome": "https://github.com/navidrome/navidrome.git",
    "protonmail__webclients": "https://github.com/ProtonMail/WebClients.git",
    "qutebrowser__qutebrowser": "https://github.com/qutebrowser/qutebrowser.git",
    "tutao__tutanota": "https://github.com/tutao/tutanota.git",
}
# openlibrary vendors infogami as a git submodule at vendor/infogami; 611 evidence citations point into it.
SUBMODULES = {"vendor/infogami": ("internetarchive__infogami", "https://github.com/internetarchive/infogami.git")}

# Which files count as "the repo's code" when we search for an identifier (git pathspecs).
# Source, templates and config; no docs/changelogs (they mention removed names), no JSON (mostly
# translation files), no vendored or minified third-party code.
SEARCH_PATHSPEC = [
    "*.py", "*.pyi", "*.js", "*.jsx", "*.mjs", "*.cjs", "*.ts", "*.tsx", "*.go", "*.proto",
    "*.rs", "*.java", "*.c", "*.h", "*.sh", "*.yml", "*.yaml", "*.toml", "*.cfg", "*.ini",
    "*.html", "*.tpl", "*.tmpl", "*.jinja", "*.jinja2", "*.j2", "*.vue", "*.sql", "*.graphql",
    ":(exclude,glob)**/vendor/**", ":(exclude,glob)**/node_modules/**",
    ":(exclude,glob)**/third_party/**", ":(exclude,glob)**/*.min.js", ":(exclude,glob)**/dist/**",
]

# Dependency manifests: used to tell the repo's own packages from third-party ones (external names).
MANIFEST_PATHSPEC = [":(glob)**/go.mod", ":(glob)**/package.json", ":(glob)**/requirements*.txt",
                     ":(exclude,glob)**/node_modules/**", ":(exclude,glob)**/vendor/**"]

NO_FETCH = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
FETCH_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0")


def git(repo_dir, *args, input_text=None, env=NO_FETCH, check=True):
    """Run one git command in repo_dir and return its stdout as text."""
    r = subprocess.run(["git", "-C", repo_dir, *args], input=input_text, capture_output=True,
                       text=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed in {repo_dir}: {r.stderr[-500:]}")
    return r.stdout


def repo_of(instance_id):
    """'instance_NodeBB__NodeBB-<sha>-v<sha>' -> 'NodeBB__NodeBB'."""
    return re.match(r"instance_(.+?__.+?)-[0-9a-f]{40}", instance_id).group(1)


def load_results():
    """Yield (model label, instance id, result.json content) for all 400 assessed trajectories."""
    root = os.path.join(ARTIFACT, "rq2", "results")
    for label in sorted(os.listdir(root)):
        for instance in sorted(os.listdir(os.path.join(root, label))):
            with open(os.path.join(root, label, instance, "result.json"), encoding="utf-8") as f:
                yield label, instance, json.load(f)


def base_commits():
    """{repo name: set of base commits} over the 100 tasks."""
    out = {}
    for _, instance, result in load_results():
        out.setdefault(repo_of(instance), set()).add(result["base_commit"])
    return out


def missing_objects(repo_dir, oids):
    """Which of these object ids are not in the clone yet (cat-file prints '<id> missing')."""
    if not oids:
        return []
    out = git(repo_dir, "cat-file", "--batch-check", input_text="\n".join(sorted(oids)) + "\n")
    return [line.split()[0] for line in out.splitlines() if line.endswith(" missing")]


def fetch_ids(repo_dir, oids, chunk=20000):
    """Download these commits or blobs from GitHub, many ids per request, never their blobs' history."""
    oids = sorted(oids)
    for start in range(0, len(oids), chunk):
        part = oids[start:start + chunk]
        r = subprocess.run(["git", "-C", repo_dir, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "origin",
                            "--no-tags", "--no-write-fetch-head", "--recurse-submodules=no",
                            "--filter=blob:none", "--stdin"],
                           input="\n".join(part) + "\n", capture_output=True, text=True, env=FETCH_ENV)
        if r.returncode != 0:
            print(f"  fetch FAILED for {len(part)} ids: {r.stderr[-400:]}", flush=True)
            return False
    return True


def du_bytes(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def check_cap():
    used = du_bytes(REPOS_DIR) if os.path.isdir(REPOS_DIR) else 0
    print(f"  repos_pro now uses {used / 1024 ** 2:,.0f} MB (cap {CAP_BYTES / 1024 ** 3:.0f} GB)", flush=True)
    if used > CAP_BYTES:
        sys.exit("STOP: over the 3 GB cap")
    return used


def cmd_clone():
    os.makedirs(REPOS_DIR, exist_ok=True)
    wanted = base_commits()
    # submodule commits come from the evidence records themselves
    sub_commits = {}
    for _, _, result in load_results():
        for ev in result["evidence"].values():
            p = ev.get("repository_provenance") or {}
            if p.get("repository_path") in SUBMODULES:
                sub_commits.setdefault(SUBMODULES[p["repository_path"]][0], set()).add(p["commit"])
    jobs = [(name, url, wanted.get(name, set())) for name, url in REPOS.items()]
    jobs += [(name, url, sub_commits.get(name, set())) for name, url in SUBMODULES.values()]
    for name, url, commits in jobs:
        repo_dir = os.path.join(REPOS_DIR, name)
        if not os.path.isdir(os.path.join(repo_dir, ".git")):
            print(f"cloning {name} ...", flush=True)
            r = subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--single-branch",
                                "--quiet", url, repo_dir], capture_output=True, text=True, env=FETCH_ENV)
            if r.returncode != 0:
                print(f"  clone FAILED: {r.stderr[-400:]}", flush=True)
                continue
        # base commits that are not on the default branch: fetch them by id (commits + folders only)
        absent = [c for c in commits
                  if subprocess.run(["git", "-C", repo_dir, "cat-file", "-e", c + "^{commit}"],
                                    capture_output=True, env=NO_FETCH).returncode != 0]
        if absent:
            print(f"  {name}: {len(absent)} base commits not on the default branch, fetching by id", flush=True)
            fetch_ids(repo_dir, absent)
        still = [c for c in commits
                 if subprocess.run(["git", "-C", repo_dir, "cat-file", "-e", c + "^{commit}"],
                                   capture_output=True, env=NO_FETCH).returncode != 0]
        print(f"  {name}: {len(commits) - len(still)}/{len(commits)} needed commits present", flush=True)
        check_cap()


def cmd_evidence():
    """Fetch every file version cited as repository evidence (all 400 trajectories, all claims)."""
    want = {}
    for _, instance, result in load_results():
        for ev in result["evidence"].values():
            p = ev.get("repository_provenance")
            if not p:
                continue
            name = repo_of(instance) if p["repository_path"] == "." else SUBMODULES[p["repository_path"]][0]
            want.setdefault(name, set()).add(p["blob_oid"])
    for name, oids in sorted(want.items()):
        repo_dir = os.path.join(REPOS_DIR, name)
        missing = missing_objects(repo_dir, oids)
        ok = fetch_ids(repo_dir, missing) if missing else True
        left = len(missing_objects(repo_dir, oids))
        print(f"{name}: {len(oids)} cited file versions, {len(missing)} downloaded now, {left} still missing"
              f" ({'ok' if ok else 'fetch error'})", flush=True)
    check_cap()


def commit_time(repo_dir, commit):
    """Committer time (unix seconds). Mailmap off: 'git show' would try to download .mailmap."""
    return int(git(repo_dir, "-c", "log.mailmap=false", "log", "-1", "--format=%ct", commit).strip())


def cmd_snapshots():
    """Pick history snapshots per repo: the default-branch commit at the start of every other year
    from 4 years before the earliest base commit, plus the newest commit. These stand in for
    'all of history' (git log -S needs every old file version, which a blobless clone does not have)."""
    out = {}
    for name, commits in sorted(base_commits().items()):
        repo_dir = os.path.join(REPOS_DIR, name)
        times = [commit_time(repo_dir, c) for c in commits]
        first_year = datetime.datetime.fromtimestamp(min(times), datetime.timezone.utc).year - 4
        head = git(repo_dir, "rev-parse", "HEAD").strip()
        head_time = commit_time(repo_dir, head)
        head_year = datetime.datetime.fromtimestamp(head_time, datetime.timezone.utc).year
        snaps = []
        for year in range(first_year, head_year + 1, 2):
            sha = git(repo_dir, "rev-list", "-1", "--first-parent", f"--before={year}-01-01T00:00:00Z",
                      "HEAD").strip()
            if sha and sha not in [s["commit"] for s in snaps]:
                snaps.append({"commit": sha, "label": f"{year}-01-01", "time": commit_time(repo_dir, sha)})
        snaps.append({"commit": head, "label": "HEAD", "time": head_time})
        out[name] = snaps
        print(f"{name}: base commits {len(commits)}, snapshots "
              + ", ".join(s['label'] for s in snaps), flush=True)
    with open(SNAPSHOTS_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {SNAPSHOTS_JSON}")


EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's id for an empty folder


def searchable_files(repo_dir, commit, pathspec=None):
    """{path: blob id} for the searchable files (SEARCH_PATHSPEC) at one commit.
    'git ls-tree' does not understand the exclude patterns, so we list the files as a diff against
    the empty folder, which uses the same pathspec rules as 'git grep' and needs no file contents."""
    out = git(repo_dir, "diff", "--raw", "--no-renames", "--abbrev=40", EMPTY_TREE, commit, "--",
              *(pathspec or SEARCH_PATHSPEC))
    files = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        new_mode, oid = meta.split()[1], meta.split()[3]
        if new_mode in ("100644", "100755"):  # regular files only (no symlinks, no submodules)
            files[path] = oid
    return files


def searchable_blobs(repo_dir, commit, pathspec=None):
    return set(searchable_files(repo_dir, commit, pathspec).values())


def cmd_trees(which):
    """Fetch the searchable files at the base commits ('base') or at the snapshot commits ('snap'),
    or the dependency manifests at the base commits ('manifest')."""
    pathspec = MANIFEST_PATHSPEC if which == "manifest" else SEARCH_PATHSPEC
    if which in ("base", "manifest"):
        jobs = {name: sorted(c) for name, c in base_commits().items()}
    else:
        with open(SNAPSHOTS_JSON) as f:
            jobs = {name: [s["commit"] for s in snaps] for name, snaps in json.load(f).items()}
    for name, commits in sorted(jobs.items()):
        repo_dir = os.path.join(REPOS_DIR, name)
        blobs = set()
        for c in commits:
            blobs |= searchable_blobs(repo_dir, c, pathspec)
        missing = missing_objects(repo_dir, blobs)
        print(f"{name}: {len(commits)} commits, {len(blobs):,} searchable file versions, "
              f"{len(missing):,} to download", flush=True)
        if missing:
            fetch_ids(repo_dir, missing)
            left = len(missing_objects(repo_dir, blobs))
            print(f"  still missing after fetch: {left}", flush=True)
        check_cap()


def cmd_size():
    total = 0
    for name in sorted(os.listdir(REPOS_DIR)):
        path = os.path.join(REPOS_DIR, name)
        if os.path.isdir(path):
            b = du_bytes(path)
            total += b
            print(f"{name:<32}{b / 1024 ** 2:>10,.0f} MB")
    print(f"{'total':<32}{total / 1024 ** 2:>10,.0f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    step = sys.argv[1]
    if step == "clone":
        cmd_clone()
    elif step == "evidence":
        cmd_evidence()
    elif step == "snapshots":
        cmd_snapshots()
    elif step == "trees":
        cmd_trees(sys.argv[2])
    elif step == "size":
        cmd_size()
    else:
        sys.exit(__doc__)
