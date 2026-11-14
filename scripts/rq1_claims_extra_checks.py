"""Helpers for checking rq1_claim_provenance.py by hand (print only; only fix-names downloads anything).

  view TRAJ BEFORE NEEDLE [CONTEXT]
      Search the tool outputs of trajectory TRAJ that come before message index BEFORE for the substring
      NEEDLE (raw text, no normalizing) and print each matching output line with the command behind it.
  none-sample N SEED
      N random DeepSeek (no-network) refuted claims whose evidence_seen_before is 'none'. For each cited
      non-trivial line, take its longest word (6+ characters) and look for it in earlier outputs about the
      same file: a cheap second opinion on whether the line matcher missed a display.
  other-version-sample N SEED
      N random DeepSeek claims whose identifier class is 'other_version' (a name missing at the base
      commit but present in older/newer snapshots), with the claim text, to judge by reading.
  fix-names CID:NAME [CID:NAME ...]
      Is NAME in the task's own upstream fix? The fix commit is the sha in the instance id; we diff base..fix
      and search the changed files' new versions. Downloads those few file versions (one batched request).

Usage (from the project root):
    .venv/bin/python -I scripts/rq1_claims_extra_checks.py view data/faultyclaims/.../x.json 68 "FakeSignal"
    .venv/bin/python -I scripts/rq1_claims_extra_checks.py none-sample 10 1
    .venv/bin/python -I scripts/rq1_claims_extra_checks.py other-version-sample 12 7
    .venv/bin/python -I scripts/rq1_claims_extra_checks.py fix-names c35238:refreshSmartPlaylist c11167:qt_64
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_claim_provenance as m  # noqa: E402
import rq1_pro_repos as rp  # noqa: E402


def view(path, before, needle, ctx=1):
    with open(path, encoding="utf-8") as f:
        messages = json.load(f)["messages"]
    cmd = {}
    for msg in messages:
        for call_id, command in m.tool_calls(msg):
            cmd[call_id] = command
    hits = 0
    for i, msg in enumerate(messages[:before]):
        if msg["role"] != "tool":
            continue
        lines = m.message_text(msg).splitlines()
        for k, line in enumerate(lines):
            if needle in line:
                hits += 1
                print(f"--- msg {i} cmd: {cmd.get(msg.get('tool_call_id'), '')[:160]!r}")
                for j in range(max(0, k - ctx), min(len(lines), k + ctx + 1)):
                    print(f"   {'>' if j == k else ' '} {lines[j][:200]}")
                break
    print(f"[{hits} earlier tool outputs contain {needle!r}]")


def load_features():
    with open(m.FEATURES, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def none_sample(n, seed):
    rows = load_features()
    pool = [r for r in rows if r["evidence_seen_before"] == "none" and r["network_none"] and r["verdict"] == "refuted"]
    tidy, tables = m.load_claims(), m.load_evidence_tables()
    for r in random.Random(seed).sample(pool, n):
        t = tidy[int(r["cid"][1:])]
        traj = m.Trajectory(os.path.join(m.PROJECT, t["trajectory_path"]))
        _, ev_table = tables[(t["model"], t["instance_id"])]
        here = r["message_index"]
        print("=" * 90)
        print(r["cid"], r["model"], "msg", here, "n_cited", r["n_cited_lines"], "any-obs:", r["evidence_seen_before_any"],
              "file_seen:", r["file_seen_before"])
        print("CLAIM:", t["claim"][:200])
        print("last earlier commands:", [traj.obs_cmd[i][:90] for i in sorted(traj.obs) if i < here][-12:])
        for ev in t["evidence"]:
            if ev["kind"] != "repository":
                continue
            prov = ev_table[ev["id"]]["repository_provenance"]
            name = r["repo"] if prov["repository_path"] == "." else m.SUBMODULES[prov["repository_path"]][0]
            lines = m.blob_text(os.path.join(m.REPOS_DIR, name), prov["blob_oid"]).splitlines()
            file_name = ev["source"].rsplit("/", 1)[-1]
            checked, examples = 0, []
            for k in range(ev["start_line"], ev["end_line"] + 1):
                line = m.norm(lines[k - 1]) if k <= len(lines) else ""
                words = sorted(re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", line), key=len, reverse=True)
                if m.is_trivial(line) or not words:
                    continue
                checked += 1
                hit = next((i for i in sorted(traj.obs) if i < here and words[0] in traj.obs[i]
                            and traj.anchored(i, file_name)), None)
                if hit is not None:
                    examples.append((k, words[0], hit))
            print(f"  {ev['source']}:{ev['start_line']}-{ev['end_line']}  lines checked {checked}, longest word seen "
                  f"in an earlier output about this file: {len(examples)}", examples[:4])


def other_version_sample(n, seed):
    rows = load_features()
    tidy = m.load_claims()
    pool = [r for r in rows if r["network_none"] and r["ident_class"] == "other_version"]
    print("pool", len(pool))
    for r in random.Random(seed).sample(pool, n):
        t = tidy[int(r["cid"][1:])]
        print("=" * 80)
        print(r["cid"], r["model"], r["verdict"], r["ogsf_type"], r["repo"], "evidence_seen_before:", r["evidence_seen_before"])
        print("CLAIM:", t["claim"][:260])
        print("QUOTE:", t["quote"][:200])
        for i in r["identifiers"]:
            if i["name"] in r["unexplained"]:
                print("   ->", i["name"], i["kind"], i["status"])


def fix_names(pairs):
    rows = {r["cid"]: r for r in load_features()}
    for pair in pairs:
        cid, word = pair.split(":", 1)
        r = rows[cid]
        fix = re.match(r"instance_.+?__.+?-([0-9a-f]{40})", r["instance_id"]).group(1)
        repo_dir = os.path.join(rp.REPOS_DIR, r["repo"])
        new = {}  # changed file -> its blob id after the fix (deleted files skipped)
        for line in rp.git(repo_dir, "diff", "--raw", "--no-renames", "--abbrev=40", r["base_commit"], fix).splitlines():
            meta, path = line.split("\t", 1)
            if meta.split()[3] != "0" * 40:
                new[path] = meta.split()[3]
        missing = rp.missing_objects(repo_dir, set(new.values()))
        if missing:
            rp.fetch_ids(repo_dir, missing)
        rx = m.word_regex(word)
        hits = [p for p, oid in new.items() if rx.search(m.blob_text(repo_dir, oid))]
        print(f"{cid} {word}: fix {fix[:10]} changes {len(new)} files ({len(missing)} downloaded); "
              f"name in the post-fix version of: {hits}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "view":
        view(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif sys.argv[1] == "none-sample":
        none_sample(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "other-version-sample":
        other_version_sample(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "fix-names":
        fix_names(sys.argv[2:])
    else:
        sys.exit(__doc__)
