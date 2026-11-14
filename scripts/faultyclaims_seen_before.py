"""First-pass "read or remembered" signal for You Wang et al.'s claims.

For every claim in claims_tidy.jsonl (made by load_faultyclaims.py), look back in the trajectory and
ask: before the agent wrote this claim, had it already SEEN the things the claim names?

  identifiers  code-like names in the claim's CoT quote (snake_case, camelCase, dotted.names,
               CONSTANTS, `backticked` text). An identifier counts as seen if it (or the last part of
               a dotted name) appears in the task text or in an earlier tool output.
  files        the claim's related_files. A file counts as seen if its path or file name appears in
               the task text, an earlier command, or an earlier tool output.

A claim whose names never appeared before it is a candidate for "remembered" (or guessed). This is
only a screening signal: it cannot tell memory from guessing, and a name can be seen without the
fact about it being seen. The traps are what turn this into a causal test.

Usage (from the project root):
    python3 -I scripts/faultyclaims_seen_before.py data/faultyclaims/claims_tidy.jsonl \
        data/faultyclaims/claims_seen_before.jsonl
"""

import collections
import json
import os
import re
import sys

# Code-like names. Plain English words never match these shapes.
CODE_PATTERNS = [
    r"`([^`\n]{2,80})`",                                   # anything in backticks
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b",  # dotted.name or file.ext
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b",                   # snake_case
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",                   # CONSTANT_CASE
    r"\b[a-z]+[A-Z][A-Za-z0-9]*\b",                         # camelCase
    r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b",                 # PascalCase with an inner capital
    r"\b[A-Z]{2,}[a-z][A-Za-z0-9]{2,}\b",                   # names like IClientWellKnown, HTTPServer
]


def identifiers(text):
    """Return the code-like names in a piece of text (at least 4 characters long)."""
    found = []
    for pattern in CODE_PATTERNS:
        for match in re.finditer(pattern, text):
            token = match.group(1) if match.groups() else match.group(0)
            token = token.strip().rstrip("().,:;").lstrip("(")
            if " " in token:  # backticked phrase: keep only its code-like words
                found += [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", token) if len(t) >= 4]
            elif len(token) >= 4 and not re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", token):  # skip "e.g."
                found.append(token)
    return list(dict.fromkeys(found))  # unique, in order


def message_texts(messages):
    """Split a trajectory into: the task text, and per message what the agent could read or did.

    Returns (task, outputs, commands): outputs[i] is the tool output text of message i (or ""),
    commands[i] is the shell commands the agent issued in message i (or "")."""
    task = next(m["content"] for m in messages if m["role"] == "user")
    outputs, commands = [], []
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        outputs.append(content if m["role"] == "tool" else "")
        cmds = []
        for call in m.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments") or ""
            cmds.append(args if isinstance(args, str) else json.dumps(args))
        commands.append("\n".join(cmds))
    return task, outputs, commands


def first_index(needle, texts):
    """Index of the first text that contains needle, or None."""
    for i, t in enumerate(texts):
        if needle in t:
            return i
    return None


def main(tidy_path, out_path, project_root):
    rows_by_traj = collections.defaultdict(list)
    with open(tidy_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows_by_traj[r["trajectory_path"]].append(r)

    groups = collections.defaultdict(collections.Counter)
    with open(out_path, "w", encoding="utf-8") as out:
        for traj_path, rows in rows_by_traj.items():
            with open(os.path.join(project_root, traj_path), encoding="utf-8") as f:
                messages = json.load(f)["messages"]
            task, outputs, commands = message_texts(messages)
            seen_text = [o + "\n" + c for o, c in zip(outputs, commands)]  # for files
            cache_out, cache_file = {}, {}  # first message index per name (None = never)

            for r in rows:
                here = r["message_index"]  # the claim's own assistant message

                # Identifiers: seen if in the task, or in a tool output before this message.
                idents = identifiers(r["quote"])
                unseen = []
                for name in idents:
                    last = name.split(".")[-1]
                    keys = [name] + ([last] if last != name and len(last) >= 4 else [])
                    in_task = any(k in task for k in keys)
                    if not in_task:
                        for k in keys:
                            if k not in cache_out:
                                cache_out[k] = first_index(k, outputs)
                        earliest = min([cache_out[k] for k in keys if cache_out[k] is not None], default=None)
                        if earliest is None or earliest >= here:
                            unseen.append(name)

                # Files: seen if the path or file name was in the task, a command or an output before.
                files_unseen = []
                for path in r["related_files"]:
                    keys = [path, os.path.basename(path)]
                    if any(k in task for k in keys):
                        continue
                    for k in keys:
                        if k not in cache_file:
                            cache_file[k] = first_index(k, seen_text)
                    earliest = min([cache_file[k] for k in keys if cache_file[k] is not None], default=None)
                    if earliest is None or earliest >= here:
                        files_unseen.append(path)

                rec = {"unit_id": r["unit_id"], "identifiers": idents, "identifiers_unseen": unseen,
                       "files_unseen": files_unseen}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Which comparison groups this claim belongs to.
                names = ["all claims"]
                if r["verdict"] == "supported":
                    names.append("supported")
                if r["faulty"]:
                    names.append("refuted (agent)")
                if r["faulty"] and r["proof_pattern"] in ("INTERFACE", "EXECUTION"):
                    names.append("refuted INTERFACE/EXECUTION")
                if r["faulty"] and r["network_none"]:
                    names.append("refuted, no-network runs")
                if r["manual_confirmed"]:
                    names.append("manual " + r["error_form"])
                    if r["error_form"] in ("Misstatement", "Misreference"):
                        names.append("manual Misst.+Misref.")
                        if r["network_none"]:
                            names.append("manual Misst.+Misref., no-net")
                if r["manual_reviewed"] and not r["manual_confirmed"]:
                    names.append("manual overturned")
                if r["key_claim"]:
                    names.append("key faulty claims")
                for g in names:
                    c = groups[g]
                    c["n"] += 1
                    if idents:
                        c["with_idents"] += 1
                        c["some_unseen"] += bool(unseen)
                    if r["related_files"]:
                        c["with_files"] += 1
                        c["file_unseen"] += bool(files_unseen)

    order = ["all claims", "supported", "refuted (agent)", "refuted, no-network runs",
             "refuted INTERFACE/EXECUTION", "manual Misstatement", "manual Misreference",
             "manual Misst.+Misref.", "manual Misst.+Misref., no-net", "manual Overgeneralization",
             "manual Omission", "manual overturned", "key faulty claims"]
    print(f"Wrote {out_path}\n")
    print(f"{'group':<30}{'claims':>8}{'w/ names':>10}{'some name unseen':>18}{'w/ files':>10}{'file unseen':>13}")
    for g in order:
        c = groups[g]
        if not c["n"]:
            continue
        pct_u = 100 * c["some_unseen"] / c["with_idents"] if c["with_idents"] else 0
        pct_f = 100 * c["file_unseen"] / c["with_files"] if c["with_files"] else 0
        print(f"{g:<30}{c['n']:>8,}{c['with_idents']:>10,}{c['some_unseen']:>10,} ({pct_u:4.1f}%)"
              f"{c['with_files']:>10,}{c['file_unseen']:>6,} ({pct_f:4.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 -I scripts/faultyclaims_seen_before.py <claims_tidy.jsonl> <out.jsonl>")
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1], sys.argv[2], project)
