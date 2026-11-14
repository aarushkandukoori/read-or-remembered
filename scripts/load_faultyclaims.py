"""Load You Wang et al.'s labeled claims (arXiv 2610.32338) into one tidy JSONL file.

Their Zenodo artifact (DOI 10.5281/zenodo.23093905, file reasoning_replication.tar.xz) has
the labels spread over several files. This script joins them, one row per claim
("reasoning unit"), 76,537 rows in total.

Where each piece comes from (paths inside reasoning_replication/):
  rq1/results/<model>/<instance>/reasoning_units.json   the claims (LLM-extracted), O/G/S/F type,
                                                         CoT quote + character span, step, key-claim link
  rq2/results/<model>/<instance>/result.json            the verdict of their assessment agent
                                                         (supported / refuted / unresolved), the
                                                         explanation, and the cited evidence
  rq2/fi_verification.json                              manual check of the 896 refuted claims in 40
                                                         sampled trajectories: confirmed or not, and the
                                                         paper's error form (Misreference, Misstatement,
                                                         Omission, Overgeneralization)
  rq3/key_faulty_claims.json                            the 182 key faulty claims (linked to patch defects)
                                                         with the corrected statement used in the replay
  rq3/rerollout/rerollout_outcomes.json                 replay results for each key claim
  data/resolution_status.json                           resolved / unresolved per trajectory
  data/<model dir>/<instance>.json                      the trajectory itself (mini-swe-agent format)

Two different "category" labels exist, so the output keeps them apart:
  error_form     the paper's taxonomy (Table 3, Fig. 5b). Manual. Only for the 773 confirmed faulty
                 claims in the 40 reviewed trajectories (10 tasks x 4 models). Null everywhere else.
  proof_pattern  a label their assessment agent gave every refuted claim (CONTRACT, INTERFACE,
                 EXECUTION, GUARANTEE, NECESSITY, OTHER; defined in src/rq2/prompt.md). Automatic.
                 Not described in the paper.
  ogsf_type      O/G/S/F: what the claim is about (paper Table 2, Fig. 5a). For every claim.

Usage (from the project root):
    python3 -I scripts/load_faultyclaims.py data/faultyclaims/reasoning_replication \
        data/faultyclaims/claims_tidy.jsonl

The artifact is downloaded data: this script only reads it as JSON, it never runs anything from it.
"""

import collections
import json
import os
import sys

# The artifact uses two names per model: a short label (rq1/rq2 folders) and a folder name for
# the trajectories (data/). key_faulty_claims.json uses a third spelling. The paper names are last.
MODELS = {
    # label:        (trajectory folder,  name in key_faulty_claims.json,  name in the paper)
    "deepseek-v32": ("deepseek-v3.2", "DeepSeek-V3.2", "DeepSeek-V3.2"),
    "flash": ("deepseek-flash", "deepseek-flash", "DeepSeek-V4.1-Flash"),
    "qwen36": ("qwen3.6-27b", "qwen3.6-27b", "Qwen3.6-27B"),
    "qwen38": ("qwen3.8-27b", "qwen3.8-27b", "Qwen3.8-27B"),
}

# Numbers printed in the paper, used only to check that the join reproduces them.
PAPER = {
    "units": 76537, "supported": 62587, "refuted": 8540, "unresolved": 5410,
    "refuted_per_model": {"deepseek-v32": 1789, "flash": 2865, "qwen36": 1179, "qwen38": 2707},
    "reviewed": 896, "confirmed": 773,
    "error_forms": {"Misstatement": 412, "Overgeneralization": 191, "Misreference": 87, "Omission": 83},
    "key_claims": 182, "key_claims_refuted": 150,
}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_evidence(evidence_ids, evidence_table):
    """Turn evidence ids like 'B001-E002' into small records: where the evidence came from."""
    out = []
    for eid in evidence_ids or []:
        e = evidence_table.get(eid, {})
        rec = {"id": eid, "kind": e.get("kind"), "source": e.get("source"),
               "start_line": e.get("start_line"), "end_line": e.get("end_line")}
        if e.get("kind") == "web":
            rec["url"] = (e.get("web_provenance") or {}).get("url")
        out.append(rec)
    return out


def main(root, out_path, project_root):
    # 1) Small lookup tables that do not depend on the instance.
    status = read_json(os.path.join(root, "data", "resolution_status.json"))

    # Manual check: fi_id looks like "<model>::<instance>::step_0007::unit_02".
    manual = {e["fi_id"]: e for e in read_json(os.path.join(root, "rq2", "fi_verification.json"))}

    # Key faulty claims, keyed by their id "<name>::<instance>::KFI-01".
    key_claims = {}
    for name, instances in read_json(os.path.join(root, "rq3", "key_faulty_claims.json")).items():
        for instance, record in instances.items():
            for fi in record["fis"]:
                key_claims[fi["key_fi_id"]] = fi

    # Replay outcomes, keyed by "single_faulty_claim/<model dir>/<instance>/KFI-01.json".
    replays = read_json(os.path.join(root, "rq3", "rerollout", "rerollout_outcomes.json"))

    rows = 0
    counts = collections.defaultdict(collections.Counter)  # counts[model][what] = n
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for label, (traj_dir, _, paper_name) in MODELS.items():  # middle name not needed here
            rq1_dir = os.path.join(root, "rq1", "results", label)
            for instance in sorted(os.listdir(rq1_dir)):
                # 2) Load the three files for this trajectory.
                units = read_json(os.path.join(rq1_dir, instance, "reasoning_units.json"))
                result = read_json(os.path.join(root, "rq2", "results", label, instance, "result.json"))
                traj_file = os.path.join(root, "data", traj_dir, instance + ".json")
                trajectory = read_json(traj_file)
                messages = trajectory["messages"]
                n_steps = sum(1 for m in messages if m.get("role") == "assistant")
                traj_rel = os.path.relpath(traj_file, project_root)
                # How the agent's container was set up. The DeepSeek runs had no network and no git
                # history; the Qwen runs (stock mini-swe-agent 2.1.0) had neither restriction.
                env = trajectory["info"]["config"]["environment"]
                network_none = "--network=none" in (env.get("run_args") or [])
                git_pruned = bool(env.get("prune_git_history"))

                # RQ2 stores units per step with a 1-based index; rebuild the RQ1 id to join them.
                verdicts = {}
                for step in result["steps"]:
                    for u in step["units"]:
                        verdicts[f"{instance}::{step['step_id']}::unit_{u['unit_index']:02d}"] = u

                # 3) One output row per claim.
                for unit in units:
                    v = verdicts[unit["id"]]  # every unit has a verdict (checked: 76,537 of 76,537)
                    unit_id = f"{label}::{unit['id']}"
                    m = manual.get(unit_id)  # only refuted claims in the 40 reviewed trajectories
                    key_id = unit.get("key_fi_id")
                    key = key_claims.get(key_id) if key_id else None
                    replay = None
                    if key_id:
                        replay = replays.get(
                            f"single_faulty_claim/{traj_dir}/{instance}/{key_id.split('::')[-1]}.json")
                    span = unit.get("source_span") or {}
                    row = {
                        "model": label,
                        "model_name": paper_name,
                        "instance_id": instance,
                        "unit_id": unit_id,
                        "step": unit["source_step"]["assistant_ordinal"],  # 1-based agent step
                        "step_id": unit["source_step"]["step_id"],
                        "message_index": unit["source_step"]["message_index"],  # index in "messages"
                        "n_steps": n_steps,
                        "claim": unit["claim"],  # normalized by their LLM
                        "quote": unit["original_text"],  # verbatim from the CoT
                        "quote_start": span.get("start"),  # character offsets in reasoning_content
                        "quote_end": span.get("end"),
                        "ogsf_type": unit["ogsf"],
                        "related_files": unit["related_files"],
                        "verdict": v["evidence_status"],  # supported / refuted / unresolved
                        "faulty": bool(v["FI"]),  # True exactly when refuted
                        "error_form": m.get("error_form") if m else None,
                        "manual_reviewed": m is not None,
                        "manual_confirmed": m["FI"] if m else None,
                        "proof_pattern": v.get("fi_type"),
                        "evidence_finding": v["evidence_finding"],
                        "rationale": v["rationale"],
                        "evidence": short_evidence(v.get("evidence_ids"), result.get("evidence", {})),
                        "key_claim": key_id is not None,
                        "key_fi_id": key_id,
                        "key_quote": key["quote"] if key else None,
                        "key_correction": key["fi_statement_corrected"] if key else None,
                        "replay_targeted_fix": replay.get("targeted_fix_repaired") if replay else None,
                        "replay_more_tests": replay.get("more_tests") if replay else None,
                        "replay_resolved": replay.get("became_resolved") if replay else None,
                        "trajectory_resolved": status[traj_dir][instance] == "resolved",
                        "network_none": network_none,  # agent container had no network access
                        "git_history_pruned": git_pruned,  # agent could not read upstream history
                        "trajectory_path": traj_rel,  # relative to the project root
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1

                    # 4) Counts for the summary.
                    c = counts[label]
                    c["units"] += 1
                    c["verdict:" + row["verdict"]] += 1
                    if row["faulty"]:
                        c["faulty:" + row["ogsf_type"]] += 1
                        c["pattern:" + str(row["proof_pattern"])] += 1
                    c["network_none"] += row["network_none"]
                    if row["manual_reviewed"]:
                        c["reviewed"] += 1
                    if row["manual_confirmed"]:
                        c["confirmed"] += 1
                        c["form:" + row["error_form"]] += 1
                        c["form_type:" + row["error_form"] + ":" + row["ogsf_type"]] += 1
                        if row["error_form"] in ("Misstatement", "Misreference") and row["related_files"]:
                            c["mm_with_files"] += 1
                    if row["key_claim"]:
                        c["key"] += 1
                        c["key_verdict:" + row["verdict"]] += 1
                        if row["manual_reviewed"]:
                            c["key_reviewed"] += 1
                        if row["error_form"]:
                            c["key_form:" + row["error_form"]] += 1
                        if row["replay_targeted_fix"]:
                            c["key_replay_fixed"] += 1
                        if row["replay_resolved"]:
                            c["key_replay_resolved"] += 1
            counts[label]["trajectories"] = len(os.listdir(rq1_dir))

    print_summary(rows, counts, out_path)


def total(counts, key):
    return sum(c[key] for c in counts.values())


def print_summary(rows, counts, out_path):
    labels = list(MODELS)
    print(f"Wrote {rows:,} claims to {out_path}\n")

    print("Claims and verdicts per model (assessment agent, all 400 trajectories)")
    print(f"{'model':<14}{'traj':>6}{'units':>9}{'supported':>11}{'refuted':>9}{'unresolved':>12}{'err rate':>10}")
    for lab in labels:
        c = counts[lab]
        rate = 100 * c["verdict:refuted"] / c["units"]
        print(f"{lab:<14}{c['trajectories']:>6}{c['units']:>9,}{c['verdict:supported']:>11,}"
              f"{c['verdict:refuted']:>9,}{c['verdict:unresolved']:>12,}{rate:>9.2f}%")
    print(f"{'all':<14}{total(counts, 'trajectories'):>6}{total(counts, 'units'):>9,}"
          f"{total(counts, 'verdict:supported'):>11,}{total(counts, 'verdict:refuted'):>9,}"
          f"{total(counts, 'verdict:unresolved'):>12,}")
    print("Claims from runs with no network access: "
          + ", ".join(f"{lab}={counts[lab]['network_none']:,}" for lab in labels) + "\n")

    print("Refuted claims by O/G/S/F type (paper Fig. 5a)")
    for lab in labels:
        c = counts[lab]
        print(f"  {lab:<14}" + "  ".join(f"{t}={c['faulty:' + t]:,}" for t in "OGSF"))
    print("  all           " + "  ".join(f"{t}={total(counts, 'faulty:' + t):,}" for t in "OGSF") + "\n")

    print("Refuted claims by automatic proof_pattern (artifact only, not in the paper)")
    patterns = ["EXECUTION", "INTERFACE", "GUARANTEE", "CONTRACT", "NECESSITY", "OTHER"]
    for lab in labels:
        c = counts[lab]
        print(f"  {lab:<14}" + "  ".join(f"{p}={c['pattern:' + p]}" for p in patterns))
    print("  all           " + "  ".join(f"{p}={total(counts, 'pattern:' + p)}" for p in patterns) + "\n")

    forms = ["Misstatement", "Misreference", "Overgeneralization", "Omission"]
    print("Manual error forms (paper Fig. 5b): only the 40 reviewed trajectories (10 tasks x 4 models)")
    print(f"{'model':<14}{'reviewed':>9}{'confirmed':>10}" + "".join(f"{f[:12]:>14}" for f in forms))
    for lab in labels:
        c = counts[lab]
        print(f"{lab:<14}{c['reviewed']:>9}{c['confirmed']:>10}" + "".join(f"{c['form:' + f]:>14}" for f in forms))
    print(f"{'all':<14}{total(counts, 'reviewed'):>9}{total(counts, 'confirmed'):>10}"
          + "".join(f"{total(counts, 'form:' + f):>14}" for f in forms) + "\n")

    print("Error form x O/G/S/F type (paper Fig. 5c), all models")
    for f in forms:
        print(f"  {f:<20}" + "  ".join(f"{t}={total(counts, 'form_type:' + f + ':' + t)}" for t in "OGSF"))
    mm = total(counts, "form:Misstatement") + total(counts, "form:Misreference")
    print(f"  Misstatement + Misreference (explicit factual errors): {mm}, "
          f"of which {total(counts, 'mm_with_files')} name a related file\n")

    print("Key faulty claims (RQ3)")
    for lab in labels:
        c = counts[lab]
        print(f"  {lab:<14}key={c['key']:<4} refuted by agent={c['key_verdict:refuted']:<4} "
              f"replay fixed defect={c['key_replay_fixed']:<4} replay resolved={c['key_replay_resolved']:<4} "
              f"in reviewed sample={c['key_reviewed']}")
    print(f"  all           key={total(counts, 'key')} refuted by agent={total(counts, 'key_verdict:refuted')} "
          f"replay fixed defect={total(counts, 'key_replay_fixed')} replay resolved={total(counts, 'key_replay_resolved')} "
          f"in reviewed sample={total(counts, 'key_reviewed')}")
    print("  key claims by manual error form: "
          + ", ".join(f"{f}={total(counts, 'key_form:' + f)}" for f in forms) + "\n")

    print("Check against the paper's printed numbers")
    checks = [
        ("units", total(counts, "units"), PAPER["units"]),
        ("supported", total(counts, "verdict:supported"), PAPER["supported"]),
        ("refuted", total(counts, "verdict:refuted"), PAPER["refuted"]),
        ("unresolved", total(counts, "verdict:unresolved"), PAPER["unresolved"]),
        ("manually reviewed", total(counts, "reviewed"), PAPER["reviewed"]),
        ("manually confirmed", total(counts, "confirmed"), PAPER["confirmed"]),
        ("key claims", total(counts, "key"), PAPER["key_claims"]),
        ("key claims refuted by agent", total(counts, "key_verdict:refuted"), PAPER["key_claims_refuted"]),
    ]
    checks += [(f"refuted {lab}", counts[lab]["verdict:refuted"], n) for lab, n in PAPER["refuted_per_model"].items()]
    checks += [(f, total(counts, "form:" + f), n) for f, n in PAPER["error_forms"].items()]
    for name, got, want in checks:
        print(f"  {name:<30}{got:>8,}  paper {want:>8,}  {'ok' if got == want else 'MISMATCH'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 -I scripts/load_faultyclaims.py <reasoning_replication dir> <out.jsonl>")
    artifact_root = os.path.abspath(sys.argv[1])
    # The project root is the folder that contains scripts/ (trajectory paths are stored relative to it).
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(artifact_root, sys.argv[2], project)
