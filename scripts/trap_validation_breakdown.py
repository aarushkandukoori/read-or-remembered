"""Break the Docker validation of the trap catalog down by trap type and by loud/quiet.

Reads outputs/traps/validation.json (from scripts/docker_validate_traps.py report) and
traps/catalog/index.json (type, type title and loud/quiet of each pair). Writes
outputs/traps/validation_breakdown.json and prints the tables.

Usage: .venv/bin/python -I scripts/trap_validation_breakdown.py
"""
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "outputs/traps/validation.json"
INDEX = ROOT / "traps/catalog/index.json"
OUT = ROOT / "outputs/traps/validation_breakdown.json"

CHECK_ORDER = ["original_nofix", "original_gold", "trap_nofix", "trap_gold",
               "trap_memory", "placebo_nofix", "placebo_gold", "placebo_memory"]


def frac(check):
    """'passed/total' of the FAIL_TO_PASS tests in one check."""
    total = check["f2p_passed"] + len(check["f2p_failed"])
    return f"{check['f2p_passed']}/{total}"


def effect(loud_or_quiet):
    """loud, quiet, or both (T8 in the index: loud by keyword name, quiet by positional value)."""
    return loud_or_quiet if loud_or_quiet in ("loud", "quiet") else "both"


def main():
    val = json.loads(VALIDATION.read_text())
    index = json.loads(INDEX.read_text())
    meta = {(e["instance_id"], e["trap_id"]): e for e in index["entries"]}

    rows = []
    for p in val["pairs"]:
        e = meta[(p["instance_id"], p["trap_id"])]
        assert e["type"] == p["type"] and e["loud_or_quiet"] == p["loud_or_quiet"], p["trap_id"]
        c = p["checks"]
        all_ran = all(c[k]["status"] == "ran" for k in CHECK_ORDER)
        rows.append(OrderedDict(
            instance_id=p["instance_id"], trap_id=p["trap_id"], type=e["type"],
            type_title=e["type_title"], effect=effect(e["loud_or_quiet"]),
            loud_or_quiet_raw=e["loud_or_quiet"],
            n_F2P=p["n_F2P"], n_P2P=p["n_P2P"],
            # target (FAIL_TO_PASS) tests passed, as "passed/total"
            trap_nofix=frac(c["trap_nofix"]), trap_gold=frac(c["trap_gold"]),
            upstream_fix_in_trap=frac(c["trap_memory"]),
            upstream_fix_in_placebo=frac(c["placebo_memory"]),
            # regression (PASS_TO_PASS) tests the upstream fix breaks
            p2p_broken_by_upstream_fix_in_trap=len(c["trap_memory"]["p2p_failed"]),
            p2p_broken_by_upstream_fix_in_placebo=len(c["placebo_memory"]["p2p_failed"]),
            all_8_checks_ran=all_ran,
            n_checks_ok=sum(bool(c[k]["ok"]) for k in CHECK_ORDER),
            failed_checks=p["failed_checks"], extra_checks_failed=p["extra_checks_failed"],
            verdict=p["verdict"],
            # earlier local (non-Docker) check from the catalog index; None = not run
            local_all_good=(e["local_check"] or {}).get("all_good")))

    def group(key):
        out = OrderedDict()
        for r in sorted(rows, key=lambda r: r[key]):
            g = out.setdefault(r[key], OrderedDict(built=0, valid=0, invalid_pairs=[]))
            g["built"] += 1
            if r["verdict"] == "VALID":
                g["valid"] += 1
            else:
                g["invalid_pairs"].append(f"{r['instance_id']}/{r['trap_id']}")
        return out

    by_type = group("type")
    for t, g in by_type.items():
        g["title"] = next(r["type_title"] for r in rows if r["type"] == t)
    summary = OrderedDict(
        source=["outputs/traps/validation.json", "traps/catalog/index.json"],
        command=".venv/bin/python -I scripts/trap_validation_breakdown.py",
        n_pairs=len(rows),
        n_tasks=len({r["instance_id"] for r in rows}),
        n_valid=sum(r["verdict"] == "VALID" for r in rows),
        n_all_8_checks_ok=sum(r["n_checks_ok"] == 8 for r in rows),
        n_placebo_upstream_fix_breaks_p2p=sum(r["p2p_broken_by_upstream_fix_in_placebo"] > 0 for r in rows),
        invalid_fail_only_check_b=all(r["failed_checks"] == ["trap_memory"] and not r["extra_checks_failed"]
                                      for r in rows if r["verdict"] != "VALID"),
        local_vs_docker=OrderedDict(
            local_ran=sum(r["local_all_good"] is not None for r in rows),
            local_all_good=sum(r["local_all_good"] is True for r in rows),
            disagree=[f"{r['instance_id']}/{r['trap_id']}: local {r['local_all_good']}, Docker {r['verdict']}"
                      for r in rows if r["local_all_good"] != (r["verdict"] == "VALID")]),
        by_type=by_type,
        by_effect=group("effect"),
        pilot=OrderedDict(instance_id=val["pilot"]["instance_id"], verdict=val["pilot"]["verdict"],
                          failed_checks=val["pilot"]["failed_checks"]),
        pairs=rows)
    OUT.write_text(json.dumps(summary, indent=1) + "\n")

    print(f"pairs {summary['n_pairs']} in {summary['n_tasks']} tasks; VALID {summary['n_valid']}; "
          f"all 8 checks ok {summary['n_all_8_checks_ok']}; invalid fail only (b): "
          f"{summary['invalid_fail_only_check_b']}; pairs where the upstream fix breaks a P2P test in the "
          f"placebo: {summary['n_placebo_upstream_fix_breaks_p2p']}; pilot {summary['pilot']['verdict']}")
    print("local vs Docker:", json.dumps(summary["local_vs_docker"]))
    print("\n| type | title | built | valid | invalid |\n|---|---|---|---|---|")
    for t, g in by_type.items():
        print(f"| {t} | {g['title']} | {g['built']} | {g['valid']} | {', '.join(g['invalid_pairs']) or '-'} |")
    print("\n| effect | built | valid | invalid |\n|---|---|---|---|")
    for t, g in summary["by_effect"].items():
        print(f"| {t} | {g['built']} | {g['valid']} | {', '.join(g['invalid_pairs']) or '-'} |")
    print("\n| task | trap | type | effect | no fix | gold | upstream fix, trap | upstream fix, placebo "
          "| P2P broken (trap) | checks ok | verdict |")
    for r in rows:
        print(f"| {r['instance_id']} | {r['trap_id']} | {r['type']} | {r['effect']} | {r['trap_nofix']} "
              f"| {r['trap_gold']} | {r['upstream_fix_in_trap']} | {r['upstream_fix_in_placebo']} "
              f"| {r['p2p_broken_by_upstream_fix_in_trap']} | {r['n_checks_ok']}/8 | {r['verdict']} |")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
