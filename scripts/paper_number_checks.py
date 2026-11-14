"""Re-count the few numbers the paper skeleton (paper/) states as already known.

The skeleton copies most numbers from notes/pilot-2026-11-13.md and notes/agent-targeted-candidates.md.
This script re-derives the ones that come straight from data files, so the draft does not rest on
copied text alone. It only reads files; it writes nothing.

Usage (from the project root):
    .venv/bin/python -I scripts/paper_number_checks.py

What it prints:
  1. You Wang et al.'s claims (data/faultyclaims/claims_tidy.jsonl): number of claim units, the
     manual error-form labels (Misstatement, Misreference, ...), and which models ran with the
     network off and pruned git history.
  2. The go-redis IdleTimeout claim used as the opening example (the full record).
  3. Agent-targeted facts (outputs/agent_targeted_facts.jsonl): model-level facts, how many were
     unseen, in the memory tier ("no use seen") and strong, and in how many tasks.
  4. DeepSWE sample (outputs/deepswe/sample_trials.csv): number of trials and models.
"""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def yes(x):
    """The tidy file stores some flags as True and some as the string 'True'."""
    return x is True or x == "True"


# ---- 1. You Wang et al.'s claims ----
claims = [json.loads(line) for line in open(ROOT / "data/faultyclaims/claims_tidy.jsonl", encoding="utf-8")]
print("claim units:", len(claims))
forms = collections.Counter(c["error_form"] for c in claims if c["error_form"] not in (None, "None"))
print("manual error-form labels:", dict(forms), "total", sum(forms.values()))
print("Misstatement + Misreference:", forms["Misstatement"] + forms["Misreference"])
verdicts = collections.Counter(c["verdict"] for c in claims)
print("verdicts:", dict(verdicts))
per_model = collections.defaultdict(collections.Counter)
for c in claims:
    per_model[c["model_name"]][(yes(c["network_none"]), yes(c["git_history_pruned"]))] += 1
for name, counts in sorted(per_model.items()):
    print(f"  {name}: (network_none, git_history_pruned) -> {dict(counts)}")

# ---- 2. The go-redis claim ----
for c in claims:
    if "IdleTimeout" in c["claim"] and "5 minutes" in c["claim"]:
        print("\ngo-redis claim:")
        for key in ("model_name", "instance_id", "step", "n_steps", "claim", "verdict", "error_form",
                    "manual_confirmed", "evidence_finding", "rationale", "trajectory_resolved",
                    "network_none", "git_history_pruned"):
            print(f"  {key}: {c[key]}")

# ---- 2b. What the go-redis run had seen before the claim (Figure 1a) ----
# The trajectory is from You Wang et al.'s artifact (CC-BY-4.0). Steps = assistant messages.
traj_path = ROOT / ("data/faultyclaims/reasoning_replication/data/deepseek-flash/"
                    "instance_flipt-io__flipt-492cc0b158200089dceede3b1aba0ed28df3fb1d.json")
msgs = json.load(open(traj_path, encoding="utf-8"))["messages"]


def step_of(i):
    """Step number of message i = how many assistant messages there are up to and including it."""
    return sum(1 for m in msgs[:i + 1] if m.get("role") == "assistant")


claim_msg = 94  # message_index of the claim in claims_tidy.jsonl
for i, m in enumerate(msgs[:claim_msg]):
    if m.get("role") == "tool" and "Default is 30 minutes" in (m.get("content") or ""):
        command = json.loads(msgs[i - 1]["tool_calls"][0]["function"]["arguments"])["command"]
        print(f"  first output with 'Default is 30 minutes': message {i} (step {step_of(i - 1)}), command: {command}")
        break
reasoning = msgs[claim_msg].get("reasoning_content") or ""
print(f"  claim message {claim_msg} = step {step_of(claim_msg)} of {step_of(len(msgs) - 1)}")
print("  hidden reasoning contains 'I also have a memory of':", "I also have a memory of" in reasoning)
print("  visible text of that step:", repr(msgs[claim_msg].get("content")))
earlier_outputs = " ".join(str(m.get("content") or "") for m in msgs[:claim_msg] if m.get("role") == "tool")
print("  'IdleTimeout' in any tool output before the claim:", "IdleTimeout" in earlier_outputs)

# ---- 3. Agent-targeted facts ----
# Same rule as per_model_fact() in agent_targeted_candidates.py: only the trap-relevant fact types,
# and a fact used on several patch lines counts once, at the line the agent wrote first.
TRAP_TYPES = ("arg_order", "default", "keyword", "import", "raises", "exception_class", "attribute")
best = {}
for line in open(ROOT / "outputs/agent_targeted_facts.jsonl", encoding="utf-8"):
    f = json.loads(line)
    if f["fact_type"] not in TRAP_TYPES:
        continue  # rows like "no repo facts" carry no fact
    key = (f["model"], f["instance_id"], f["fact_key"])
    cut = f["write_step"] if f.get("write_step") is not None else (f.get("n_steps") or 0) + 1
    if key not in best or cut < best[key][0]:
        best[key] = (cut, f)
facts = {k: v[1] for k, v in best.items()}
print("\nmodel-level facts (model x task x fact):", len(facts))
for flag in ("unseen", "no_use", "strong"):
    hits = [k for k, v in facts.items() if v[flag]]
    print(f"  {flag}: {len(hits)} facts in {len({k[1] for k in hits})} tasks")
# Appendix table: relied on / unseen / no use seen / strong, per fact type and model.
models = sorted({k[0] for k in facts})
print("  per type (relied / unseen / no use / strong):")
for ftype in TRAP_TYPES:
    cells = []
    for m in models + ["all"]:
        sub = [v for k, v in facts.items() if v["fact_type"] == ftype and (m == "all" or k[0] == m)]
        cells.append(f"{len(sub)}/{sum(bool(v['unseen']) for v in sub)}/"
                     f"{sum(bool(v['no_use']) for v in sub)}/{sum(bool(v['strong']) for v in sub)}")
    print(f"    {ftype:16s}", "  ".join(f"{m}={c}" for m, c in zip(models + ["all"], cells)))

# Cross-model agreement: per candidate (task x fact), how many models had it unseen / no use / strong.
cands = [json.loads(line) for line in open(ROOT / "outputs/agent_targeted_candidates.jsonl", encoding="utf-8")]
print("candidates:", len(cands), "in", len({c["instance_id"] for c in cands}), "tasks")
for field in ("n_unseen", "n_no_use", "n_strong"):
    for k in (1, 2, 3):
        exactly = sum(1 for c in cands if int(c[field]) == k)
        tasks_at_least = len({c["instance_id"] for c in cands if int(c[field]) >= k})
        print(f"  {field}: exactly {k} models = {exactly} candidates; tasks with >= {k} = {tasks_at_least}")

# ---- 4. DeepSWE sample ----
rows = list(csv.DictReader(open(ROOT / "outputs/deepswe/sample_trials.csv", encoding="utf-8")))
print("\nDeepSWE sample trials:", len(rows), "models:", len({r["model"] for r in rows}),
      "model x effort configs:", len({(r["model"], r["reasoning_effort"]) for r in rows}))
for source in sorted({r["source"] for r in rows}):
    sub = [r for r in rows if r["source"] == source]
    print(f"  source {source}: {len(sub)} trials on {len({r['task_name'] for r in sub})} tasks")
with_reasoning = sum(1 for r in rows if float(r["frac_steps_with_reasoning"] or 0) > 0)
print("  trials with reasoning text in at least one step:", with_reasoning)

# ---- 5. Pilot trap validation and edit sizes ----
# Changed lines = '+' or '-' lines of the edit diff, without the '+++'/'---' headers.
# (validate_trap.py and trap_pipeline.py use l[:1] in '+-', which also matches the empty string after
# the diff's last newline, so their counts are one higher: 79/77 in the notes are 78/76 here.)
def changed_lines(path):
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    changed = sum(1 for l in lines if l[:1] in ("+", "-") and l[:3] not in ("+++", "---"))
    files = sum(1 for l in lines if l.startswith("+++ "))
    return changed, files

print()
for task in ("django__django-11149", "sympy__sympy-13615"):
    for version in ("trap", "placebo"):
        changed, files = changed_lines(ROOT / "work" / task / f"{version}.edit.diff")
        print(f"{task} {version}: {changed} changed lines in {files} files")
django = json.load(open(ROOT / "outputs/trap_validation_django__django-11149.json", encoding="utf-8"))
for version, res in django.items():
    print("django__django-11149", version, {k: res[k] for k in ("no fix", "gold'", "memory fix")})
sympy = json.load(open(ROOT / "work/sympy__sympy-13615/validation.json", encoding="utf-8"))
for version, res in sympy.items():
    print("sympy__sympy-13615", version, {k: f"F2P {v['F2P']}, P2P {v['P2P']}" for k, v in res.items()})
