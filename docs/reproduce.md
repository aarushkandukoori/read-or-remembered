# Reproducing the results

Everything below runs on a laptop (tested on an Apple M4, 24 GB) from the project root. Only the agent runs on traps
call a paid API (DeepSeek, a few cents per run). Each analysis has a write-up in `notes/` with its exact commands,
outputs and caveats; this page is the short version, in order.

## 0. Setup

    uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt

Raw data is downloaded, not stored in git (all under `data/`, gitignored).

## 1. Data

    # public mini-swe-agent runs (with reasoning) on SWE-bench Verified
    python3 -I scripts/fetch_runs.py data/raw 20251201_mini-v1.17.1_deepseek-v3.2-reasoner 20260217_mini-v2.0.0_kimi-k2-5-high 20260217_mini-v2.0.0_minimax-2-5-high
    hf download princeton-nlp/SWE-bench_Verified --repo-type dataset --local-dir data/meta/swebv
    git clone --filter=blob:none --no-checkout https://github.com/<org>/<repo>.git data/repos/<org>_<repo>   # per Verified repo
    # You Wang et al.'s artifact (Zenodo 10.5281/zenodo.23093905, CC-BY-4.0), unpacked into data/faultyclaims/
    python3 -I scripts/load_faultyclaims.py data/faultyclaims/reasoning_replication data/faultyclaims/claims_tidy.jsonl
    # DeepSWE trials (Datacurve); see the script's --help for the sample we used
    python3 -I scripts/fetch_deepswe.py --help

## 2. RQ1: provenance

    # claims (You Wang et al.'s SWE-bench Pro runs): notes/rq1-claims.md
    python3 -I scripts/rq1_pro_repos.py clone && python3 -I scripts/rq1_pro_repos.py evidence && python3 -I scripts/rq1_pro_repos.py snapshots
    python3 -I scripts/rq1_pro_repos.py trees base && python3 -I scripts/rq1_pro_repos.py trees manifest && python3 -I scripts/rq1_pro_repos.py trees snap
    .venv/bin/python -I scripts/rq1_claim_provenance.py features && .venv/bin/python -I scripts/rq1_claim_provenance.py summary
    # repository facts at scale (SWE-bench Verified + DeepSWE): notes/rq1-facts.md
    .venv/bin/python -I scripts/rq1_clone_deepswe.py data/deepswe data/repos_deepswe
    .venv/bin/python -I scripts/rq1_fact_provenance.py index swebv && .venv/bin/python -I scripts/rq1_fact_provenance.py index deepswe
    .venv/bin/python -I scripts/rq1_fact_provenance.py vocab
    .venv/bin/python -I scripts/rq1_fact_provenance.py run swebv --workers 4 && .venv/bin/python -I scripts/rq1_fact_provenance.py run deepswe --workers 3
    .venv/bin/python -I scripts/rq1_fact_provenance.py summarize
    # facts the final patches rely on (agent-targeted trap candidates): notes/agent-targeted-candidates.md
    .venv/bin/python -I scripts/agent_targeted_candidates.py extract && .venv/bin/python -I scripts/agent_targeted_candidates.py report
    # error-type annotation vs the 773 human labels: blind batches, LLM labels, agreement
    python3 -I scripts/make_annotation_batches.py && python3 -I scripts/annotation_agreement.py
    # provenance classes for the 257 no-network Misstatement/Misreference claims
    python3 -I scripts/make_provenance_batches.py
    # LLM-annotated labels: outputs/provenance/labels_A (all 257), labels_B (44, independent); then
    .venv/bin/python -I scripts/provenance_summary.py            # summary.json, tables in notes/provenance.md

## 3. RQ2: memory traps

    .venv/bin/python -m pytest -q tests            # trap builder unit tests
    # pilot traps (django-11149, sympy-13615): build, validate, run, grade, analyze
    .venv/bin/python -I scripts/trap_pipeline.py prepare sympy__sympy-13615
    .venv/bin/python -I scripts/trap_pipeline.py validate sympy__sympy-13615
    RR_NONET=1 RR_TAG=-nonet .venv/bin/python -I scripts/trap_pipeline.py run sympy__sympy-13615 trap 1 deepseek/deepseek-flash
    # agent-targeted traps (django-15563, django-16100): scripts/targeted/, write-up notes/targeted-traps.md
    .venv/bin/python -I scripts/targeted/targeted.py --help
    # trap catalog: Docker validation in the official SWE-bench images, then the breakdown by type
    .venv/bin/python -I scripts/docker_validate_traps.py run --deadline 03:03
    .venv/bin/python -I scripts/docker_validate_traps.py report
    .venv/bin/python -I scripts/trap_validation_breakdown.py     # notes/trap-validation.md

## 4. RQ3: disclosure

    .venv/bin/python -I scripts/rq3_disclosure.py

## 4b. RQ4: RecallAudit

    .venv/bin/python -I scripts/recall_audit_sample.py           # items to label (data/recall_audit/batches)
    # LLM-annotated labels go to data/recall_audit/labels/b_NN.jsonl; then
    .venv/bin/python -I scripts/recall_audit_train.py            # metrics.json + models/recall-audit/ (notes/recall-audit.md)

## 5. Memorization probes (local MLX models)

    python3 -I scripts/probe_memory.py build
    for m in qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b qwen2.5-coder-1.5b qwen2.5-coder-7b qwen3-1.7b r1-distill-1.5b; do python3 -I scripts/probe_memory.py run $m; done

## 6. Figures

    python3 scripts/make_figures.py

## 7. Release bundle

    .venv/bin/python -I scripts/build_release_data.py 0.2        # release/rr-derived-data-v0.2.zip
