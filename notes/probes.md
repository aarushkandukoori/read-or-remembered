# Memorization probes with local models (E5, Nov 13 2026)

Script: `scripts/probe_memory.py` (build, run, report). Outputs: `outputs/probes/probes.jsonl` (one row per probe
item with every model's numbers and answers), `outputs/probes/summary.json`, `outputs/probes/fig_*.csv`.
Intermediate files: `data/probes_work/` (gitignored). No paid APIs, no network except the model downloads.

## Bottom line

148 probe items (2 pilot facts, 63 agent-relied facts, 9 catalog trap facts, 74 paired random controls) x 8 local
models, about 90 minutes of model time on the M4 (the GPU was shared with another project's MLX job the whole time).

- **Every model prefers the real line over the trapped one for most facts, but it does the same for random
  functions no agent used.** Controls: memory strength 6.15 to 9.76 nats, trapped line less likely in 80% to 96% of
  controls. So "the model prefers the real fact" is mostly general code knowledge plus what the 40 lines of
  context show, not proof that a specific fact was memorized.
- **Facts the agents relied on are only a little more expected than their own controls:** +1.24 to +3.01 nats
  (all 74 pairs); the 95% interval excludes 0 for Qwen2.5-0.5B, 3B, 7B, Coder-7B and Qwen3-1.7B. After removing pairs
  where the context already shows the fact (46 pairs left), only Qwen2.5-7B stays above 0 (+2.73 [0.18, 5.38]).
- **Placebo check points the same way.** For agent facts, in Qwen2.5-1.5B, 3B, 7B and both coder models, the trap
  costs 2.48 to 3.79 nats more than an edit that keeps every caller-facing fact (adding `**kwargs`), intervals above
  0 (Qwen2.5-0.5B, Qwen3-1.7B and the R1 distill: intervals include 0); for controls the difference is near 0 (-2.01
  to +1.89, every interval includes 0).
- **Size gradient is general, not specific.** From 0.5B to 7B, memory strength rises by +3.38 nats [1.58, 4.98] on
  agent facts and by +3.52 [2.23, 4.84] on controls: bigger models know repo code better across the board. The
  pilot facts show the steepest climb (+11.36): `get_permission_codename(action, opts)` goes from -0.70 (0.5B
  prefers the trapped order) to +13.85 (7B); `sift(seq, keyfunc)` from -0.67 to +7.49. The memory tier shows no
  gradient (+0.27 [-3.46, 3.85]): it is already high at 0.5B (15.59 nats), largely from conventions visible in
  context (7 of 15 show the fact in the 40 lines; 9 of 15 come from one DeepSeek run, sympy-19040).
- **Coder vs general.** Likelihood: no difference at 7B (agent facts -0.20 [-1.22, 0.78]); Coder-1.5B is above
  1.5B by about the same amount on facts (+1.33) and controls (+1.15). Closed-book recall: Coder-7B beats 7B on agent
  facts (+14 points [2, 27]; memory tier +27 points [7, 53]) but not on controls (-1 point).
- **Closed-book recall is weak and does not explain the likelihood preference.** "Fact correct" is 10% to 38% for
  the facts and 4% to 26% for the controls; full exact signatures are at most 27% in any group. No model recalled
  either pilot signature (7B: `def get_permission_codename(perm_name, model_opts, obj=None):`), yet with the file
  context the same 7B prefers the real `(action, opts)` order by 13.85 nats. Facts beat controls on recall by
  0 (Qwen2.5-7B) to +15 points; the interval excludes 0 for Coder-1.5B (+14.9 [4.1, 24.3]) and Qwen3-1.7B (+14.9
  [6.8, 24.3]).
- **Catalog trap facts (n=9) are not especially remembered by these models.** Memory strength 1.37 to 6.84 nats,
  below their paired controls in 7 of 8 models (-3.87 to +0.14, every interval includes 0), and the placebo beats
  the trap in only 22% to 44% of trap items. (The T7/T8 trap names are hand-picked natural renames, the control
  renames are mechanical singular/plural toggles, so this comparison is tilted toward the controls.)
- **The two measures agree.** Memory strength is 1.68 to 5.44 nats higher on items whose closed-book answer got the
  fact right than on items where it did not.

What it means for the project: small open models show plenty of "repo-shaped" expectation, but very little of it
is specific to the facts agents used without reading, and almost none of it is retrievable as an exact signature
without the file in view. Whether the agents' much larger models store these facts has to be measured on the
agents themselves (the trap runs); these probes are a lower-capacity reference point, not that measurement.

## Results (all tables printed by `python3 -I scripts/probe_memory.py report`)

Groups: pilot (2), agent:memory (15), agent:unseen (24), agent:seen (24), trap (9), control (74). Memory strength =
logp(original) - logp(trapped) in nats. "fact not in context" drops pairs where the 40 context lines already show
the fact (same two names in order, same `name=default`, the keyword, or the same module).

### Memory strength by group (mean logp(original) - logp(trapped), nats; 95% bootstrap CI; share of items where the trapped line is less likely)

| model | pilot | agent:memory | agent:unseen | agent:seen | trap | control |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -0.69 [-0.70, -0.67] 0% (n=2) | 15.59 [9.02, 22.42] 100% (n=15) | 8.08 [3.82, 14.09] 83% (n=24) | 7.90 [5.61, 10.65] 96% (n=24) | 1.41 [-1.02, 4.21] 44% (n=9) | 6.15 [4.50, 7.86] 85% (n=74) |
| qwen2.5-1.5b | 0.53 [0.34, 0.72] 100% (n=2) | 14.36 [9.89, 19.08] 100% (n=15) | 8.87 [4.77, 13.89] 88% (n=24) | 10.46 [8.14, 13.06] 100% (n=24) | 2.78 [1.30, 4.36] 89% (n=9) | 7.75 [6.00, 9.56] 95% (n=74) |
| qwen2.5-3b | 7.40 [3.34, 11.47] 100% (n=2) | 17.14 [12.06, 22.78] 100% (n=15) | 10.09 [5.97, 15.36] 88% (n=24) | 13.65 [10.34, 17.66] 100% (n=24) | 5.96 [1.72, 11.04] 89% (n=9) | 9.09 [7.17, 11.01] 93% (n=74) |
| qwen2.5-7b | 10.67 [7.49, 13.86] 100% (n=2) | 15.86 [11.75, 20.08] 100% (n=15) | 11.21 [7.63, 15.66] 96% (n=24) | 13.48 [11.08, 16.01] 100% (n=24) | 6.66 [3.13, 10.97] 89% (n=9) | 9.67 [7.74, 11.60] 92% (n=74) |
| qwen2.5-coder-1.5b | 6.15 [5.77, 6.53] 100% (n=2) | 16.11 [11.95, 20.76] 100% (n=15) | 10.57 [6.41, 16.06] 88% (n=24) | 11.17 [8.36, 14.30] 96% (n=24) | 3.40 [0.94, 5.87] 67% (n=9) | 8.90 [6.94, 10.91] 96% (n=74) |
| qwen2.5-coder-7b | 7.49 [3.97, 11.01] 100% (n=2) | 17.20 [13.00, 22.05] 100% (n=15) | 11.12 [7.62, 15.13] 92% (n=24) | 12.19 [9.77, 14.79] 100% (n=24) | 6.84 [2.86, 12.24] 89% (n=9) | 9.76 [7.96, 11.60] 95% (n=74) |
| qwen3-1.7b | 6.36 [1.98, 10.74] 100% (n=2) | 18.83 [11.29, 27.58] 100% (n=15) | 12.19 [6.65, 19.34] 83% (n=24) | 13.66 [9.97, 17.33] 92% (n=24) | 2.42 [-1.12, 6.39] 67% (n=9) | 9.69 [7.57, 11.93] 95% (n=74) |
| r1-distill-1.5b | -0.13 [-5.94, 5.68] 50% (n=2) | 14.12 [7.51, 21.72] 80% (n=15) | 10.17 [5.02, 18.00] 83% (n=24) | 8.57 [5.76, 11.43] 92% (n=24) | 1.37 [-3.00, 5.35] 56% (n=9) | 7.87 [5.55, 10.41] 80% (n=74) |

### Placebo check (mean logp(placebo) - logp(trapped), nats; share where the placebo beats the trap; mean cost of the placebo edit logp(original) - logp(placebo))

| model | agent | trap | control |
|---|---|---|---|
| qwen2.5-0.5b | 1.62 [-0.13, 3.65] 49%; placebo cost 8.18 | -4.27 [-9.72, -0.19] 33%; placebo cost 5.69 | -1.21 [-3.01, 0.57] 42%; placebo cost 7.36 |
| qwen2.5-1.5b | 2.72 [0.87, 4.73] 64%; placebo cost 8.06 | -2.25 [-5.74, 0.28] 33%; placebo cost 5.04 | 0.34 [-1.28, 2.01] 51%; placebo cost 7.41 |
| qwen2.5-3b | 3.61 [1.36, 5.95] 64%; placebo cost 9.51 | -0.66 [-6.78, 5.03] 22%; placebo cost 6.62 | 0.80 [-1.24, 2.84] 53%; placebo cost 8.29 |
| qwen2.5-7b | 3.79 [1.90, 5.68] 67%; placebo cost 9.39 | -0.01 [-4.55, 4.95] 44%; placebo cost 6.67 | 1.89 [-0.06, 4.00] 61%; placebo cost 7.78 |
| qwen2.5-coder-1.5b | 2.48 [0.52, 4.61] 57%; placebo cost 9.63 | -2.81 [-5.66, -0.36] 33%; placebo cost 6.21 | 0.33 [-1.42, 2.18] 38%; placebo cost 8.58 |
| qwen2.5-coder-7b | 2.93 [1.30, 4.55] 75%; placebo cost 10.05 | -1.29 [-7.23, 4.72] 33%; placebo cost 8.13 | 1.20 [-0.43, 3.00] 50%; placebo cost 8.56 |
| qwen3-1.7b | 2.02 [-0.08, 4.53] 52%; placebo cost 12.32 | -4.31 [-9.29, 0.01] 33%; placebo cost 6.73 | -0.35 [-3.00, 2.28] 43%; placebo cost 10.03 |
| r1-distill-1.5b | -0.71 [-3.47, 2.27] 41%; placebo cost 11.22 | -5.34 [-13.16, 0.60] 22%; placebo cost 6.71 | -2.01 [-4.97, 0.73] 42%; placebo cost 9.88 |

### Each fact against its own control (same repo, commit, kind of edit): memory strength fact - control (95% CI), and closed-book recall accuracy fact vs control

| model | all facts | agent:memory | agent:unseen | agent:seen | trap | all facts, fact not in context |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 2.35 [0.02, 4.73] (n=74); recall 10% vs 4% | 6.22 [-0.15, 14.21] (n=15); recall 7% vs 0% | 2.15 [-1.81, 7.30] (n=24); recall 4% vs 4% | 2.61 [-0.22, 5.37] (n=24); recall 21% vs 8% | -1.90 [-4.70, 1.38] (n=9); recall 0% vs 0% | 1.43 [-0.64, 3.71] (n=46); recall 13% vs 6% |
| qwen2.5-1.5b | 1.78 [-0.04, 3.72] (n=74); recall 26% vs 18% | 3.55 [-0.41, 8.34] (n=15); recall 27% vs 7% | 0.84 [-2.29, 4.35] (n=24); recall 25% vs 12% | 3.42 [0.08, 6.43] (n=24); recall 33% vs 33% | -1.45 [-3.72, 0.87] (n=9); recall 11% vs 11% | 1.64 [-0.77, 4.34] (n=46); recall 30% vs 22% |
| qwen2.5-3b | 3.01 [0.56, 5.62] (n=74); recall 22% vs 14% | 5.86 [-0.28, 12.73] (n=15); recall 20% vs 20% | 1.06 [-3.18, 5.84] (n=24); recall 21% vs 4% | 4.94 [0.91, 9.08] (n=24); recall 33% vs 21% | -0.46 [-6.76, 5.59] (n=9); recall 0% vs 11% | 1.99 [-1.06, 5.24] (n=46); recall 28% vs 15% |
| qwen2.5-7b | 2.65 [0.67, 4.66] (n=74); recall 26% vs 26% | 2.74 [-1.38, 7.00] (n=15); recall 20% vs 20% | 2.22 [-1.38, 5.76] (n=24); recall 42% vs 17% | 4.14 [0.54, 7.50] (n=24); recall 21% vs 33% | 0.14 [-5.58, 5.42] (n=9); recall 11% vs 33% | 2.73 [0.18, 5.38] (n=46); recall 26% vs 30% |
| qwen2.5-coder-1.5b | 1.99 [-0.10, 4.22] (n=74); recall 30% vs 15% | 4.63 [0.86, 8.94] (n=15); recall 27% vs 20% | 1.77 [-2.36, 6.09] (n=24); recall 29% vs 4% | 2.96 [-0.31, 6.41] (n=24); recall 46% vs 21% | -3.87 [-11.29, 2.02] (n=9); recall 0% vs 11% | 1.06 [-1.66, 3.96] (n=46); recall 37% vs 15% |
| qwen2.5-coder-7b | 2.32 [0.39, 4.32] (n=74); recall 38% vs 24% | 5.24 [0.96, 9.88] (n=15); recall 47% vs 7% | 1.90 [-1.82, 5.63] (n=24); recall 29% vs 21% | 2.80 [-0.20, 5.46] (n=24); recall 54% vs 38% | -1.51 [-6.87, 4.10] (n=9); recall 11% vs 33% | 1.46 [-0.87, 3.83] (n=46); recall 41% vs 30% |
| qwen3-1.7b | 2.98 [0.21, 5.98] (n=74); recall 22% vs 7% | 3.96 [-3.36, 12.90] (n=15); recall 13% vs 13% | 3.28 [-0.79, 8.03] (n=24); recall 17% vs 4% | 4.88 [0.67, 9.11] (n=24); recall 38% vs 8% | -2.93 [-6.76, 0.85] (n=9); recall 11% vs 0% | 2.12 [-0.61, 5.09] (n=46); recall 30% vs 6% |
| r1-distill-1.5b | 1.24 [-1.40, 3.92] (n=74) | -1.53 [-9.15, 6.80] (n=15) | 1.88 [-2.76, 7.17] (n=24) | 4.91 [2.05, 7.92] (n=24) | -3.10 [-7.12, 1.19] (n=9) | 0.93 [-1.31, 3.17] (n=46) |

### Closed-book recall (fact correct: right order / default / keyword present / module; full exact = every parameter name, order and default right)

| model | pilot | agent:memory | agent:unseen | agent:seen | trap | control |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0% (exact 0%, n=2) | 7% (exact 0%, n=15) | 4% (exact 0%, n=24) | 21% (exact 4%, n=24) | 0% (exact 0%, n=9) | 4% (exact 0%, n=74) |
| qwen2.5-1.5b | 0% (exact 0%, n=2) | 27% (exact 27%, n=15) | 25% (exact 8%, n=24) | 33% (exact 17%, n=24) | 11% (exact 11%, n=9) | 18% (exact 8%, n=74) |
| qwen2.5-3b | 0% (exact 0%, n=2) | 20% (exact 13%, n=15) | 21% (exact 4%, n=24) | 33% (exact 12%, n=24) | 0% (exact 0%, n=9) | 14% (exact 7%, n=74) |
| qwen2.5-7b | 0% (exact 0%, n=2) | 20% (exact 7%, n=15) | 42% (exact 12%, n=24) | 21% (exact 17%, n=24) | 11% (exact 0%, n=9) | 26% (exact 14%, n=74) |
| qwen2.5-coder-1.5b | 0% (exact 0%, n=2) | 27% (exact 20%, n=15) | 29% (exact 4%, n=24) | 46% (exact 12%, n=24) | 0% (exact 0%, n=9) | 15% (exact 7%, n=74) |
| qwen2.5-coder-7b | 0% (exact 0%, n=2) | 47% (exact 27%, n=15) | 29% (exact 8%, n=24) | 54% (exact 17%, n=24) | 11% (exact 11%, n=9) | 24% (exact 11%, n=74) |
| qwen3-1.7b | 0% (exact 0%, n=2) | 13% (exact 7%, n=15) | 17% (exact 4%, n=24) | 38% (exact 12%, n=24) | 11% (exact 11%, n=9) | 7% (exact 3%, n=74) |
| r1-distill-1.5b | - | - | - | - | - | - |

### Gradients: paired per-item differences between models (memory strength, nats; recall, points)

| contrast | group | n | memory strength diff [95% CI] | recall diff [95% CI] |
|---|---|---|---|---|
| qwen2.5-7b minus qwen2.5-0.5b | agent | 63 | 3.38 [1.58, 4.98] | 0.17 [0.05, 0.30] |
| qwen2.5-7b minus qwen2.5-0.5b | agent:memory | 15 | 0.27 [-3.46, 3.85] | 0.13 [0.00, 0.33] |
| qwen2.5-7b minus qwen2.5-0.5b | control | 74 | 3.52 [2.23, 4.84] | 0.22 [0.12, 0.32] |
| qwen2.5-7b minus qwen2.5-0.5b | pilot | 2 | 11.36 [8.16, 14.56] | 0.00 [0.00, 0.00] |
| qwen2.5-7b minus qwen2.5-0.5b | trap | 9 | 5.24 [1.03, 10.38] | 0.11 [0.00, 0.33] |
| qwen2.5-7b minus qwen2.5-1.5b | agent | 63 | 2.40 [1.28, 3.43] | 0.00 [-0.11, 0.11] |
| qwen2.5-7b minus qwen2.5-1.5b | agent:memory | 15 | 1.50 [-1.21, 4.23] | -0.07 [-0.27, 0.13] |
| qwen2.5-7b minus qwen2.5-1.5b | control | 74 | 1.91 [0.90, 2.94] | 0.08 [0.00, 0.18] |
| qwen2.5-7b minus qwen2.5-1.5b | pilot | 2 | 10.14 [7.15, 13.13] | 0.00 [0.00, 0.00] |
| qwen2.5-7b minus qwen2.5-1.5b | trap | 9 | 3.87 [0.69, 7.90] | 0.00 [-0.33, 0.33] |
| qwen2.5-3b minus qwen2.5-0.5b | agent | 63 | 3.33 [1.71, 5.02] | 0.14 [0.05, 0.24] |
| qwen2.5-3b minus qwen2.5-0.5b | agent:memory | 15 | 1.56 [-1.67, 4.54] | 0.13 [0.00, 0.33] |
| qwen2.5-3b minus qwen2.5-0.5b | control | 74 | 2.94 [1.95, 4.05] | 0.10 [0.03, 0.18] |
| qwen2.5-3b minus qwen2.5-0.5b | pilot | 2 | 8.09 [4.01, 12.17] | 0.00 [0.00, 0.00] |
| qwen2.5-3b minus qwen2.5-0.5b | trap | 9 | 4.55 [0.45, 10.06] | 0.00 [0.00, 0.00] |
| qwen2.5-coder-7b minus qwen2.5-7b | agent | 63 | -0.20 [-1.22, 0.78] | 0.14 [0.02, 0.27] |
| qwen2.5-coder-7b minus qwen2.5-7b | agent:memory | 15 | 1.34 [-1.01, 3.29] | 0.27 [0.07, 0.53] |
| qwen2.5-coder-7b minus qwen2.5-7b | control | 74 | 0.10 [-0.73, 0.89] | -0.01 [-0.12, 0.10] |
| qwen2.5-coder-7b minus qwen2.5-7b | pilot | 2 | -3.18 [-3.52, -2.85] | 0.00 [0.00, 0.00] |
| qwen2.5-coder-7b minus qwen2.5-7b | trap | 9 | 0.18 [-1.79, 1.97] | 0.00 [-0.33, 0.33] |
| qwen2.5-coder-1.5b minus qwen2.5-1.5b | agent | 63 | 1.33 [0.26, 2.34] | 0.06 [0.00, 0.14] |
| qwen2.5-coder-1.5b minus qwen2.5-1.5b | agent:memory | 15 | 1.75 [-1.16, 4.70] | 0.00 [0.00, 0.00] |
| qwen2.5-coder-1.5b minus qwen2.5-1.5b | control | 74 | 1.15 [0.28, 2.06] | -0.03 [-0.11, 0.05] |
| qwen2.5-coder-1.5b minus qwen2.5-1.5b | pilot | 2 | 5.62 [5.04, 6.19] | 0.00 [0.00, 0.00] |
| qwen2.5-coder-1.5b minus qwen2.5-1.5b | trap | 9 | 0.62 [-1.01, 2.18] | -0.11 [-0.33, 0.00] |
| qwen2.5-coder-7b minus qwen2.5-coder-1.5b | agent | 63 | 0.86 [-0.34, 1.95] | 0.08 [-0.03, 0.17] |
| qwen2.5-coder-7b minus qwen2.5-coder-1.5b | agent:memory | 15 | 1.09 [-1.26, 3.44] | 0.20 [0.00, 0.40] |
| qwen2.5-coder-7b minus qwen2.5-coder-1.5b | control | 74 | 0.86 [-0.12, 1.85] | 0.10 [-0.01, 0.19] |
| qwen2.5-coder-7b minus qwen2.5-coder-1.5b | pilot | 2 | 1.34 [-2.56, 5.24] | 0.00 [0.00, 0.00] |
| qwen2.5-coder-7b minus qwen2.5-coder-1.5b | trap | 9 | 3.44 [-0.05, 8.10] | 0.11 [0.00, 0.33] |
| qwen3-1.7b minus qwen2.5-1.5b | agent | 63 | 3.55 [1.79, 5.53] | -0.05 [-0.16, 0.06] |
| qwen3-1.7b minus qwen2.5-1.5b | agent:memory | 15 | 4.48 [0.32, 9.80] | -0.13 [-0.33, 0.00] |
| qwen3-1.7b minus qwen2.5-1.5b | control | 74 | 1.93 [0.89, 2.95] | -0.11 [-0.19, -0.03] |
| qwen3-1.7b minus qwen2.5-1.5b | pilot | 2 | 5.83 [1.26, 10.40] | 0.00 [0.00, 0.00] |
| qwen3-1.7b minus qwen2.5-1.5b | trap | 9 | -0.36 [-2.75, 2.27] | 0.00 [0.00, 0.00] |
| r1-distill-1.5b minus qwen2.5-1.5b | agent | 63 | -0.28 [-2.23, 1.79] | - |
| r1-distill-1.5b minus qwen2.5-1.5b | agent:memory | 15 | -0.23 [-4.36, 4.37] | - |
| r1-distill-1.5b minus qwen2.5-1.5b | control | 74 | 0.12 [-1.43, 1.72] | - |
| r1-distill-1.5b minus qwen2.5-1.5b | pilot | 2 | -0.67 [-6.28, 4.95] | - |
| r1-distill-1.5b minus qwen2.5-1.5b | trap | 9 | -1.41 [-5.14, 1.78] | - |

### Pilot traps (logp original / trapped / placebo; memory strength; recalled answer)

| model | fact | lp orig | lp trap | lp placebo | strength | recall answer | order right? |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | get_permission_codename | -31.20 | -30.49 | -35.17 | -0.70 | `def get_permission_codename(perm_name):` | False |
| qwen2.5-0.5b | sift | -37.15 | -36.47 | -43.00 | -0.67 | `def sift(ary, key=0, reverse=False):` | False |
| qwen2.5-1.5b | get_permission_codename | -26.73 | -27.45 | -32.15 | 0.73 | `def get_permission_codename(perm):` | False |
| qwen2.5-1.5b | sift | -38.12 | -38.46 | -42.80 | 0.34 | `def sift(sequences, key):` | False |
| qwen2.5-3b | get_permission_codename | -13.62 | -25.09 | -20.54 | 11.47 | `def get_permission_codename(verb, opts, obj=None):` | False |
| qwen2.5-3b | sift | -31.98 | -35.32 | -35.68 | 3.34 | `def sift(ordered, pred):` | False |
| qwen2.5-7b | get_permission_codename | -7.64 | -21.50 | -19.50 | 13.85 | `def get_permission_codename(perm_name, model_opts, obj=None):` | False |
| qwen2.5-7b | sift | -10.05 | -17.54 | -16.98 | 7.49 | `def sift( iterable, keyfunc ):` | False |
| qwen2.5-coder-1.5b | get_permission_codename | -16.91 | -22.68 | -25.00 | 5.77 | `def get_permission_codename(perm, app_label):` | False |
| qwen2.5-coder-1.5b | sift | -23.72 | -30.25 | -29.18 | 6.53 | `def sift( iterable, key=None, default=None ):` | False |
| qwen2.5-coder-7b | get_permission_codename | -11.44 | -22.45 | -20.60 | 11.01 | `def get_permission_codename(verb, model=None):` | False |
| qwen2.5-coder-7b | sift | -17.98 | -21.95 | -23.96 | 3.97 | `def sift(Iterable, key=None, default=None):` | False |
| qwen3-1.7b | get_permission_codename | -43.60 | -45.58 | -54.90 | 1.98 | `def get_permission_codename(perm):` | False |
| qwen3-1.7b | sift | -28.95 | -39.69 | -34.45 | 10.74 | `def sift(ary, key=None, reverse=False):` | False |
| r1-distill-1.5b | get_permission_codename | -51.16 | -56.84 | -58.58 | 5.68 | `-` | None |
| r1-distill-1.5b | sift | -56.21 | -50.27 | -60.26 | -5.94 | `-` | None |

Other checks (from `summary.json`):
- Batched vs one-at-a-time greedy recall on the same prompts (signature answers, first runs kept for comparison):
  identical text in 84.4% (0.5B, n=90), 80.0% (1.5B, n=90), 90.5% (3B, n=21) and 67.8% (Coder-1.5B, n=90). Greedy
  decoding is sensitive to the small numeric differences batching brings; all reported recall numbers use the
  batched answers, the same method for every model.
- Imports, strict vs loose: "fact correct" needs the defining module; `import_valid` (the stated module exists and
  mentions the name, so re-exports count; star re-exports are missed, so it is a lower bound) is higher for
  controls, e.g. Qwen2.5-7B facts 19% strict / 29% valid, controls 38% / 62%. Models often answer with a valid
  package-level import (`from django.db.models import F`, `from sympy import Mul`).
- Tokenizer boundary problems: 0 (every line was scored from the first token after the context).
- Independent recomputation (a separate script reading the raw result files) matched `summary.json` to 3 decimals
  for all 8 models (control means, paired fact-minus-control differences, pilot strengths).
- Scorer check (LLM-checked, not hand-labeled): I read 20 scored Qwen2.5-1.5B answers; all 20 scores followed the
  rules above (the rules themselves are strict: renamed parameters count as wrong, re-exports count as wrong).


## Question

When a coding agent states or uses a repo fact it never read (argument order, a default value, a keyword name,
the module a name is imported from), is that the kind of fact language models carry in their weights? And do the
facts the agents relied on without reading look more "remembered" than random facts of the same repos?

**These local models are not the agents studied.** DeepSeek-V3.2, Kimi K2.5 and MiniMax M2.5 are far too big to
run here. We probe one open family at several sizes (Qwen2.5 0.5B, 1.5B, 3B, 7B, general and coder versions) plus
two extras (Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B) to see how such memory grows with size and with code
training. The numbers say what small models remember; they do not measure the agents' memory directly.

## Probe items

Built by `python3 -I scripts/probe_memory.py build` (14 s, deterministic: fixed seeds).

| group | what | how many |
|---|---|---|
| pilot | the two pilot traps: django `get_permission_codename(action, opts)` at django-11149's base, sympy `sift(seq, keyfunc, ...)` at sympy-13615's base | 2 |
| agent | facts from `outputs/agent_targeted_facts.jsonl` of the four probe-able types (argument order, default, keyword, import). All memory-tier facts that could be built, plus 6 per type sampled from the "unseen" tier (definition not seen, a use was) and 6 per type from the "seen" tier (definition seen before writing) | 63 (15 memory, 24 unseen, 24 seen) |
| trap | trap specs from `traps/catalog/*/*/spec.json` that are one of the probed kinds: T1 `mod_inverse`, `get_primary_key_column`; T2 `_keep_coeff` sign, `split_having_qualify` must_group_by; T4 `KroneckerDelta`; T7 `salted_hmac` secret, `deconstructible` path; T8 `_create_project_state`, `get_full_path` (keyword renamed and bool default negated) | 9 |
| control | for every fact above, one random function (or import line) from the same repo and base commit, whose name no agent's patch relied on (any model, any task), probed with the same kind of edit | 74 |

A "fact" is distinct per (task base commit, fact). Memory-tier facts dropped: 10 defaults whose value is `None`
("not set"; changing it is not a fair trap) or whose target is ambiguous (`<2 classes>.drop_vars`), and 3 imports
with no one-line absolute import of the name anywhere in the repo (`ValuesListIterable`, `dmp_one_p` only appear in
multi-line imports, `from sympy import asin` is the package's top-level API). Of the 15 memory-tier facts that
remain, 9 come from one DeepSeek run (sympy-19040: `dmp_div`, `dmp_mul`, `dmp_pow`, `dmp_sub`, `dmp_from_dict` and
their imports).

Controls: random library file first (tests, docs, examples, vendored code excluded), then a random function in it
that fits the edit (argument order: first two parameters after self are required; default: a plain bool, int,
float or listed string default; keyword: a parameter with a default). Import controls: a random one-line absolute
`from pkg.mod import Name` where `Name` is a public function or class really defined in `pkg.mod`.

### The three versions of the probed line

The probed line is the real `def` header (all its lines, for multi-line signatures) or the real import line, at
the task's base commit (read with `git show` from `data/repos`, lazy fetching off).

| fact type | trapped line (fact changed) | placebo line (fact kept) |
|---|---|---|
| argument order | the two relied-on parameters swapped: `def get_permission_codename(opts, action):` | `**kwargs` added: `def get_permission_codename(action, opts, **kwargs):` |
| default | the default changed: bool flipped, int 0/1 swapped or doubled, float doubled, a listed string swapped (`'strict'` -> `'ignore'`) | `**kwargs` added |
| keyword | the parameter renamed singular/plural (`code` -> `codes`, `opts` -> `opt`) | `**kwargs` added |
| import | the module replaced by a sibling module that does not mention the name (`django.utils.encoding` -> `django.utils.archive`) | one more real public name from the same module added to the import |

The placebo keeps every fact a caller relies on (all old calls and imports still work), like the rr-targeted
keyword placebo that adds `**_legacy_kwargs`. If a function already has `**kwargs`, `*args` is added instead.

Prompt for the likelihood probe: a one-line comment `# file: <path>`, then the 40 real lines before the probed
line (fewer near the top of a file). The path comment is our addition (agents always know the path).

## Measures

- **(a) Likelihood.** logp(line) = sum over the line's tokens (newline included) of the log-probability the model
  gives each token after the prompt (natural log, nats). **Memory strength = logp(original) - logp(trapped)**: > 0
  means the real fact is preferred. Placebo check: logp(placebo) - logp(trapped) (> 0: the trap costs more than an
  edit that keeps the fact) and the placebo's own cost logp(original) - logp(placebo). The prompt is run once and
  its key/value cache reused for the three lines (`check` mode: same numbers as three plain passes within 0.084 nats
  on 12 line scores from 4 items with Qwen2.5-0.5B; the gap comes from 4-bit/float16 numerics).
- **(b) Closed-book recall.** Chat prompt, no code shown: "In Django 3.0 (2019), what is the exact signature of
  `django.contrib.auth.get_permission_codename`? Reply with only the Python def line, nothing else." (imports:
  "what is the Python import statement for `force_str`?"). Version and year come from the SWE-bench Verified task.
  The answer is prefilled with `def <name>(` (or `from`) so every model answers in the same format; greedy
  decoding; at most 80 new tokens (30 for imports); generation stops at the end of the def/import line or once the
  answer lists 4 more parameters than the real signature (small models repeat parameters until the limit; such
  answers are wrong either way). Scoring: **fact correct** = both relied-on parameters named and in the right order
  / the parameter named with the right default / the keyword present / the name imported from the real module.
  **Full exact** = every parameter name, order and default right (self/cls ignored). For imports the stated module
  must be the defining module (re-exports such as `from django.db.models import F` count as wrong for "fact correct";
  `import_valid` in summary.json counts them).
- **(c) Gradients.** Same items for every model; paired per-item differences between models (7B minus 0.5B, coder
  minus general at 1.5B and 7B, ...) with bootstrap intervals.
- Statistics: means with 95% percentile bootstrap intervals (2,000 resamples over items, seed 0). Facts are compared
  with their own controls item by item (paired). Items are not fully independent (several facts can share one
  signature, e.g. five facts on `LogisticRegression.__init__`), so the intervals are somewhat too narrow.

## Commands (exact)

```
# models (5.47 GB new, mlx-community 4-bit; the other five were already cached)
python3 -I -c "from huggingface_hub import snapshot_download
for r in ['mlx-community/Qwen2.5-0.5B-Instruct-4bit','mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit','mlx-community/Qwen2.5-Coder-7B-Instruct-4bit']:
    print(snapshot_download(r, allow_patterns=['*.json','*.safetensors','*.txt']))"
python3 -I scripts/probe_memory.py build
python3 -I scripts/probe_memory.py check qwen2.5-0.5b
# final runs (each run = likelihood pass, then one batched recall pass; both resume where they stopped)
for m in qwen2.5-3b qwen2.5-7b qwen2.5-coder-7b qwen2.5-0.5b qwen2.5-1.5b qwen2.5-coder-1.5b qwen3-1.7b r1-distill-1.5b; do
  python3 -I scripts/probe_memory.py run $m; done
# trap catalog, add-only (06:40 and again around 07:30), each followed by the same loop (only new items get scored)
python3 -I scripts/probe_memory.py build --catalog-anyway
python3 -I scripts/probe_memory.py report
```

Environment: Apple M4, 24 GB; system Python 3.14.6, mlx 0.32.3, mlx-lm 0.32.0, transformers 5.13.1. Model
snapshots: Qwen2.5-0.5B-Instruct-4bit a5339a41, Qwen2.5-1.5B-Instruct-4bit 8b403126, Qwen2.5-3B-Instruct-4bit
4f83f8f1, Qwen2.5-7B-Instruct-4bit c26a38f6, Qwen2.5-Coder-1.5B-Instruct-4bit b3252a2f, Qwen2.5-Coder-7B-Instruct-4bit
019cc73c, Qwen3-1.7B-4bit 3b1b1768, DeepSeek-R1-Distill-Qwen-1.5B-4bit 933185be.

## Trap catalog timing

`traps/catalog/READY` never appeared. I waited the full 60 minutes (to 06:40) while running the other groups, then
added the specs that existed (4 probe-able traps), and around 07:30 added the ones written since (5 more), each time
with `build --catalog-anyway` (add-only: earlier items unchanged) and a catch-up `run` for every model. Skipped
spec types: T3 renames (4; a rename has no fair automatic rule for controls and the recall question names the
function), T5 exception swaps and T6 return conventions (not facts on a signature or import line). T4
`get_level_tags` was skipped because no one-line import of it exists at the base commit. Specs written after that
are not included.

## Caveats

- **Not the agents' models.** Qwen2.5 (0.5B to 7B), Qwen2.5-Coder, Qwen3-1.7B and an R1 distill, all 4-bit
  (`mlx-community`). The agents (DeepSeek-V3.2, Kimi K2.5, MiniMax M2.5) are much larger and trained on more recent
  data; their memory could be far stronger. These probes cannot say what the agents remembered.
- **Context, not memory, drives much of the likelihood signal.** The 40 lines often show conventions (sympy's
  `dmp_*(f, g, u, K)`) or sibling imports from the same module. The "fact not in context" filter is a simple regex
  and cannot catch looser cues (similar names, the docstring below the line is not shown, but the comment above is).
- **Small and lopsided sets.** Memory tier n=15, 9 of them from one task (sympy-19040, one DeepSeek run); trap n=9
  across 5 types; pilot n=2. Results for these groups are case studies. Several items share a signature (five facts
  on `LogisticRegression.__init__`, three on `force_str`), so items are not independent and the bootstrap
  intervals are too narrow.
- **Rule-made trap lines.** Keyword traps use singular/plural toggles (`code` -> `codes`, `verbose` -> `verboses`),
  import traps a random sibling module, default traps fixed flips; some of these are unnatural and cost likelihood
  for that reason alone. Facts and their controls use the same rules (except catalog T7/T8, see above).
- **Placebo length.** The placebo adds tokens (`, **kwargs` or one more imported name), the trap mostly does not, so
  the placebo's cost includes predicting extra tokens. Read "placebo beats trap" as a rough scale, not a clean null.
- **Recall scoring is strict** (names must match exactly; a valid re-export is "wrong" for the strict import score)
  and the answer is prefilled with `def <name>(`. The Qwen2.5 chat template adds its default system prompt. The R1
  distill was scored on likelihood only (its answers need long thinking first). Batched greedy decoding gives the
  same text as one-at-a-time decoding for 68% to 91% of answers (see above).
- **Build reproducibility.** Which import line is picked for an import fact depends on which test-file blobs are
  already downloaded in `data/repos` (other agents fetch more over time), so a fresh `build --fresh` can pick other
  lines for a few import items. The exact probe set used is saved in `data/probes_work/items.jsonl` and in full
  (contexts and all three lines) in `outputs/probes/probes.jsonl`; `build` without `--fresh` only adds new items.
- **Runtime.** About 90 minutes of wall-clock model time in total, including passes that were replaced: a first
  one-question-at-a-time recall pass (kept only for the batched-vs-sequential check), one aborted start of the
  0.5B run before the stop rule for rambling answers was added, and the 7B-8bit model was skipped (the other
  project's job held about 10 GB of GPU memory). The machine also swapped heavily near the end (17.5 of 18.4 GB swap
  in use), which affected speed only.

