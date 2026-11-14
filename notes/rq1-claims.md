# RQ1: was the code behind a claim READ before the claim? (You Wang et al.'s SWE-bench Pro claims, Nov 13 2026)

Free, local, deterministic. No API calls. Data: You Wang et al.'s artifact (arXiv 2610.32338, Zenodo, CC-BY-4.0),
100 SWE-bench Pro tasks x 4 models = 400 mini-swe-agent trajectories, 76,537 claims with verdicts and cited evidence.
Main analysis = the two DeepSeek models (DeepSeek-V3.2, DeepSeek-V4.1-Flash), which ran with `--network=none` and pruned
git history. The two Qwen models (Qwen3.6-27B, Qwen3.8-27B) had network and full git history: reported separately.

## Bottom line

- **Most faulty claims come after the agent has seen at least part of the cited code.** DeepSeek refuted claims with
  repository evidence: all cited lines seen before the claim 46.0%, some 45.5%, **none 8.6%** (95% CI 7.7-9.5).
  Supported controls: 60.0% / 33.8% / **6.2%** (5.5-7.0). So "never looked" is rare for both. Refuted claims differ
  mainly by partial reading (some: 45.5% vs 33.8%); "none" is only 2.4 points higher.
- **Explicit factual errors (Misstatement + Misreference, no network, n=257) look the same:** all 46.9%, some 46.0%,
  none 7.0% (4.3-11.3). This points mostly to read-but-misread or read-only-part, not to pure recall.
- **The memory signal is in the names.** DeepSeek refuted claims name a code identifier that does not exist at the base
  commit (and is not in the task, not seen or written earlier in the run) 9.2% of the time vs 4.8% for supported claims;
  a name that exists only in OLDER or NEWER versions of the repo: 3.6% vs 1.8%. For claims about the original
  implementation (type O) the gap is larger: 9.4% vs 3.6%, and other-version names 4.0% vs 1.1% (about 3.6x).
- Qwen (network on) shows the same pattern with more "none": refuted 12.9% vs supported 7.9%.

## What was computed (scripts/rq1_claim_provenance.py)

Claims: all 8,540 refuted claims + a random sample of 8,540 supported claims (`random.Random(0).sample` over the
supported rows in file order) = 17,080 rows. `cid = 'c%05d' % i`, i = 0-based row in `data/faultyclaims/claims_tidy.jsonl`.

1. **Repo and base commit** per trajectory: `base_commit` in the artifact's `rq2/results/<model>/<instance>/result.json`
   (100 tasks, 100 distinct base commits, 11 repos). Each repository citation also records the exact file version
   (`repository_provenance.blob_oid`), so the cited lines are read from git by blob id. 36,611 citations read;
   36,610 reproduce the artifact's own sha256 of the file and of the excerpt (the one mismatch is a PNG test fixture).
2. **evidence_seen_before** (all / some / none / no_repo_evidence): every non-trivial cited line (union over the
   claim's repository citations, a line cited twice counts once) is looked up in the tool outputs (role `tool`)
   with message index < the claim's `message_index`.
   - whitespace collapsed; tool display prefixes ignored (line numbers from cat -n / nl / grep -n, `path:12:` from
     grep -rn, `+`/`-`/`>` from diffs and pytest);
   - trivial lines skipped: shorter than 4 characters, or only punctuation, numbers, keywords and literals
     (`}`, `else:`, `return nil, err`);
   - **anchored** (main): the output must be about the same file (its file name appears in that output or in the
     command that produced it). `evidence_seen_before_any` drops this condition (sensitivity check);
   - no_repo_evidence: no repository citation (3,148 claims) or only trivial cited lines (4 claims).
3. **file_seen_before**: same classes per cited file: was ANY non-trivial line of that file version shown before?
4. **Identifiers**: backticked or code-shaped names in claim + quote (snake_case, camelCase, dotted.names, file paths
   with an extension). Symbols: whole-word match in the repo's code files at the base commit (source, templates,
   config; no docs, no JSON, no vendored/minified code), read from git in one pass per repo (156,939 file versions).
   Paths: `git ls-tree` at the base commit. Missing names then get:
   - external: the name's first part (the whole name if it has no dots) is a standard-library/builtin name or is
     imported by the repo from a third-party package (Python/JS/TS/Go import lines at the base commit; the repo's own
     packages come from its go.mod/package.json files);
   - stale / future / both / never: found only in older / only in newer / in both / in none of the history
     snapshots (default-branch commit at Jan 1 of every other year from 4 years before the repo's first base commit,
     plus the newest commit; 5-7 per repo). **Deviation from the task:** `git log -S` over all refs needs every
     historical file version; a blobless clone does not have them and fetching them would break the 3 GB cap, so
     the snapshots stand in for history (a name that lived only between two snapshots is missed and counts as never).
   - flags: in_task, seen_before (earlier tool output), written_before (earlier agent command), in_own_command.
     "Unexplained missing" = missing at base, not external, none of the four flags. "other_version" = at least one
     unexplained name that is stale/future/both.
5. Summary tables with 95% Wilson intervals (`outputs/rq1/claims_summary.json`).

## Commands (run from the project root, in this order)

```
python3 -I scripts/rq1_pro_repos.py clone        # 11 blobless clones + internetarchive/infogami (openlibrary's vendor/infogami submodule)
python3 -I scripts/rq1_pro_repos.py evidence     # 5,421 cited file versions, all fetched
python3 -I scripts/rq1_pro_repos.py snapshots    # writes data/repos_pro/snapshots.json
python3 -I scripts/rq1_pro_repos.py trees base   # code files at the 100 base commits
python3 -I scripts/rq1_pro_repos.py trees manifest
python3 -I scripts/rq1_pro_repos.py trees snap   # code files at the 67 snapshot commits
python3 -I scripts/rq1_pro_repos.py size         # 630 MB in total (cap 3 GB)
.venv/bin/python -I scripts/rq1_claim_provenance.py features    # 3 min 17 s; 17,080 rows
.venv/bin/python -I scripts/rq1_claim_provenance.py summary
.venv/bin/python -I scripts/rq1_claim_provenance.py handcheck 15 0
.venv/bin/python -I scripts/rq1_claims_extra_checks.py none-sample 10 1
.venv/bin/python -I scripts/rq1_claims_extra_checks.py other-version-sample 12 7
.venv/bin/python -I scripts/rq1_claims_extra_checks.py fix-names c35238:refreshSmartPlaylist c11167:qt_64 c33900:aesGCM c29326:AuditBackoffTimeout c28102:collectDeviceData c05301:ExactVersionMatch c22557:mapstructure c16123:HAS_SELINUX c21657:FLIPT_CACHE_MEMORY_ENABLED
GIT_NO_LAZY_FETCH=1 git -C data/repos_pro/gravitational__teleport grep -l -w -F collectDeviceData <base|HEAD> -- '*.go'   # same for flipt/mapstructure
.venv/bin/python -I scripts/rq1_claims_extra_checks.py view <trajectory.json> <message index> "<text>"   # used for the hand check
```

`features` was run twice; the second run's output is byte-identical to the first.

## Results

Cells: n (% of claims WITH repository evidence [95% Wilson CI]); no_repo_evidence is a plain count.

### Refuted vs supported x evidence_seen_before (anchored)

| subset | verdict | n | all | some | none | no_repo_evidence |
|---|---|---|---|---|---|---|
| DeepSeek, no network (main) | refuted | 4654 | 1803 (46.0% [44.4-47.5]) | 1784 (45.5% [43.9-47.0]) | 336 (8.6% [7.7-9.5]) | 731 |
| DeepSeek, no network (main) | supported | 4695 | 2242 (60.0% [58.4-61.5]) | 1264 (33.8% [32.3-35.3]) | 233 (6.2% [5.5-7.0]) | 956 |
| Qwen, network on | refuted | 3886 | 1311 (41.0% [39.3-42.7]) | 1477 (46.2% [44.4-47.9]) | 412 (12.9% [11.8-14.1]) | 686 |
| Qwen, network on | supported | 3845 | 1807 (58.9% [57.2-60.7]) | 1017 (33.2% [31.5-34.9]) | 242 (7.9% [7.0-8.9]) | 779 |
| all models | refuted | 8540 | 3114 (43.7% [42.6-44.9]) | 3261 (45.8% [44.6-46.9]) | 748 (10.5% [9.8-11.2]) | 1417 |
| all models | supported | 8540 | 4049 (59.5% [58.3-60.7]) | 2281 (33.5% [32.4-34.6]) | 475 (7.0% [6.4-7.6]) | 1735 |

When a claim is "some", about half of its cited lines were seen (mean share: DeepSeek refuted 0.55, supported 0.59).
Looser matching (any earlier output, not tied to the file): DeepSeek none 6.3% [5.6-7.1] refuted vs 5.1% [4.4-5.8]
supported; Qwen 9.2% vs 6.3%. file_seen_before (any part of each cited file shown): DeepSeek refuted all/some/none
69.0% / 24.3% / 6.7%, supported 79.6% / 15.8% / 4.6%.

Claims about the original implementation only (type O, DeepSeek): refuted (n=2,321) all 44.1%, some 45.4%,
none 10.4% [9.2-11.9]; supported (n=1,990) all 60.6%, some 32.2%, none 7.2% [6.1-8.5].

### The 773 manually confirmed faulty claims: error form x evidence_seen_before (anchored)

| subset | error form | n | all | some | none | no_repo_evidence |
|---|---|---|---|---|---|---|
| all models | Misreference | 87 | 25 (34.7% [24.8-46.2]) | 42 (58.3% [46.8-69.0]) | 5 (6.9% [3.0-15.2]) | 15 |
| all models | Misstatement | 412 | 139 (41.1% [36.0-46.4]) | 159 (47.0% [41.8-52.4]) | 40 (11.8% [8.8-15.7]) | 74 |
| all models | Omission | 83 | 39 (59.1% [47.0-70.1]) | 19 (28.8% [19.3-40.6]) | 8 (12.1% [6.3-22.1]) | 17 |
| all models | Overgeneralization | 191 | 78 (46.4% [39.1-54.0]) | 80 (47.6% [40.2-55.1]) | 10 (5.9% [3.3-10.6]) | 23 |
| DeepSeek | Misreference | 41 | 12 (34.3% [20.8-50.8]) | 21 (60.0% [43.6-74.5]) | 2 (5.7% [1.6-18.6]) | 6 |
| DeepSeek | Misstatement | 216 | 88 (49.4% [42.2-56.7]) | 77 (43.3% [36.2-50.6]) | 13 (7.3% [4.3-12.1]) | 38 |
| DeepSeek | Omission | 50 | 25 (65.8% [49.9-78.8]) | 7 (18.4% [9.2-33.4]) | 6 (15.8% [7.4-30.4]) | 12 |
| DeepSeek | Overgeneralization | 112 | 46 (48.4% [38.6-58.3]) | 43 (45.3% [35.6-55.3]) | 6 (6.3% [2.9-13.1]) | 17 |
| Qwen | Misreference | 46 | 13 (35.1% [21.8-51.2]) | 21 (56.8% [40.9-71.3]) | 3 (8.1% [2.8-21.3]) | 9 |
| Qwen | Misstatement | 196 | 51 (31.9% [25.1-39.5]) | 82 (51.2% [43.6-58.9]) | 27 (16.9% [11.9-23.4]) | 36 |
| Qwen | Omission | 33 | 14 (50.0% [32.6-67.4]) | 12 (42.9% [26.5-60.9]) | 2 (7.1% [2.0-22.7]) | 5 |
| Qwen | Overgeneralization | 79 | 32 (43.8% [33.1-55.2]) | 37 (50.7% [39.5-61.8]) | 4 (5.5% [2.1-13.3]) | 6 |

### Model x network x verdict (anchored)

| model | verdict | n | all | some | none | no_repo_evidence |
|---|---|---|---|---|---|---|
| DeepSeek-V3.2 (network none) | refuted | 1789 | 624 (39.4% [37.0-41.8]) | 777 (49.0% [46.6-51.5]) | 183 (11.6% [10.1-13.2]) | 205 |
| DeepSeek-V3.2 (network none) | supported | 1559 | 665 (50.8% [48.1-53.5]) | 530 (40.5% [37.9-43.2]) | 114 (8.7% [7.3-10.4]) | 250 |
| DeepSeek-V4.1-Flash (network none) | refuted | 2865 | 1179 (50.4% [48.4-52.4]) | 1007 (43.0% [41.1-45.1]) | 153 (6.5% [5.6-7.6]) | 526 |
| DeepSeek-V4.1-Flash (network none) | supported | 3136 | 1577 (64.9% [63.0-66.8]) | 734 (30.2% [28.4-32.1]) | 119 (4.9% [4.1-5.8]) | 706 |
| Qwen3.6-27B (network on) | refuted | 1179 | 539 (51.8% [48.8-54.9]) | 413 (39.7% [36.8-42.7]) | 88 (8.5% [6.9-10.3]) | 139 |
| Qwen3.6-27B (network on) | supported | 1430 | 644 (56.8% [53.9-59.7]) | 392 (34.6% [31.9-37.4]) | 98 (8.6% [7.1-10.4]) | 296 |
| Qwen3.8-27B (network on) | refuted | 2707 | 772 (35.7% [33.8-37.8]) | 1064 (49.3% [47.1-51.4]) | 324 (15.0% [13.6-16.6]) | 547 |
| Qwen3.8-27B (network on) | supported | 2415 | 1163 (60.2% [58.0-62.4]) | 625 (32.4% [30.3-34.5]) | 144 (7.4% [6.4-8.7]) | 483 |

### Misstatement + Misreference, no-network subset (You Wang's suggested scope)

257 claims (216 Misstatement, 41 Misreference; 112 DeepSeek-V3.2, 145 Flash), from the 20 DeepSeek trajectories of
the 10 manually reviewed tasks. Controls = other claims from the same 20 trajectories.

| subset | n | all | some | none | no_repo_evidence |
|---|---|---|---|---|---|
| M+M no-net, anchored | 257 | 100 (46.9% [40.4-53.6]) | 98 (46.0% [39.5-52.7]) | 15 (7.0% [4.3-11.3]) | 44 |
| M+M no-net, any output | 257 | 100 (46.9% [40.4-53.6]) | 101 (47.4% [40.8-54.1]) | 12 (5.6% [3.2-9.6]) | 44 |
| M+M no-net, file seen | 257 | 160 (75.1% [68.9-80.4]) | 44 (20.7% [15.8-26.6]) | 9 (4.2% [2.2-7.8]) | 44 |
| same trajectories: refuted, not M+M | 243 | 109 (53.4% [46.6-60.2]) | 82 (40.2% [33.7-47.0]) | 13 (6.4% [3.8-10.6]) | 39 |
| same trajectories: supported control | 463 | 221 (60.1% [55.0-64.9]) | 117 (31.8% [27.2-36.7]) | 30 (8.2% [5.8-11.4]) | 95 |
| Qwen M+M (network on) | 242 | 64 (32.5% [26.3-39.3]) | 103 (52.3% [45.3-59.2]) | 30 (15.2% [10.9-20.9]) | 45 |

Identifiers in this subset: 212 claims name at least one identifier; 25 (11.8% [8.1-16.8]) name an unexplained
missing one, 10 (4.7% [2.6-8.5]) an other-version one (same-trajectory supported controls: 24/359 = 6.7%, 8 = 2.2%;
Qwen M+M: 35/210 = 16.7%, 16 = 7.6%). Examples (names only): `ReportConf` (vuls, future), `IdleTimeout` and
`NetTimeout` (flipt, future), `FLAG_NOT_FOUND` (flipt, future), `config.DiffMode*` (vuls, never),
`DBSystemCockroachDB` (never; the real otel constant is spelled `DBSystemCockroachdb`).

### Identifiers (claim level)

| subset | verdict | claims with identifiers | unexplained missing name | other-version name |
|---|---|---|---|---|
| DeepSeek (main) | refuted | 3600 | 331 (9.2% [8.3-10.2]) | 128 (3.6% [3.0-4.2]) |
| DeepSeek (main) | supported | 3575 | 173 (4.8% [4.2-5.6]) | 66 (1.8% [1.5-2.3]) |
| DeepSeek, type O | refuted | 1852 | 174 (9.4% [8.2-10.8]) | 75 (4.0% [3.2-5.1]) |
| DeepSeek, type O | supported | 1595 | 57 (3.6% [2.8-4.6]) | 18 (1.1% [0.7-1.8]) |
| DeepSeek, type F (fix) | refuted | 1190 | 131 (11.0%) | 46 (3.9%) |
| DeepSeek, type F (fix) | supported | 926 | 107 (11.6%) | 44 (4.8%) |
| Qwen | refuted | 3103 | 309 (10.0% [8.9-11.1]) | 92 (3.0% [2.4-3.6]) |
| Qwen | supported | 3175 | 143 (4.5% [3.8-5.3]) | 59 (1.9% [1.4-2.4]) |

All identifier statuses, DeepSeek refuted: base 8,119, future 915, never 609, external 83, both 30, stale 26 (most
future names are new interfaces named in the task text, so they are "explained"). Unexplained ones: never 255,
future 125, stale 16, both 9. Type F claims name things the agent is about to create, so missing names there are
expected and do not separate refuted from supported. Of the 128 DeepSeek refuted claims with an other-version name,
52 had seen ALL cited evidence lines, 41 some, 9 none, 26 had no repository evidence.

## Hand check (15 random claims, LLM-judged)

`handcheck 15 0` picks 15 random analyzed claims with repository evidence. For each one I (Claude, an LLM, not a
human annotator) opened the trajectory, read every command before the claim, and searched the earlier outputs for
distinctive pieces of the cited lines (`rq1_claims_extra_checks.py view`), then gave my own all/some/none.

| cid | model | verdict | script (anchored) | my judgment | note |
|---|---|---|---|---|---|
| c76032 | qwen38 | refuted | some 34/46 | some | stubs.py `FakeSignal` lines never shown |
| c33196 | flash | refuted | all 124/124 | all | sed ranges cover every cited range |
| c67703 | qwen38 | refuted | some 19/21 | some | requirements.txt lines never shown |
| c36587 | flash | refuted | all 67/67 | all | full `cat` of useShare.ts at msg 55 |
| c02946 | deepseek-v32 | supported | some 55/69 | some | voice-broadcast/index.ts never opened |
| c20689 | flash | supported | all 31/31 | all | sed 430-590 at msg 7 |
| c45160 | qwen36 | supported | some 9/16 | some | call sites in LoggedInView/RoomView never shown |
| c43179 | qwen36 | supported | some 7/15 | some | `User.getStatus` definition never shown |
| c35236 | flash | supported | all 17/17 | all | gofmt-aligned lines matched after whitespace collapse |
| c70320 | qwen38 | supported | none 0/15 | none | claim at msg 5, only `ls` before |
| c74260 | qwen38 | refuted | none 0/10 | none | app.py never opened; one identical line came from a test file |
| c25406 | flash | supported | all 23/23 | all | sed 1-360 of converter.go |
| c42549 | qwen36 | supported | some 35/49 | some | routes/index.js (`POST /compose`) never shown |
| c30745 | flash | supported | all 22/22 | all | XML fixture shown at msgs 13/179/181 |
| c52097 | qwen36 | refuted | some 15/30 | some | test file shown only after the claim |

**Agreement: 15/15 with the anchored (main) class; 14/15 with the looser any-output class** (c74260: a line identical
to the cited one appeared in a different file's output, so the loose version said "some" although app.py was never
opened). Only 2 of the 15 were "none", so this small check says little about the error rate of that class.

Extra check of the "none" class (`none-sample 10 1`, 10 random DeepSeek refuted "none" claims): for each cited line,
its longest word was searched in earlier outputs about the same file. No missed display found. The two candidate hits
I opened were (a) c15045: the agent had run `cat src/topics/tags.js`, but mini-swe-agent cut the middle of the long
output and the cited lines 292-294 were in the elided part, and (b) c31865: a `grep -l` that printed only the file
name. The other hits were generic words (`github`, `version`). LLM-judged.

Extra check of the "other_version" class (`other-version-sample 12 7`, LLM-judged by reading the claims): 9 of 12 name
a real repo identifier that exists only in another version, in a way consistent with recall; 3 of 12 are noise
(`anyOf`, a JSON-Schema keyword; `MediaWiki` and `DistinguishedName`, words that occur in comments). So roughly a
quarter of other-version flags are false alarms. For 9 of the sampled names I then checked the task's own upstream
fix (`fix-names`: diff base..fix commit from the instance id, search the changed files' new versions; 81 file
versions downloaded in total). **5 flagged names are introduced by the very fix the agent was asked to write:**
`refreshSmartPlaylist`, `qt_64`, `aesGCM`, `defaults.AuditBackoffTimeout` (fix proposals that use the upstream names)
and `collectDeviceData` (a refuted type-O claim that describes the original code as if the fix were already there).
`ExactVersionMatch` and `FLIPT_CACHE_MEMORY_ENABLED` are not in the fix (other versions only), `mapstructure` only in
the fix's go.sum, and `HAS_SELINUX` exists only in older ansible (its claim literally says "the recalled
implementation"). Natural naming could produce some of these names without memory; these DeepSeek runs had no
network and pruned git history, so the names did not come from a download or `git log`. Two names were also checked
with `git grep`: `collectDeviceData` (teleport) is in 0 Go files at the base commit and 6 at HEAD; `mapstructure`
(flipt) in 0 at base and 24 at HEAD.

## Caveats

- **Seen is not the same as understood, and unseen is not proof of memory.** "none" means the cited lines were not
  displayed; the claim may still come from reasoning, naming conventions, the task text (field
  `n_lines_in_task_text` counts cited lines that also appear in the task text), or memory.
- **The cited evidence is the assessment agent's choice** (an LLM with tools, You Wang et al.'s RQ2). Ranges vary in
  breadth; a wide range makes "all" harder and "some" more likely. "some" claims saw about half of their cited lines.
- **Lines are compared at the base commit.** If the agent had already edited a file, later outputs show the edited
  version and unchanged lines still match, changed lines do not.
- Truncated tool outputs (mini-swe-agent cuts the middle of long outputs) are handled correctly: elided lines count as
  unseen (see c15045). Unusual displays (e.g. `cat -A`, Python `repr`, cut-off long lines) are not recognized and
  would count as unseen.
- Anchoring uses the file NAME, so generic names (index.js, __init__.py, utils.py) anchor loosely. The unanchored
  variant is in every table of the JSON and lowers "none" by 1-4 points (DeepSeek refuted 8.6% -> 6.3%,
  Qwen refuted 12.9% -> 9.2%).
- Identifier checks are literal and word-based: shorthand names (`paidSingle` for `paidSingleUserUpsellConfig`) count
  as missing; words in comments count as existing; the snapshot history is coarse (every 2 years + HEAD); external
  detection only knows builtin/stdlib names and names the repo itself imports, so e.g. a stdlib function the repo
  never imported (`lru_cache` in old ansible) is classified by the repo's history instead.
  About 3 of 12 sampled other-version flags were noise.
- **Wilson intervals treat claims as independent.** Claims from one trajectory are correlated (400 trajectories,
  100 tasks), so the true uncertainty is larger, especially for the small M+M and 773 subsets.
- DeepSeek vs Qwen differ in network/git access AND in model, so the network effect is not isolated. In Qwen runs,
  files fetched from the network count as observations (they are real outputs the agent saw).
- Misstatement/Misreference labels exist only for the 40 manually reviewed trajectories (10 tasks x 4 models).
- The identifier "future" class overlaps with the task's new-interface names; those are excluded only when the exact
  name appears in the task text.
- `data/repos_pro/` (630 MB) is not covered by `.gitignore` (only `data/repos/` is); it should be added before any
  `git add`. I did not edit `.gitignore` (not my file).

## Files

- scripts/rq1_pro_repos.py (download side: clones, batched blob fetches, snapshot choice)
- scripts/rq1_claim_provenance.py (analysis; never downloads, runs git with GIT_NO_LAZY_FETCH=1)
- scripts/rq1_claims_extra_checks.py (hand-check helpers; prints only)
- outputs/rq1/claims_features.jsonl (17,080 rows; ids, features, identifier names; no claim text)
- outputs/rq1/claims_summary.json (all tables above, both matching variants, file_seen, O/G/S/F splits)
- data/repos_pro/ (12 blobless clones + snapshots.json; gitignore it)
