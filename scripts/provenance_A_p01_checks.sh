#!/usr/bin/env bash
# Re-runs the version checks behind annotator A's LLM-judged labels for outputs/provenance/batches/p_01.jsonl
# (written to outputs/provenance/labels_A/p_01.jsonl). Read-only: prints what each check found.
#
# Usage (from the project root):
#   bash scripts/provenance_A_p01_checks.sh            # git checks only (offline, local blobless clones)
#   bash scripts/provenance_A_p01_checks.sh DL_DIR     # also download a few public upstream files into DL_DIR
#
# The git checks need blobs that were fetched once with
#   .venv/bin/python -I scripts/rq1_claims_extra_checks.py fix-names <cid>:<name>
# or with "git fetch origin --filter=blob:none --stdin" (blob ids from git ls-tree). Nothing here fetches.
set -u
R=data/repos_pro
export GIT_NO_LAZY_FETCH=1   # never download inside this script

echo "== NodeBB fix 445b70d: what the upstream fix validates (c14483, c14486, c14493, c14494)"
git -C $R/NodeBB__NodeBB diff 565ca3cc3b78 445b70deda20201b7d9a68f7224da751b3db728c -- src/api/users.js src/api/chats.js \
  | grep -E '^\+.*(if \(!|throw)'

echo "== NodeBB sortedSetScore for a missing member, base + snapshots (c14551)"
for c in 565ca3cc3b78 1199336b0ba7f626d0e80744b05ac47c0f633384 d1700c40e2b5503146f70f5f3c307022c7197722 \
         43e7cc0afb5d7df0bfc9d9434e631fc4bf63731e a08474a9fdd380b1e72b49b788b82b66edcce096; do
  echo "-- $c"; git -C $R/NodeBB__NodeBB show $c:src/database/redis/sorted.js | grep -A7 'sortedSetScore = '
done

echo "== element-web fix a692fe2: CreateRoomDialog changes (c02740, c02741)"
git -C $R/element-hq__element-web diff 9d9c55d92e98 a692fe21811f88d92e8f7047fc615e4f1f986b0f \
  -- src/components/views/dialogs/CreateRoomDialog.tsx | grep -E '^[+-]' | head -40

echo "== element-web package name/version at base and snapshots (c19714)"
for c in 9d9c55d92e98 66080b57512ad5fa8cf6826ef12ceb05ceea295a 85e2f063ac812a2b5826bee73cd084b949beca62 \
         e98529824e919ac4d29b3818333005106a56e74b; do
  echo "$c $(git -C $R/element-hq__element-web show $c:package.json | grep -m2 -E '"(name|version)"' | tr -d '\n ')"
done

echo "== flipt go.mod replace lines at snapshots and base (c03272)"
for c in 5438b455b049306971f8672d2f57a5e626de8a19 441d1450811476f920ee16b27c0de7a9a546d404 d38a357b67ce; do
  echo "-- $c"; git -C $R/flipt-io__flipt show $c:go.mod | grep -E '^replace|=> \./'
done

echo "== flipt fix 492cc0b: new RedisCacheConfig fields (c20900, c20914, c03380)"
git -C $R/flipt-io__flipt diff d38a357b67ce 492cc0b158200089dceede3b1aba0ed28df3fb1d -- internal/config/cache.go \
  | grep -E '^\+\s+[A-Z][A-Za-z]+ '

if [ $# -ge 1 ]; then
  # Public upstream files (data, never executed): go-redis and matrix-js-sdk at the tags the labels cite.
  D="$1"; mkdir -p "$D"
  for spec in v8.11.5/internal/pool/pool.go:v8_pool.go v9.0.5/internal/pool/pool.go:v9_0_5_pool.go \
              v7.4.1/options.go:v7_options.go v8.11.5/options.go:v8_options.go v9.0.5/options.go:v9_0_5_options.go \
              v9.7.0/options.go:v9_7_0_options.go; do
    curl -sS --max-time 30 -o "$D/${spec##*:}" "https://raw.githubusercontent.com/redis/go-redis/${spec%%:*}"
  done
  echo "== go-redis internal/pool Options: IdleTimeout (v8) vs ConnMaxIdleTime (v9.0.5) (c20898)"
  grep -n 'IdleTimeout\|ConnMaxIdleTime' "$D/v8_pool.go" "$D/v9_0_5_pool.go" | grep -v '//'
  echo "== go-redis ReadTimeout switch per version (c03323, c03351, c03363)"
  for f in v7_options.go v8_options.go v9_0_5_options.go v9_7_0_options.go; do
    echo "-- $f"; grep -A7 'switch opt.ReadTimeout' "$D/$f"
  done
  for v in v12.0.0 v26.1.0 v34.0.0; do
    curl -sS --max-time 60 -o "$D/mjs_client_$v.ts" "https://raw.githubusercontent.com/matrix-org/matrix-js-sdk/$v/src/client.ts"
    echo "== matrix-js-sdk $v doesServerForceEncryptionForPreset (c02687)"
    grep -A4 'public async doesServerForceEncryptionForPreset' "$D/mjs_client_$v.ts"
  done
fi
