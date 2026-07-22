#!/usr/bin/env bash
set -eo pipefail

# Build compact files (tier x detector x hasADD) on the login node.
# Each compact packs, per selected object, only the arrays training needs, so a
# job reads one local file instead of ~1.5M FUSE opens/epoch. Depends on the
# per-category weight files, so (re)build those first. Source the LCG view.

usage() {
  cat <<'EOF'
Usage: ./run_build_compact.sh [--tier aod|mini|both] [--hasadd all|eq0|eq1|all-modes]
                              [--weight-root DIR] [--output-root DIR] [--python BIN]
Defaults: tier=both  hasadd=all-modes
          weight-root=/eos/user/y/yeo/4l/weights/categories
          output-root=/eos/user/y/yeo/4l/compact  python=python3
Builds tier x {eb,ee} x hasadd compacts (both/all-modes = 12 files; mini = 6).
EOF
}

WROOT="/eos/user/y/yeo/4l/weights/categories"
OUT="/eos/user/y/yeo/4l/compact"
PY="${PYTHON:-python3}"
TIER="both"
HASADD="all-modes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2;;
    --hasadd) HASADD="$2"; shift 2;;
    --weight-root) WROOT="$2"; shift 2;;
    --output-root) OUT="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${TIER}" in
  aod) TIERS=("AOD");; mini) TIERS=("MiniAOD");; both) TIERS=("AOD" "MiniAOD");;
  *) echo "--tier must be aod|mini|both" >&2; exit 1;;
esac
case "${HASADD}" in
  all-modes) MODES=("all" "eq0" "eq1");; all|eq0|eq1) MODES=("${HASADD}");;
  *) echo "--hasadd must be all|eq0|eq1|all-modes" >&2; exit 1;;
esac

declare -A TT=( [AOD]="GenTrk" [MiniAOD]="Lost,PF,GSF" )

for tier in "${TIERS[@]}"; do
  for det in eb ee; do
    for mode in "${MODES[@]}"; do
      w="${WROOT}/${tier}_${mode}/category_weights.h5"
      o="${OUT}/${tier}_${det}_${mode}.h5"
      if [ -f "${o}" ]; then
        echo "=== ${tier} ${det} ${mode} -> ${o} (SKIP: exists) ==="
        continue
      fi
      echo "=== ${tier} ${det} ${mode} -> ${o} ==="
      "${PY}" "${SCRIPT_DIR}/build_compact_dataset.py" --weight-h5 "${w}" \
        --detector "${det}" --track-types "${TT[$tier]}" --output-path "${o}"
    done
  done
done
echo done.
