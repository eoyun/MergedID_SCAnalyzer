#!/usr/bin/env bash
set -eo pipefail

# Build all 6 per-category weight files (tier x hasADD) on the login node.
# EB and EE live inside each file. Source the cvmfs LCG view before running.

usage() {
  cat <<'EOF'
Usage: ./run_build_category_weights.sh [--data-root DIR] [--output-root DIR]
                                       [--python BIN] [--background-max-events N]
                                       [--tier aod|mini|both] [--hasadd all|eq0|eq1|all-modes]
Defaults: data-root=/eos/user/y/yeo/4l/data  output-root=/eos/user/y/yeo/4l/weights/categories
          python=python3  background-max-events=1000000  tier=both  hasadd=all-modes
EOF
}

DATA_ROOT="/eos/user/y/yeo/4l/data"
OUTPUT_ROOT="/eos/user/y/yeo/4l/weights/categories"
PYTHON_BIN="python3"
BG_MAX="1000000"
TIER="both"
HASADD="all-modes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --background-max-events) BG_MAX="$2"; shift 2;;
    --tier) TIER="$2"; shift 2;;
    --hasadd) HASADD="$2"; shift 2;;
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

for tier in "${TIERS[@]}"; do
  for mode in "${MODES[@]}"; do
    out="${OUTPUT_ROOT}/${tier}_${mode}/category_weights.h5"
    echo "=== ${tier} ${mode} -> ${out} ==="
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_category_weights.py" \
      --input-dir "${DATA_ROOT}/${tier}" \
      --output-path "${out}" \
      --hasadd-mode "${mode}" \
      --background-max-events "${BG_MAX}"
  done
done
echo "done."
