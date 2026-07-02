#!/usr/bin/env bash
set -eo pipefail
# Build all 12 compact files (tier x detector x hasADD) on the login node.
WROOT="/eos/user/y/yeo/4l/weights/categories"
OUT="/eos/user/y/yeo/4l/compact"
PY="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A TT=( [AOD]="GenTrk" [MiniAOD]="Lost,PF,GSF" )
for tier in AOD MiniAOD; do
  for det in eb ee; do
    for mode in all eq0 eq1; do
      w="${WROOT}/${tier}_${mode}/category_weights.h5"
      o="${OUT}/${tier}_${det}_${mode}.h5"
      echo "=== ${tier} ${det} ${mode} -> ${o} ==="
      "${PY}" "${SCRIPT_DIR}/build_compact_dataset.py" --weight-h5 "${w}" \
        --detector "${det}" --track-types "${TT[$tier]}" --output-path "${o}"
    done
  done
done
echo done.
