#!/usr/bin/env bash
# Build the low-cut (pT>=20) EVALUATION sample: eval selection -> eval compacts.
# signal = 20% of all signal (no pT cut); background = 20% of min(1e6, pT>=20 pool).
# Login-node, resumable (compacts skip existing). Source the LCG view inside.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export HDF5_USE_FILE_LOCKING=FALSE

WROOT=/eos/user/y/yeo/4l/weights/eval_20gev
CROOT=/eos/user/y/yeo/4l/compact_eval_20gev
DATA=/eos/user/y/yeo/4l/data

echo "=========== [$(date)] STEP 1/2: eval selection -> ${WROOT} ==========="
for tier in AOD MiniAOD; do
  for mode in all eq0 eq1; do
    out="${WROOT}/${tier}_${mode}/category_weights.h5"
    if [ -f "${out}" ]; then echo "=== ${tier} ${mode} (SKIP: exists) ==="; continue; fi
    echo "=== ${tier} ${mode} -> ${out} ==="
    python3 build_eval_sample.py --input-dir "${DATA}/${tier}" \
      --output-path "${out}" --hasadd-mode "${mode}"
  done
done

echo "=========== [$(date)] STEP 2/2: eval compacts -> ${CROOT} ==========="
./run_build_compact.sh --weight-root "${WROOT}" --output-root "${CROOT}"

echo "=========== [$(date)] DONE ==========="
