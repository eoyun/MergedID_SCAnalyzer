#!/usr/bin/env bash
# Build v9/cross_v3 (pT>200) data: per-category weights -> compacts.
# Login-node, resumable (compacts skip existing). Source the LCG view inside.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export HDF5_USE_FILE_LOCKING=FALSE

WROOT=/eos/user/y/yeo/4l/weights/categories_pt200
CROOT=/eos/user/y/yeo/4l/compact_pt200

echo "=========== [$(date)] STEP 1/2: weights (pt-min 200) -> ${WROOT} ==========="
./run_build_category_weights.sh --pt-min 200 --output-root "${WROOT}"

echo "=========== [$(date)] STEP 2/2: compacts -> ${CROOT} ==========="
./run_build_compact.sh --weight-root "${WROOT}" --output-root "${CROOT}"

echo "=========== [$(date)] DONE ==========="
