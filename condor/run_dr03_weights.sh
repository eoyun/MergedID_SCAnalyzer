#!/usr/bin/env bash
# Condor: dr03 (cropped) category weights for ONE pt_min.
#   signal=HToEleEle_dr03, background=W,Z; data=MiniAOD_cropped; pT=per-object reco SC ET.
#   arguments = "<pt_min>"
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
ptmin="$1"; [[ -z "$ptmin" ]] && { echo "usage: run_dr03_weights.sh <pt_min>" >&2; exit 2; }
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u
command -v xrdcp >/dev/null 2>&1 || { echo "[fatal] no xrdcp" >&2; exit 66; }
scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"; export TMPDIR="$scratch" HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR="$scratch/mpl"
DATA="/eos/user/y/yeo/4l/data/MiniAOD_cropped"
WEOS="/eos/user/y/yeo/4l/weights/categories_dr03_pt${ptmin}/dr03_all/category_weights.h5"
wlocal="${scratch}/category_weights.h5"
echo "=========== [$(date)] dr03 weights (SC-ET) pt>=${ptmin} ==========="
python3 build_category_weights_gun.py --input-dir "$DATA" --output-path "$wlocal" \
    --pt-min "$ptmin" --signal-dir HToEleEle_dr03 --workers 16
xrdcp -f -p "$wlocal" "root://eosuser.cern.ch/${WEOS}"
echo "=========== [$(date)] DONE dr03 weights pt>=${ptmin} ==========="
