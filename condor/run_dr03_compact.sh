#!/usr/bin/env bash
# Condor: one dr03 (cropped) compact. args: "<pt_min> <det>"  det in {eb,ee}
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
ptmin="$1"; det="$2"; [[ -z "$ptmin" || -z "$det" ]] && { echo usage >&2; exit 2; }
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u
command -v xrdcp >/dev/null 2>&1 || { echo "[fatal] no xrdcp" >&2; exit 66; }
scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"; export TMPDIR="$scratch" HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR="$scratch/mpl"
TT="Lost,PF,GSF"
WEOS="/eos/user/y/yeo/4l/weights/categories_dr03_pt${ptmin}/dr03_all/category_weights.h5"
CEOS="/eos/user/y/yeo/4l/compact_dr03_pt${ptmin}/dr03_${det}.h5"
wlocal="$scratch/category_weights.h5"; clocal="$scratch/dr03_${det}.h5"
xrdcp -f "root://eosuser.cern.ch/${WEOS}" "$wlocal"
# fewer workers for ee (big ES planes 24x768) to bound memory
wk=8; [[ "$det" == "ee" ]] && wk=4
echo "=========== [$(date)] dr03 compact pt${ptmin} ${det} (workers=$wk) ==========="
python3 build_compact_dataset_gun.py --weight-h5 "$wlocal" --detector "$det" \
    --track-types "$TT" --output-path "$clocal" --workers $wk
xrdcp -f -p "$clocal" "root://eosuser.cern.ch/${CEOS}"
echo "=========== [$(date)] DONE dr03 compact pt${ptmin} ${det} ==========="
