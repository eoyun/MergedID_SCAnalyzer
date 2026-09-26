#!/usr/bin/env bash
# Condor executable: build ONE gun-heavy compact (one pt_min, one detector).
#   arguments = "<pt_min> <det>"   det in {eb,ee}
#
# Reads the gun-heavy category weight file from EOS (xrdcp -> local), builds the
# compact with build_compact_dataset_gun.py (pt = reco SC ET, a_mass = continuous),
# xrdcp the compact to EOS. Session-independent (survives logout).
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
ptmin="$1"; det="$2"
if [[ -z "$ptmin" || -z "$det" ]]; then echo "usage: run_gunheavy_compact.sh <pt_min> <det>" >&2; exit 2; fi

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
if ! command -v xrdcp >/dev/null 2>&1; then echo "[fatal] no xrdcp" >&2; exit 66; fi

scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
export TMPDIR="$scratch"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${scratch}/mplconfig"

TT="Lost,PF,GSF"
WEOS="/eos/user/y/yeo/4l/weights/categories_gunheavy_pt${ptmin}/gunheavy_all/category_weights.h5"
CEOS="/eos/user/y/yeo/4l/compact_gunheavy_pt${ptmin}/gunheavy_${det}.h5"
wlocal="${scratch}/category_weights.h5"
clocal="${scratch}/gunheavy_${det}.h5"

echo "=========== [$(date)] fetch weight ${WEOS} ==========="
xrdcp -f "root://eosuser.cern.ch/${WEOS}" "$wlocal"

echo "=========== [$(date)] compact gunheavy pt${ptmin} ${det} ==========="
python3 build_compact_dataset_gun.py --weight-h5 "$wlocal" --detector "$det" \
    --track-types "$TT" --output-path "$clocal" --workers 8

echo "[stage-out] compact -> ${CEOS}"
xrdcp -f -p "$clocal" "root://eosuser.cern.ch/${CEOS}"
echo "=========== [$(date)] DONE gunheavy compact pt${ptmin} ${det} ==========="
