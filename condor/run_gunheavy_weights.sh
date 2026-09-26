#!/usr/bin/env bash
# Condor executable: build GUN-HEAVY category weights for ONE pt_min.
#   arguments = "<pt_min>"
#
#   signal     = particle_gun_heavy   (continuous A mass, A->ee)
#   background = W, Z
#   pT variable = per-object RECO supercluster ET (SC_energyT_sum / EE_seed_energyT_sum)
#                 -> data-applicable, cut + binning both on this per-object SC ET.
#   category   = all
#
# Reads raw via robust_h5_open (FUSE + xrdcp fallback), writes weight to LOCAL
# scratch, xrdcp it to EOS. Session-independent (survives logout).
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
ptmin="$1"
if [[ -z "$ptmin" ]]; then echo "usage: run_gunheavy_weights.sh <pt_min>" >&2; exit 2; fi

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
if ! command -v xrdcp >/dev/null 2>&1; then echo "[fatal] no xrdcp" >&2; exit 66; fi

scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
export TMPDIR="$scratch"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${scratch}/mplconfig"

DATA="/eos/user/y/yeo/4l/data/MiniAOD"
WEOS="/eos/user/y/yeo/4l/weights/categories_gunheavy_pt${ptmin}/gunheavy_all/category_weights.h5"
wlocal="${scratch}/category_weights.h5"

echo "=========== [$(date)] gunheavy weights (SC-ET) pt>=${ptmin} ==========="
python3 build_category_weights_gun.py --input-dir "$DATA" --output-path "$wlocal" \
    --pt-min "$ptmin" --workers 16
echo "[stage-out] weight -> ${WEOS}"
xrdcp -f -p "$wlocal" "root://eosuser.cern.ch/${WEOS}"
echo "=========== [$(date)] DONE gunheavy weights pt>=${ptmin} ==========="
