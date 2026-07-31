#!/usr/bin/env bash
# Condor executable: build the TRAINING data for ONE (tier, mode) at a given pT cut.
#   arguments = "<tier> <mode> <pt_min>"   tier in {AOD,MiniAOD}, mode in {all,eq0,eq1}
#
# signal = all objects with event pT >= pt_min; background = 1M random from that
# pool (build_category_weights). Reads raw via robust_h5_open (FUSE + xrdcp
# fallback), writes weight + 2 compacts to LOCAL scratch, xrdcp them to EOS.
# Session-independent (survives logout), unlike a login-node build.
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
tier="$1"; mode="$2"; ptmin="$3"
if [[ -z "$tier" || -z "$mode" || -z "$ptmin" ]]; then
  echo "usage: run_train_build.sh <tier> <mode> <pt_min>" >&2; exit 2; fi

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
if ! command -v xrdcp >/dev/null 2>&1; then echo "[fatal] no xrdcp" >&2; exit 66; fi

scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
export TMPDIR="$scratch"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${scratch}/mplconfig"

case "$tier" in
  AOD) TT="GenTrk";;
  MiniAOD) TT="Lost,PF,GSF";;
  *) echo "[fatal] bad tier $tier" >&2; exit 2;;
esac

DATA="/eos/user/y/yeo/4l/data/${tier}"
WEOS="/eos/user/y/yeo/4l/weights/categories_pt${ptmin}/${tier}_${mode}/category_weights.h5"
CEOS="/eos/user/y/yeo/4l/compact_pt${ptmin}"
wlocal="${scratch}/category_weights.h5"

echo "=========== [$(date)] weights ${tier} ${mode} pt>=${ptmin} ==========="
python3 build_category_weights.py --input-dir "$DATA" --output-path "$wlocal" \
    --hasadd-mode "$mode" --pt-min "$ptmin"
echo "[stage-out] weight -> ${WEOS}"
xrdcp -f -p "$wlocal" "root://eosuser.cern.ch/${WEOS}"

for det in eb ee; do
  clocal="${scratch}/${tier}_${det}_${mode}.h5"
  echo "=========== [$(date)] compact ${tier} ${det} ${mode} ==========="
  python3 build_compact_dataset.py --weight-h5 "$wlocal" --detector "$det" \
      --track-types "$TT" --output-path "$clocal" --workers 8
  echo "[stage-out] compact -> ${CEOS}/${tier}_${det}_${mode}.h5"
  xrdcp -f -p "$clocal" "root://eosuser.cern.ch/${CEOS}/${tier}_${det}_${mode}.h5"
  rm -f "$clocal"
done

echo "=========== [$(date)] DONE ${tier} ${mode} pt>=${ptmin} ==========="
