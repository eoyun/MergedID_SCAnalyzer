#!/usr/bin/env bash
# Condor executable: build the pT>=20 eval sample for ONE (tier, mode).
#   arguments = "<tier> <mode>"   tier in {AOD,MiniAOD}, mode in {all,eq0,eq1}
#
# Reads raw HDF5 from EOS via the FUSE mount (build_*.py use robust_h5_open, which
# falls back to xrdcp per file when FUSE hangs/fails). Writes the weight file and
# the two compacts to LOCAL scratch, then stages them out to EOS via xrdcp (EOS
# FUSE *writes* from workers are unreliable). Session-independent (unlike a login
# node tmux build), so it survives logout.
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
tier="$1"; mode="$2"
if [[ -z "$tier" || -z "$mode" ]]; then echo "usage: run_eval_build.sh <tier> <mode>" >&2; exit 2; fi

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
WEOS="/eos/user/y/yeo/4l/weights/eval_20gev/${tier}_${mode}/category_weights.h5"
CEOS="/eos/user/y/yeo/4l/compact_eval_20gev"

wlocal="${scratch}/category_weights.h5"

echo "=========== [$(date)] eval selection ${tier} ${mode} ==========="
python3 build_eval_sample.py --input-dir "$DATA" --output-path "$wlocal" --hasadd-mode "$mode"
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

echo "=========== [$(date)] DONE ${tier} ${mode} ==========="
