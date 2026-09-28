#!/usr/bin/env bash
# Condor executable: build the SHARED 3-class 20 GeV eval compacts (eb + ee) for gunheavy.
# No arguments. Builds the eval selection (gun/haa/bkg, SC ET>=20) once, then a compact per
# detector, xrdcp'd to EOS. Session-independent.
set -eo pipefail
echo "[host] $(hostname)"
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
command -v xrdcp >/dev/null 2>&1 || { echo "[fatal] no xrdcp" >&2; exit 66; }
scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
export TMPDIR="$scratch" HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR="${scratch}/mpl"

DATA="/eos/user/y/yeo/4l/data/MiniAOD"
sel="${scratch}/gunheavy_eval_sel.h5"
echo "=========== [$(date)] eval selection (SC ET>=20, gun/haa/bkg) ==========="
python3 build_category_weights_gun_eval.py --input-dir "$DATA" --output-path "$sel" --pt-min 20 --workers 16

for det in eb ee; do
  c="${scratch}/gunheavy_eval_${det}.h5"
  echo "=========== [$(date)] eval compact ${det} ==========="
  python3 build_compact_dataset_gun.py --weight-h5 "$sel" --detector "$det" \
      --track-types "Lost,PF,GSF" --output-path "$c" --workers 8
  xrdcp -f -p "$c" "root://eosuser.cern.ch//eos/user/y/yeo/4l/compact_gunheavy_eval/gunheavy_eval_${det}.h5"
  rm -f "$c"
done
echo "=========== [$(date)] DONE eval compacts ==========="
