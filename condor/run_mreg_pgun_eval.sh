#!/usr/bin/env bash
# Condor: one mreg_v1 particle-gun eval -> EPS plots (+npz).
#   arguments = "<mtype> <det> <run> <compact> <noes:0|1>"
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
mt="$1"; det="$2"; run="$3"; cpt="$4"; noes="$5"
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
command -v xrdcp >/dev/null 2>&1 || { echo "[fatal] no xrdcp" >&2; exit 66; }
scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
export TMPDIR="$scratch" HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR="$scratch/mpl"

clocal="$scratch/$cpt"
xrdcp -f "root://eosuser.cern.ch//eos/user/y/yeo/4l/compact_pgun/$cpt" "$clocal"
rundir="$scratch/rundir"; mkdir -p "$rundir"
xrdcp -f "root://eosuser.cern.ch//eos/user/y/yeo/4l/runs/mreg_v1/$run/best_model.pt" "$rundir/best_model.pt"

out="$scratch/out"; mkdir -p "$out"
esflag=""; [[ "$noes" == "1" ]] && esflag="--no-es"
python3 -c "import torch;print('[gpu]', torch.cuda.is_available())" || true
python3 eval_pgun.py --model-type "$mt" --detector "$det" --run-dir "$rundir" \
    --compact "$clocal" --output-dir "$out" --track-types "Lost,PF,GSF" --dpi 200 --ext .eps $esflag

EOUT="/eos/user/y/yeo/4l/runs/mreg_v1_eval_pgun/$run"
for f in "$out"/*; do
  [[ -f "$f" ]] || continue
  xrdcp -f -p "$f" "root://eosuser.cern.ch/${EOUT}/$(basename "$f")"
done
echo "DONE $run"
