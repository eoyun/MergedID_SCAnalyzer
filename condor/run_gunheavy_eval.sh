#!/usr/bin/env bash
# Condor executable: evaluate ONE trained gunheavy_v1 run on the shared 3-class eval compact.
#   arguments = "<det> <no_es:0|1> <run_name>"   det in {eb,ee}
# Stages the eval compact + the run's best_model.pt from EOS, runs eval_gunheavy.py (6 plots),
# xrdcp the plots to EOS runs/gunheavy_v1_eval/<run_name>/. Session-independent.
set -eo pipefail
echo "[host] $(hostname)  [args] $*"
det="$1"; noes="$2"; run="$3"
if [[ -z "$det" || -z "$noes" || -z "$run" ]]; then echo "usage: <det> <no_es:0|1> <run_name>" >&2; exit 2; fi
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
command -v xrdcp >/dev/null 2>&1 || { echo "[fatal] no xrdcp" >&2; exit 66; }
scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
export TMPDIR="$scratch" HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR="${scratch}/mpl"

CEOS="/eos/user/y/yeo/4l/compact_gunheavy_eval/gunheavy_eval_${det}.h5"
clocal="${scratch}/eval_${det}.h5"
echo "[stage-in] ${CEOS}"
xrdcp -f "root://eosuser.cern.ch/${CEOS}" "$clocal"

rundir="${scratch}/rundir"; mkdir -p "$rundir"
RBASE="/eos/user/y/yeo/4l/runs/gunheavy_v1/${run}"
xrdcp -f "root://eosuser.cern.ch/${RBASE}/best_model.pt" "${rundir}/best_model.pt"
# training history (for the training-history plot); best-effort
xrdcp -f "root://eosuser.cern.ch/${RBASE}/history.json" "${rundir}/history.json" || echo "[warn] no history.json"

esflag=""; [[ "$noes" == "1" ]] && esflag="--no-es"
out="${scratch}/evalout"; mkdir -p "$out"
python3 -c "import torch;print('[gpu]',torch.cuda.is_available())" || true
python3 eval_gunheavy.py --detector "$det" --compact "$clocal" --run-dir "$rundir" \
    --output-dir "$out" $esflag --track-types "Lost,PF,GSF" --batch-size 64 --num-workers 8

EOUT="/eos/user/y/yeo/4l/runs/gunheavy_v1_eval/${run}"
for f in "$out"/*; do
  [[ -f "$f" ]] || continue
  xrdcp -f -p "$f" "root://eosuser.cern.ch/${EOUT}/$(basename "$f")"
done
echo "=========== [$(date)] DONE eval ${run} -> ${EOUT} ==========="
