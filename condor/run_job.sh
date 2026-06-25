#!/usr/bin/env bash
# Condor executable: source the cvmfs LCG view, then run the python job.
# arguments = "<script.py> <args...>" (passed verbatim as argv).
set -eo pipefail
echo "[host] $(hostname)"
echo "[args] $*"
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export MPLCONFIGDIR=/tmp/mplconfig_${USER:-job}
python3 -c "import torch; print('[gpu]', torch.cuda.is_available(), torch.cuda.device_count())" || true
exec python3 "$@"
