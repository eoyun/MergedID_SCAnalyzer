#!/usr/bin/env bash
# Condor executable.
# arguments = "<script.py> ... --output-dir <EOS_PATH> ..."
#
# EOS FUSE writes from batch workers are unreliable (CERN batchdocs: FUSE is not
# recommended for jobs -> "Level 2 not synchronized" / stat errors). So we write
# the job's output to LOCAL scratch and stage it out to EOS via xrootd (xrdcp)
# at the end. Reads of the weight/raw HDF5 still go through the FUSE mount
# (h5py can't read root:// URLs); condor max_retries covers occasional read hiccups.
set -eo pipefail
echo "[host] $(hostname)"
echo "[args] $*"
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export MPLCONFIGDIR="/tmp/mplconfig_${USER:-job}"

if ! command -v xrdcp >/dev/null 2>&1; then
  echo "[fatal] xrdcp not found in environment" >&2
  exit 66
fi

# Rewrite --output-dir to a local sandbox path; remember the intended EOS path.
args=("$@")
eos_out=""
scratch="${_CONDOR_SCRATCH_DIR:-$PWD}"
local_out="${scratch}/jobout"
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--output-dir" ]]; then
    eos_out="${args[$((i+1))]}"
    args[$((i+1))]="${local_out}"
  fi
done
mkdir -p "${local_out}"

# Give the trainer the real EOS output dir so it can push/pull a rolling
# checkpoint (last.pt/best_model.pt) each epoch and resume after an eviction.
if [[ -n "${eos_out}" ]]; then
  args+=("--resume-eos-dir" "${eos_out}")
fi

# Stage the compact dataset (single file) to local scratch and read it locally
# (avoids per-object FUSE reads during training).
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--compact" ]]; then
    eos_compact="${args[$((i+1))]}"
    local_compact="${scratch}/$(basename "${eos_compact}")"
    echo "[stage-in] ${eos_compact} -> ${local_compact}"
    xrdcp -f "root://eosuser.cern.ch/${eos_compact}" "${local_compact}"
    args[$((i+1))]="${local_compact}"
  fi
done

python3 -c "import torch; print('[gpu]', torch.cuda.is_available(), torch.cuda.device_count())" || true

set +e
python3 "${args[@]}"
status=$?
set -e
if [[ ${status} -ne 0 ]]; then
  echo "[python] failed rc=${status}; not staging out" >&2
  exit ${status}
fi

# ---- stage local output out to EOS via xrootd ----
if [[ -n "${eos_out}" ]]; then
  url="root://eosuser.cern.ch/${eos_out}"   # eos_out starts with /eos -> //eos
  echo "[stage-out] ${local_out} -> ${url}"
  shopt -s nullglob
  for f in "${local_out}"/*; do
    [[ -f "$f" ]] || continue
    ok=0
    for attempt in 1 2 3; do
      if xrdcp -f -p "$f" "${url}/$(basename "$f")"; then ok=1; break; fi
      echo "[xrdcp] $(basename "$f") attempt ${attempt} failed; retrying" >&2
      sleep 20
    done
    if [[ ${ok} -ne 1 ]]; then
      echo "[xrdcp] FAILED permanently: $f" >&2
      exit 65
    fi
  done
  echo "[stage-out] done"
fi
