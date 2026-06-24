#!/usr/bin/env bash

set -euo pipefail

# Thin tier-aware driver around build_event_level_weights.py.
#
# The new dataset layout is:
#   <data-root>/AOD/{W,Z,signal}/*.h5
#   <data-root>/MiniAOD/{W,Z,signal}/*.h5
#
# build_event_level_weights.py already balances background vs signal across pT
# bins for the event/EB/EE cases given a directory that contains W/, Z/ and
# signal/. This script just resolves a tier name to that directory and to a
# per-tier output directory, then runs the builder once. One tier -> one
# event_level_weights.h5 (with event, EB and EE cases inside).

usage() {
  cat <<'EOF'
Usage:
  ./run_build_weights.sh --tier aod|mini [options]

Required:
  --tier                  aod (-> AOD) or mini (-> MiniAOD)

Common:
  --data-root             dataset root holding AOD/ and MiniAOD/
                          (default: /eos/user/y/yeo/4l/data)
  --output-root           root for per-tier weight outputs
                          (default: outputs/weights)
  --python                python executable (default: python3)
  --background-max-events cap on background events (passthrough; default: all)
  --event-pt-mode         max|leading|mean (passthrough; default: max)
  --seed                  random seed (passthrough; default: 1234)

Output:
  <output-root>/<TIER>/event_level_weights.h5  (+ summary CSVs + config.json)

Examples:
  ./run_build_weights.sh --tier mini
  ./run_build_weights.sh --tier aod --output-root /eos/user/y/yeo/4l/weights
EOF
}

TIER=""
DATA_ROOT="/eos/user/y/yeo/4l/data"
OUTPUT_ROOT="outputs/weights"
PYTHON_BIN="python3"
BACKGROUND_MAX_EVENTS=""
EVENT_PT_MODE=""
SEED=""

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "${value}" ]]; then
    echo "Missing value for ${flag}" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      require_value "$1" "${2-}"; TIER="$2"; shift 2;;
    --data-root)
      require_value "$1" "${2-}"; DATA_ROOT="$2"; shift 2;;
    --output-root)
      require_value "$1" "${2-}"; OUTPUT_ROOT="$2"; shift 2;;
    --python)
      require_value "$1" "${2-}"; PYTHON_BIN="$2"; shift 2;;
    --background-max-events)
      require_value "$1" "${2-}"; BACKGROUND_MAX_EVENTS="$2"; shift 2;;
    --event-pt-mode)
      require_value "$1" "${2-}"; EVENT_PT_MODE="$2"; shift 2;;
    --seed)
      require_value "$1" "${2-}"; SEED="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; usage >&2; exit 1;;
  esac
done

if [[ -z "${TIER}" ]]; then
  echo "--tier is required" >&2
  usage >&2
  exit 1
fi

# Normalize tier name to the on-disk directory name.
case "${TIER,,}" in
  aod)            TIER_DIR="AOD";;
  mini|miniaod)   TIER_DIR="MiniAOD";;
  *)
    echo "--tier must be 'aod' or 'mini' (got '${TIER}')" >&2
    exit 1;;
esac

# Resolve the script directory so the builder is found regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDER="${SCRIPT_DIR}/build_event_level_weights.py"
if [[ ! -f "${BUILDER}" ]]; then
  # Fall back to cwd (e.g. when transferred into a Condor scratch sandbox).
  BUILDER="build_event_level_weights.py"
fi

INPUT_DIR="${DATA_ROOT}/${TIER_DIR}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TIER_DIR}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 1
fi
for sub in W Z signal; do
  if [[ ! -d "${INPUT_DIR}/${sub}" ]]; then
    echo "Missing '${sub}/' under ${INPUT_DIR}." >&2
    if [[ "${sub}" == "signal" ]]; then
      echo "build_event_level_weights.py needs signal/ to balance signal vs background; rerun when it is present." >&2
    fi
    exit 1
  fi
done

ARGS=(
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${BACKGROUND_MAX_EVENTS}" ]]; then
  ARGS+=(--background-max-events "${BACKGROUND_MAX_EVENTS}")
fi
if [[ -n "${EVENT_PT_MODE}" ]]; then
  ARGS+=(--event-pt-mode "${EVENT_PT_MODE}")
fi
if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

echo "[tier]    ${TIER_DIR}"
echo "[input]   ${INPUT_DIR}"
echo "[output]  ${OUTPUT_DIR}"
echo "[python]  ${PYTHON_BIN}"
echo "[builder] ${BUILDER}"

"${PYTHON_BIN}" "${BUILDER}" "${ARGS[@]}"

echo
echo "done: ${OUTPUT_DIR}/event_level_weights.h5"
