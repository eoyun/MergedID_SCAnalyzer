#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_holdout_fusion_pipeline.sh \
    --detector eb \
    --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
    --output-root outputs/pipeline_eb

Required arguments:
  --detector                eb or ee
  --weight-h5               path to event_level_weights.h5
  --output-root             root directory for image/track/fusion outputs

Common arguments:
  --python                  python executable (default: ./.venv/bin/python)
  --epochs                  max epochs for image and track training (default: 20)
  --num-workers             dataloader workers for image/track/fusion (default: 0)
  --seed                    random seed shared by image and track (default: 42)
  --train-frac              train fraction shared by image and track (default: 0.8)
  --val-frac                validation fraction shared by image and track (default: 0.1)
  --threshold-mode          max-f1 or youden (default: max-f1)
  --weight-key              override detector-specific weight dataset
  --debug                   enable debug mode for image and track
  --debug-max-events-per-sample  debug event cap per sample (default: 12)

Image arguments:
  --image-model             resnet18, resnet34, or resnet50 (default: resnet18)
  --image-batch-size        image batch size (default: 16)
  --image-lr                image learning rate (default: 1e-4)
  --image-weight-decay      image weight decay (default: 1e-4)
  --image-monitor           auc or f1 (default: auc)
  --image-patience          image early stopping patience (default: 5)
  --disable-log-scale       disable image channel log1p preprocessing

Track arguments:
  --track-batch-size        track batch size (default: 64)
  --track-lr                track learning rate (default: 1e-4)
  --track-weight-decay      track weight decay (default: 1e-4)
  --track-monitor           auc or f1 (default: auc)
  --track-patience          track early stopping patience (default: 5)
  --max-points              max track points per object (default: 16)
  --embed-dim               transformer embedding dim (default: 128)
  --depth                   transformer depth (default: 4)
  --num-heads               transformer attention heads (default: 4)
  --mlp-ratio               transformer mlp ratio (default: 4.0)
  --dropout                 transformer dropout (default: 0.1)
  --disable-log-track-pt    use raw track pT instead of log1p

Fusion arguments:
  --prediction-source       auto, csv, or rescore (default: csv)
  --fusion-num-workers      dataloader workers for fusion (default: same as --num-workers)
  --fusion-image-batch-size optional image batch size override for fusion rescoring
  --fusion-track-batch-size optional track batch size override for fusion rescoring

Example:
  ./run_holdout_fusion_pipeline.sh \
    --detector eb \
    --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
    --output-root outputs/holdout_eb \
    --epochs 20 \
    --num-workers 4 \
    --prediction-source csv
EOF
}

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "${value}" ]]; then
    echo "Missing value for ${flag}" >&2
    exit 1
  fi
}

PYTHON_BIN="./.venv/bin/python"
DETECTOR=""
WEIGHT_H5=""
OUTPUT_ROOT=""

EPOCHS="20"
NUM_WORKERS="0"
SEED="42"
TRAIN_FRAC="0.8"
VAL_FRAC="0.1"
THRESHOLD_MODE="max-f1"
WEIGHT_KEY=""
DEBUG=0
DEBUG_MAX_EVENTS_PER_SAMPLE="12"

IMAGE_MODEL="resnet18"
IMAGE_BATCH_SIZE="16"
IMAGE_LR="1e-4"
IMAGE_WEIGHT_DECAY="1e-4"
IMAGE_MONITOR="auc"
IMAGE_PATIENCE="5"
DISABLE_LOG_SCALE=0

TRACK_BATCH_SIZE="64"
TRACK_LR="1e-4"
TRACK_WEIGHT_DECAY="1e-4"
TRACK_MONITOR="auc"
TRACK_PATIENCE="5"
MAX_POINTS="16"
EMBED_DIM="128"
DEPTH="4"
NUM_HEADS="4"
MLP_RATIO="4.0"
DROPOUT="0.1"
DISABLE_LOG_TRACK_PT=0

PREDICTION_SOURCE="csv"
FUSION_NUM_WORKERS=""
FUSION_IMAGE_BATCH_SIZE=""
FUSION_TRACK_BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detector)
      require_value "$1" "${2-}"
      DETECTOR="$2"
      shift 2
      ;;
    --weight-h5)
      require_value "$1" "${2-}"
      WEIGHT_H5="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --epochs)
      require_value "$1" "${2-}"
      EPOCHS="$2"
      shift 2
      ;;
    --num-workers)
      require_value "$1" "${2-}"
      NUM_WORKERS="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "${2-}"
      SEED="$2"
      shift 2
      ;;
    --train-frac)
      require_value "$1" "${2-}"
      TRAIN_FRAC="$2"
      shift 2
      ;;
    --val-frac)
      require_value "$1" "${2-}"
      VAL_FRAC="$2"
      shift 2
      ;;
    --threshold-mode)
      require_value "$1" "${2-}"
      THRESHOLD_MODE="$2"
      shift 2
      ;;
    --weight-key)
      require_value "$1" "${2-}"
      WEIGHT_KEY="$2"
      shift 2
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --debug-max-events-per-sample)
      require_value "$1" "${2-}"
      DEBUG_MAX_EVENTS_PER_SAMPLE="$2"
      shift 2
      ;;
    --image-model)
      require_value "$1" "${2-}"
      IMAGE_MODEL="$2"
      shift 2
      ;;
    --image-batch-size)
      require_value "$1" "${2-}"
      IMAGE_BATCH_SIZE="$2"
      shift 2
      ;;
    --image-lr)
      require_value "$1" "${2-}"
      IMAGE_LR="$2"
      shift 2
      ;;
    --image-weight-decay)
      require_value "$1" "${2-}"
      IMAGE_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --image-monitor)
      require_value "$1" "${2-}"
      IMAGE_MONITOR="$2"
      shift 2
      ;;
    --image-patience)
      require_value "$1" "${2-}"
      IMAGE_PATIENCE="$2"
      shift 2
      ;;
    --disable-log-scale)
      DISABLE_LOG_SCALE=1
      shift
      ;;
    --track-batch-size)
      require_value "$1" "${2-}"
      TRACK_BATCH_SIZE="$2"
      shift 2
      ;;
    --track-lr)
      require_value "$1" "${2-}"
      TRACK_LR="$2"
      shift 2
      ;;
    --track-weight-decay)
      require_value "$1" "${2-}"
      TRACK_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --track-monitor)
      require_value "$1" "${2-}"
      TRACK_MONITOR="$2"
      shift 2
      ;;
    --track-patience)
      require_value "$1" "${2-}"
      TRACK_PATIENCE="$2"
      shift 2
      ;;
    --max-points)
      require_value "$1" "${2-}"
      MAX_POINTS="$2"
      shift 2
      ;;
    --embed-dim)
      require_value "$1" "${2-}"
      EMBED_DIM="$2"
      shift 2
      ;;
    --depth)
      require_value "$1" "${2-}"
      DEPTH="$2"
      shift 2
      ;;
    --num-heads)
      require_value "$1" "${2-}"
      NUM_HEADS="$2"
      shift 2
      ;;
    --mlp-ratio)
      require_value "$1" "${2-}"
      MLP_RATIO="$2"
      shift 2
      ;;
    --dropout)
      require_value "$1" "${2-}"
      DROPOUT="$2"
      shift 2
      ;;
    --disable-log-track-pt)
      DISABLE_LOG_TRACK_PT=1
      shift
      ;;
    --prediction-source)
      require_value "$1" "${2-}"
      PREDICTION_SOURCE="$2"
      shift 2
      ;;
    --fusion-num-workers)
      require_value "$1" "${2-}"
      FUSION_NUM_WORKERS="$2"
      shift 2
      ;;
    --fusion-image-batch-size)
      require_value "$1" "${2-}"
      FUSION_IMAGE_BATCH_SIZE="$2"
      shift 2
      ;;
    --fusion-track-batch-size)
      require_value "$1" "${2-}"
      FUSION_TRACK_BATCH_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DETECTOR}" || -z "${WEIGHT_H5}" || -z "${OUTPUT_ROOT}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${DETECTOR}" != "eb" && "${DETECTOR}" != "ee" ]]; then
  echo "--detector must be 'eb' or 'ee'" >&2
  exit 1
fi

if [[ -z "${FUSION_NUM_WORKERS}" ]]; then
  FUSION_NUM_WORKERS="${NUM_WORKERS}"
fi

IMAGE_OUTPUT_DIR="${OUTPUT_ROOT}/image_${DETECTOR}"
TRACK_OUTPUT_DIR="${OUTPUT_ROOT}/track_${DETECTOR}"
FUSION_OUTPUT_DIR="${OUTPUT_ROOT}/fusion_${DETECTOR}"

mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
  --detector "${DETECTOR}"
  --weight-h5 "${WEIGHT_H5}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --train-frac "${TRAIN_FRAC}"
  --val-frac "${VAL_FRAC}"
  --threshold-mode "${THRESHOLD_MODE}"
  --debug-max-events-per-sample "${DEBUG_MAX_EVENTS_PER_SAMPLE}"
)

if [[ -n "${WEIGHT_KEY}" ]]; then
  COMMON_ARGS+=(--weight-key "${WEIGHT_KEY}")
fi

if [[ "${DEBUG}" -eq 1 ]]; then
  COMMON_ARGS+=(--debug)
fi

IMAGE_ARGS=(
  "${COMMON_ARGS[@]}"
  --output-dir "${IMAGE_OUTPUT_DIR}"
  --model "${IMAGE_MODEL}"
  --batch-size "${IMAGE_BATCH_SIZE}"
  --lr "${IMAGE_LR}"
  --weight-decay "${IMAGE_WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --monitor "${IMAGE_MONITOR}"
  --patience "${IMAGE_PATIENCE}"
)

if [[ "${DISABLE_LOG_SCALE}" -eq 1 ]]; then
  IMAGE_ARGS+=(--disable-log-scale)
fi

TRACK_ARGS=(
  "${COMMON_ARGS[@]}"
  --output-dir "${TRACK_OUTPUT_DIR}"
  --batch-size "${TRACK_BATCH_SIZE}"
  --lr "${TRACK_LR}"
  --weight-decay "${TRACK_WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --monitor "${TRACK_MONITOR}"
  --patience "${TRACK_PATIENCE}"
  --max-points "${MAX_POINTS}"
  --embed-dim "${EMBED_DIM}"
  --depth "${DEPTH}"
  --num-heads "${NUM_HEADS}"
  --mlp-ratio "${MLP_RATIO}"
  --dropout "${DROPOUT}"
)

if [[ "${DISABLE_LOG_TRACK_PT}" -eq 1 ]]; then
  TRACK_ARGS+=(--disable-log-track-pt)
fi

FUSION_ARGS=(
  --image-run-dir "${IMAGE_OUTPUT_DIR}"
  --track-run-dir "${TRACK_OUTPUT_DIR}"
  --output-dir "${FUSION_OUTPUT_DIR}"
  --num-workers "${FUSION_NUM_WORKERS}"
  --threshold-mode "${THRESHOLD_MODE}"
  --prediction-source "${PREDICTION_SOURCE}"
)

if [[ -n "${FUSION_IMAGE_BATCH_SIZE}" ]]; then
  FUSION_ARGS+=(--image-batch-size "${FUSION_IMAGE_BATCH_SIZE}")
fi

if [[ -n "${FUSION_TRACK_BATCH_SIZE}" ]]; then
  FUSION_ARGS+=(--track-batch-size "${FUSION_TRACK_BATCH_SIZE}")
fi

echo "[1/3] train image model"
"${PYTHON_BIN}" train_resnet_image_classifier.py "${IMAGE_ARGS[@]}"

echo "[2/3] train track model"
"${PYTHON_BIN}" train_point_transformer_track_classifier.py "${TRACK_ARGS[@]}"

echo "[3/3] train fusion model"
"${PYTHON_BIN}" train_fusion_ensemble.py "${FUSION_ARGS[@]}"

echo
echo "image run  : ${IMAGE_OUTPUT_DIR}"
echo "track run  : ${TRACK_OUTPUT_DIR}"
echo "fusion run : ${FUSION_OUTPUT_DIR}"
