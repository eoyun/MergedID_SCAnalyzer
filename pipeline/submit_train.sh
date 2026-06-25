#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
python3 -m pipeline.make_submit
mkdir -p condor/logs
echo "[submit] track (12)"; condor_submit condor/train_track.sub
echo "[submit] image (18)"; condor_submit condor/train_image.sub
echo "Track + image submitted. Run pipeline/submit_fusion.sh after they finish."
