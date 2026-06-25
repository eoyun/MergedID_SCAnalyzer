#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
python3 -m pipeline.make_submit
echo "[submit] fusion (18)"; condor_submit condor/fusion.sub
