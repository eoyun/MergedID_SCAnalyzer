#!/usr/bin/env bash
# Test: can a Condor worker create a dir + write a file on EOS (with SendCredential)?
set -eo pipefail
P="${1:-0}"
echo "[host]   $(hostname)"
echo "[whoami] $(whoami)"
echo "[klist]"
klist 2>&1 | head -5 || echo "  (no kerberos ticket on worker)"
TARGET="/eos/user/y/yeo/4l/runs/kerberos/proc_${P}"
echo "[mkdir]  ${TARGET}"
mkdir -p "${TARGET}"
echo "written by proc ${P} on $(hostname) at $(date)" > "${TARGET}/hello.txt"
echo "[ok] wrote ${TARGET}/hello.txt"
ls -l "${TARGET}"
