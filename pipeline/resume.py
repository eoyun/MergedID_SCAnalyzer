"""Best-effort checkpoint resume across Condor evictions.

Training writes to LOCAL scratch (run_job.sh rewrites --output-dir), and Condor
only stages that out to EOS on *successful* exit. A job evicted mid-training
therefore loses everything and restarts from epoch 1. To survive eviction we
push a rolling `last.pt` (and the current `best_model.pt`) to EOS every epoch
via xrdcp, and on (re)start we pull them back and continue.

All functions are best-effort: EOS/xrdcp hiccups log a warning and never crash
training. If nothing is on EOS (first attempt) we simply start fresh.
"""
import subprocess
from pathlib import Path

EOS_PREFIX = "root://eosuser.cern.ch/"


def _xrdcp(src, dst):
    try:
        r = subprocess.run(["xrdcp", "-f", "-p", src, dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _url(eos_dir, name):
    return f"{EOS_PREFIX}{str(eos_dir).rstrip('/')}/{name}"


def stage_in(eos_dir, local_dir, names):
    """Pull checkpoint files EOS->local. Returns the set of names fetched."""
    got = set()
    for name in names:
        if _xrdcp(_url(eos_dir, name), str(Path(local_dir) / name)):
            got.add(name)
    return got


def push(eos_dir, local_path):
    """Push one local file -> EOS. Returns True on success (best-effort)."""
    local_path = Path(local_path)
    ok = _xrdcp(str(local_path), _url(eos_dir, local_path.name))
    if not ok:
        print(f"[resume] warning: failed to push {local_path.name} to EOS")
    return ok
