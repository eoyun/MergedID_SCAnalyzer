#!/usr/bin/env python3
"""Slim every raw HDF5 in place, resumably and safely.

For each file: skip if already slimmed (drop datasets absent); else slim to local
scratch, verify byte-identical against the original, copy the slim to an EOS temp
next to the original, then atomically rename it over the original. The original is
only replaced AFTER verification, so an interruption never leaves a corrupt file
(a half-written temp is discarded; unprocessed files keep their originals).

Run on the login node (EOS FUSE writes are fine there). Parallel via a spawn Pool.
"""
import argparse
import glob
import os
import shutil

import h5py
import multiprocessing as mp

from slim_raw import slim_file, verify, DROP


def already_slim(path):
    try:
        with h5py.File(path, "r") as h:
            return not any(d in h for d in DROP)
    except Exception:
        return False


def process_one(args):
    path, scratch = args
    try:
        if already_slim(path):
            return (path, "skip")
        base = os.path.basename(path)
        local_tmp = os.path.join(scratch, f"{base}.{os.getpid()}.slim")
        slim_file(path, local_tmp)
        ok, probs = verify(path, local_tmp)          # byte-identical kept, drops gone
        if not ok:
            os.remove(local_tmp)
            return (path, "VERIFY_FAIL:" + ";".join(probs[:2]))
        eos_tmp = path + ".slimtmp"
        shutil.copyfile(local_tmp, eos_tmp)          # write slim next to original (EOS)
        os.replace(eos_tmp, path)                    # atomic rename over original
        os.remove(local_tmp)
        return (path, "ok")
    except Exception as e:
        return (path, "ERR:" + repr(e))


def iter_files(data_root, tiers):
    for tier in tiers:
        for split in ("W", "Z", "signal"):
            yield from sorted(glob.glob(f"{data_root}/{tier}/{split}/*.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/eos/user/y/yeo/4l/data")
    ap.add_argument("--tier", choices=("AOD", "MiniAOD", "both"), default="both")
    ap.add_argument("--scratch", default="/tmp/yeo/slim_scratch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="process only first N (testing)")
    ap.add_argument("--files", nargs="*", default=None, help="explicit file list (testing)")
    args = ap.parse_args()

    os.makedirs(args.scratch, exist_ok=True)
    tiers = ["AOD", "MiniAOD"] if args.tier == "both" else [args.tier]
    files = args.files if args.files else list(iter_files(args.data_root, tiers))
    if args.limit:
        files = files[:args.limit]
    print(f"[slim] {len(files)} files, {args.workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    tally = {"ok": 0, "skip": 0, "fail": 0}
    done = 0
    with ctx.Pool(args.workers) as pool:
        for path, status in pool.imap_unordered(process_one, ((f, args.scratch) for f in files)):
            done += 1
            if status == "ok":
                tally["ok"] += 1
            elif status == "skip":
                tally["skip"] += 1
            else:
                tally["fail"] += 1
                print(f"  [FAIL] {os.path.basename(path)}: {status}", flush=True)
            if done % 500 == 0:
                print(f"  progress {done}/{len(files)}  {tally}", flush=True)
    print(f"[slim] done: {tally}", flush=True)


if __name__ == "__main__":
    main()
