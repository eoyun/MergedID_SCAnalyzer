#!/usr/bin/env python3
"""One-time migration: add object->event maps (EB/EE_ele_event_idx) into existing
category weight files, so training no longer opens raw files at startup.

Runs on the login node with the system python3 (h5py). Reads each raw file's
event-index arrays (parallel, to overlap EOS-fuse latency) and writes them into
the matching group of each weight file. New weight files already carry these
(build_category_weights writes them); this is only for files built before.
"""
import argparse
import sys
from multiprocessing import Pool

import h5py
import numpy as np

KEYS = ["EB_ele_event_idx", "EE_ele_event_idx"]


def read_one(args):
    sp, st, raw = args
    try:
        with h5py.File(raw, "r") as r:
            return sp, st, {k: r[k][:].astype(np.int64) for k in KEYS}
    except Exception as e:  # noqa: BLE001
        return sp, st, {"__error__": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weight_files", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    for wf in args.weight_files:
        with h5py.File(wf, "r") as f:
            groups = [(sp, st, str(f["files"][sp][st].attrs["input_file"]))
                      for sp in f["files"] for st in f["files"][sp]]
        print(f"[{wf}] {len(groups)} groups", flush=True)

        with Pool(args.workers) as pool:
            results = pool.map(read_one, groups)

        errs = [(sp, st, d["__error__"]) for sp, st, d in results if "__error__" in d]
        if errs:
            print(f"  {len(errs)} read errors, e.g. {errs[:2]}", file=sys.stderr, flush=True)

        n, skip = 0, 0
        with h5py.File(wf, "a") as f:
            for sp, st, d in results:
                if "__error__" in d:
                    skip += 1
                    continue
                g = f["files"][sp][st]
                wlen = g["EB_weight_split"].shape[0]
                if d["EB_ele_event_idx"].shape[0] != wlen:
                    print(f"  LEN MISMATCH {sp}/{st}", file=sys.stderr, flush=True)
                    skip += 1
                    continue
                for k in KEYS:
                    if k in g:
                        del g[k]
                    g.create_dataset(k, data=d[k], compression="gzip")
                n += 1
        print(f"  wrote event_idx into {n} groups (skipped {skip})", flush=True)


if __name__ == "__main__":
    main()
