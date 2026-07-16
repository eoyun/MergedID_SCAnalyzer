#!/usr/bin/env python3
"""Per-sample object counts split by hasAdditionalTrk.

Run with NO arguments to scan the full dataset and print, for each tier:
an EB table, a divider, then an EE table -- columns: sample, total,
hasADD=1, hasADD=0.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    iter_h5_files,
    extract_source_path,
    extract_process_name,
    classify_sample,
)

HA_KEY = {"eb": "hasAdditionalTrk_EB", "ee": "hasAdditionalTrk_EE"}
W = 52


def scan(input_dir, max_files=None):
    counts = {"eb": defaultdict(lambda: [0, 0, 0]),
              "ee": defaultdict(lambda: [0, 0, 0])}
    for fi, (split, path) in enumerate(iter_h5_files(Path(input_dir))):
        if max_files is not None and fi >= max_files:
            break
        with h5py.File(path, "r") as h:
            sid = classify_sample(
                split, extract_process_name(extract_source_path(h.attrs["source_root_files"])))
            for det in ("eb", "ee"):
                ha = np.asarray(h[HA_KEY[det]][:])
                c = counts[det][sid]
                c[0] += int(ha.size)
                c[1] += int((ha == 1).sum())
                c[2] += int((ha == 0).sum())
    return counts


def print_table(counts_det, title):
    print(f"[{title}]")
    print(f"{'sample':<24}{'total':>10}{'hasADD=1':>10}{'hasADD=0':>10}")
    rows = sorted(counts_det.items(), key=lambda kv: (kv[0] == "background", kv[0]))
    tt = t1 = t0 = 0
    for sid, (tot, n1, n0) in rows:
        print(f"{sid:<24}{tot:>10,}{n1:>10,}{n0:>10,}")
        tt += tot; t1 += n1; t0 += n0
    print(f"{'TOTAL':<24}{tt:>10,}{t1:>10,}{t0:>10,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/eos/user/y/yeo/4l/data")
    ap.add_argument("--tier", choices=("aod", "mini", "both"), default="both")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()
    tiers = {"aod": ["AOD"], "mini": ["MiniAOD"], "both": ["AOD", "MiniAOD"]}[args.tier]
    for tier in tiers:
        print("=" * W)
        print(f"  {tier}")
        print("=" * W)
        counts = scan(f"{args.data_root}/{tier}", args.max_files)
        print_table(counts["eb"], "EB")
        print("-" * W)
        print_table(counts["ee"], "EE")
        print()


if __name__ == "__main__":
    main()
