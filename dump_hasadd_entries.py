#!/usr/bin/env python3
"""Dump per-object entries from the RAW tier data, filtered by hasAdditionalTrk.

For each selected object (electron supercluster candidate) it emits:
  raw_file, sample_id, object_idx, event_idx, hasAdditionalTrk, pt

- sample_id: W/Z -> "background", signal -> "signal_H{h}_A{a}" (parsed from source).
- pt: the event pt (max A pT per event, same definition the weight builder uses)
  mapped onto each object via the {DET}_ele_event_idx object->event map.
- hasADD filter: all | eq0 (==0) | eq1 (==1).

Print to screen (--limit rows + a per-sample count) or write the full list to CSV
(--output). This is a diagnostic dump; it does not touch weights or compacts.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    iter_h5_files,
    extract_source_path,
    extract_process_name,
    classify_sample,
    derive_event_pt,
)

DET_EVENT_IDX = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}
HASADD_KEY = {"eb": "hasAdditionalTrk_EB", "ee": "hasAdditionalTrk_EE"}
FIELDS = ["raw_file", "sample_id", "object_idx", "event_idx", "hasAdditionalTrk", "pt"]


def hasadd_mask(values, mode):
    v = np.asarray(values)
    if mode == "eq1":
        return v == 1
    if mode == "eq0":
        return v == 0
    return np.ones(v.shape, dtype=bool)


def iter_entries(input_dir, detector, mode, event_pt_mode="max", max_files=None):
    ev_key = DET_EVENT_IDX[detector]
    ha_key = HASADD_KEY[detector]
    for fi, (split, path) in enumerate(iter_h5_files(Path(input_dir))):
        if max_files is not None and fi >= max_files:
            break
        with h5py.File(path, "r") as h:
            source = extract_source_path(h.attrs["source_root_files"])
            sample_id = classify_sample(split, extract_process_name(source))
            n_events = int(h["eventId"].shape[0])
            event_pt = derive_event_pt(n_events, h["A_pT"][:],
                                       h["A_event_idx"][:].astype(np.int64),
                                       mode=event_pt_mode)
            det_ev = h[ev_key][:].astype(np.int64)
            ha = np.asarray(h[ha_key][:])
            obj_pt = event_pt[det_ev]
            for o in np.nonzero(hasadd_mask(ha, mode))[0]:
                yield {
                    "raw_file": str(path),
                    "sample_id": sample_id,
                    "object_idx": int(o),
                    "event_idx": int(det_ev[o]),
                    "hasAdditionalTrk": int(ha[o]),
                    "pt": float(obj_pt[o]),
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="tier dir with W/ Z/ signal/")
    ap.add_argument("--detector", choices=("eb", "ee"), required=True)
    ap.add_argument("--hasadd", choices=("all", "eq0", "eq1"), required=True)
    ap.add_argument("--event-pt-mode", choices=("max", "leading", "mean"), default="max")
    ap.add_argument("--output", default=None, help="CSV path; if omitted, print to screen")
    ap.add_argument("--limit", type=int, default=20, help="rows to print when no --output")
    ap.add_argument("--max-files", type=int, default=None, help="scan only the first N files")
    args = ap.parse_args()

    gen = iter_entries(args.input_dir, args.detector, args.hasadd,
                       args.event_pt_mode, args.max_files)

    if args.output:
        n = 0
        counts = Counter()
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for e in gen:
                w.writerow(e)
                counts[e["sample_id"]] += 1
                n += 1
        print(f"[dump] {n} objects -> {args.output}")
        for s, c in sorted(counts.items()):
            print(f"    {s:22} {c}")
    else:
        header = "  ".join(f"{f}" for f in FIELDS)
        print(header)
        counts = Counter()
        for i, e in enumerate(gen):
            counts[e["sample_id"]] += 1
            if i < args.limit:
                print(f'{e["raw_file"].split("/")[-1]}  {e["sample_id"]}  '
                      f'obj={e["object_idx"]}  ev={e["event_idx"]}  '
                      f'hasADD={e["hasAdditionalTrk"]}  pt={e["pt"]:.2f}')
        total = sum(counts.values())
        print(f"\n[summary] total selected objects = {total}  "
              f"(shown {min(args.limit, total)})")
        for s, c in sorted(counts.items()):
            print(f"    {s:22} {c}")


if __name__ == "__main__":
    main()
