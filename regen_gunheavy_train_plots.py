#!/usr/bin/env python3
"""Redraw the per-run TRAINING-result CLASSIFICATION plots in a chosen format (default
.eps) from each run's val/test_predictions.csv, reusing the shared plot helpers with an
explicit output path (no training code modified). Replicates save_unweighted_test_plots:

  roc_test_unw, roc_test, score_distribution_test_unw,
  efficiency_vs_pt_test_bkg{2p5,05,10}_unw   (WP threshold on validation background)

Mass-regression training plots are handled by regen_gunheavy_mass.py (0-60).

Usage: regen_gunheavy_train_plots.py [--eps|--pdf|--png] [run_dir ...]
       default fmt .eps, default dirs runs/gunheavy_v1/*/
"""
import csv
import glob
import os
import sys

import numpy as np

from train_resnet_image_classifier import (
    save_roc_plot, save_score_distribution, save_efficiency_plot,
    threshold_for_bkg_eff_unw, compute_efficiency_by_pt)
from build_event_level_weights import PT_BIN_EDGES_WITH_OVERFLOW

RUNS = "/eos/user/y/yeo/4l/runs/gunheavy_v1"
PT_EDGES = np.asarray(PT_BIN_EDGES_WITH_OVERFLOW, dtype=np.float64)
FAKE_WPS = (0.025, 0.05, 0.10)


def _load(path):
    cols = {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        data = {k: [] for k in r.fieldnames}
        for row in r:
            for k in r.fieldnames:
                data[k].append(row[k])
    for k, v in data.items():
        try:
            cols[k] = np.array(v, dtype=np.float64)
        except ValueError:
            cols[k] = np.array(v)
    return cols


def _ptag(tgt):
    pct = tgt * 100.0
    return (f"{int(round(pct)):02d}" if abs(pct - round(pct)) < 1e-9
            else f"{pct:g}".replace(".", "p"))


def redo(run_dir, ext):
    tp = os.path.join(run_dir, "test_predictions.csv")
    if not os.path.exists(tp):
        return False
    test = _load(tp)
    vp = os.path.join(run_dir, "val_predictions.csv")
    val = _load(vp) if os.path.exists(vp) else test

    vt, vs = val["label"], val["score"]
    tt, ts, tpt = test["label"], test["score"], test["pt"]
    vfin = np.isfinite(vs); vt, vs = vt[vfin], vs[vfin]
    tfin = np.isfinite(ts); tt, ts, tpt = tt[tfin], ts[tfin], tpt[tfin]
    ones = np.ones_like(ts, dtype=np.float64)
    w = test["weight"][tfin].astype(np.float64) if "weight" in test else ones

    save_roc_plot(tt, ts, ones, os.path.join(run_dir, f"roc_test_unw{ext}"))
    save_roc_plot(tt, ts, w, os.path.join(run_dir, f"roc_test{ext}"))
    save_score_distribution(tt, ts, ones, os.path.join(run_dir, f"score_distribution_test_unw{ext}"),
                            normalize=True, ylabel="a.u.")
    for tgt in FAKE_WPS:
        thr = threshold_for_bkg_eff_unw(vt, vs, tgt)
        rows = compute_efficiency_by_pt(tt, ts, ones, tpt, thr, PT_EDGES)
        save_efficiency_plot(rows, os.path.join(run_dir, f"efficiency_vs_pt_test_bkg{_ptag(tgt)}_unw{ext}"),
                             thr, min_denom=20)
    print(f"[ok] {os.path.basename(run_dir.rstrip('/'))}", flush=True)
    return True


def main():
    args = sys.argv[1:]
    ext = ".eps"
    for flag, e in (("--eps", ".eps"), ("--pdf", ".pdf"), ("--png", ".png")):
        if flag in args:
            args.remove(flag); ext = e
    dirs = args or sorted(glob.glob(RUNS + "/*/"))
    print(f"[fmt] {ext}", flush=True)
    ok = skip = 0
    for d in dirs:
        if redo(d, ext):
            ok += 1
        else:
            skip += 1
            print(f"[skip: no test_predictions.csv] {os.path.basename(d.rstrip('/'))}")
    print(f"done: {ok} redrawn, {skip} skipped")


if __name__ == "__main__":
    main()
