#!/usr/bin/env python3
"""Regenerate the FULL standard evaluation plot set for every completed mreg run
into a NEW folder (default runs/<campaign>_eval/<run>/), identically for all configs:

  classification (unweighted):
    roc_test_unw.png, score_distribution_test_unw.png,
    efficiency_vs_pt_test_bkg{2p5,05,10}_unw.png (+json)  [signal eff + bkg fake-rate]
  mass regression:
    mass_pred_vs_true_test.png (log), mass_pred_vs_true_linear_test.png,
    mass_resolution_vs_mass_test.png, mass_pred_hist_per_point_test.png

Pure post-processing from each run's val/test_predictions.csv (no model run).
Usage: regen_mreg_eval.py [campaign ...]   (default: mreg_v1)
"""
import csv
import glob
import os
import sys

import numpy as np

import multitask_common as mc
from train_resnet_image_classifier import save_unweighted_test_plots
from build_event_level_weights import PT_BIN_EDGES_WITH_OVERFLOW

RUNS = "/eos/user/y/yeo/4l/runs"
PT_EDGES = np.asarray(PT_BIN_EDGES_WITH_OVERFLOW, dtype=np.float64)


def _load(path):
    """Return dict of numpy columns from a predictions CSV (by header name)."""
    cols = {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        data = {k: [] for k in r.fieldnames}
        for row in r:
            for k in r.fieldnames:
                data[k] = data.get(k, [])
                data[k].append(row[k])
    for k, v in data.items():
        try:
            cols[k] = np.array(v, dtype=np.float64)
        except ValueError:
            cols[k] = np.array(v)
    return cols


def process(run_dir, out_dir):
    tp = os.path.join(run_dir, "test_predictions.csv")
    if not os.path.exists(tp):
        return False
    test = _load(tp)
    if "m_true" not in test or "pt" not in test:
        return False
    os.makedirs(out_dir, exist_ok=True)
    # val for the WP threshold; fall back to test if no val CSV
    vp = os.path.join(run_dir, "val_predictions.csv")
    val = _load(vp) if os.path.exists(vp) else test

    # --- classification: unweighted roc / score / efficiency (WP 2.5/5/10%) ---
    save_unweighted_test_plots(out_dir, val["label"], val["score"],
                               test["label"], test["score"], test["pt"], PT_EDGES)

    # --- mass regression (signal only) ---
    sig = test["is_sig"].astype(float) > 0.5 if "is_sig" in test else np.ones(len(test["m_true"]), bool)
    mc.save_mass_regression_plots(out_dir, test["m_true"][sig], test["m_pred"][sig], suffix="_test")
    return True


def main():
    camps = sys.argv[1:] or ["mreg_v1"]
    for camp in camps:
        base = f"{RUNS}/{camp}"
        eval_base = f"{RUNS}/{camp}_eval"
        ok = skip = 0
        for d in sorted(glob.glob(base + "/*/")):
            if not os.path.exists(d + "test_metrics.json"):
                continue
            name = os.path.basename(d.rstrip("/"))
            if process(d, os.path.join(eval_base, name)):
                ok += 1
                print(f"[ok] {camp}/{name}")
            else:
                skip += 1
                print(f"[skip] {camp}/{name} (missing csv/cols)")
        print(f"{camp}: {ok} plotted -> {eval_base}/, {skip} skipped")


if __name__ == "__main__":
    main()
