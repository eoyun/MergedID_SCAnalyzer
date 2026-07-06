"""Redraw efficiency-vs-pT at a FIXED background fake-rate working point,
without retraining.

Operating point: pick the score threshold so that the (weighted) background
efficiency = TARGET (default 0.10). The threshold is chosen on the VALIDATION
set (val_predictions.csv) to avoid choosing the operating point on the same
test sample we then plot -- consistent with the training pipeline, which also
selects its threshold on validation. We then apply that fixed threshold to the
test set and redraw signal/background efficiency vs pT (with error bars).

Outputs go to efficiency_vs_pt_test_bkg{NN}.{png,json} so the original max-f1
plots are left untouched.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_resnet_image_classifier import (
    compute_efficiency_by_pt,
    save_efficiency_plot,
)
from regenerate_efficiency_plots import reconstruct_edges, load_predictions


def threshold_for_bkg_eff(y_true, y_score, weights, target):
    """Smallest threshold with weighted background efficiency <= target.

    Background efficiency(thr) = sum_w(bkg & score>=thr) / sum_w(bkg) is
    monotonically decreasing in thr, so the target sits at the (1-target)
    weighted quantile of the background score distribution.
    """
    bkg = (y_true == 0)
    s = y_score[bkg]
    w = weights[bkg]
    order = np.argsort(s)
    s, w = s[order], w[order]
    cdf = np.cumsum(w) / w.sum()          # fraction with score <= s
    # want fraction with score >= thr == target  ->  cdf just below (1-target)
    idx = int(np.searchsorted(cdf, 1.0 - target, side="left"))
    idx = min(idx, len(s) - 1)
    return float(s[idx])


def integrated_bkg_eff(y_true, y_score, weights, thr):
    bkg = (y_true == 0)
    passed = bkg & (y_score >= thr)
    denom = weights[bkg].sum()
    return float(weights[passed].sum() / denom) if denom > 0 else float("nan")


def process_run(run_dir, target):
    run_dir = Path(run_dir)
    val_csv = run_dir / "val_predictions.csv"
    test_csv = run_dir / "test_predictions.csv"
    eff_json = run_dir / "efficiency_vs_pt_test.json"
    for p in (val_csv, test_csv, eff_json):
        if not p.exists():
            return f"skip ({p.name} missing)"

    v_score, v_true, v_w, _ = load_predictions(val_csv)
    thr = threshold_for_bkg_eff(v_true, v_score, v_w, target)

    t_score, t_true, t_w, t_pt = load_predictions(test_csv)
    edges = reconstruct_edges(eff_json)
    rows = compute_efficiency_by_pt(t_true, t_score, t_w, t_pt, thr, edges)

    realized = integrated_bkg_eff(t_true, t_score, t_w, thr)
    tag = f"bkg{int(round(target * 100)):02d}"
    save_efficiency_plot(rows, run_dir / f"efficiency_vs_pt_test_{tag}.png", thr)
    with (run_dir / f"efficiency_vs_pt_test_{tag}.json").open("w") as fh:
        json.dump({"target_bkg_eff": target, "threshold": thr,
                   "realized_test_bkg_eff": realized, "rows": rows}, fh, indent=2)
    return (f"ok thr={thr:.4f} (val bkg_eff={target:.2f}, "
            f"test bkg_eff={realized:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/eos/user/y/yeo/4l/runs/v5")
    ap.add_argument("--glob", default="*/")
    ap.add_argument("--target", type=float, default=0.10,
                    help="target background efficiency / fake rate")
    args = ap.parse_args()
    root = Path(args.runs_root)
    for d in sorted(x for x in root.glob(args.glob) if x.is_dir()):
        print(f"[{d.name}] {process_run(d, args.target)}")


if __name__ == "__main__":
    main()
