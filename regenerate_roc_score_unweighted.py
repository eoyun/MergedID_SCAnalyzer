"""Regenerate UNWEIGHTED ROC and score-distribution plots from saved
test_predictions.csv, without retraining.

Training writes roc_test.png / score_distribution_test.png using the per-object
(training-balance) weights. Those weights are a training device, not a physical
rate, so the unweighted (raw per-object) versions are often the more meaningful
view. This reuses the exact plotting functions with weights set to 1 and writes:
  roc_test_unw.png, score_distribution_test_unw.png
"""
import argparse
from pathlib import Path

import numpy as np

from regenerate_efficiency_plots import load_predictions
from train_resnet_image_classifier import save_roc_plot, save_score_distribution


def regenerate_run(run_dir):
    run_dir = Path(run_dir)
    csv_path = run_dir / "test_predictions.csv"
    if not csv_path.exists():
        return "skip (no test_predictions.csv)"
    score, label, weight, _pt = load_predictions(csv_path)
    ones = np.ones_like(weight)
    auc = save_roc_plot(label, score, ones, run_dir / "roc_test_unw.png")
    save_score_distribution(label, score, ones, run_dir / "score_distribution_test_unw.png")
    return f"ok (unweighted AUC={auc:.4f}, {len(label)} objs)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/eos/user/y/yeo/4l/runs/v6")
    ap.add_argument("--glob", default="*/")
    args = ap.parse_args()
    root = Path(args.runs_root)
    for d in sorted(x for x in root.glob(args.glob) if x.is_dir()):
        print(f"[{d.name}] {regenerate_run(d)}")


if __name__ == "__main__":
    main()
