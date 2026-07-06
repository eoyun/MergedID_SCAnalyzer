"""Regenerate efficiency-vs-pT plots (now with error bars) from saved
test_predictions.csv, WITHOUT re-training.

Older runs wrote efficiency_vs_pt_test.{png,json} before error bars existed.
Everything needed to redraw them is already on disk:
  - score, label, weight, pt  <- test_predictions.csv
  - threshold                 <- test_metrics.json
  - pt bin edges              <- reconstructed from the existing efficiency json

We reuse the exact library functions so the recomputed efficiencies match the
originals bit-for-bit and only gain the efficiency_err field + error bars.
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


def reconstruct_edges(eff_json_path):
    rows = json.loads(Path(eff_json_path).read_text())
    sig = [r for r in rows if r["class_name"] == "signal"]
    sig.sort(key=lambda r: r["bin_index"])
    edges = [float(sig[0]["pt_low"])]
    for r in sig:
        hi = r["pt_high"]
        edges.append(np.inf if hi == "inf" else float(hi))
    return edges


def load_predictions(csv_path):
    score, label, weight, pt = [], [], [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            score.append(float(row["score"]))
            label.append(int(float(row["label"])))
            weight.append(float(row["weight"]))
            pt.append(float(row["pt"]))
    return (np.array(score), np.array(label),
            np.array(weight), np.array(pt))


def regenerate_run(run_dir):
    run_dir = Path(run_dir)
    csv_path = run_dir / "test_predictions.csv"
    eff_json = run_dir / "efficiency_vs_pt_test.json"
    metrics = run_dir / "test_metrics.json"
    for p in (csv_path, eff_json, metrics):
        if not p.exists():
            return f"skip ({p.name} missing)"

    threshold = float(json.loads(metrics.read_text())["threshold"])
    edges = reconstruct_edges(eff_json)
    y_score, y_true, weights, pt = load_predictions(csv_path)

    rows = compute_efficiency_by_pt(y_true, y_score, weights, pt, threshold, edges)
    save_efficiency_plot(rows, run_dir / "efficiency_vs_pt_test.png", threshold)
    with eff_json.open("w") as fh:
        json.dump(rows, fh, indent=2)
    return f"ok ({len(y_true)} objs, thr={threshold:.3f}, {len(edges)-1} bins)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/eos/user/y/yeo/4l/runs/v5")
    ap.add_argument("--glob", default="*/",
                    help="subdir glob under runs-root (default all run dirs)")
    args = ap.parse_args()
    root = Path(args.runs_root)
    dirs = sorted(d for d in root.glob(args.glob) if d.is_dir())
    for d in dirs:
        print(f"[{d.name}] {regenerate_run(d)}")


if __name__ == "__main__":
    main()
