"""Collect fusion test metrics across all categories into one table.

Reads each runs/fusion_<name>/test_metrics.json (keys: test_auc, test_f1,
component_metrics{image_test_auc, track_test_auc, fusion_test_auc}).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_RUNS_ROOT = "/eos/user/y/yeo/4l/runs"
FIELDS = ["name", "image_test_auc", "track_test_auc", "fusion_test_auc", "test_f1"]


def collect(runs_root):
    runs_root = Path(runs_root)
    rows = []
    for d in sorted(runs_root.glob("fusion_*")):
        metrics_path = d / "test_metrics.json"
        if not metrics_path.is_file():
            continue
        m = json.loads(metrics_path.read_text())
        comp = m.get("component_metrics", {})
        rows.append({
            "name": d.name[len("fusion_"):],
            "image_test_auc": comp.get("image_test_auc"),
            "track_test_auc": comp.get("track_test_auc"),
            "fusion_test_auc": m.get("test_auc", comp.get("fusion_test_auc")),
            "test_f1": m.get("test_f1"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    args = ap.parse_args()
    rows = collect(args.runs_root)
    if not rows:
        print("No fusion metrics found yet.", file=sys.stderr)
        return
    w = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
