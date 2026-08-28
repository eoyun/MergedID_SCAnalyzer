#!/usr/bin/env python3
"""Regenerate the LINEAR-scale predicted-vs-true A-mass plot from an existing
test_predictions.csv (columns must include m_true, m_pred, is_sig). Writes
mass_pred_vs_true_linear_test.png next to it. Pure post-processing (no model run).
Run over given run dirs, or default = all runs/mreg_v*/ that have the CSV."""
import csv
import glob
import os
import sys

import numpy as np
import multitask_common as mc


def process(run_dir):
    p = os.path.join(run_dir, "test_predictions.csv")
    if not os.path.exists(p):
        return False
    mt, mp = [], []
    with open(p) as fh:
        for row in csv.DictReader(fh):
            if "m_true" not in row or "m_pred" not in row:
                return False
            if "is_sig" in row and row["is_sig"] not in ("1", "1.0", "True"):
                continue
            try:
                mt.append(float(row["m_true"])); mp.append(float(row["m_pred"]))
            except ValueError:
                continue
    if not mt:
        return False
    mc.save_mass_pred_vs_true_linear(run_dir, np.array(mt), np.array(mp), suffix="_test")
    return True


def main():
    run_dirs = sys.argv[1:] or sorted(glob.glob("/eos/user/y/yeo/4l/runs/mreg_v*/*/"))
    ok = skip = 0
    for d in run_dirs:
        if process(d):
            ok += 1
        else:
            skip += 1
    print(f"done: {ok} linear plots written, {skip} skipped (no CSV / no m_true,m_pred)")


if __name__ == "__main__":
    main()
