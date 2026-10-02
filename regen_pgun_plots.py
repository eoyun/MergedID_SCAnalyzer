#!/usr/bin/env python3
"""Redraw mreg_v1 particle-gun eval plots from each run's pgun_predictions.npz, in any
format (default .eps) WITHOUT re-scoring on GPU. Reuses eval_pgun.save_pgun_plots and
the shared score-distribution helper.

Usage: regen_pgun_plots.py [--eps|--pdf|--png] [run_dir ...]
       default fmt .eps, default dirs runs/mreg_v1_eval_pgun/*/
"""
import glob
import os
import sys

import numpy as np

import eval_pgun as E
from train_resnet_image_classifier import save_score_distribution

RUNS = "/eos/user/y/yeo/4l/runs/mreg_v1_eval_pgun"


def redo(d, ext):
    npz = os.path.join(d, "pgun_predictions.npz")
    if not os.path.exists(npz):
        return False
    z = np.load(npz)
    m_true, m_pred, score = z["m_true"], z["m_pred"], z["score"]
    tag = os.path.basename(d.rstrip("/"))
    E.PLOT_EXT = ext
    E.save_pgun_plots(d, m_true, m_pred, tag)
    save_score_distribution(np.ones_like(score, dtype=np.int64), score,
                            np.ones_like(score, dtype=np.float64),
                            os.path.join(d, "pgun_score_distribution" + ext),
                            normalize=True, ylabel="a.u.")
    print(f"[ok] {tag} n={len(m_true)}", flush=True)
    return True


def main():
    args = sys.argv[1:]
    ext = ".eps"
    for fl, e in (("--eps", ".eps"), ("--pdf", ".pdf"), ("--png", ".png")):
        if fl in args:
            args.remove(fl); ext = e
    dirs = args or sorted(glob.glob(RUNS + "/*/"))
    print(f"[fmt] {ext}", flush=True)
    ok = skip = 0
    for d in dirs:
        if redo(d, ext):
            ok += 1
        else:
            skip += 1
            print(f"[skip: no npz] {os.path.basename(d.rstrip('/'))}")
    print(f"done: {ok} redrawn, {skip} skipped")


if __name__ == "__main__":
    main()
