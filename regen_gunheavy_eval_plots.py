#!/usr/bin/env python3
"""Redraw gunheavy eval plots from each run's saved eval_predictions.npz WITHOUT
re-scoring on GPU. Reuses eval_gunheavy's plot functions, so any plotting change
(error bars, ranges, 1D-vs-2D, ...) is applied cheaply to finished eval dirs.

By default redraws ALL npz-derived plots; pass plot names to limit. Also removes the
stale signal 2D linear plots (superseded by the 1D per-mass distribution).

Usage: regen_gunheavy_eval_plots.py [eval_dir ...]   (default: runs/gunheavy_v1_eval/*/)
"""
import glob
import os
import sys

import numpy as np

import eval_gunheavy as E

RUNS = "/eos/user/y/yeo/4l/runs/gunheavy_v1_eval"


def redo(d):
    npz = os.path.join(d, "eval_predictions.npz")
    if not os.path.exists(npz):
        return False
    z = np.load(npz, allow_pickle=True)
    score, pt, a_mass, m_pred = z["score"], z["pt"], z["a_mass"], z["m_pred"]
    sid = np.array([s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in z["sample_id"]])
    gun, haa, bkg = sid == "gun", sid == "haa", sid == "background"
    tag = os.path.basename(d.rstrip("/"))

    E.plot_9curve(d, pt, score, gun, haa, bkg, tag)
    E.plot_mass_linear(d, a_mass[gun], m_pred[gun], f"{tag} gun", "mass_pred_vs_true_linear_gun")
    E.plot_signal_pred_1d(d, a_mass[haa], m_pred[haa], f"{tag} signal")
    E.plot_score_dist(d, score, gun, haa, bkg)
    E.plot_roc(d, score, gun, haa, bkg)
    # remove stale 2D signal linear plots (replaced by mass_pred_dist_signal.png)
    for suf in ("", "_0to10", "_0to5"):
        p = os.path.join(d, f"mass_pred_vs_true_linear_signal{suf}.png")
        if os.path.exists(p):
            os.remove(p)
    print(f"[ok] {tag}  gun={int(gun.sum())} haa={int(haa.sum())} bkg={int(bkg.sum())}", flush=True)
    return True


def main():
    dirs = sys.argv[1:] or sorted(glob.glob(RUNS + "/*/"))
    ok = skip = 0
    for d in dirs:
        if redo(d):
            ok += 1
        else:
            skip += 1
            print(f"[skip: no npz] {os.path.basename(d.rstrip('/'))}")
    print(f"done: {ok} redrawn, {skip} skipped")


if __name__ == "__main__":
    main()
