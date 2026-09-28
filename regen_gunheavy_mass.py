#!/usr/bin/env python3
"""Redraw the gun-heavy mass-regression plots with a mass range matched to the gun
(0-60 GeV), from each run's test_predictions.csv. The shared multitask_common plot
helpers hardcode the OLD discrete H->AA range (max ~10-12 GeV) and CLIP m_true at 11,
which piles every gun object with true mass > 11 GeV (~2/3 of them) onto the axis edge
in the linear plot. This regen leaves those shared helpers (and the discrete-mreg plots
they serve) untouched and just redraws the gun runs with range [0, 60] (linear) and
[0.1, 60] (log). Signal-only (is_sig==1). Overwrites the standard filenames in-place.

Usage: regen_gunheavy_mass.py [run_dir ...]   (default: all runs/gunheavy_v1/*/)
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/eos/user/y/yeo/4l/runs/gunheavy_v1"
LIN = [0.0, 60.0]
LOG = [0.1, 60.0]


def _load_signal(run_dir):
    p = os.path.join(run_dir, "test_predictions.csv")
    if not os.path.exists(p):
        return None
    mt, mp, sig = [], [], []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            mt.append(float(r["m_true"])); mp.append(float(r["m_pred"]))
            sig.append(int(r["is_sig"]))
    mt = np.asarray(mt); mp = np.asarray(mp); sig = np.asarray(sig).astype(bool)
    keep = sig & np.isfinite(mt) & np.isfinite(mp) & (mt > 0)
    return mt[keep], mp[keep]


def redraw(run_dir):
    got = _load_signal(run_dir)
    if got is None:
        return False
    m_true, m_pred = got
    tag = os.path.basename(run_dir.rstrip("/"))

    # ---- linear pred-vs-true (0-60) ----
    plt.figure(figsize=(6, 6))
    plt.hist2d(np.clip(m_true, *LIN), np.clip(m_pred, *LIN), bins=60, range=[LIN, LIN], cmin=1)
    plt.plot(LIN, LIN, "r--", lw=1); plt.xlim(LIN); plt.ylim(LIN)
    plt.xlabel("true $m_A$ [GeV]"); plt.ylabel("predicted $m_A$ [GeV]")
    plt.colorbar(label="objects"); plt.title(f"{tag} — mass pred vs true (linear)")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "mass_pred_vs_true_linear_test.png")); plt.close()

    # ---- log pred-vs-true (0.1-60) ----
    edges = np.logspace(np.log10(LOG[0]), np.log10(LOG[1]), 60)
    plt.figure(figsize=(6, 6))
    plt.hist2d(np.clip(m_true, *LOG), np.clip(m_pred, *LOG), bins=[edges, edges], cmin=1)
    plt.plot(LOG, LOG, "r--", lw=1)
    plt.xscale("log"); plt.yscale("log"); plt.xlim(LOG); plt.ylim(LOG)
    plt.xlabel("true $m_A$ [GeV]"); plt.ylabel("predicted $m_A$ [GeV]")
    plt.colorbar(label="objects"); plt.title(f"{tag} — mass pred vs true (log)")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "mass_pred_vs_true_test.png")); plt.close()

    # ---- resolution & bias vs true mass (log-spaced bins over the gun range) ----
    be = np.geomspace(max(m_true.min(), 0.1), min(m_true.max(), 60.0), 15)
    cen, res, bias = [], [], []
    for i in range(len(be) - 1):
        sel = (m_true >= be[i]) & (m_true < be[i + 1])
        if sel.sum() < 20:
            continue
        rel = (m_pred[sel] - m_true[sel]) / m_true[sel]
        cen.append(np.sqrt(be[i] * be[i + 1])); res.append(rel.std()); bias.append(rel.mean())
    plt.figure(figsize=(7, 5))
    plt.plot(cen, res, "o-", label="resolution  std[(pred-true)/true]")
    plt.plot(cen, bias, "s--", label="bias  mean[(pred-true)/true]")
    plt.axhline(0, color="gray", lw=0.8); plt.xscale("log")
    plt.xlabel("true $m_A$ [GeV]"); plt.ylabel("relative residual")
    plt.title(f"{tag} — mass resolution/bias vs mass"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "mass_resolution_vs_mass_test.png")); plt.close()

    mae = float(np.abs(m_pred - m_true).mean())
    print(f"[ok] {tag}: n={len(m_true)} true[{m_true.min():.2f},{m_true.max():.2f}] "
          f"pred[{m_pred.min():.2f},{m_pred.max():.2f}] MAE={mae:.3f}", flush=True)
    return True


def main():
    run_dirs = sys.argv[1:] or sorted(glob.glob(RUNS + "/*/"))
    ok = skip = 0
    for d in run_dirs:
        if redraw(d):
            ok += 1
        else:
            skip += 1
            print(f"[skip: no test_predictions.csv] {os.path.basename(d.rstrip('/'))}")
    print(f"done: {ok} redrawn, {skip} skipped")


if __name__ == "__main__":
    main()
