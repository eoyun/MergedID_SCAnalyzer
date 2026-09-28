#!/usr/bin/env python3
"""One combined plot per run: signal efficiency (solid) + background fake-rate
(dashed) vs pT, overlaid for the 3 background fake-rate WPs (2.5/5/10%), from the
20 GeV unweighted eval JSONs. WP -> color; signal solid, background dashed.
Writes efficiency_fakerate_combined_allWP_unw_20GeV_sample.png into each run dir.
Run over given run dirs, or default = all runs/cross_v4/*/.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPS = [("2p5", "2.5%", "tab:blue"), ("05", "5%", "tab:orange"), ("10", "10%", "tab:green")]
SUFFIX = "_20GeV_sample"
MIN_DENOM = 20


def _series(rows, class_name):
    x, y, xerr, ylo, yhi = [], [], [], [], []
    for r in rows:
        if r["class_name"] != class_name or r.get("denom", 0) < MIN_DENOM:
            continue
        eff = r["efficiency"]
        if not np.isfinite(eff):
            continue
        hi = np.inf if r["pt_high"] == "inf" else float(r["pt_high"])
        lo = float(r["pt_low"])
        x.append(lo if np.isinf(hi) else 0.5 * (lo + hi))
        y.append(eff)
        xerr.append(0.0 if np.isinf(hi) else 0.5 * (hi - lo))
        err = r.get("efficiency_err", float("nan"))
        if np.isfinite(err):
            ylo.append(min(err, eff)); yhi.append(min(err, 1.0 - eff))
        else:
            ylo.append(0.0); yhi.append(0.0)
    return x, y, xerr, ylo, yhi


def process(run_dir):
    data = {}
    for tag, _, _ in WPS:
        p = os.path.join(run_dir, f"efficiency_vs_pt_test_bkg{tag}_unw{SUFFIX}.json")
        if not os.path.exists(p):
            return False
        j = json.load(open(p))
        data[tag] = j if isinstance(j, list) else j["rows"]

    name = os.path.basename(run_dir.rstrip("/"))
    plt.figure(figsize=(8, 6))
    for tag, label, color in WPS:
        xs, ys, xe, lo, hi = _series(data[tag], "signal")
        if xs:
            plt.errorbar(xs, ys, yerr=[lo, hi], xerr=xe, marker="o", ms=4, lw=1.6,
                         ls="-", color=color, capsize=2, label=f"signal eff  {label}")
        xb, yb, xeb, lob, hib = _series(data[tag], "background")
        if xb:
            plt.errorbar(xb, yb, yerr=[lob, hib], xerr=xeb, marker="s", ms=3, lw=1.3,
                         ls="--", color=color, capsize=2, label=f"bkg fake  {label}")
    ax = plt.gca()
    plt.xscale("log"); plt.ylim(0.0, 1.05)
    plt.xlabel("pT [GeV]"); plt.ylabel("efficiency / fake-rate")
    # CMS Private label (top-left, above the axes)
    ax.text(0.0, 1.01, r"$\bf{CMS}$ $\it{Private\ Work}$", transform=ax.transAxes,
            fontsize=13, va="bottom", ha="left")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2, loc="center left", bbox_to_anchor=(0.01, 0.36))
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"efficiency_fakerate_combined_allWP_unw{SUFFIX}.png"))
    plt.close()
    return True


def main():
    run_dirs = sys.argv[1:] or sorted(glob.glob("/eos/user/y/yeo/4l/runs/cross_v4/*/"))
    ok = skip = 0
    for d in run_dirs:
        if process(d):
            ok += 1
            print(f"[ok] {os.path.relpath(d, '/eos/user/y/yeo/4l/runs')}")
        else:
            skip += 1
            print(f"[skip: missing WP json] {d}")
    print(f"done: {ok} combined plots, {skip} skipped")


if __name__ == "__main__":
    main()
