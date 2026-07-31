#!/usr/bin/env python3
"""Combine the three per-working-point efficiency plots of the pT>=20 eval sample
into TWO overlay plots per run:
  * efficiency_vs_pt_signal_allWP_unw_20GeV_sample.png     - signal efficiency,
    one curve per background fake-rate WP (2.5 / 5 / 10 %).
  * efficiency_vs_pt_background_allWP_unw_20GeV_sample.png - background fake-rate,
    one curve per WP.
Pure post-processing: reads the existing efficiency_vs_pt_test_bkg{2p5,05,10}_unw_
20GeV_sample.json files (no model re-run). Run over specific run dirs, or with no
args over every runs/*/*/ that has all three JSONs.
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


def _series(rows, class_name, min_denom=MIN_DENOM):
    """Extract (x, y, xerr, yerr_lo, yerr_hi) for one class, matching the style of
    save_efficiency_plot (bin centers, half-width x-errors, clipped y-errors,
    low-statistics bins dropped)."""
    x, y, xerr, ylo, yhi = [], [], [], [], []
    for r in rows:
        if r["class_name"] != class_name:
            continue
        if min_denom and r.get("denom", 0) < min_denom:
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
        if not np.isfinite(err):
            ylo.append(0.0); yhi.append(0.0)
        else:
            ylo.append(min(err, eff)); yhi.append(min(err, 1.0 - eff))
    return x, y, xerr, ylo, yhi


def _load(run_dir):
    data = {}
    for tag, _, _ in WPS:
        p = os.path.join(run_dir, f"efficiency_vs_pt_test_bkg{tag}_unw{SUFFIX}.json")
        if not os.path.exists(p):
            return None
        j = json.load(open(p))
        data[tag] = j if isinstance(j, list) else j["rows"]
    return data


def _plot_class(run_dir, data, class_name, out_name, ylabel, title):
    plt.figure(figsize=(8, 6))
    for tag, label, color in WPS:
        x, y, xerr, ylo, yhi = _series(data[tag], class_name)
        if not x:
            continue
        plt.errorbar(x, y, yerr=[ylo, yhi], xerr=xerr, marker="o", linestyle="-",
                     capsize=3, elinewidth=1, color=color,
                     label=f"bkg fake-rate {label}")
    plt.xlabel("pT [GeV]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, out_name))
    plt.close()


def process(run_dir):
    data = _load(run_dir)
    if data is None:
        return False
    name = os.path.basename(run_dir.rstrip("/"))
    _plot_class(run_dir, data, "signal",
                f"efficiency_vs_pt_signal_allWP_unw{SUFFIX}.png",
                "Signal efficiency",
                f"Signal efficiency vs pT (20 GeV sample) - {name}")
    _plot_class(run_dir, data, "background",
                f"efficiency_vs_pt_background_allWP_unw{SUFFIX}.png",
                "Background fake rate",
                f"Background fake-rate vs pT (20 GeV sample) - {name}")
    return True


def main():
    if len(sys.argv) > 1:
        run_dirs = sys.argv[1:]
    else:
        run_dirs = sorted({os.path.dirname(p) for p in glob.glob(
            "/eos/user/y/yeo/4l/runs/*/*/efficiency_vs_pt_test_bkg05_unw" + SUFFIX + ".json")})
    ok = skip = 0
    for d in run_dirs:
        if process(d):
            ok += 1
            print(f"[ok] {os.path.relpath(d, '/eos/user/y/yeo/4l/runs')}")
        else:
            skip += 1
            print(f"[skip: missing WP json] {d}")
    print(f"done: {ok} run(s) plotted, {skip} skipped")


if __name__ == "__main__":
    main()
