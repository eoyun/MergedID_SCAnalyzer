#!/usr/bin/env python3
"""Plot the efficiency NUMERATOR (objects passing the WP threshold) and DENOMINATOR
(all objects in the bin) per pT bin, for the pT>=20 eval sample. Two plots per run:
  num_denom_signal_unw_20GeV_sample.png
  num_denom_background_unw_20GeV_sample.png
Each shows the denominator (WP-independent) plus the numerator at each background
fake-rate WP (2.5/5/10%). Pure post-processing of the existing
efficiency_vs_pt_test_bkg{2p5,05,10}_unw_20GeV_sample.json files.
Run over given run dirs, or default = all v8/cross_v2/v10/cross_v4 runs.
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


def _load(run_dir):
    data = {}
    for tag, _, _ in WPS:
        p = os.path.join(run_dir, f"efficiency_vs_pt_test_bkg{tag}_unw{SUFFIX}.json")
        if not os.path.exists(p):
            return None
        j = json.load(open(p))
        data[tag] = j if isinstance(j, list) else j["rows"]
    return data


def _xy(rows, class_name, field):
    x, y = [], []
    for r in rows:
        if r["class_name"] != class_name:
            continue
        v = r.get(field, 0.0)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            v = 0.0
        hi = np.inf if r["pt_high"] == "inf" else float(r["pt_high"])
        lo = float(r["pt_low"])
        x.append(lo if np.isinf(hi) else 0.5 * (lo + hi))
        y.append(float(v))
    return np.array(x), np.array(y)


def _plot(run_dir, data, class_name, out_name, title):
    plt.figure(figsize=(8, 6))
    # denominator (same across WPs) -> take from the first WP json
    xd, yd = _xy(data[WPS[0][0]], class_name, "denom")
    m = yd > 0
    plt.step(xd[m], yd[m], where="mid", color="black", lw=2.0,
             label="denominator (all)")
    # numerator per WP
    for tag, label, color in WPS:
        xn, yn = _xy(data[tag], class_name, "numer")
        mm = yd > 0
        plt.plot(xn[mm], yn[mm], marker="o", ms=4, lw=1.4, color=color,
                 label=f"numerator @ {label}")
    plt.xlabel("pT [GeV]", fontsize=12)
    plt.ylabel("N objects", fontsize=12)
    plt.title(title)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, out_name))
    plt.close()


def process(run_dir):
    data = _load(run_dir)
    if data is None:
        return False
    name = os.path.basename(run_dir.rstrip("/"))
    _plot(run_dir, data, "signal",
          f"num_denom_signal_unw{SUFFIX}.png",
          f"Signal numerator/denominator vs pT (20 GeV sample) - {name}")
    _plot(run_dir, data, "background",
          f"num_denom_background_unw{SUFFIX}.png",
          f"Background numerator/denominator vs pT (20 GeV sample) - {name}")
    return True


def main():
    if len(sys.argv) > 1:
        run_dirs = sys.argv[1:]
    else:
        run_dirs = []
        for fam in ("v8", "cross_v2", "v10", "cross_v4"):
            run_dirs += sorted(glob.glob(f"/eos/user/y/yeo/4l/runs/{fam}/*/"))
    ok = skip = 0
    for d in run_dirs:
        if process(d):
            ok += 1
        else:
            skip += 1
            print(f"[skip: missing WP json] {d}")
    print(f"done: {ok} run(s) plotted, {skip} skipped")


if __name__ == "__main__":
    main()
