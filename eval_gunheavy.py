#!/usr/bin/env python3
"""Evaluate a trained gunheavy_v1 cross-attention model on the 3-class 20 GeV eval compact
(build_category_weights_gun_eval.py -> build_compact_dataset_gun.py). Scores every object,
splits by sample_id (gun / haa / background), and writes 6 plots:

  1. eff_fake_9curve_20GeV.png   : 3 WP (bkg fake 2.5/5/10%, GLOBAL threshold) x
                                   {bkg fake rate, gun efficiency, signal(H->AA) efficiency}
                                   vs reco SC ET  -> 9 curves in one plot.
  2. mass_pred_vs_true_linear_gun.png     : true vs pred m_A, linear, range [0,60], gun
  3. mass_pred_vs_true_linear_signal.png  : true vs pred m_A, linear, range [0,60], H->AA
  4. score_distribution_3class.png : gun / signal / background score overlay (normalized)
  5. roc_log.png     : ROC (FPR log axis)    gun-vs-bkg + signal-vs-bkg
  6. roc_linear.png  : ROC (FPR linear axis) gun-vs-bkg + signal-vs-bkg

Model is NOT retrained. WP threshold is a single global value (fraction of ALL background
passing = WP), matching the 18-way _20GeV_sample convention.
"""
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc as sk_auc

import multitask_common as mc
from pipeline.compact_manifest import load_compact_manifest
from train_fusion_cross_attention import CrossAttentionFusion, CombinedDataset, collate_combined
from train_resnet_image_classifier import (
    DetectorObjectDataset, image_in_channels, resolve_num_workers)
from track_point_transformer import TrackPointCloudDataset, point_feature_dim, parse_track_types
from build_event_level_weights import PT_BIN_EDGES_WITH_OVERFLOW

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WPS = [(2.5, "tab:blue"), (5.0, "tab:orange"), (10.0, "tab:green")]
ET_EDGES = np.asarray(PT_BIN_EDGES_WITH_OVERFLOW, dtype=np.float64)
MIN_DENOM = 20
MASS_LIN = [0.0, 60.0]


def build_model(detector, tt, include_es, a):
    in_ch = image_in_channels(detector, tt, include_es, include_track=False)
    bb = CrossAttentionFusion(in_channels=in_ch, point_feat_dim=point_feature_dim(tt),
                              backbone=a.backbone, d_model=a.d_model, num_heads=a.num_heads,
                              depth=a.depth, track_depth=a.track_depth, dropout=a.dropout)
    bb.head = nn.Identity()
    return mc.MultiTaskModel(bb, 2 * a.d_model)


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    S, R, O = [], [], []
    for b in loader:
        cls, reg = model(b["image"].to(device), b["points"].to(device), b["mask"].to(device))
        S.append(torch.sigmoid(cls).cpu().numpy())
        R.append(reg.cpu().numpy())
        O.append(b["object_idx"].numpy())
    return np.concatenate(S), np.concatenate(R), np.concatenate(O)


def _eff_vs_et(pt, sel_mask, passed):
    """efficiency (mean of `passed`) of objects in sel_mask, per ET bin."""
    x, y = [], []
    for i in range(len(ET_EDGES) - 1):
        lo, hi = ET_EDGES[i], ET_EDGES[i + 1]
        inb = sel_mask & (pt >= lo) & (pt < hi)
        d = int(inb.sum())
        if d < MIN_DENOM:
            continue
        x.append(lo if np.isinf(hi) else 0.5 * (lo + hi))
        y.append(float(passed[inb].mean()))
    return np.asarray(x), np.asarray(y)


def plot_9curve(out, pt, score, gun, haa, bkg, tag):
    plt.figure(figsize=(9, 6))
    for wp, color in WPS:
        thr = np.quantile(score[bkg], 1.0 - wp / 100.0)   # global WP threshold from bkg
        passed = score >= thr
        for mask, ls, mk, lbl in [(bkg, ":", "s", "bkg fake"),
                                   (gun, "-", "o", "gun eff"),
                                   (haa, "--", "^", "signal eff")]:
            x, y = _eff_vs_et(pt, mask, passed)
            if x.size:
                plt.plot(x, y, ls=ls, marker=mk, ms=3, lw=1.4, color=color,
                         label=f"{lbl} {wp:g}%")
    ax = plt.gca()
    plt.xscale("log"); plt.ylim(0.0, 1.05)
    plt.xlabel("reco SC $E_T$ [GeV]"); plt.ylabel("efficiency / fake-rate")
    ax.text(0.0, 1.01, r"$\bf{CMS}$ $\it{Private\ Work}$", transform=ax.transAxes,
            fontsize=13, va="bottom", ha="left")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=3, loc="center left", bbox_to_anchor=(0.01, 0.5))
    plt.tight_layout(); plt.savefig(os.path.join(out, "eff_fake_9curve_20GeV.png")); plt.close()


def plot_mass_linear(out, m_true, m_pred, tag, fname):
    fin = np.isfinite(m_true) & np.isfinite(m_pred) & (m_true > 0)
    m_true, m_pred = m_true[fin], m_pred[fin]
    plt.figure(figsize=(6, 6))
    plt.hist2d(np.clip(m_true, *MASS_LIN), np.clip(m_pred, *MASS_LIN),
               bins=60, range=[MASS_LIN, MASS_LIN], cmin=1)
    plt.plot(MASS_LIN, MASS_LIN, "r--", lw=1); plt.xlim(MASS_LIN); plt.ylim(MASS_LIN)
    plt.xlabel("true $m_A$ [GeV]"); plt.ylabel("predicted $m_A$ [GeV]")
    plt.colorbar(label="objects"); plt.title(f"{tag} — mass pred vs true (linear)")
    mae = float(np.abs(m_pred - m_true).mean()) if len(m_true) else float("nan")
    plt.text(0.05, 0.92, f"n={len(m_true)}\nMAE={mae:.2f} GeV", transform=plt.gca().transAxes,
             va="top", fontsize=9, bbox=dict(fc="white", alpha=0.7))
    plt.tight_layout(); plt.savefig(os.path.join(out, fname)); plt.close()
    return mae


def plot_score_dist(out, score, gun, haa, bkg):
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    for mask, lbl, col in [(bkg, "background (W/Z)", "tab:red"),
                           (haa, "signal (H→AA)", "tab:green"),
                           (gun, "particle gun (heavy)", "tab:blue")]:
        if mask.sum():
            plt.hist(score[mask], bins=bins, histtype="step", lw=1.8, density=True,
                     color=col, label=f"{lbl} (n={int(mask.sum())})")
    plt.yscale("log")
    plt.xlabel("classification score"); plt.ylabel("a.u.")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out, "score_distribution_3class.png")); plt.close()


def plot_roc(out, score, gun, haa, bkg):
    curves = {}
    for name, sig in [("particle gun (heavy)", gun), ("signal (H→AA)", haa)]:
        y = np.concatenate([np.ones(int(sig.sum())), np.zeros(int(bkg.sum()))])
        s = np.concatenate([score[sig], score[bkg]])
        fpr, tpr, _ = roc_curve(y, s)
        curves[name] = (fpr, tpr, sk_auc(fpr, tpr))
    for scale, fname in [("log", "roc_log.png"), ("linear", "roc_linear.png")]:
        plt.figure(figsize=(6.5, 6))
        for name, (fpr, tpr, au) in curves.items():
            plt.plot(fpr, tpr, lw=1.8, label=f"{name}  AUC={au:.4f}")
        if scale == "log":
            plt.xscale("log"); plt.xlim(1e-4, 1.0)
        else:
            plt.xlim(0, 1)
        plt.ylim(0, 1.02)
        plt.xlabel("background fake rate (FPR)"); plt.ylabel("signal efficiency (TPR)")
        plt.grid(True, alpha=0.3, which="both"); plt.legend(fontsize=10, loc="lower right")
        plt.title(f"ROC ({scale} FPR)")
        plt.tight_layout(); plt.savefig(os.path.join(out, fname)); plt.close()
    return {k: float(v[2]) for k, v in curves.items()}


def plot_history(out, hist, tag):
    """Training history from the run's history.json: loss (train+val), val AUC,
    val mass MAE vs epoch."""
    if not hist:
        return
    ep = [h.get("epoch") for h in hist]
    tl = [h.get("train_loss") for h in hist]
    vl = [h.get("val_loss") for h in hist]
    au = [h.get("val_auc") for h in hist]
    mm = [h.get("val_mass_mae") for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(ep, tl, "o-", ms=3, label="train")
    ax[0].plot(ep, vl, "s-", ms=3, label="val")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, au, "o-", ms=3, color="tab:green")
    ax[1].set_title("val AUC"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3)
    ax[2].plot(ep, mm, "o-", ms=3, color="tab:red")
    ax[2].set_title("val mass MAE [GeV]"); ax[2].set_xlabel("epoch"); ax[2].grid(alpha=0.3)
    fig.suptitle(f"{tag} — training history")
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_history.png")); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--compact", required=True, help="3-class eval compact")
    p.add_argument("--run-dir", required=True, help="dir with best_model.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-es", action="store_true")
    p.add_argument("--track-types", default="Lost,PF,GSF")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--track-depth", type=int, default=2)
    p.add_argument("--max-points", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tt = parse_track_types(a.track_types)
    include_es = not a.no_es
    os.makedirs(a.output_dir, exist_ok=True)

    fe, sm, c2s = load_compact_manifest(a.compact)
    man = {k: np.concatenate([sm[s][k] for s in ("train", "val", "test")]) for k in sm["train"]}
    img = DetectorObjectDataset(a.detector, fe, man, track_types=tt,
                                include_es=include_es, include_track=False)
    trk = TrackPointCloudDataset(detector=a.detector, file_entries=fe, manifest=man,
                                 max_points=a.max_points, track_types=tt)
    ds = CombinedDataset(img, trk)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=resolve_num_workers(a.num_workers),
                        pin_memory=(device.type == "cuda"), collate_fn=collate_combined)

    model = build_model(a.detector, tt, include_es, a).to(device)
    ck = torch.load(os.path.join(a.run_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    score, reg, oidx = infer(model, loader, device)
    m_pred = mc.inv_transform(reg)

    # per-object truth from the compact, indexed by compact row (object_idx)
    with h5py.File(a.compact, "r") as f:
        pt_all = f["pt"][:]; am_all = f["a_mass"][:]
        sid_all = np.array([s.decode() if isinstance(s, bytes) else s for s in f["sample_id"][:]])
    pt = pt_all[oidx]; a_mass = am_all[oidx]; sid = sid_all[oidx]
    gun = sid == "gun"; haa = sid == "haa"; bkg = sid == "background"
    tag = os.path.basename(a.run_dir.rstrip("/"))
    print(f"[eval] {tag}: gun={gun.sum()} haa={haa.sum()} bkg={bkg.sum()}", flush=True)

    plot_9curve(a.output_dir, pt, score, gun, haa, bkg, tag)
    mae_gun = plot_mass_linear(a.output_dir, a_mass[gun], m_pred[gun], f"{tag} gun",
                               "mass_pred_vs_true_linear_gun.png")
    mae_haa = plot_mass_linear(a.output_dir, a_mass[haa], m_pred[haa], f"{tag} signal",
                               "mass_pred_vs_true_linear_signal.png")
    plot_score_dist(a.output_dir, score, gun, haa, bkg)
    aucs = plot_roc(a.output_dir, score, gun, haa, bkg)
    import json
    hist_path = os.path.join(a.run_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            plot_history(a.output_dir, json.load(fh), tag)
    else:
        print(f"[warn] no history.json in {a.run_dir}; skipping training-history plot", flush=True)
    with open(os.path.join(a.output_dir, "eval_summary.json"), "w") as fh:
        json.dump({"tag": tag, "n_gun": int(gun.sum()), "n_haa": int(haa.sum()),
                   "n_bkg": int(bkg.sum()), "mae_gun": mae_gun, "mae_haa": mae_haa,
                   "auc_gun_vs_bkg": aucs.get("particle gun (heavy)"),
                   "auc_haa_vs_bkg": aucs.get("signal (H→AA)")}, fh, indent=2)
    print(f"[done] {tag} -> {a.output_dir}  auc={aucs}", flush=True)


if __name__ == "__main__":
    main()
