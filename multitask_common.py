#!/usr/bin/env python3
"""Shared building blocks for the multi-task (classification + A-mass regression)
models. Imported by train_multitask_{image,track,cross}.py. The existing single-task
trainers/backbones are NOT modified; each backbone's final head is swapped to
nn.Identity() so it returns a pooled feature vector, and MultiTaskModel adds a
classification head (BCE) and a regression head (MAE on log m_A, masked to signal)."""
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASS_POINTS = (0.4, 1.0, 2.0, 5.0, 10.0)


# --------------------------------------------------------------------------- #
# Mass target: parse from sample id, log transform
# --------------------------------------------------------------------------- #
def sample_id_to_mass(sample_id):
    """A mass (GeV) from a signal sample id like 'signal_H2000_A0p4'; None for background."""
    if sample_id is None:
        return None
    s = sample_id.decode() if isinstance(sample_id, (bytes, bytearray)) else str(sample_id)
    if "background" in s:
        return None
    m = re.search(r"_A(\d+p\d+|\d+)(?:_|$)", s)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def transform_mass(m):        # GeV -> network units (natural log)
    return np.log(m)


def inv_transform(y):         # network units -> GeV
    return np.exp(y)


def mass_targets_from_codes(sample_codes, code_to_sample):
    """Return (log_mass[float32], signal_mask[float32]) for a batch of sample_code ints.
    Background (unparseable) -> log_mass 0.0, mask 0.0."""
    n = len(sample_codes)
    logm = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(sample_codes):
        m = sample_id_to_mass(code_to_sample[int(c)])
        if m is not None:
            logm[i] = np.log(m)
            mask[i] = 1.0
    return logm, mask


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def multitask_loss(cls_logit, reg_mu, label, mass_target, sig_mask, weight=None, lam=1.0):
    """BCE(classification) + lam * MAE(log-mass) restricted to signal objects.
    Returns (total, bce_detached, mae_detached)."""
    label = label.float()
    if weight is None:
        weight = torch.ones_like(label)
    bce = F.binary_cross_entropy_with_logits(cls_logit, label, weight=weight, reduction="mean")
    denom = sig_mask.sum().clamp_min(1.0)
    mae = ((reg_mu - mass_target).abs() * sig_mask).sum() / denom
    return bce + lam * mae, bce.detach(), mae.detach()


# --------------------------------------------------------------------------- #
# Metrics (masses in GeV)
# --------------------------------------------------------------------------- #
def regression_metrics(m_true, m_pred):
    m_true = np.asarray(m_true, dtype=np.float64)
    m_pred = np.asarray(m_pred, dtype=np.float64)
    ae = np.abs(m_pred - m_true)
    return {"mae": float(ae.mean()), "mre": float((ae / m_true).mean())}


def regression_metrics_per_point(m_true, m_pred):
    """Per true-mass-point resolution (std of relative residual) and bias (mean)."""
    m_true = np.asarray(m_true, dtype=np.float64)
    m_pred = np.asarray(m_pred, dtype=np.float64)
    out = {}
    for pt in np.unique(np.round(m_true, 6)):
        sel = np.round(m_true, 6) == pt
        rel = (m_pred[sel] - m_true[sel]) / m_true[sel]
        out[float(pt)] = {"n": int(sel.sum()),
                          "bias": float(rel.mean()),
                          "resolution": float(rel.std()),
                          "mae": float(np.abs(m_pred[sel] - m_true[sel]).mean())}
    return out


# --------------------------------------------------------------------------- #
# Model: backbone (head swapped to Identity) + cls head + reg head
# --------------------------------------------------------------------------- #
def make_heads(feat_dim, dropout=0.1):
    cls_head = nn.Linear(feat_dim, 1)
    reg_head = nn.Sequential(
        nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(feat_dim, 1))
    return cls_head, reg_head


class MultiTaskModel(nn.Module):
    """Wrap a backbone whose final head is nn.Identity (so it returns a pooled
    [B, feat_dim] feature). Adds a classification head and a regression head.
    forward(*inputs) -> (cls_logit[B], reg_mu[B]). *inputs go straight to the
    backbone (image: (x,); track: (points, mask); cross: (image, points, mask))."""
    def __init__(self, backbone, feat_dim, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.cls_head, self.reg_head = make_heads(feat_dim, dropout)

    def forward(self, *inputs):
        feat = self.backbone(*inputs)
        return self.cls_head(feat).squeeze(1), self.reg_head(feat).squeeze(1)


# --------------------------------------------------------------------------- #
# Plots (masses in GeV)
# --------------------------------------------------------------------------- #
def save_mass_regression_plots(output_dir, m_true, m_pred, suffix=""):
    """pred-vs-true, resolution-vs-mass, and per-point predicted-mass histograms."""
    m_true = np.asarray(m_true, dtype=np.float64)
    m_pred = np.asarray(m_pred, dtype=np.float64)
    fin = np.isfinite(m_pred) & np.isfinite(m_true) & (m_true > 0)
    m_true, m_pred = m_true[fin], m_pred[fin]

    plt.figure(figsize=(6, 6))
    plt.hist2d(m_true, np.clip(m_pred, 1e-3, None), bins=40, cmin=1)
    lims = [0.3, 12]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xscale("log"); plt.yscale("log"); plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("true m_A [GeV]"); plt.ylabel("predicted m_A [GeV]")
    plt.title("Predicted vs true A mass"); plt.colorbar(label="objects")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"mass_pred_vs_true{suffix}.png")); plt.close()

    per = regression_metrics_per_point(m_true, m_pred)
    pts = sorted(per.keys())
    res = [per[p]["resolution"] for p in pts]
    bias = [per[p]["bias"] for p in pts]
    plt.figure(figsize=(7, 5))
    plt.plot(pts, res, "o-", label="resolution  std[(pred-true)/true]")
    plt.plot(pts, bias, "s--", label="bias  mean[(pred-true)/true]")
    plt.axhline(0, color="gray", lw=0.8)
    plt.xscale("log"); plt.xlabel("true m_A [GeV]"); plt.ylabel("relative residual")
    plt.title("Mass resolution / bias vs true mass"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"mass_resolution_vs_mass{suffix}.png")); plt.close()

    plt.figure(figsize=(8, 6))
    for p in pts:
        sel = np.round(m_true, 6) == p
        plt.hist(np.clip(m_pred[sel], 0, 12), bins=60, range=(0, 12),
                 histtype="step", label=f"true {p:g} GeV")
        plt.axvline(p, color="gray", ls=":", lw=0.8)
    plt.xlabel("predicted m_A [GeV]"); plt.ylabel("objects")
    plt.title("Predicted mass per true mass point"); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"mass_pred_hist_per_point{suffix}.png")); plt.close()
