# A-mass Regression (multi-task) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add A-boson mass regression to the merged-electron models as a multi-task head (classification + regression), in new files, reusing the existing backbones and the pt50 compacts.

**Architecture:** Each existing backbone (image ResNet / track transformer / cross-attention fusion) has its final 1-logit head replaced by `nn.Identity()` so it returns a pooled feature vector; a small wrapper adds a classification head (BCE) and a regression head (MAE on `log(m_A)`, masked to signal). Mass targets are derived at run time from `sample_code → sample_id → m_A` (no compact rebuild). No explicit pT input (avoids gen-pT truth leak).

**Tech Stack:** Python 3.13 (cvmfs LCG_109_cuda view), PyTorch (fp32 + TF32), h5py, numpy, matplotlib, pytest, HTCondor GPU.

**Environment for every command:**
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh   # run `set +u` first if in a strict shell
export HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR=/tmp/yeo/mpl TORCHDYNAMO_DISABLE=1 OPENBLAS_NUM_THREADS=1
cd /afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer
```

---

## File Structure

- **Create `multitask_common.py`** — the only new logic module: mass parsing/transform, the 2-head wrapper, the masked multi-task loss, regression metrics, and regression plots. Imported by all three trainers.
- **Create `train_multitask_image.py`** — image multi-task trainer (reuses `DetectorObjectDataset`, `build_resnet` from `train_resnet_image_classifier.py`).
- **Create `train_multitask_track.py`** — track multi-task trainer (reuses `TrackPointCloudDataset`, `collate_track_point_cloud_batch`, `PointCloudTransformerClassifier`).
- **Create `train_multitask_cross.py`** — cross-attention multi-task trainer (reuses `CombinedDataset`, `collate_combined`, `CrossAttentionFusion`).
- **Create `tests/test_multitask.py`** — unit tests for the pure functions + model-wrapper smoke tests.
- **Create `condor/mreg_{image,track,cross}.{txt,sub}`** + `condor/logs/mreg_v1/` — training jobs → `runs/mreg_v1/`.
- **Do NOT modify** any existing file. All reuse is via import.

Backbone → feature extraction (verified attribute names):
- image: `m = build_resnet(name, in_channels); m.fc = nn.Identity()` → feature dim `_RESNET_FEAT_CHANNELS[name]` (512 for resnet18/34, 2048 for resnet50/101/152), importable from `train_fusion_cross_attention`.
- track: `m = PointCloudTransformerClassifier(...); m.head = nn.Identity()` → feature dim `embed_dim`.
- cross: `m = CrossAttentionFusion(...); m.head = nn.Identity()` → feature dim `2 * d_model`.

---

## Task 1: Mass parsing + log transform (`multitask_common.py`)

**Files:**
- Create: `multitask_common.py`
- Test: `tests/test_multitask.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multitask.py
import numpy as np
import multitask_common as mc

def test_sample_id_to_mass():
    assert mc.sample_id_to_mass("signal_H2000_A0p4") == 0.4
    assert mc.sample_id_to_mass("signal_H250_A1") == 1.0
    assert mc.sample_id_to_mass("signal_H750_A10") == 10.0
    assert mc.sample_id_to_mass("signal_H2000_A5") == 5.0
    assert mc.sample_id_to_mass("background") is None
    assert mc.sample_id_to_mass(b"signal_H250_A2") == 2.0  # bytes

def test_mass_transform_roundtrip():
    for m in (0.4, 1.0, 2.0, 5.0, 10.0):
        assert abs(mc.inv_transform(mc.transform_mass(m)) - m) < 1e-6

def test_mass_targets_from_codes():
    code_to_sample = {0: "background", 1: "signal_H2000_A0p4", 2: "signal_H250_A10"}
    logm, mask = mc.mass_targets_from_codes(np.array([0, 1, 2, 1]), code_to_sample)
    assert mask.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert abs(logm[1] - np.log(0.4)) < 1e-6
    assert abs(logm[2] - np.log(10.0)) < 1e-6
    assert logm[0] == 0.0  # background sentinel
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_multitask.py -k "sample_id_to_mass or roundtrip or from_codes" -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'multitask_common'`).

- [ ] **Step 3: Implement**

```python
# multitask_common.py
import re
import numpy as np


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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_multitask.py -k "sample_id_to_mass or roundtrip or from_codes" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add multitask_common.py tests/test_multitask.py
git commit -m "feat(mreg): mass parsing + log transform for A-mass regression"
```

---

## Task 2: Masked multi-task loss (`multitask_common.py`)

**Files:**
- Modify: `multitask_common.py`
- Test: `tests/test_multitask.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_multitask.py
import torch

def test_multitask_loss_masks_background():
    cls = torch.zeros(4)                    # logits
    reg = torch.tensor([9.0, 0.0, 0.0, 0.0])  # only idx0 is signal; wrong pred there
    label = torch.tensor([1., 0., 0., 0.])
    logm = torch.tensor([0.0, 0.0, 0.0, 0.0])   # true log-mass for idx0 = 0.0
    mask = torch.tensor([1., 0., 0., 0.])
    total, bce, mae = mc.multitask_loss(cls, reg, label, logm, mask, lam=1.0)
    # only idx0 contributes to MAE: |9-0| / 1 = 9
    assert abs(mae.item() - 9.0) < 1e-5
    # all-signal-masked-out -> MAE 0 (no divide-by-zero)
    _, _, mae0 = mc.multitask_loss(cls, reg, label, logm, torch.zeros(4), lam=1.0)
    assert mae0.item() == 0.0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_multitask.py -k multitask_loss -q`
Expected: FAIL (`AttributeError: module 'multitask_common' has no attribute 'multitask_loss'`).

- [ ] **Step 3: Implement**

```python
# append to multitask_common.py
import torch
import torch.nn.functional as F


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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_multitask.py -k multitask_loss -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multitask_common.py tests/test_multitask.py
git commit -m "feat(mreg): masked multi-task loss (BCE + MAE on log-mass, signal-only)"
```

---

## Task 3: Regression metrics (`multitask_common.py`)

**Files:**
- Modify: `multitask_common.py`
- Test: `tests/test_multitask.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_multitask.py
def test_regression_metrics():
    m_true = np.array([1.0, 2.0, 5.0, 10.0])
    m_pred = np.array([1.1, 1.8, 5.5, 9.0])
    d = mc.regression_metrics(m_true, m_pred)     # inputs in GeV
    assert abs(d["mae"] - np.mean([0.1, 0.2, 0.5, 1.0])) < 1e-6
    assert abs(d["mre"] - np.mean([0.1, 0.1, 0.1, 0.1])) < 1e-6

def test_regression_metrics_per_point():
    m_true = np.array([1.0, 1.0, 10.0])
    m_pred = np.array([1.2, 0.8, 9.0])
    per = mc.regression_metrics_per_point(m_true, m_pred)
    assert set(per.keys()) == {1.0, 10.0}
    assert abs(per[1.0]["resolution"] - np.std([0.2, -0.2])) < 1e-6   # std of (pred-true)/true
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_multitask.py -k "regression_metrics" -q`
Expected: FAIL (attribute error).

- [ ] **Step 3: Implement**

```python
# append to multitask_common.py
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_multitask.py -k "regression_metrics" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multitask_common.py tests/test_multitask.py
git commit -m "feat(mreg): regression metrics (MAE/MRE, per-mass-point resolution/bias)"
```

---

## Task 4: Two-head model wrapper (`multitask_common.py`)

**Files:**
- Modify: `multitask_common.py`
- Test: `tests/test_multitask.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_multitask.py
import torch.nn as nn

def test_multitask_wrapper_shapes():
    class DummyBackbone(nn.Module):          # returns a [B, feat] feature
        def __init__(self, feat): super().__init__(); self.lin = nn.Linear(8, feat)
        def forward(self, x): return self.lin(x)
    model = mc.MultiTaskModel(DummyBackbone(16), feat_dim=16)
    x = torch.randn(5, 8)
    cls, reg = model(x)
    assert cls.shape == (5,) and reg.shape == (5,)
    assert torch.isfinite(cls).all() and torch.isfinite(reg).all()
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_multitask.py -k wrapper -q`
Expected: FAIL (attribute error).

- [ ] **Step 3: Implement**

```python
# append to multitask_common.py
import torch.nn as nn


def make_heads(feat_dim, dropout=0.1):
    cls_head = nn.Linear(feat_dim, 1)
    reg_head = nn.Sequential(
        nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(feat_dim, 1))
    return cls_head, reg_head


class MultiTaskModel(nn.Module):
    """Wrap a backbone whose final head is nn.Identity (so it returns a pooled
    [B, feat_dim] feature). Adds a classification head and a regression head.
    forward(*inputs) -> (cls_logit[B], reg_mu[B]). *inputs are passed straight to
    the backbone (image: (x,); track: (points, mask); cross: (image, points, mask))."""
    def __init__(self, backbone, feat_dim, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.cls_head, self.reg_head = make_heads(feat_dim, dropout)

    def forward(self, *inputs):
        feat = self.backbone(*inputs)
        return self.cls_head(feat).squeeze(1), self.reg_head(feat).squeeze(1)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_multitask.py -k wrapper -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multitask_common.py tests/test_multitask.py
git commit -m "feat(mreg): MultiTaskModel wrapper (backbone feature -> cls + reg heads)"
```

---

## Task 5: Regression plots (`multitask_common.py`)

**Files:**
- Modify: `multitask_common.py`
- Test: `tests/test_multitask.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_multitask.py
def test_save_mass_plots(tmp_path):
    rng = np.random.default_rng(0)
    m_true = np.repeat([0.4, 1.0, 2.0, 5.0, 10.0], 50)
    m_pred = m_true * rng.normal(1.0, 0.2, size=m_true.size)
    mc.save_mass_regression_plots(str(tmp_path), m_true, m_pred, suffix="")
    for f in ("mass_pred_vs_true.png", "mass_resolution_vs_mass.png",
              "mass_pred_hist_per_point.png"):
        assert (tmp_path / f).exists()
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_multitask.py -k save_mass_plots -q`
Expected: FAIL (attribute error).

- [ ] **Step 3: Implement**

```python
# append to multitask_common.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASS_POINTS = (0.4, 1.0, 2.0, 5.0, 10.0)


def save_mass_regression_plots(output_dir, m_true, m_pred, suffix=""):
    """Write pred-vs-true, resolution-vs-mass, and per-point predicted-mass histograms
    (all masses in GeV). `suffix` appended before .png (e.g. '_test')."""
    m_true = np.asarray(m_true, dtype=np.float64)
    m_pred = np.asarray(m_pred, dtype=np.float64)
    fin = np.isfinite(m_pred) & np.isfinite(m_true) & (m_true > 0)
    m_true, m_pred = m_true[fin], m_pred[fin]

    # 1) pred vs true (2D hist, log-log)
    plt.figure(figsize=(6, 6))
    plt.hist2d(m_true, np.clip(m_pred, 1e-3, None), bins=40, cmin=1)
    lims = [0.3, 12]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xscale("log"); plt.yscale("log"); plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("true m_A [GeV]"); plt.ylabel("predicted m_A [GeV]")
    plt.title("Predicted vs true A mass"); plt.colorbar(label="objects")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"mass_pred_vs_true{suffix}.png")); plt.close()

    # 2) resolution + bias vs true mass point
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

    # 3) predicted-mass histogram per true mass point
    plt.figure(figsize=(8, 6))
    for p in pts:
        sel = np.round(m_true, 6) == p
        plt.hist(np.clip(m_pred[sel], 0, 12), bins=60, range=(0, 12),
                 histtype="step", label=f"true {p:g} GeV")
        plt.axvline(p, color="gray", ls=":", lw=0.8)
    plt.xlabel("predicted m_A [GeV]"); plt.ylabel("objects")
    plt.title("Predicted mass per true mass point"); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"mass_pred_hist_per_point{suffix}.png")); plt.close()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_multitask.py -k save_mass_plots -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multitask_common.py tests/test_multitask.py
git commit -m "feat(mreg): mass regression plots (pred-vs-true, resolution, per-point hist)"
```

---

## Task 6: Image multi-task trainer (`train_multitask_image.py`)

**Files:**
- Create: `train_multitask_image.py`
- Test: `tests/test_multitask.py` (CPU smoke test of the run-epoch step)

**Design:** Reuse `train_resnet_image_classifier.py`'s argument parsing pattern, `DetectorObjectDataset`, compact manifest loading, `build_resnet`, and `save_roc_plot`. Build the backbone, set `model.fc = nn.Identity()`, wrap in `MultiTaskModel`. The dataset already yields `label`, `weight`, `pt`, `sample_code`; compute the mass target from `sample_code` + `code_to_sample` in the epoch loop.

- [ ] **Step 1: Write the failing smoke test**

```python
# append to tests/test_multitask.py
def test_image_multitask_forward_and_loss():
    import torch, torch.nn as nn
    from train_fusion_cross_attention import _RESNET_FEAT_CHANNELS
    from train_resnet_image_classifier import build_resnet
    backbone = build_resnet("resnet18", in_channels=1)
    backbone.fc = nn.Identity()
    model = mc.MultiTaskModel(backbone, feat_dim=_RESNET_FEAT_CHANNELS["resnet18"])
    x = torch.randn(4, 1, 512, 512)
    cls, reg = model(x)
    assert cls.shape == (4,) and reg.shape == (4,)
    code_to_sample = {0: "background", 1: "signal_H250_A5"}
    logm, mask = mc.mass_targets_from_codes(np.array([1, 0, 1, 0]), code_to_sample)
    total, bce, mae = mc.multitask_loss(cls, reg, torch.tensor([1.,0,1,0]),
                                        torch.tensor(logm), torch.tensor(mask))
    assert torch.isfinite(total)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/test_multitask.py -k image_multitask -q`
Expected: FAIL until `build_resnet`/import path resolves in the test (it should pass once `multitask_common` + imports exist; if `build_resnet` signature differs, fix the call). This test does not import `train_multitask_image` yet — it validates the wrapping recipe the trainer will use.

- [ ] **Step 3: Implement `train_multitask_image.py`**

Create the file with this structure (fill the data-loading block by mirroring `train_resnet_image_classifier.py:main()` lines 1026-1115, which build `file_entries`, `code_to_sample`, `split_manifests`, and `DetectorObjectDataset` train/val/test loaders):

```python
#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

import multitask_common as mc
from train_fusion_cross_attention import _RESNET_FEAT_CHANNELS
from train_resnet_image_classifier import (
    build_resnet, DetectorObjectDataset, image_in_channels, set_seed,
    resolve_num_workers, save_roc_plot, save_score_distribution, safe_roc_auc,
    DEFAULT_PREFETCH_FACTOR,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_epoch(model, loader, device, code_to_sample, lam, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot = 0.0; n = 0
    S_score, S_label, S_mpred, S_mtrue, S_sig = [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        logm_np, mask_np = mc.mass_targets_from_codes(
            batch["sample_code"].numpy(), code_to_sample)
        mass = torch.from_numpy(logm_np).to(device)
        sig = torch.from_numpy(mask_np).to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            cls, reg = model(x)
            loss, bce, mae = mc.multitask_loss(cls, reg, label, mass, sig, weight, lam)
            if is_train:
                loss.backward(); optimizer.step()
        tot += float(loss.detach().cpu()); n += 1
        S_score.append(torch.sigmoid(cls).detach().cpu().numpy())
        S_label.append(batch["label"].numpy())
        S_mpred.append(mc.inv_transform(reg.detach().cpu().numpy()))
        S_mtrue.append(mc.inv_transform(logm_np))
        S_sig.append(mask_np)
    pack = {"loss": tot / max(n, 1),
            "score": np.concatenate(S_score), "label": np.concatenate(S_label),
            "weight": np.ones(len(np.concatenate(S_label))),
            "m_pred": np.concatenate(S_mpred), "m_true": np.concatenate(S_mtrue),
            "is_sig": np.concatenate(S_sig).astype(bool)}
    pack["auc"] = safe_roc_auc(pack["label"], pack["score"], pack["weight"])
    return pack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--compact", required=True)
    p.add_argument("--weight-h5", required=True)          # only for compat; unused when --compact given
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--track-types", default="GenTrk")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--no-es", action="store_true")
    p.add_argument("--no-track-channels", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--resume-eos-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    detector = args.detector
    track_types = tuple(t for t in args.track_types.split(",") if t)
    include_es = not args.no_es
    include_track = not args.no_track_channels
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nworkers = resolve_num_workers(args.num_workers)

    # ---- data: reuse the compact manifest loader ----
    from pipeline.compact_manifest import load_compact_manifest
    from train_resnet_image_classifier import rebalance_manifest_class_weights
    file_entries, split_manifests, code_to_sample = load_compact_manifest(args.compact)
    for s in ("train", "val", "test"):
        split_manifests[s], _ = rebalance_manifest_class_weights(split_manifests[s])

    def ds(split):
        return DetectorObjectDataset(detector, file_entries, split_manifests[split],
                                     track_types=track_types, include_es=include_es,
                                     include_track=include_track)
    kw = dict(batch_size=args.batch_size, num_workers=nworkers,
              pin_memory=(device.type == "cuda"))
    if nworkers > 0:
        kw["persistent_workers"] = True; kw["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    train_loader = DataLoader(ds("train"), shuffle=True, **kw)
    val_loader = DataLoader(ds("val"), shuffle=False, **kw)
    test_loader = DataLoader(ds("test"), shuffle=False, **kw)

    in_ch = image_in_channels(detector, track_types, include_es, include_track)
    backbone = build_resnet(args.model, in_channels=in_ch)
    backbone.fc = nn.Identity()
    model = mc.MultiTaskModel(backbone, _RESNET_FEAT_CHANNELS[args.model]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best = -np.inf; best_path = output_dir / "best_model.pt"
    if args.resume_eos_dir:
        from pipeline import resume as _r
        _r.stage_in(args.resume_eos_dir, output_dir, ["best_model.pt", "last.pt"])
    hist = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, code_to_sample, args.lam, opt)
        va = run_epoch(model, val_loader, device, code_to_sample, args.lam, None)
        va_mass = mc.regression_metrics(va["m_true"][va["is_sig"]], va["m_pred"][va["is_sig"]])
        sched.step(va["auc"] if np.isfinite(va["auc"]) else -1.0)
        hist.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                     "val_auc": va["auc"], "val_mass_mae": va_mass["mae"]})
        print(f"[epoch {epoch}] val_auc={va['auc']:.4f} val_mass_mae={va_mass['mae']:.4f}", flush=True)
        ck = {"epoch": epoch, "model_state_dict": model.state_dict()}
        torch.save(ck, output_dir / "last.pt")
        if np.isfinite(va["auc"]) and va["auc"] > best:
            best = va["auc"]; torch.save(ck, best_path)
        if args.resume_eos_dir:
            from pipeline import resume as _r
            _r.push(args.resume_eos_dir, output_dir / "last.pt")
            if best_path.exists(): _r.push(args.resume_eos_dir, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test = run_epoch(model, test_loader, device, code_to_sample, args.lam, None)
    sig = test["is_sig"]
    save_roc_plot(test["label"], test["score"], test["weight"], output_dir / "roc_test.png")
    mc.save_mass_regression_plots(str(output_dir), test["m_true"][sig], test["m_pred"][sig], suffix="_test")
    metrics = {"test_auc": test["auc"], **{f"mass_{k}": v for k, v in
               mc.regression_metrics(test["m_true"][sig], test["m_pred"][sig]).items()}}
    metrics["mass_per_point"] = mc.regression_metrics_per_point(test["m_true"][sig], test["m_pred"][sig])
    with (output_dir / "test_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    with (output_dir / "history.json").open("w") as fh:
        json.dump(hist, fh, indent=2)
    import csv
    with (output_dir / "test_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["score", "label", "m_true", "m_pred", "is_sig"])
        for i in range(len(test["score"])):
            w.writerow([test["score"][i], int(test["label"][i]),
                        test["m_true"][i], test["m_pred"][i], int(sig[i])])
    print("[done] wrote plots + metrics to", output_dir, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify (syntax + import + CPU smoke)**

Run:
```bash
python -c "import ast; ast.parse(open('train_multitask_image.py').read()); print('syntax OK')"
python -c "import train_multitask_image; print('import OK')"
python -m pytest tests/test_multitask.py -k image_multitask -q
```
Expected: `syntax OK`, `import OK`, test PASS.

- [ ] **Step 5: Commit**

```bash
git add train_multitask_image.py tests/test_multitask.py
git commit -m "feat(mreg): image multi-task trainer (BCE + log-mass MAE, reuse compact_pt50)"
```

---

## Task 7: Track multi-task trainer (`train_multitask_track.py`)

**Files:**
- Create: `train_multitask_track.py`

**Design:** Same as Task 6 but track inputs. Reuse `TrackPointCloudDataset`, `collate_track_point_cloud_batch`, `point_feature_dim`, `parse_track_types` from `track_point_transformer.py`; backbone = `PointCloudTransformerClassifier(...)`, set `model.head = nn.Identity()`, feat_dim = `args.embed_dim`. `run_epoch` calls `model(points, mask)`; batch provides `points`, `mask`, `label`, `weight`, `sample_code`. Loaders use `collate_fn=collate_track_point_cloud_batch`.

- [ ] **Step 1: Write `train_multitask_track.py`**

Copy Task 6's file and change ONLY:
- imports:
```python
from track_point_transformer import (
    PointCloudTransformerClassifier, TrackPointCloudDataset,
    collate_track_point_cloud_batch, point_feature_dim, parse_track_types)
from train_resnet_image_classifier import set_seed, resolve_num_workers, save_roc_plot, safe_roc_auc, DEFAULT_PREFETCH_FACTOR
```
- args: replace image args with `--embed-dim` (128), `--depth` (4), `--num-heads` (4), `--mlp-ratio` (4.0), `--dropout` (0.1), `--max-points` (16); keep `--detector/--compact/--weight-h5/--output-dir/--track-types/--epochs/--batch-size/--lr/--weight-decay/--num-workers/--lam/--seed/--resume-eos-dir/--no-early-stopping`.
- dataset:
```python
def ds(split):
    return TrackPointCloudDataset(detector=detector, file_entries=file_entries,
        manifest=split_manifests[split], max_points=args.max_points, track_types=track_types)
kw["collate_fn"] = collate_track_point_cloud_batch
```
- backbone:
```python
backbone = PointCloudTransformerClassifier(
    input_dim=point_feature_dim(track_types), embed_dim=args.embed_dim,
    depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout)
backbone.head = nn.Identity()
model = mc.MultiTaskModel(backbone, args.embed_dim).to(device)
```
- in `run_epoch`, replace the input line with:
```python
points = batch["points"].to(device, non_blocking=True)
mask = batch["mask"].to(device, non_blocking=True)
...
cls, reg = model(points, mask)
```
(no `image` key; everything else identical).

- [ ] **Step 2: Verify (syntax + import)**

Run:
```bash
python -c "import ast; ast.parse(open('train_multitask_track.py').read()); print('syntax OK')"
python -c "import train_multitask_track; print('import OK')"
```
Expected: both OK.

- [ ] **Step 3: CPU smoke test (1 forward+backward)**

Run:
```bash
python - <<'PY'
import torch, torch.nn as nn, numpy as np
import multitask_common as mc
from track_point_transformer import PointCloudTransformerClassifier, point_feature_dim
b = PointCloudTransformerClassifier(input_dim=point_feature_dim(("GenTrk",)), embed_dim=32, depth=1, num_heads=4)
b.head = nn.Identity()
m = mc.MultiTaskModel(b, 32)
pts = torch.randn(4, 5, point_feature_dim(("GenTrk",))); mask = torch.ones(4,5,dtype=torch.bool); mask[0]=False
cls, reg = m(pts, mask)
assert cls.shape==(4,) and reg.shape==(4,) and torch.isfinite(reg).all()
print("track multitask smoke OK")
PY
```
Expected: `track multitask smoke OK`.

- [ ] **Step 4: Commit**

```bash
git add train_multitask_track.py
git commit -m "feat(mreg): track multi-task trainer"
```

---

## Task 8: Cross-attention multi-task trainer (`train_multitask_cross.py`)

**Files:**
- Create: `train_multitask_cross.py`

**Design:** Same skeleton; cross inputs. Reuse `CombinedDataset`, `collate_combined`, `CrossAttentionFusion` from `train_fusion_cross_attention.py`, plus `DetectorObjectDataset`/`TrackPointCloudDataset` to build the combined dataset (see `train_fusion_cross_attention.py:make_loader`, lines 307-319). backbone = `CrossAttentionFusion(...)`, set `model.head = nn.Identity()`, feat_dim = `2 * args.d_model`. `run_epoch` calls `model(image, points, mask)`.

- [ ] **Step 1: Write `train_multitask_cross.py`**

Copy Task 6's file; change:
- imports:
```python
from train_fusion_cross_attention import (
    CrossAttentionFusion, CombinedDataset, collate_combined, _RESNET_FEAT_CHANNELS)
from train_resnet_image_classifier import (
    DetectorObjectDataset, image_in_channels, set_seed, resolve_num_workers,
    save_roc_plot, safe_roc_auc, DEFAULT_PREFETCH_FACTOR)
from track_point_transformer import TrackPointCloudDataset, point_feature_dim, parse_track_types
```
- args: add `--backbone` (resnet18), `--d-model` (256), `--num-heads` (4), `--depth` (2), `--track-depth` (2), `--max-points` (16), `--dropout` (0.1); keep the common ones.
- dataset:
```python
def ds(split):
    img = DetectorObjectDataset(detector, file_entries, split_manifests[split],
        track_types=track_types, include_es=include_es, include_track=False)
    trk = TrackPointCloudDataset(detector=detector, file_entries=file_entries,
        manifest=split_manifests[split], max_points=args.max_points, track_types=track_types)
    return CombinedDataset(img, trk)
kw["collate_fn"] = collate_combined
```
- backbone:
```python
in_ch = image_in_channels(detector, track_types, include_es, include_track=False)
backbone = CrossAttentionFusion(in_channels=in_ch, point_feat_dim=point_feature_dim(track_types),
    backbone=args.backbone, d_model=args.d_model, num_heads=args.num_heads,
    depth=args.depth, track_depth=args.track_depth, dropout=args.dropout)
backbone.head = nn.Identity()
model = mc.MultiTaskModel(backbone, 2 * args.d_model).to(device)
```
- in `run_epoch`:
```python
image = batch["image"].to(device, non_blocking=True)
points = batch["points"].to(device, non_blocking=True)
mask = batch["mask"].to(device, non_blocking=True)
...
cls, reg = model(image, points, mask)
```

- [ ] **Step 2: Verify (syntax + import)**

Run:
```bash
python -c "import ast; ast.parse(open('train_multitask_cross.py').read()); print('syntax OK')"
python -c "import train_multitask_cross; print('import OK')"
```
Expected: both OK.

- [ ] **Step 3: CPU smoke test**

Run:
```bash
python - <<'PY'
import torch, torch.nn as nn
import multitask_common as mc
from train_fusion_cross_attention import CrossAttentionFusion
from track_point_transformer import point_feature_dim
b = CrossAttentionFusion(in_channels=1, point_feat_dim=point_feature_dim(("GenTrk",)),
    backbone="resnet18", d_model=32, num_heads=4, depth=1, track_depth=1)
b.head = nn.Identity()
m = mc.MultiTaskModel(b, 2*32)
img = torch.randn(3,1,64,64); pts = torch.randn(3,5,point_feature_dim(("GenTrk",)))
mask = torch.zeros(3,5,dtype=torch.bool); mask[1,:2]=True
cls, reg = m(img, pts, mask)
assert cls.shape==(3,) and reg.shape==(3,) and torch.isfinite(reg).all()
print("cross multitask smoke OK")
PY
```
Expected: `cross multitask smoke OK`.

- [ ] **Step 4: Commit**

```bash
git add train_multitask_cross.py
git commit -m "feat(mreg): cross-attention multi-task trainer"
```

---

## Task 9: Condor job files (`condor/mreg_*`)

**Files:**
- Create: `condor/mreg_image.txt`, `condor/mreg_image.sub`, `condor/mreg_track.txt`, `condor/mreg_track.sub`, `condor/mreg_cross.txt`, `condor/mreg_cross.sub`
- Create dir: `condor/logs/mreg_v1/`

- [ ] **Step 1: Generate the arg-line txt files from the existing pt50 txt files**

The run list mirrors v8/cross_v2 (pt50). Reuse the compacts and category structure by transforming the existing txt files: swap the script name, point outputs to `runs/mreg_v1/`, and add `--lam 1.0`.

```bash
cd /afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer/condor
mkdir -p logs/mreg_v1
# image: same args as v8_image but new script + output dir + lam
sed -e 's#^train_resnet_image_classifier.py#train_multitask_image.py#' \
    -e 's#runs/v8/image_#runs/mreg_v1/image_#g' \
    -e 's#$# --lam 1.0#' v8_image.txt > mreg_image.txt
sed -e 's#^train_point_transformer_track_classifier.py#train_multitask_track.py#' \
    -e 's#runs/v8/track_#runs/mreg_v1/track_#g' \
    -e 's#$# --lam 1.0#' v8_track.txt > mreg_track.txt
sed -e 's#^train_fusion_cross_attention.py#train_multitask_cross.py#' \
    -e 's#runs/cross_v2/cross_#runs/mreg_v1/cross_#g' \
    -e 's#$# --lam 1.0#' cross_v2.txt > mreg_cross.txt
wc -l mreg_image.txt mreg_track.txt mreg_cross.txt   # expect 18, 12, 18
head -1 mreg_image.txt
```
Expected: 18/12/18 lines; first image line starts `train_multitask_image.py ... --output-dir /eos/user/y/yeo/4l/runs/mreg_v1/image_aod_eb_all ... --lam 1.0`.

- [ ] **Step 2: Create the .sub files**

For each family create `condor/mreg_<fam>.sub` by copying the matching v8/cross_v2 sub and adding the new trainer + `multitask_common.py` to `transfer_input_files`, pointing logs to `logs/mreg_v1/`, and queueing from the new txt. Example `mreg_image.sub`:

```bash
sed -e 's#train_resnet_image_classifier.py, #train_resnet_image_classifier.py, train_multitask_image.py, multitask_common.py, #' \
    -e 's#condor/logs/v8/#condor/logs/mreg_v1/#g' \
    -e 's#queue arguments from condor/v8_image.txt#queue arguments from condor/mreg_image.txt#' \
    v8_image.sub > mreg_image.sub
sed -e 's#train_resnet_image_classifier.py, #train_resnet_image_classifier.py, train_multitask_track.py, multitask_common.py, #' \
    -e 's#condor/logs/v8/#condor/logs/mreg_v1/#g' \
    -e 's#queue arguments from condor/v8_track.txt#queue arguments from condor/mreg_track.txt#' \
    v8_track.sub > mreg_track.sub
sed -e 's#train_fusion_cross_attention.py, #train_fusion_cross_attention.py, train_multitask_cross.py, multitask_common.py, #' \
    -e 's#condor/logs/cross_v2/#condor/logs/mreg_v1/#g' \
    -e 's#queue arguments from condor/cross_v2.txt#queue arguments from condor/mreg_cross.txt#' \
    cross_v2.sub > mreg_cross.sub
grep -E "transfer_input_files|queue|logs/mreg" mreg_image.sub mreg_track.sub mreg_cross.sub
```
Expected: each sub transfers its `train_multitask_*.py` + `multitask_common.py`, logs under `logs/mreg_v1/`, queues from the matching `mreg_*.txt`.

- [ ] **Step 3: Commit**

```bash
cd /afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer
git add condor/mreg_image.txt condor/mreg_image.sub condor/mreg_track.txt condor/mreg_track.sub condor/mreg_cross.txt condor/mreg_cross.sub
git commit -m "feat(mreg): condor jobs for multi-task training (mreg_v1 on pt50)"
```

---

## Task 10: End-to-end validation on Condor (1 job, then full)

**Files:** none (submission only).

- [ ] **Step 1: Submit ONE track job to validate the full path on GPU**

```bash
cd /afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer
grep track_aod_eb_all condor/mreg_track.txt > condor/mreg_TEST.txt
sed 's#queue arguments from condor/mreg_track.txt#queue arguments from condor/mreg_TEST.txt#' condor/mreg_track.sub > condor/mreg_TEST.sub
condor_submit condor/mreg_TEST.sub
```
Expected: `1 job(s) submitted`.

- [ ] **Step 2: When it finishes, verify outputs**

Run (poll `condor_q`; when gone, check):
```bash
ls /eos/user/y/yeo/4l/runs/mreg_v1/track_aod_eb_all/
python -c "import json; d=json.load(open('/eos/user/y/yeo/4l/runs/mreg_v1/track_aod_eb_all/test_metrics.json')); print('AUC', d['test_auc'], 'mass_mae', d['mass_mae']); print('per_point', {k:round(v['resolution'],3) for k,v in d['mass_per_point'].items()})"
```
Expected: `test_metrics.json`, `roc_test.png`, `mass_pred_vs_true_test.png`, `mass_resolution_vs_mass_test.png`, `mass_pred_hist_per_point_test.png`, `test_predictions.csv`. AUC finite (classification still works); `mass_mae` finite; per-point resolution present for the 5 masses (expect larger at 0.4 GeV).

- [ ] **Step 3: If the test job is healthy, submit the full set**

```bash
condor_rm $(condor_q -af ClusterId | tail -1)   # remove the test job cluster if still queued; else skip
condor_submit condor/mreg_image.sub
condor_submit condor/mreg_track.sub
condor_submit condor/mreg_cross.sub
condor_q -totals
```
Expected: 18 + 12 + 18 = 48 jobs submitted.

- [ ] **Step 4: When all finish, sanity-scan**

```bash
python - <<'PY'
import glob, json, os
for d in sorted(glob.glob("/eos/user/y/yeo/4l/runs/mreg_v1/*/")):
    p = os.path.join(d, "test_metrics.json")
    if not os.path.exists(p): print("MISSING", os.path.basename(d.rstrip("/"))); continue
    m = json.load(open(p))
    print(os.path.basename(d.rstrip("/")), "AUC=%.3f"%m["test_auc"], "mass_mae=%.3f"%m["mass_mae"])
PY
```
Expected: 48 rows; AUCs comparable to the pure classifiers (multi-task shouldn't wreck classification); mass_mae finite everywhere.

- [ ] **Step 5: Commit any run-config notes (optional) and update the handoff doc**

```bash
git add docs/PROJECT_HANDOFF.md   # add an mreg_v1 row if desired
git commit -m "docs: note mreg_v1 multi-task mass-regression campaign" || echo "nothing to commit"
```

---

## Notes for the executor

- **Do not modify** `train_resnet_image_classifier.py`, `track_point_transformer.py`, or `train_fusion_cross_attention.py`. All reuse is by import; the head→Identity swap happens on the instantiated object.
- `run_job.sh` stages `--compact` and `--weight-h5` in and injects `--resume-eos-dir`. Keep `--weight-h5` in the arg lines (it comes from the copied txt) even though the trainer ignores it when `--compact` is given — it must exist on EOS to stage.
- EOS reads can be flaky; if a training job dies on an I/O error, `max_retries` re-runs it (the compacts are single files staged via xrdcp, so this is rarely an issue).
- Mass loss weight `λ` is fixed at 1.0 for v1; if classification AUC degrades badly vs the pure classifiers, lower `λ` (e.g. 0.3) and resubmit.
```
