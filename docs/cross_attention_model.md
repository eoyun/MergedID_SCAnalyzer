# Cross-Attention Fusion Model for Merged-Electron ID

A two-modality (calorimeter image + track point cloud) classifier that identifies
**merged electrons** in H → AA → 4ℓ events. This document describes the model in
`train_fusion_cross_attention.py`.

---

## 1. Physics motivation

In H → AA → 4ℓ, when the pseudoscalar **A** is light and highly boosted, its two
electrons come out extremely collimated. The detector often reconstructs the pair as
a **single object** (one supercluster + a small cluster of tracks) instead of two.
We call that object a *merged electron*.

The task is a per-object **binary classification**:

- **Signal** — a real merged electron (a genuine A → ee).
- **Background** — a fake: a single electron or a jet faking one, from W/Z events.

Barrel (**EB**) and endcap (**EE**) are trained as separate models because their
detector geometry and inputs differ.

---

## 2. The two views of one object

Each object is described by two complementary "views":

**(a) Calorimeter image** — the electromagnetic shower shape.
- EB: `SC_energy`, a 32×32 crop of the supercluster's ECAL energy.
- EE: `EE_seed_energy` (32×32) plus two preshower planes
  `ES_seed_plane1` (16×512) and `ES_seed_plane2` (512×16). `--no-es` drops the
  preshower and uses the seed image only.
- Intuition: a merged electron tends to leave a wider / double-cored shower than a
  single electron.

**(b) Track point cloud** — the charged-particle trajectories.
- The object's associated tracks form an **unordered set of points** (0 … `max_points`).
- Each track becomes a point with features `[x, y, log(1+pT), one-hot(collection)]`,
  where `(x, y)` is its normalized position on the detector grid (∈ [-1, 1]).
- Track collections depend on the data tier: **AOD → `GenTrk`**;
  **MiniAOD → `GSF, PF, Lost`**.
- Intuition: a merged electron usually has **two or more** nearby tracks.

**Why fuse them with cross-attention?** The shower shape and the track pattern carry
*different* evidence. Rather than embedding each view independently and concatenating
(late fusion), cross-attention lets **image tokens attend to track tokens and vice
versa**, so the network can learn correlations like "wide shower *and* two tracks".

---

## 3. Architecture

```
   Calorimeter image                         Track point cloud
   [B, C, 32, 32]                             [B, N, 3+T]   (+ mask)
        │                                          │
   ResNet backbone (resnet18)                 per-point MLP embed
   → feature map [B, C', h, w]                → [B, N, d]
   → 1×1 conv → flatten                       prepend learned NULL token
   → image tokens [B, Ni, d]                  → [B, 1+N, d]
        │  (+ img-type emb)                   track self-attention (Transformer)
        │                                     → track tokens [B, 1+N, d]
        │                                          │  (+ trk-type emb)
        └───────────────┬──────────────────────────┘
                        ▼
        Bidirectional cross-attention  ×  depth layers
          image ← track   (image queries attend to track keys)
          track ← image   (track queries attend to image keys)
                        │
        ┌───────────────┴───────────────┐
   mean-pool image tokens        masked mean-pool track tokens
        └───────────────┬───────────────┘
                   concat [B, 2d] → LayerNorm → MLP head → 1 logit
```

Component details (defaults in parentheses):

- **Image tower** — a torchvision ResNet (`--backbone resnet18`) with its global pool
  and FC head removed, giving a spatial feature map; a 1×1 conv projects channels to
  `d_model` (256) and the map is flattened into image tokens. A learned "image-type"
  embedding tags them.
- **Track tower** — an MLP embeds each point to `d_model`. A single learned **`null`
  token** is prepended so an object with *zero* tracks still has a non-empty key set
  (this prevents a degenerate all-masked attention softmax). A small Transformer
  encoder (`--track-depth 2`) adds intra-track context, respecting a padding mask.
- **Cross-attention** — `--depth` (2) layers. Each layer runs two pre-LayerNorm
  blocks (`nn.MultiheadAttention` + feed-forward): one where the image attends to the
  tracks, one where the tracks attend to the image.
- **Pooling & head** — image tokens are mean-pooled; track tokens are
  **masked**-mean-pooled (padding excluded, denominator clamped ≥ 1). The two `d_model`
  vectors are concatenated, LayerNorm'd, and passed through an MLP to a single logit.
- **Loss** — sample-weighted binary cross-entropy. Each object carries a physics
  weight so the training mix matches the intended signal/background composition.

---

## 4. Inputs

The model reads a **compact HDF5** file (one per data-tier × detector × hasADD-mode),
where every row is a single object holding its calo image, its track point cloud, and
bookkeeping (`label`, `weight`, `pt`, train/val/test `split`). See
[`dataset-structure.md`](dataset-structure.md) for the full schema and how these files
are built from the raw data.

- `--compact` — the compact HDF5 (the images + tracks).
- `--weight-h5` — the per-category weight file (object selection, splits, pT bin edges).
- `--detector {eb,ee}`, `--track-types` — pick the region and track collections.

---

## 5. How to run

```bash
python train_fusion_cross_attention.py \
    --detector eb \
    --weight-h5 /eos/user/y/yeo/4l/weights/categories_pt200/AOD_all/category_weights.h5 \
    --compact   /eos/user/y/yeo/4l/compact_pt200/AOD_eb_all.h5 \
    --output-dir runs/cross_v3/cross_aod_eb_all \
    --track-types GenTrk \
    --backbone resnet18 --epochs 50 --batch-size 16 --lr 1e-4 \
    --num-workers 8 --no-early-stopping
```

Common options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--detector` | (required) | `eb` (barrel) or `ee` (endcap) |
| `--track-types` | `GSF,PF,Lost` | track collections (`GenTrk` for AOD) |
| `--no-es` | off | (EE) drop the two preshower planes |
| `--backbone` | `resnet18` | image backbone (resnet18/34/50/101/152) |
| `--d-model` / `--num-heads` | 256 / 4 | transformer width / attention heads |
| `--depth` / `--track-depth` | 2 / 2 | cross-attention layers / track self-attn layers |
| `--max-points` | 16 | max tracks kept per object |
| `--epochs` / `--batch-size` / `--lr` | 50 / 16 / 1e-4 | training schedule |
| `--monitor` | `auc` | early-stopping / best-model metric (`auc` or `f1`) |
| `--no-early-stopping` | off | run all epochs regardless of `--patience` |
| `--resume-eos-dir` | none | resume from `last.pt` staged on EOS |

**Optimizer / schedule.** AdamW (`--weight-decay 1e-4`) with `ReduceLROnPlateau` on
the monitored validation metric; optional early stopping (`--patience 5`).

**Numerical precision.** Training runs in **fp32** with **TF32** matmuls enabled on
Ampere+ GPUs (A100). An earlier fp16 mixed-precision version overflowed fp16's limited
range on some batches → `NaN` logits → crashes; fp32 keeps the full range (no overflow)
while TF32 preserves most of the speed.

---

## 6. Outputs

Written to `--output-dir` (and mirrored to EOS each epoch when `--resume-eos-dir` is set):

- `best_model.pt`, `last.pt` — checkpoints (model + optimizer + scheduler + epoch).
- `run_config.json`, `test_metrics.json` — configuration and final test AUC / F1.
- `val_predictions.csv`, `test_predictions.csv` — per-object score, label, weight, pT.
- `training_history.png` — loss / AUC vs epoch.
- ROC, score-distribution, and **efficiency-vs-pT** plots, in both weighted and
  unweighted variants. Unweighted efficiency is reported at fixed background
  fake-rate working points (**2.5%, 5%, 10%**), with the threshold chosen on the
  validation set.

---

## 7. Where it sits

This cross-attention model is one of several fusion approaches trained side by side
against single-modality baselines:

- **image-only** — ResNet on the calorimeter image (`train_resnet_image_classifier.py`).
- **track-only** — Transformer on the track point cloud (`train_point_transformer_track_classifier.py`).
- **fusion** — cross-attention (this model), plus concat and GBDT-ensemble variants.

Models are trained per data tier (AOD, MiniAOD), detector (EB, EE), and hasADD mode
(all / =0 / =1 extra tracks). Typical test AUC lands in the **~0.85 – 0.95** range
depending on region and category.
