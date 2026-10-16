# MergedID_SCAnalyzer — Cross-Attention Fusion for Merged-Electron ID

Identifies **merged electrons** (a collimated A → ee pair reconstructed as one supercluster)
and regresses the parent **A mass**. The main model is a **cross-attention fusion** network that
combines the ECAL supercluster image with the track point cloud, trained as a multi-task model
(classification + continuous mass regression).

The previous README (event-level weights, standalone image/track classifiers, stacking ensemble,
legacy regression) is kept in [`docs/README_legacy.md`](docs/README_legacy.md).

---

## Contents
1. [Model](#1-model)
2. [Pipeline overview](#2-pipeline-overview)
3. [Datasets](#3-datasets)
4. [Step-by-step](#4-step-by-step)
5. [Running on Condor](#5-running-on-condor)
6. [Evaluation and plots](#6-evaluation-and-plots)
7. [Campaigns](#7-campaigns)
8. [Environment](#8-environment)
9. [Known issues](#9-known-issues)
10. [Further documentation](#10-further-documentation)

---

## 1. Model

`CrossAttentionFusion` (`train_fusion_cross_attention.py`) wrapped by `MultiTaskModel`
(`multitask_common.py`).

```
 calo image (raw crop 32x32 or 48x48, + ES planes for EE)      track point cloud [N_t, F] + mask
        │                                                               │
 tile_to_square -> common square S (512 or 768)                 trk_in MLP (F -> d)
 ResNet backbone (avgpool/fc removed) -> [C_f, S/32, S/32]       + null token, self-attention encoder
 1x1 conv -> Ni image tokens [Ni, d]  (+ img_type)               (+ trk_type)
        └──────────── bidirectional cross-attention x depth ───────────┘
                 img <- track (img2trk),  track <- img (trk2img)
                                │
           masked-mean pool per modality -> concat [2d] -> LayerNorm
                                │ feat [512]
                 ┌──────────────┴──────────────┐
          cls_head Linear(512,1)        reg_head MLP(512->512->1)
          merged-electron logit         predicted log(m_A)
```

| Item | Default |
|---|---|
| Image backbone | ResNet-18 (avgpool and fc removed) |
| `d_model` / fused feature | 256 / 512 |
| Heads / cross-attention depth / track depth | 4 / 2 / 2 |
| Image tokens `Ni` | 256 at S=512, 576 at S=768 |
| Loss | `BCE(cls; weight) + λ · MAE(reg, log m_A) · [signal]`, λ = 1.0 |
| Optimizer | AdamW lr 1e-4, wd 1e-4, ReduceLROnPlateau on val AUC |
| Precision | fp32 + TF32 (no AMP) |

Key properties:
- **Null track token** keeps attention defined for objects with zero tracks.
- **No positional embedding**, so the number of image tokens follows the tile size automatically.
- The **mass truth is only used in the loss**; the model sees only the image and tracks.
- Background objects are masked out of the regression term.

Full explanation: [`docs/cross-attention-logic.md`](docs/cross-attention-logic.md),
line-by-line review: [`docs/gunheavy-model-code-review.md`](docs/gunheavy-model-code-review.md).

### Image input and `IMAGE_TILE_SIZE`

Every image channel (calo, ES plane 1, ES plane 2) is tiled by nearest-repeat to a common square
`S` and stacked into `[C, S, S]`. `S` must be divisible by every channel's height and width:

| Data | calo | ES planes | `S` |
|---|---|---|---|
| `MiniAOD` (32x32 crops) | 32x32 | 16x512, 512x16 | 512 (default) |
| `MiniAOD_cropped` (48x48 crops) | 48x48 | 24x768, 768x24 | **768** |

Set it with the environment variable `IMAGE_TILE_SIZE` (read in `train_resnet_image_classifier.py`):

```bash
export IMAGE_TILE_SIZE=768   # for MiniAOD_cropped
```

At S=768 the input has 2.25x the pixels of S=512, so use a smaller batch (8 instead of 16).

Image channels are `log1p(energy)` with **no per-channel normalization**. ES values are about 1000x
smaller than calo after `log1p`; per-channel normalization is a planned follow-up.

---

## 2. Pipeline overview

```
raw h5 (EOS data/<tier>/{signal_dir, W, Z})
   │  build_category_weights_gun.py      per-object SC-ET cut + pT-bin balanced weights
   ▼
category_weights.h5
   │  build_compact_dataset_gun.py       train/val/test split, images, tracks, pt, a_mass
   ▼
compact h5 (one per detector)
   │  train_multitask_cross_gun.py       cross-attention multi-task training
   ▼
runs/<campaign>/<run>/  (best_model.pt, plots, CSVs, history.json)
   │  build_category_weights_gun_eval.py + build_compact_dataset_gun.py   20 GeV eval sample
   │  eval_gunheavy.py                   score + evaluation plots
   ▼
runs/<campaign>_eval/<run>/
```

### pT variable

The pT used for the cut, the weight binning and the efficiency-vs-pT plots is the **per-object
reconstructed supercluster transverse energy**: `SC_energyT_sum` (EB) and `EE_seed_energyT_sum` (EE).
It exists in real data, unlike the generator-level `A_pT` used by the older 18-way pipeline
(the `A_*` arrays hold the parent particle per sample: A for signal, W or Z boson for background).

### Weights

Each class (signal, background) sums to 1. Within a class, every non-empty SC-ET bin gets an equal
share, split evenly over its objects:

`w = 1 / (N_nonempty_bins(class) * n(class, bin))`

so the classifier cannot use the pT spectrum as a shortcut. Background is randomly capped at
1M objects per detector.

### Regression target

`a_mass` per object = the A mass of its event. In the particle-gun samples both A's in an event have
the same mass, so no electron-to-A matching is needed. Background `a_mass = 0` and is masked.

---

## 3. Datasets

On EOS under `/eos/user/y/yeo/4l/data/`:

| Directory | Content | Role |
|---|---|---|
| `MiniAOD/particle_gun_heavy` | A → ee gun, continuous m_A 0.1–50 GeV | signal (gunheavy) |
| `MiniAOD/signal` | H → AA → 4ℓ, discrete m_A (0.4, 1, 2, 5, 10 GeV) | evaluation signal |
| `MiniAOD/W`, `MiniAOD/Z` | W → eν, Z → ee | background |
| `MiniAOD_cropped/HToEleEle_dr03` | A → ee, ΔR < 0.3, continuous m_A, 48x48 crops | signal (dr03) |
| `MiniAOD_cropped/{signal,W,Z}` | same as above, 48x48 crops | eval signal / background |

File schema: [`docs/dataset-structure.md`](docs/dataset-structure.md).

---

## 4. Step-by-step

Examples for the dr03 campaign at pT ≥ 50 GeV. Paths are on EOS.

### 4.1 Category weights

```bash
python3 build_category_weights_gun.py \
  --input-dir /eos/user/y/yeo/4l/data/MiniAOD_cropped \
  --signal-dir HToEleEle_dr03 \
  --pt-min 50 \
  --output-path weights/categories_dr03_pt50/dr03_all/category_weights.h5
```

`--signal-dir` names the signal subdirectory; `W` and `Z` are always background.
Default `--background-max-objects 1000000`. Unreadable files are skipped.

### 4.2 Compact dataset

```bash
python3 build_compact_dataset_gun.py \
  --weight-h5 weights/categories_dr03_pt50/dr03_all/category_weights.h5 \
  --detector ee --track-types Lost,PF,GSF \
  --output-path compact_dr03_pt50/dr03_ee.h5
```

The compact stores raw crops (no tiling), labels, weights, split (80/10/10), per-object `pt`
(SC ET) and `a_mass`. Run once per detector (`eb`, `ee`).

### 4.3 Training

```bash
export IMAGE_TILE_SIZE=768
python3 train_multitask_cross_gun.py \
  --detector ee --compact compact_dr03_pt50/dr03_ee.h5 \
  --output-dir runs/dr03_v1/cross_pt50_ee_es \
  --track-types Lost,PF,GSF --backbone resnet18 \
  --epochs 50 --no-early-stopping --batch-size 8 --lr 1e-4 --lam 1.0
```

- `--no-es` drops the ES planes (the `ee_noes` configuration).
- `--lam` sets the regression weight λ.
- `--resume-eos-dir` pushes/pulls `last.pt` and `best_model.pt` each epoch (set automatically by `condor/run_job.sh`).

Outputs in the run directory: `best_model.pt`, `last.pt`, `history.json`, `test_metrics.json`,
`val_predictions.csv`, `test_predictions.csv`, ROC / score / efficiency-vs-pT plots and mass
regression plots.

### 4.4 Evaluation sample (20 GeV inclusive, 3 classes)

```bash
python3 build_category_weights_gun_eval.py --pt-min 20 \
  --output-path gunheavy_eval_sel.h5
python3 build_compact_dataset_gun.py --weight-h5 gunheavy_eval_sel.h5 \
  --detector ee --track-types Lost,PF,GSF --output-path gunheavy_eval_ee.h5
```

Classes: gun signal, H → AA signal (all mass points), W/Z background, all at SC ET ≥ 20 GeV.
Caps: gun 300k, background 400k, H → AA all.

### 4.5 Evaluation

```bash
python3 eval_gunheavy.py --detector ee --compact gunheavy_eval_ee.h5 \
  --run-dir runs/gunheavy_v1/cross_pt100_ee_es \
  --output-dir runs/gunheavy_v1_eval/cross_pt100_ee_es
```

See [section 6](#6-evaluation-and-plots).

---

## 5. Running on Condor

All heavy steps run on HTCondor; do not run long jobs on the login node (its GPU is shared).
Submit from the repository root.

| Step | Executable | Submit file |
|---|---|---|
| Weights | `condor/run_gunheavy_weights.sh`, `condor/run_dr03_weights.sh` | `gunheavy_weights.sub`, `dr03_weights.sub` |
| Compact | `condor/run_gunheavy_compact.sh`, `condor/run_dr03_compact.sh` | `gunheavy_compact{,_ee}.sub`, `dr03_compact.sub` |
| Training (GPU) | `condor/run_job.sh` | `gunheavy_train.sub`, `gunheavy_lambda.sub`, `dr03_train.sub` |
| Eval sample | `condor/run_gunheavy_eval_compact.sh` | `gunheavy_eval_compact.sub` |
| Evaluation (GPU) | `condor/run_gunheavy_eval.sh` | `gunheavy_eval.sub` |
| mreg particle-gun eval (GPU) | `condor/run_mreg_pgun_eval.sh` | `mreg_pgun_eval.sub` |

```bash
condor_submit condor/dr03_train.sub
```

Arguments come from the matching `condor/*.txt` file (one job per line).

`condor/run_job.sh` stages the compact to local scratch, rewrites `--output-dir` to local scratch,
adds `--resume-eos-dir`, and copies the outputs back to EOS with `xrdcp`.
`dr03_train.sub` sets `environment = "IMAGE_TILE_SIZE=768"`.

Resources that have worked:
- EB compact: 24 GB. EE compact: 64 GB (MiniAOD), 90 GB (MiniAOD_cropped) because of the ES planes.
- Training and evaluation: 1 GPU with `Capability >= 8.0`.

---

## 6. Evaluation and plots

`eval_gunheavy.py` writes, per run:

| File | Content |
|---|---|
| `eff_fake_9curve_20GeV` | 3 working points (background fake rate 2.5 / 5 / 10 %, one global threshold) × {background fake rate, gun efficiency, H → AA efficiency} vs SC ET, with error bars |
| `mass_pred_vs_true_linear_gun{,_0to10,_0to5}` | gun true vs predicted m_A, 0–60, 0–10, 0–5 GeV |
| `mass_pred_dist_signal` | H → AA predicted-mass distribution per nominal A mass |
| `score_distribution_3class` | gun / H → AA / background scores |
| `roc_log`, `roc_linear` | gun-vs-background and H → AA-vs-background ROC |
| `training_history` | loss, val AUC, val mass MAE per epoch |
| `eval_predictions.npz` | per-object scores and masses for re-drawing |

All training and evaluation plots carry the **CMS Private Work** label.

### Output format and re-drawing

Plots default to PNG. Re-draw from saved predictions without a GPU:

| Script | Re-draws | Source |
|---|---|---|
| `regen_gunheavy_eval_plots.py [--eps\|--pdf]` | evaluation plots | `eval_predictions.npz` |
| `regen_gunheavy_train_plots.py [--eps\|--pdf]` | training ROC / score / efficiency | `val/test_predictions.csv` |
| `regen_gunheavy_mass.py [--eps]` | training mass plots, 0–60 GeV | `test_predictions.csv` |
| `regen_mreg_eval.py [--eps] [--dpi N] <campaign>` | full mreg plot set | `val/test_predictions.csv` |
| `regen_pgun_plots.py [--eps]` | mreg particle-gun plots | `pgun_predictions.npz` |

The shared plot helpers honor a `PLOT_EXT` module variable (`.png` by default).

### Notes for reading the plots
- The training-time efficiency plot sets its threshold on the training test split background
  (SC ET ≥ training cut). The 20 GeV evaluation sets it on the inclusive background. The
  inclusive background contains low-ET W/Z that a pt100 model never saw, so its threshold is higher
  and every ET bin shows lower efficiency. At the same threshold the two agree.
- The `roc_test.png` written during training uses unit weights, so it matches `roc_test_unw.png`.
  The training loss itself is weighted.

---

## 7. Campaigns

| Campaign | Signal | Data | Cuts | Notes |
|---|---|---|---|---|
| `gunheavy_v1` | `particle_gun_heavy` | MiniAOD (32x32) | pt 50 / 100 / 200 × {eb, ee_es, ee_noes} | λ = 1.0, trained and evaluated |
| `gunheavy_v2`–`v5` | `particle_gun_heavy` | MiniAOD | pt 50 / 100 × {eb, ee_es, ee_noes} | λ scan: 0.5, 0.1, 0.05, 0.01 |
| `dr03_v1` | `HToEleEle_dr03` | MiniAOD_cropped (48x48) | pt 50 × {eb, ee_es, ee_noes} | `IMAGE_TILE_SIZE=768`, batch 8 |
| `mreg_v1`–`v5` | H → AA (discrete masses) | MiniAOD / AOD | pt 50 | earlier multi-task campaign (image, track, cross) |

Run outputs are in `/eos/user/y/yeo/4l/runs/<campaign>/`, evaluation in `runs/<campaign>_eval/`.
History of earlier versions: [`docs/run-versions.md`](docs/run-versions.md).

---

## 8. Environment

```bash
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export HDF5_USE_FILE_LOCKING=FALSE
export TORCHDYNAMO_DISABLE=1
```

The system `python3` is too old; always use the LCG view.

---

## 9. Known issues

- **Corrupt files in `MiniAOD_cropped/W` and `/Z`** (about 20 truncated or with broken HDF5 headers).
  The weight builders skip them. Regenerate them if full background statistics are needed.
- **EOS quota**: checkpoints are the largest consumer. Delete `last.pt` from finished runs if the
  quota is reached (`errno 122` / `[3021]`).
- **EE compact memory**: worker results can pile up in memory; raise `request_memory` or lower `--workers`.
- **Login-node GPU** is shared with other users; jobs there can be very slow. Use Condor.
- The evaluation sample builder (`build_category_weights_gun_eval.py`) still has `particle_gun_heavy`
  hardcoded as the gun class; it needs a signal-directory option before evaluating dr03.

---

## 10. Further documentation

| File | Content |
|---|---|
| [`docs/cross-attention-logic.md`](docs/cross-attention-logic.md) | Cross-attention model logic |
| [`docs/gunheavy-model-code-review.md`](docs/gunheavy-model-code-review.md) | Line-by-line model review |
| [`docs/gunheavy-pipeline-code-review.md`](docs/gunheavy-pipeline-code-review.md) | Line-by-line data pipeline review |
| [`docs/cross_attention_model.md`](docs/cross_attention_model.md) | Original single-task cross-attention classifier |
| [`docs/dataset-structure.md`](docs/dataset-structure.md) | Raw HDF5 schema |
| [`docs/PROJECT_HANDOFF.md`](docs/PROJECT_HANDOFF.md) | Project handoff notes |
| [`docs/run-versions.md`](docs/run-versions.md) | Run version history |
| [`docs/README_legacy.md`](docs/README_legacy.md) | Previous README |
