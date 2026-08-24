# A-mass regression (multi-task) — design

Add A-boson mass regression to the merged-electron pipeline as a **multi-task**
extension of the existing classifiers, in **new files** (the classification code is
left untouched). Informed by the CMS end-to-end (E2E) A→ee mass-regression reference
the user supplied (used only for the regression mechanics: target scaling, MAE/MRE
metrics, predicted-mass histograms).

## 1. Goal & physics

Predict the pseudoscalar **A mass** for a merged electron (A→ee reconstructed as one
supercluster). A→ee is boosted, opening angle ΔR ≈ 2·m_A/pT(A); the merged shower
width + track separation encode ΔR, so the network can infer m_A from the same inputs
used for classification. Regress **continuous log(m_A)** (training masses are the
discrete set {0.4, 1, 2, 5, 10} GeV × H∈{250,750,2000}; "continuous" therefore
interpolates between these 5 points).

**No explicit pT input.** The `pt` column in the compacts is the *gen* A-boson pT —
feeding it would be a truth leak (unavailable at inference on data). The model already
sees measured energy/momentum (calo image ≈ cluster ET; per-track pT in the point
cloud), so it can learn the boost scale itself. No pT scalar is added.

## 2. Task framing — multi-task

One shared backbone per model family, **two heads**:
- **classification head** — sig/bkg logit, `BCE` (unchanged from the current models).
- **regression head** — one scalar `μ = predicted log(m_A)`, `MAE (L1)` loss, **active
  on signal objects only** (background is masked out of the mass loss; background has no A).

Total loss: `L = BCE(cls) + λ · MAE(μ, log m_A) · [is_signal]`, with `λ` a tunable
weight (start λ=1, tune on val). Target transform: `y = log(m_A)` (natural log),
inverse `m̂ = exp(μ)` for reporting. (Reference used linear `m/1.1`; we use log because
our range spans ~1.4 decades.)

## 3. Model families — all three, new files

Reuse the existing backbones by importing them; wrap each with a 2-head module. The
classification code is NOT modified.
- `train_multitask_image.py`  — ResNet backbone (`build_resnet`), calo image input.
- `train_multitask_track.py`  — point-cloud transformer backbone.
- `train_multitask_cross.py`  — cross-attention fusion backbone.

Each: take the backbone's pooled feature vector → `cls_head` (Linear→1) and
`reg_head` (small MLP→1). fp32 + TF32 (same as the current trainers). A shared
`multitask_common.py` may hold the 2-head wrapper, the masked loss, and the
regression metrics/plots to avoid duplication across the three trainers.

## 4. Data — reuse compact_pt50 (no rebuild)

- Input: `compact_pt50/{TIER}_{det}_{mode}.h5` (signal + 1M background), read by the
  existing dataset classes.
- **Mass target derived at load time** from `sample_code → sample_id → m_A` (parse the
  sample name: `..._A0p4`→0.4, `_A1`→1, `_A2`→2, `_A5`→5, `_A10`→10). Background rows
  get a sentinel mass and `is_signal=0` so they are excluded from the mass loss. **No
  compact change / rebuild needed.**
- Categories: mirror the pt50 layout (AOD/MiniAOD × eb/ee{es,noes for image} ×
  all/eq0/eq1). Note: **eq0** objects have no extra track → mass must come from shower
  width alone (weakest case); **eq1/all** carry track-separation info.

## 5. Metrics & plots (per run)

Keep the classification outputs (AUC/ROC/etc.) so the multi-task model is still a
usable classifier. Add regression outputs:
- **MAE** and **MRE** = mean |m̂−m|/m (from the reference), overall and per mass point.
- **Resolution** σ((m̂−m)/m) and **bias** ⟨(m̂−m)/m⟩ vs true mass point, and vs pT.
- **Predicted-mass histogram** per true mass point (with a line at the true mass) —
  the reference's plot, one per mass point (or overlaid).
- **pred-vs-true** 2D / profile of m̂ vs m_A.
- `test_predictions.csv` gains `m_true`, `m_pred` columns for re-plotting.

## 6. Training infra

Condor, same pattern as the classifiers (fp32/TF32, `run_job.sh`, `+JobFlavour`,
`require_gpus`). New submit files under `condor/` (e.g. `mreg_image.{txt,sub}`,
`mreg_track.*`, `mreg_cross.*`) → `runs/mreg_v1/...`. Scope of first build: the pt50
categories across the three families (~48 runs), mirroring v8+cross_v2.

## 7. Honest limitations / risks

- **5 discrete training masses** → the "regression" interpolates between 5 points;
  report **per-mass-point** resolution, not just a global number.
- **Low mass (0.4 GeV)**: ΔR is tiny → likely near-irresolvable; expect large
  resolution/bias there. Report it honestly rather than hiding in an average.
- **eq0** mass (image-only, no track separation) will be the weakest category.
- **Multi-task balance**: `λ` needs tuning; the mass loss must be correctly masked to
  signal (a common bug source — verify the mask).

## 8. Out of scope (possible later)

- Impact-parameter channels (dz/d0 significance) as in the reference — our raw has
  `track_max_dxy/dz` but they are not currently in the compacts.
- Position aux inputs (iphi/ieta) — available (`SC_ieta/iphi`) but not used now.
- **Gaussian-NLL** per-object uncertainty head — a v2 upgrade if per-object σ is wanted.
- New lower-cut (pt20 / no-cut) training data for a wider ΔR range.

## 9. Open item for the user

Categories scope: this design mirrors **all** pt50 categories (all/eq0/eq1). If you'd
rather focus the first pass on the mass-informative ones (**all** and **eq1**, skipping
the image-only **eq0**), say so and the run list shrinks accordingly.
