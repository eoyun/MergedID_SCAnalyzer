# v7 / v8 changes vs v6

All changes are **additive**: new optional flags whose **defaults reproduce v6
exactly**. No v6 code path was deleted or replaced — running without the new
flags gives identical v6 behaviour. This file records what each version adds so
v6 stays reproducible.

Status legend: **[done-code]** code written + unit-verified, not yet submitted ·
**[planned]** designed, not yet implemented.

---

## v6 (baseline, for reference)
- image = calo (+ES for EE) only, track-pT channels dropped (`--no-track-channels`)
- channel preprocessing = `log1p(clip(x,0))` only (no per-channel scaling)
- object selection = hasADD split + mass-point exclusion + bg subsample (1M); **no pt cut**
- epochs 50, `--no-early-stopping`, checkpoint resume
- RUNS_ROOT = `/eos/user/y/yeo/4l/runs/v6`

---

## v7 — ES effect study via per-channel [0,1] normalization  **[done-code]**

**Goal:** equalize channel scales so ES (raw ~0.05, log1p ~0.05) is not dominated
by calo (log1p ~7.5), then measure whether ES actually helps.

**Code change (additive):**
- `preprocess_channel(arr, log_scale, normalize01=False)` — new `normalize01` arg.
  When True: after log1p, per-image per-channel min-max to [0,1]; an all-zero
  channel stays 0 (no divide-by-zero). Default False = **v6 behaviour**.
- `DetectorObjectDataset(..., normalize_channels=False)` — new arg, threaded to
  `_build_image`.
- `train_resnet_image_classifier.py` new flag `--normalize-channels`; stored in
  `run_config.json` and `best_model.pt` (`normalize_channels`).
- `train_fusion_ensemble.py` rescore path reads `normalize_channels` from the
  image run_config (kept consistent; unused in csv mode).

Verified: v6 default max=7.508 (log1p only); v7 max=1.000; ES v6 max 0.049 -> v7 max 1.0.

**Scope (4 jobs):** EE image only, `{es, noes} x {AOD, MiniAOD}`, hasADD = `all`,
calo(+ES) only, resnet50, epochs 50, `--no-early-stopping`, `--normalize-channels`.
- **Reuses v6 compacts** (`AOD_ee_all`, `MiniAOD_ee_all`) — no new weights/compacts.
- RUNS_ROOT = `.../runs/v7`. EB excluded (no ES).
- Compare es-vs-noes test AUC to judge ES contribution.

**Not yet done:** categories/submit wiring, submission.

---

## v8 — pT >= 50 GeV cut  **[planned]**

**Goal:** same as v6 but object selection adds `event_pt >= 50 GeV`.

**Code changes (additive, planned):**
- `build_category_weights.py`: new `--pt-min` (default 0). Object mask gains
  `event_pt[det_event_idx] >= pt_min`.
- `build_event_level_weights.py`: background 1M sampling restricted to events with
  `event_pt >= pt_min` (sampling pool filtered before the draw). *(the non-trivial
  part — bg selection is event-level and must become pt-aware.)*

**Pipeline (planned):**
1. rebuild weights (6) with `--pt-min 50` -> new dir `weights/categories_pt50/`
2. rebuild compacts (12) from those -> new dir `compact_pt50/` (pt>=50 subset, smaller)
3. train v8 = same 18 image + 12 track as v6 (calo-only, log1p, no normalization),
   RUNS_ROOT = `.../runs/v8`

**Not yet done:** all of the above (code + rebuild + submit).

---

## EOS note
data/ = 722 GB is the bulk (immovable); compact 6.4, weights 2.1, all runs ~17 GB.
Headroom ~70 GB (logical). v7 ~0 (reuses compacts); v8 ~11 GB (weights+compacts+runs)
-> both fit without deletions. Reclaimable buffer if needed: v5 (5 GB) + test dirs (~1 GB).
