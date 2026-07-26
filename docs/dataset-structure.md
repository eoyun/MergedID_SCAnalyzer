# Dataset structure — context for modifying the dataset-building code

Paste this whole block as context when asking to change how the training dataset
is built. It describes the 3-stage pipeline, the exact HDF5 schemas, and the
conventions the reader code relies on. Repo:
`/afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer`.

## Task domain
Merged-electron ID for HToAATo4L (H→AA→4ℓ). Binary classification of one **reco
electron / supercluster (SC)** at a time: **signal** = gen-matched electron from
the A→ee decay, **background** = fake electron in W/Z events. Two detector regions
are trained separately: **EB** (barrel) and **EE** (endcap). Models: `image`
(ResNet on calo images), `track` (transformer over the electron's track set),
`fusion`/`cross` (both).

## Pipeline: 3 stages
```
raw *.h5  ──build_category_weights.py──►  category_weights.h5  ──build_compact_dataset.py──►  compact_*.h5
(per-event,     (selection + train/val/test          (flat per-object arrays,
 per-object)     split + per-object weight)            the actual model input)
```
Driver scripts: `run_build_category_weights.sh` (→ weights), `run_build_compact.sh`
(→ compacts). Readers: `train_resnet_image_classifier.py` (image),
`train_point_transformer_track_classifier.py` (track), `train_fusion_*.py`.

Data roots on EOS:
- raw: `/eos/user/y/yeo/4l/data/{AOD,MiniAOD}/{signal,W,Z}/*.h5`
- weights: `/eos/user/y/yeo/4l/weights/categories[_pt50|_pt200]/{TIER}_{mode}/category_weights.h5`
- compacts: `/eos/user/y/yeo/4l/compact[_pt50|_pt200]/{TIER}_{det}_{mode}.h5`

`TIER ∈ {AOD, MiniAOD}`, `det ∈ {eb, ee}`, `mode ∈ {all, eq0, eq1}` (hasADD mode).

## The per-object `pt` (IMPORTANT — it is NOT the electron pT)
`pt` = **event-level generator A-boson pT**, broadcast to every object in that
event. Computed by `derive_event_pt(n_events, A_pT, A_event_idx, mode="max")`
(`build_event_level_weights.py:258`): for each event take the **max `A_pT`** over
the A particles in it. Every SC in that event gets that event's value. So:
- the `--pt-min` preselection cut is on this event A-pT,
- the efficiency-vs-pT plots and the `effective_pt_bin_edges` are on this value,
- it is the same for all objects sharing an event (not a per-electron quantity).

## hasADD modes
Each object has `hasAdditionalTrk_{EB|EE}` ∈ {0,1} (has ≥1 extra track besides the
seed). Modes select which objects enter a category:
- `all` — both, `eq0` — only hasADD==0, `eq1` — only hasADD==1.
For the **track** model, hasADD==0 means an (almost) empty track set — this is the
source of the earlier nan-score bug (empty point cloud → nan logit).

Approach-b signal mass-point exclusions (hard-coded in the selection): `eq1`
excludes A0p4/A1; `eq0` excludes H250_A10.

## Background sampling (object-level, per detector, per mode)
`background_sampling = per_detector_per_mode_object`. For each (detector, mode):
**signal = ALL** eligible objects; **background = a random subsample of up to
`--background-max-events` (default 1,000,000)** from the full eligible background
pool (all of it if fewer, e.g. eq1). EB and EE are sampled independently (1M each).
Eligible = hasADD matches mode AND (pt_min≤0 OR event A-pT ≥ pt_min) AND not an
excluded mass point.

---

## Stage 1 — RAW `*.h5` (input, read-only; already slimmed)
Per file: many events; per detector, `N_{EB|EE}` objects. First axis = object index
within file (unless noted). Prefix `EB`/`EE` picks the region. `TIER` decides which
track collections exist: **AOD → `GenTrk`**, **MiniAOD → `Lost,PF,GSF`**.

Calo images (float32, lzf):
- EB: `SC_energy` (N, 32, 32) — cropped SC ECAL energy.
- EE: `EE_seed_energy` (N, 32, 32) **plus** preshower `ES_seed_plane1_energy`
  (N, 16, 512) and `ES_seed_plane2_energy` (N, 512, 16).
- (unused variants also present: `*_energyT`, `*_energyZ`, `SC_time`, sums.)

Tracks (variable-length; stored as HDF5 object/vlen arrays, one 1-D array per object):
- `{EB|EE}_track_pt_{GenTrk|PF|GSF}_idx` and `..._val` (N,). `_val[j]` = that
  object's track pT list; `_idx[j]` = matching bin indices.
- also `_n_`, `_qpt_`, `_max_dxy_`, `_max_dz_` (idx/val) — mostly unused by current models.

Bookkeeping:
- `hasAdditionalTrk_{EB|EE}` (N,) — 0/1 (see modes above).
- `{EB|EE}_ele_event_idx` (N,) — object → event index in this file.
- `A_pT`, `A_eta`, `A_event_idx`, `A_mass`, ... (per gen-A arrays).
- per-event: `eventId`, `runId`, `lumiId`, `mHgen`, `numRecoEles`, `numGenEles`.
- file attrs: `sample_H_mass`, `sample_A_mass` (signal only), `sc_crop_size=32`, etc.

Key helpers: `raw_source_keys(detector, track_types)` returns
`{calo:[SC_energy|EE_seed_energy], es:[]|[ES_seed_plane1,ES_seed_plane2],
track:[{PREFIX}_track_pt_{t} ...]}`; `calo_key(detector)` returns the calo dataset name.

## Stage 2 — `category_weights.h5` (selection + split + weight)
Per-file groups: `files/{sample}/{stem}/` with:
- `EB_ele_event_idx` (n_sel,), `EB_weight_split` (n_sel,) — selected EB object
  indices and their **split-encoded weight** (sign/magnitude encodes train/val/test
  membership + the sample weight); same for `EE_*`; `event_pt` (n_events,).
File attrs: `hasadd_mode`, `pt_min`, `background_sampling`, `background_max_objects`,
`background_selected_objects_{eb,ee}`, `weight_basis=object`, `seed`,
`requested_pt_bin_edges`, `effective_pt_bin_edges` (the requested edges + overflow).
`PT_BIN_EDGES = [0,20,30,50,60,70,80,100,125,150,200,250,300,350,400,500,600,...]`.

## Stage 3 — `compact_{TIER}_{det}_{mode}.h5` (the model input)
One flat row per selected object (`n` rows), gzip-compressed. Built by
`build_compact_dataset.py`. Common datasets (all length `n`):
`label` (i8, 1=signal/0=bkg), `weight` (f4), `pt` (f4, the event A-pT above),
`split` (i1: train/val/test), `file_idx` (i4), `object_idx` (i4), `event_idx` (i4),
`sample_code` (i2), `sample_id` (vlen str), `raw_file` (vlen str).

Region-specific payload:
- **EB compact**: `SC_energy` (n, 32, 32) + tracks `{EB}_track_pt_{types}_idx/_val` (vlen).
- **EE compact**: `EE_seed_energy` (n, 32, 32) + `ES_seed_plane1_energy` (n, 16, 512)
  + `ES_seed_plane2_energy` (n, 512, 16) + tracks `{EE}_track_pt_{types}_idx/_val` (vlen).

File attrs: `detector`, `hasadd_mode`, `track_types`, `seed`, `train_frac=0.8`,
`val_frac=0.1`, `requested_pt_bin_edges`, `effective_pt_bin_edges`.

Splits: train/val/test = 0.8/0.1/0.1, derived at the **event** level (all objects of
an event stay in the same split) with `seed=42`, encoded in `split`.

## How readers consume the compact
- image model: reads the calo image(s); `--no-track-channels` = calo-only (v6+);
  `--no-es` (EE) drops the two ES planes; preprocessing = `log1p(clip≥0)` then tile
  32×32 → 512×512 (`repeat_to_512`); v7 `--normalize-channels` adds per-channel [0,1].
- track model: reads `{PREFIX}_track_pt_{types}_val` as an unordered point set
  (up to `--max-points`), feeds a Transformer encoder classifier. Empty set
  (hasADD==0) → must be handled to avoid nan.
- All models select the calo/track keys via `--detector` and `--track-types`, and
  use `pt` + `effective_pt_bin_edges` for the efficiency-vs-pT outputs.

## When you modify the dataset build, keep these invariants
1. Object-level rows; `label/weight/pt/split/file_idx/object_idx/event_idx/
   sample_code/sample_id/raw_file` present and length-`n` consistent.
2. `pt` stays the event A-pT (unless the change is explicitly about redefining pt) so
   the pt-cut and efficiency binning stay meaningful; keep `effective_pt_bin_edges`.
3. Splits stay event-level (no leakage of an event across splits), `seed=42`.
4. Per-(detector, mode): signal = all, background = ≤`background_max_events` random.
5. Region payload keys unchanged (`SC_energy` for EB; `EE_seed_energy` +
   two `ES_seed_plane*` for EE) unless the change is about the images themselves —
   reader code keys off these exact names.
6. Trace every downstream reader (image/track/fusion trainers) after a schema change.
