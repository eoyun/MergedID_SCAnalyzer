# 18-way Condor Training Pipeline — Design

Date: 2026-06-24

## Goal

Train the image + track + fusion stack across **18 categories**, on GPU via
HTCondor, with stage-by-stage submission. The 18 categories come from these
axes:

- **tier**: AOD, MiniAOD
- **detector**: EB, EE
- **ES** (EE only): ES image channels included vs excluded — an *input* choice,
  not a data split
- **hasAdditionalTrk** (per-object 0/1 flag): `all` (union), `eq0`, `eq1`

Counts:

| stage  | axes                                   | count |
|--------|----------------------------------------|-------|
| image  | tier × det × (EE: ES) × hasADD         | 18    |
| track  | tier × det × hasADD  (**ES-independent**) | 12 |
| fusion | one per image variant (shares track)   | 18    |

EB = `tier(2) × hasADD(3) = 6`; EE = `tier(2) × ES(2) × hasADD(3) = 12`; → 18.
ES never enters the track point cloud, so each EE `ES-on`/`ES-off` pair shares
one trained track model → 12 track trainings, not 18.

## Confirmed data / environment facts

- Data root: `/eos/user/y/yeo/4l/data/{AOD,MiniAOD}/{W,Z,signal}`.
  AOD: W=3619, Z=4827, signal=1500. MiniAOD: W=608, Z=1341, signal=1500.
  15 signal mass points: `H{250,750,2000} × A{0p4,1,2,5,10}`.
- Track schema differs by tier: **AOD = GenTrk** (GSF/PF dropped per decision),
  **MiniAOD = GSF, PF, Lost**. Both tiers also carry GSF/PF physically.
- `hasAdditionalTrk_EB` / `hasAdditionalTrk_EE`: per-object 0/1 (value 1 is rare).
- `ES_seed_plane1/2_energy` etc.: preshower images, EE only.
- Interpreter: cvmfs `LCG_109_cuda` el9 view
  (`/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh`) —
  torch 2.10 + CUDA + h5py + sklearn. Source with `set +u` (it references unbound
  vars). System `python3` is too old (lacks `int | None`).
- Condor pool has abundant GPUs (A100 / H100 / H200 / V100 / T4).
- Baseline tier-level weights already built at
  `/eos/user/y/yeo/4l/weights/{AOD,MiniAOD}/event_level_weights.h5`
  (event/EB/EE balanced to 1.0). Per-category weights are a separate sub-project.

## Decisions

- **Orchestration**: stage-by-stage Condor submit scripts (not DAGMan).
  Stage 1 submits all track (12) + image (18) jobs; stage 2 submits the 18
  fusion jobs after stage 1 completes.
- **Weights**: built by a **separate prerequisite sub-project** (hasADD filter +
  ~1M resampling → 12 weight files keyed by tier×det×hasADD, ES-independent).
  The training pipeline only *consumes* per-category weight files.
- **Output root**: `/eos/user/y/yeo/4l/runs`, with per-category subdirs.
- **Interpreter**: the cvmfs LCG view above, sourced inside each Condor job.

## Decomposition (build order)

### Sub-project 1 — Per-category weights (prerequisite, separate)
Build **12 object-level balanced weight files** keyed by
`tier × detector × hasADD` (ES-independent). This requires moving the existing
event-based balancing to an **object-based** scheme (because `hasAdditionalTrk`
is a per-object flag) with a per-object `hasAdditionalTrk` filter
(`all | eq0 | eq1`) and ~1M **background** resampling (signal is the scarce class
— used in full; EE signal totals only ~99k).

Balancing (per category): background class sum = 1, signal class sum = 1, with
the signal total split **equally among the included mass points**. Object pT
binning keeps the existing equal-weight-per-nonempty-bin rule.

**Per-category signal mass-point inclusion** (decision 2026-06-24, approach b):

| hasADD | included signal mass points | count |
|--------|------------------------------|-------|
| `all`  | all (`A{0p4,1,2,5,10} × H{250,750,2000}`) | 15 |
| `eq1`  | exclude `A0p4`, `A1` → `A{2,5,10} × H{250,750,2000}` | 9 |
| `eq0`  | exclude `H250_A10` | 14 |

Rationale: `hasADD==1` selects resolved (wide-opening-angle) topologies, so the
merged points `A0p4/A1` barely populate `eq1` (single/double-digit object counts
in some EE mass points). `hasADD==0` represents merged topology, so the maximally
resolved `H250_A10` is dropped. Excluded mass points are removed entirely from
that category's training set and from its balancing. Background (no mass points)
is always included.

FYI noted: in `eq0`, `H2000_A10` / `H750_A10` (EE) actually have smaller object
counts than `H250_A10`; only `H250_A10` is excluded per decision (physics
intent, not pure statistics).

This sub-project is tracked separately and built first. Until ready, training
can fall back to the baseline tier weights + train-time
`rebalance_manifest_class_weights`.

Signal hasADD==1 fractions are nearly identical between AOD and MiniAOD (the
MiniAOD hasADD==1 reduction is a *background-only* effect: soft additional tracks
are pruned from `lostTracks`, while signal's hard second-electron track survives
in both tiers). Confirmed via track-multiplicity: AOD `GenTrk` covers 98.6% of
EB objects (~1.05 trk/obj), MiniAOD `lostTracks` only 7.1% (~0.08 trk/obj), both
with the same 5 GeV pT floor.

### Sub-project 2 — Script parameterization (code changes)
Add the new axes as explicit options so one script body serves all categories:

1. **Track types, tier-aware** (`track_point_transformer.py`):
   replace the hardcoded `TRACK_TYPES = ("GSF","PF","Lost")` and
   `POINT_FEATURE_DIM = 6` with a `--track-types` option
   (`gsf,pf,lost` | `gentrk`); `POINT_FEATURE_DIM = 3 + len(track_types)`
   (x, y, pT + per-type one-hot). Persist the chosen types in the checkpoint so
   fusion rebuilds the model correctly.
2. **Image channels, tier/ES-aware** (`train_resnet_image_classifier.py`):
   derive the channel list from `(tier, detector, es_flag)` instead of the
   hardcoded `[SC,lost,pf,gsf]` / `[EE_seed,es1,es2,lost,pf,gsf]`. AOD uses the
   single GenTrk track channel; `es_flag=off` drops `es1,es2`. `in_channels` is
   computed from the channel list and stored in the checkpoint.
   Channel sets:
   - AOD·EB: `[SC_energy, GenTrk]`
   - AOD·EE·ES: `[EE_seed, es1, es2, GenTrk]`
   - AOD·EE·noES: `[EE_seed, GenTrk]`
   - Mini·EB: `[SC_energy, Lost, PF, GSF]`
   - Mini·EE·ES: `[EE_seed, es1, es2, Lost, PF, GSF]`
   - Mini·EE·noES: `[EE_seed, Lost, PF, GSF]`
3. **hasADD selection** (`build_object_manifest`, shared by image/track/fusion):
   add `--hasadd-mode all|eq0|eq1`; apply `object_mask &= hasadd_mask` where
   `hasadd_mask` comes from `raw["hasAdditionalTrk_{EB|EE}"][:] (== 1 | == 0)`.
   Single change point propagates to all consumers.
4. **Fusion, tier-aware** (`train_fusion_ensemble.py`):
   - `build_fusion_features` / CSV columns currently hardcode
     `n_gsf,n_pf,n_lost,sum_pt_*`; make them derive from the track run's
     persisted track-type list (AOD → `n_gentrk,sum_pt_gentrk`).
   - `validate_compatible_runs` must ignore the `es` flag (track runs have none)
     while still enforcing detector/seed/splits/weight-sidecar compatibility, so
     an EE ES-on and EE ES-off fusion can both consume the shared track run.

### Sub-project 3 — Category config + Condor submit + results
1. **`pipeline/categories.py`** — single source of truth. Enumerates the 48 job
   specs (18 image, 12 track, 18 fusion) with: tier, detector, es, hasADD,
   track-types, channel list, weight-h5 path, output dir, and (for fusion) the
   image-run + shared-track-run it depends on. Naming:
   - image/fusion: `{tier}_{det}[_es|_noes]_{hasadd}`
     (e.g. `aod_eb_all`, `mini_ee_es_eq1`, `aod_ee_noes_eq0`)
   - track: `{tier}_{det}_{hasadd}` (e.g. `aod_ee_all`)
2. **`pipeline/make_submit.py`** — generates Condor submit files from the config
   using `queue ... from <list>` to fan out a stage in one cluster. Each job:
   `request_gpus=1`, sources the LCG view, `transfer_input_files` the repo
   scripts, reads data from EOS, writes runs to `/eos/user/y/yeo/4l/runs`.
3. **`pipeline/submit_train.sh`** — submit stage 1 (12 track + 18 image).
   **`pipeline/submit_fusion.sh`** — submit stage 2 (18 fusion) once stage 1 is
   done.
4. **`pipeline/collect_results.py`** — gather the 18 fusion `test_metrics.json`
   (weighted AUC / F1) into one summary table.

## Output layout

```
/eos/user/y/yeo/4l/runs/
  image_<cat>/    # 18
  track_<cat>/    # 12
  fusion_<cat>/   # 18
```

## Open items / risks

- `hasADD=eq1` subsets are very small (value 1 is rare; AOD·EE had 0 in a sampled
  file). Those models may train on little data — accepted per decision.
- Per-category weight balancing semantics (sub-project 1) need care: hasADD
  filtering changes the object population, so weights must be rebuilt on the
  filtered set (or rely on train-time rebalance as a fallback).
- Fusion track-feature dimensionality and the prediction-CSV columns differ
  between AOD (1 track type) and MiniAOD (3) — must be data-driven from the
  persisted track-type list, not hardcoded.
