# Compact Training Dataset — Design (for review)

Date: 2026-07-02
Status: **DRAFT for review** — not implemented.

## Problem (evidence)

Training jobs read every object's input arrays from the **raw** HDF5 files on
EOS via the FUSE mount. With `shuffle=True` and a small file-handle cache, each
of ~1.5M objects/epoch triggers a file open of a scattered raw file → on the
order of **~1.5M FUSE opens per epoch**. FUSE open latency dominates:

- Track canary (tiny model) did **not** finish 1 epoch in 21 min; GPU ~0%,
  workers spawned but data-loading-bound.
- Measured on one raw file (272 MB): the training only needs
  **~4.1 KB/object (EB image)** or **~69.7 KB/object (EE image; ES planes
  16×512 dominate)**; track needs even less (sparse arrays only). The rest of
  each 272 MB raw file is unused datasets.

So staging the whole raw tier (~98 GB MiniAOD) is wasteful — most of it is never
read. The real issue is **open-thrash over FUSE**, and the fix is to read the
needed data from **one compact, local/sequential file** instead of scattered raw.

CERN batchdocs confirm FUSE is not recommended for batch I/O; recommended pattern
is xrootd stage-in (tar many files into one). There is no simple, reliable way to
stream HDF5 over xrootd (h5py file-like over xrootd is fragile/slow).

## Goal

Materialize, **once per category**, a single compact HDF5 holding only the arrays
each model needs, for only the selected (weight>0) objects, plus the full
per-object manifest. Training then:
- loads the manifest arrays (small), and
- reads each object's input arrays from this **one** file (staged to local
  scratch, or read sequentially),

eliminating both the startup raw scan and the per-object FUSE open-thrash. Built
once, reused across all 100 epochs and by both the image and track models.

## Design

### One compact file per `(tier, detector, hasADD)` → 12 files
ES only changes which image channels are *used*, not the data — so store the ES
planes always; the `--no-es` image variant simply ignores them. This dedups
(ES stored once) and lets **image (both ES on/off) and track share one file**.
- image categories (18) and track categories (12) all map onto these 12 files.

### Compact file contents (per file)
Flat, object-indexed datasets (row i = object i), in a fixed order:
- **manifest**: `file_idx, object_idx, event_idx, label, weight, pt, sample_code,
  split` (split ∈ {0=train,1=val,2=test}), `sample_id`, `raw_file` (strings).
- **image inputs**: `SC_energy` or `EE_seed_energy` (32×32); for EE also
  `ES_seed_plane1/2` when present.
- **track inputs**: the per-object sparse track arrays (`{DET}_track_pt_{TYPE}_idx/val`)
  for the tier's track types, stored as variable-length rows; plus precomputed
  `track_counts`, `track_sum_pt`.
- Attributes: tier, detector, hasADD, track_types, seed, split fractions,
  `effective_pt_bin_edges`, source weight-h5 path.

Selection + split are taken from the **category weight file** (weight_split>0 =
selected; `EB/EE_ele_event_idx` now stored there) + the existing event-level
split logic — so the compact builder does not re-derive physics, it just packs.

### Builder: `build_compact_dataset.py`
Per `(tier, detector, hasADD)`:
1. Load weight file (weights, event_idx, event_pt) → object selection + manifest
   + event-level split (reuse existing `build_event_registry`/`split`/`manifest`).
2. Iterate raw files **once each, sequentially** (parallelizable across files),
   extract the needed arrays for that file's selected objects, append to compact.
3. Write in manifest order; training shuffles indices in memory (no disk shuffle).

Reads each raw file exactly once (one-time cost, ≈ the old startup scan but done
a single time total, not 30× and not every epoch).

### Training changes
Add a `--compact <file>` data source:
- manifest = the compact's flat arrays (no raw scan, no per-file weight loop).
- `DetectorObjectDataset` / `TrackPointCloudDataset` read object i's input arrays
  from the compact file (indexed) instead of opening raw. `run_job.sh` optionally
  stages the compact file to local scratch first (one file → trivial).

### Sizes (measured KB/obj × ~selected objects)
- **track**: ~1–2 KB/obj → **a few GB/category**
- **EB image**: ~4 KB/obj → **~5 GB/category**
- **EE image, ES excluded**: ~5 KB/obj → few GB
- **EE image, ES included**: ~70 KB/obj → **~tens of GB** (ES planes are
  inherently large). Fits local scratch (worker `/` had 320 GB free), or stage.

## Build & storage
- Build on the login node with multiprocessing across files (overlap FUSE read
  latency), or as CPU condor jobs. One-time per category.
- Store compact files on EOS (small for most); training stages the single file
  to local scratch via xrdcp (fast, one file).

## Decisions (resolved 2026-07-02)
1. **Granularity: 12 files** per `(tier,det,hasADD)`, shared by image+track (ES
   stored always; `--no-es` ignores it). Measured footprint ~183 GB total vs
   ~202 GB for the 30-file scheme (12-file saves ~19 GB / ~10% by not duplicating
   calo+track into separate ES-off image files and separate track files).
2. **ES: keep full 16×512 resolution.** This is why EE is large: ES = plane1
   (16×512) + plane2 (512×16) = 16,384 float32 = **64 KB/object** (16× the 4 KB
   calo). EE-`all` ≈ 570k objects × ~69 KB ≈ **~39 GB/file**; 6 EE files ≈ 157 GB.
3. **Build once on the login node** (parallel across files), store on EOS; jobs
   load. Not per-job, not per-epoch.
4. **Path**: `/eos/user/y/yeo/4l/compact/<tier>_<det>_<hasadd>.h5`.

### Measured selected-object counts (weight>0), for sizing
| category | n_EB | n_EE | EB compact | EE compact (ES) |
|----------|-----:|-----:|-----------:|----------------:|
| MiniAOD_all | 1,348,012 | 569,700 | ~6.7 GB | ~39 GB |
| MiniAOD_eq0 | 1,157,969 | 530,032 | ~5.8 GB | ~37 GB |
| MiniAOD_eq1 |   180,722 |  34,590 | ~0.9 GB | ~2.4 GB |
| AOD_all     | 1,346,834 | 570,055 | ~6.7 GB | ~39 GB |
| AOD_eq0     | 1,137,880 | 528,215 | ~5.7 GB | ~36 GB |
| AOD_eq1     |   197,047 |  36,758 | ~1.0 GB | ~2.5 GB |

Per-job stage-in = one file (≤ ~39 GB) to local scratch (worker `/` had 320 GB
free). Total EOS footprint ~183 GB (confirm personal quota).

## Validation plan
Build ONE compact (e.g. MiniAOD·eb·all), run a 1-epoch canary reading it, and
**measure the actual 1-epoch wall time** + confirm stage-out/outputs. Only scale
out after the canary shows a fast epoch. (No full run before the compact path is
proven end-to-end.)
