# Project handoff — Merged-electron ID (HToAATo4L)

Self-contained orientation for someone (or a fresh Claude) picking this up cold.
Read [`dataset-structure.md`](dataset-structure.md) and
[`cross_attention_model.md`](cross_attention_model.md) alongside this file — they
cover the data schema and the cross-attention model in depth; this file covers the
whole pipeline, the campaigns, the infra, and the current state.

- **Repo**: `/afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer`, git branch `lxplus`.
- **EOS data root**: `/eos/user/y/yeo/4l/` (1 TB *logical* quota; `eos quota` "logi bytes" is the real usage; raw data uses lzf+float32).
- **Compute**: CERN lxplus + HTCondor GPU. Env = cvmfs LCG view
  `source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh`
  (source with `set +u`; `export HDF5_USE_FILE_LOCKING=FALSE`). System python3 is too old — always use the LCG view.

---

## 1. The physics task

Binary classification of **one reco electron / supercluster (SC) at a time**:
- **signal** = a genuine *merged electron* (a boosted A→ee reconstructed as one object) in H→AA→4ℓ,
- **background** = a fake electron in W/Z events.

Barrel (**EB**) and endcap (**EE**) are trained as **separate** models (different inputs/geometry).

`hasADD` = whether the object has ≥1 extra track. Categories split on it:
`all` (both), `eq0` (hasADD==0), `eq1` (hasADD==1). Approach-b mass-point exclusions:
`eq1` drops A0p4/A1; `eq0` drops H250_A10.

**Important — the per-object `pt`**: it is NOT the electron pT. It is the **event-level
generator A-boson pT** (`derive_event_pt(A_pT, A_event_idx, mode="max")` = max A_pT in
the event), broadcast to every object in that event. The `--pt-min` training cuts and
the efficiency-vs-pT axes are all on this value. (Merged-electron signal is intrinsically
high-pT/boosted, so there is essentially no signal below ~40 GeV — a physical fact, not a bug.)

---

## 2. Data pipeline (3 stages)

```
raw *.h5  ──build_category_weights.py──►  category_weights.h5  ──build_compact_dataset.py──►  compact_*.h5
(per-event, per-object)   (object selection + train/val/test          (flat per-object arrays,
                           split + per-object weight)                    the actual model input)
```

- **raw**: `/eos/user/y/yeo/4l/data/{AOD,MiniAOD}/{signal,W,Z}/*.h5`. AOD track collection =
  `GenTrk`; MiniAOD = `Lost,PF,GSF` (AOD tracks are ~5× denser → hasADD==1 far more common).
- **weights**: `/eos/user/y/yeo/4l/weights/categories[_pt50|_pt100|_pt200]/{TIER}_{mode}/category_weights.h5`
- **compacts**: `/eos/user/y/yeo/4l/compact[_pt50|_pt100|_pt200]/{TIER}_{det}_{mode}.h5`

`TIER∈{AOD,MiniAOD}`, `det∈{eb,ee}`, `mode∈{all,eq0,eq1}`.

**Background sampling** (object-level, per detector, per mode): signal = ALL eligible;
background = up to `--background-max-events` (default 1,000,000) random from the eligible
(pt≥pt_min) pool, EB and EE independently. Splits are event-level (no leakage), 0.8/0.1/0.1, seed 42.

The compact schema (image + tracks + `label/weight/pt/split/...`) is in `dataset-structure.md`.
Object selection in the weight file is encoded as `{EB,EE}_weight_split > 0`.

### Builder scripts
- `build_category_weights.py --input-dir <tier> --output-path <...> --hasadd-mode <mode> --pt-min <N>`
- `build_compact_dataset.py --weight-h5 <...> --detector <eb|ee> --track-types <...> --output-path <...> [--workers N]`
- login-node drivers: `run_build_category_weights.sh`, `run_build_compact.sh`, `build_pt200_data.sh` (a pt-cut end-to-end driver).
- **Prefer Condor for the build now** (see §5): `condor/run_train_build.sh` + `condor/build_pt100.{txt,sub}`.

---

## 3. Models & trainers

Three families, all per (tier, det, mode); each trainer reads a compact and writes plots+checkpoints:
- **image** — ResNet on the calo image. `train_resnet_image_classifier.py`. Flags: `--no-track-channels`
  (calo-only, v6+), `--no-es` (EE: drop preshower), `--model resnet50`, `--normalize-channels` (v7 experiment).
- **track** — Transformer over the track point cloud. `train_point_transformer_track_classifier.py`
  (+ `track_point_transformer.py`). Each track → point `[x,y,log(1+pT),one-hot(collection)]`.
- **cross** — bidirectional cross-attention fusion of image+tracks. `train_fusion_cross_attention.py`.
  See `cross_attention_model.md`.
- also `train_fusion_concat.py`, `train_fusion_ensemble.py` (GBDT over frozen image+track), `train_fusion_gbdt.py`.

`train_resnet_image_classifier.py` is the **shared hub**: the other trainers import its dataset,
manifest, metric, and plotting helpers (`build_object_manifest`, `save_roc_plot`,
`save_efficiency_plot`, `save_unweighted_test_plots`, `save_eval_sample_plots`, `safe_roc_auc`, …).

### Per-run outputs (in each `runs/.../<run>/` dir)
`best_model.pt`, `last.pt`, `run_config.json`, `test_metrics.json`, `history.json`,
`{val,test}_predictions.csv`, `training_history.png`, and plots:
- weighted: `roc_test.png`, `score_distribution_test.png`, `efficiency_vs_pt_test.png`
- **unweighted** (auto-emitted at end of training via `save_unweighted_test_plots`):
  `roc_test_unw.png`, `score_distribution_test_unw.png` (area-normalized, y="a.u."),
  `efficiency_vs_pt_test_bkg{2p5,05,10}_unw.png/.json` — efficiency at background
  **fake-rate working points 2.5/5/10%** (threshold chosen on validation).

---

## 4. Campaigns (pT-cut versions) & the 20 GeV eval

| version | pT cut | data | run dir | status |
|---|---|---|---|---|
| **v8** (image+track) | ≥50 | categories_pt50 / compact_pt50 | `runs/v8` | done (30/30) |
| **cross_v2** (cross) | ≥50 | " | `runs/cross_v2` | done (18/18) |
| **v9** (image+track) | ≥200 | categories_pt200 / compact_pt200 | `runs/v9` | done (30/30) |
| **cross_v3** (cross) | ≥200 | " | `runs/cross_v3` | done (18/18) |
| **v10** (image+track) | ≥100 | categories_pt100 / compact_pt100 | `runs/v10` | training submitted |
| **cross_v4** (cross) | ≥100 | " | `runs/cross_v4` | training submitted |

(v5–v7 were earlier image experiments; v6 = calo-only baseline, v7 = per-channel [0,1] normalization.)

Each version = 30 image+track runs (image: AOD/Mini × eb/ee{es,noes} × all/eq0/eq1 = 18; track:
AOD/Mini × eb/ee × all/eq0/eq1 = 12) + 18 cross runs. So per pT cut there are **48 runs**; three cuts = **144** (96 done, 48 in flight for pt100).

### The 20 GeV low-cut evaluation (cross-cut generalization test)
Score the ALREADY-TRAINED models on a fresh **pT≥20** sample to see how they behave below
their training cut (mainly probes low-pT **background fake-rate**; signal is intrinsically high-pT).
- **Eval sample**: `build_eval_sample.py` → `weights/eval_20gev`, `compact_eval_20gev` (12 compacts,
  shared by all versions). signal = 20% of ALL signal (no pT cut); background = 20% of min(1e6, pT≥20 pool).
- **Scoring**: each trainer has `--eval-only` (+ `--eval-suffix _20GeV_sample`): loads `best_model.pt`,
  scores the ENTIRE eval compact, writes unweighted ROC/score/efficiency(WP 2.5/5/10%) with the
  `_20GeV_sample` suffix; the WP threshold is taken on **this sample's own background**. `save_eval_sample_plots`.
- **Combined overlay**: `combine_wp_plots.py` reads the 3 WP JSONs per run and writes 2 overlays:
  `efficiency_vs_pt_signal_allWP_unw_20GeV_sample.png` and `..._background_allWP_...png`.
- **Status**: v8/v9/cross_v2/cross_v3 (96 sets) fully eval'd + combined plots done.
  v10/cross_v4 eval configs ready (`condor/eval20_v10_*`, `eval20_cross_v4`) — submit after their training.

---

## 5. HTCondor infrastructure (key gotchas)

Submit files live in `condor/`; logs in `condor/logs/<campaign>/`.

- **GPU training**: executable `condor/run_job.sh`; `require_gpus=(Capability>=8.0)` (A100),
  `request_gpus=1`, `MY.SendCredential=True` (forwards the Kerberos cred so the worker can reach EOS),
  `+JobFlavour="nextweek"` (50-epoch jobs exceed 24h "tomorrow" → killed by SYSTEM_PERIODIC_REMOVE).
  `run_job.sh` rewrites `--output-dir` to local scratch, **xrdcp-stages the compact + weight IN**,
  injects `--resume-eos-dir=<eos out>` (eviction-safe: each epoch pushes `last.pt`/`best_model.pt` to
  EOS; on restart it resumes), and **xrdcp-stages all outputs OUT** at the end. Submit txt lists one
  arg-line per run; `queue arguments from condor/<x>.txt`.
- **Data build on Condor** (do this instead of login-node builds): executables
  `condor/run_train_build.sh` (weights+compacts for a pT cut, args `<tier> <mode> <pt_min>`) and
  `condor/run_eval_build.sh` (the 20 GeV eval sample). They read raw via FUSE (robust, see below) and
  **xrdcp results to EOS** (EOS FUSE *writes* from workers are unreliable). 6 CPU jobs (tier×mode).
- **EOS FUSE is unreliable and intermittently degrades.** Symptoms seen: FUSE open of a specific raw
  file HANGS forever (no error), and xrdcp of it returns exit 54 ("unauthorized identity") — transient,
  file-specific, comes and goes over days. Never trust a plain `h5py.File(eos_path)` in a long build.
  → `robust_h5_open()` in `build_event_level_weights.py`: bounded FUSE open (thread timeout) → xrdcp-to-local
  fallback (subprocess timeout + retries) → raises `SkippableFileError` only if all fail; callers
  (`scan_file_metadata`, `build_eval_sample`, `build_compact_dataset`, `build_category_weights`) then
  **log + skip** that file. This is why the builds now survive a flaky EOS.
- **Login-node detached builds (tmux/nohup) are fragile here** — they died repeatedly (the real cause was
  the EOS file-hang above, sometimes mis-attributed to AFS token loss). For unattended completion use
  **Condor**; if you must build on the login node, keep the session up and rely on skip-existing resume.

---

## 6. Numerical precision — fp32/TF32 (why, and the NaN history)

Training originally used **fp16 autocast + GradScaler**. On some batches the attention/activations
overflowed fp16's ~65504 range → `inf` → softmax `NaN` in the logits. That NaN score then crashed
`safe_roc_auc` (`roc_auc_score` → "Input contains NaN") **mid-training**, and also produced broken
efficiency plots (threshold came out `nan` → every bin eff=0). This hit track/cross `all`/`eq0`
categories hardest (empty/degenerate track sets) and even some image runs.

**Fixes (committed, current state):**
- All 5 trainers now train in **fp32** with **TF32** matmul/cudnn enabled on Ampere+
  (`torch.backends.cuda.matmul.allow_tf32=True`) — fp32 range (no overflow, NaN-safe) at ~bf16 speed.
  autocast/GradScaler removed. bf16 was considered but fp32/TF32 chosen for simplicity + full stability.
- `safe_roc_auc`, `save_roc_plot`, `threshold_for_bkg_eff_unw`, `save_unweighted_test_plots` all filter
  non-finite scores so a stray NaN can never crash the metric/plot path again.
- v9/cross_v3 (fp32) came out with **zero** broken plots; the older fp16 v8/cross_v2 plots were
  regenerated from the (clean) prediction CSVs via `regenerate_efficiency_plots.py` /
  reusing `save_unweighted_test_plots` from CSV.

The cross model itself already guards the empty-point-cloud case (a learned `null` track token so the
attention key set is never all-masked; masked-mean pool with `clamp_min(1.0)`); the NaN was fp16 overflow,
not a structural empty-set bug (verified by repro: a fresh model does not NaN on empty input).

---

## 7. How to run each stage (pT cut = N, e.g. 100)

```bash
# 0. env
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh   # set +u first
export HDF5_USE_FILE_LOCKING=FALSE

# 1. build data on Condor (weights + compacts) -> categories_ptN, compact_ptN
#    (edit condor/build_ptN.txt lines "TIER MODE N"; wrapper = condor/run_train_build.sh)
condor_submit condor/build_pt100.sub          # 6 jobs; verify 6 weights + 12 compacts on EOS

# 2. train  -> runs/v10, runs/cross_v4  (48 jobs; fp32)
condor_submit condor/v10_track.sub            # 12
condor_submit condor/v10_image.sub            # 18
condor_submit condor/cross_v4.sub             # 18

# 3. eval the trained models on the shared 20 GeV sample -> *_20GeV_sample plots (48 jobs)
condor_submit condor/eval20_v10_track.sub
condor_submit condor/eval20_v10_image.sub
condor_submit condor/eval20_cross_v4.sub
python3 combine_wp_plots.py                    # 2 overlay plots per run (globs all runs)
```

New cut → copy the pattern: add `condor/build_ptN.txt`, and sed the `v9_*`/`cross_v3`/`eval20_v9_*`
files (`pt200→ptN`, `v9→v??`, `cross_v3→cross_v?`) as was done for v10/cross_v4.

### Verifying / re-plotting
- Completion of a run = `test_metrics.json` exists. Efficiency plot "broken" = all signal-eff rows 0
  (nan threshold). Scan: look for that across `runs/*/*/efficiency_vs_pt_test_bkg05_unw*.json`.
- Regenerate plots from the clean CSVs without re-running the model:
  `regenerate_efficiency_plots.py`, `regenerate_roc_score_unweighted.py`, or call
  `save_unweighted_test_plots` / `save_eval_sample_plots` on `{val,test}_predictions.csv`.

---

## 8. Current state (as of this handoff)

- **Done**: v8, cross_v2 (pt50); v9, cross_v3 (pt200) — all trained; all efficiency plots clean; all 96
  eval'd on the 20 GeV sample with per-WP + combined overlay plots.
- **In flight**: v10 (pt100 image+track, clusters submitted) and cross_v4 (pt100 cross) — 48 training
  jobs just submitted. Their data (`categories_pt100`, `compact_pt100`) is built and verified
  (pt min=100, background=1M).
- **Next**: when v10/cross_v4 training finishes, submit `eval20_v10_*` + `eval20_cross_v4`, then
  `combine_wp_plots.py`. Cluster IDs are ephemeral — check `condor_q` and the run dirs, not old IDs.

## 9. Scratch data-slimming note
Raw files were slimmed (`slim_raw.py`/`slim_all.py`): full-detector arrays (ECAL/HBHE/full-EE/full-ES)
were dropped, keeping only the cropped SC/seed/ES_seed images + tracks. This cut raw from ~722 GB to
~140 GB. The cropped-image training inputs are byte-identical; the dropped arrays are recoverable only
by re-converting from ROOT if ever needed.
