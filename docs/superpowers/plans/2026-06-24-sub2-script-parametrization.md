# Sub-Project 2: Script Parametrization (track-types + ES) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the image, track, and fusion scripts accept a per-category
`--track-types` knob (AOD=`GenTrk`, MiniAOD=`Lost,PF,GSF`) and an image `--no-es`
toggle, so one code path serves all 18 image / 12 track / 18 fusion categories.

**Architecture:** A single comma-separated `--track-types` argument drives: the
track point-cloud feature dimension and one-hot width, the image track-channel
set, the track prediction-CSV `n_<type>`/`sum_pt_<type>` columns, and the fusion
features. Track types are persisted in each checkpoint/run_config so the fusion
step rebuilds models and features without guessing. The image `--no-es` flag
drops the two ES preshower channels for EE.

**Tech Stack:** Python 3.13 (cvmfs `LCG_109_cuda` el9 view), torch, h5py, numpy, pytest.

**Out of scope (confirmed unnecessary):**
- hasADD selection — the per-category weight files already encode it as
  `weight == 0`, and training/registry/manifest all gate on `weight > 0`.
- `validate_compatible_runs` ES handling — ES is not in its compared keys, and
  ES-on/off image runs share the same category weight file + key as the track run.

**Environment:** source the LCG view before any python/pytest:
```bash
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u
```

**Backward compatibility:** defaults preserve current behavior —
`--track-types` defaults to `GSF,PF,Lost` (track) / the detector-appropriate
MiniAOD set (image), ES included by default.

---

## File Structure

- Modify: `track_point_transformer.py` — dynamic feature dim, track-type-driven
  point cloud, CSV columns.
- Modify: `train_point_transformer_track_classifier.py` — `--track-types` arg,
  checkpoint persistence.
- Modify: `train_resnet_image_classifier.py` — channel builder from
  `(detector, track_types, include_es)`, `--track-types`/`--no-es` args,
  computed `in_channels`, checkpoint persistence.
- Modify: `train_fusion_ensemble.py` — track-type-driven features + CSV columns,
  reading `track_types` from the track checkpoint/run_config.
- Test: `tests/test_parametrization.py` — unit tests for the pure helpers.

Track-type name conventions (HDF5 keys): `{EB|EE}_track_pt_{TYPE}_{idx,val}` with
`TYPE` in `{GSF, PF, Lost, GenTrk}`. The image uses the same `TYPE` tokens.

---

## Task 1: Parametrize track point cloud by track types

**Files:**
- Modify: `track_point_transformer.py`
- Test: `tests/test_parametrization.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parametrization.py`:

```python
import numpy as np
import track_point_transformer as tpt


def test_point_feature_dim():
    assert tpt.point_feature_dim(("GSF", "PF", "Lost")) == 6
    assert tpt.point_feature_dim(("GenTrk",)) == 4


def test_sparse_points_onehot_width_matches_track_types():
    idx = np.array([0, 600], dtype=np.int64)
    val = np.array([10.0, 20.0], dtype=np.float32)
    # 1 track type -> dim 4 (x,y,pt,onehot[0])
    pts1, _ = tpt.sparse_track_arrays_to_points(idx, val, track_type_id=0, n_track_types=1)
    assert pts1.shape == (2, 4)
    assert np.allclose(pts1[:, 3], 1.0)
    # 3 track types, id=2 -> dim 6, onehot in column 5
    pts3, _ = tpt.sparse_track_arrays_to_points(idx, val, track_type_id=2, n_track_types=3)
    assert pts3.shape == (2, 6)
    assert np.allclose(pts3[:, 3 + 2], 1.0)
    assert pts3[:, 3] == 0  # noqa


def test_parse_track_types():
    assert tpt.parse_track_types("Lost,PF,GSF") == ("Lost", "PF", "GSF")
    assert tpt.parse_track_types("GenTrk") == ("GenTrk",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_parametrization.py -v`
Expected: FAIL (`AttributeError: ... 'point_feature_dim'`).

- [ ] **Step 3: Implement**

In `track_point_transformer.py`:

Replace the module constant block:
```python
TRACK_TYPES = ("GSF", "PF", "Lost")
POINT_FEATURE_DIM = 6
```
with:
```python
TRACK_TYPES = ("GSF", "PF", "Lost")          # default (MiniAOD) track collections
POINT_FEATURE_DIM = 3 + len(TRACK_TYPES)     # default; real dim derived from track_types


def parse_track_types(s):
    return tuple(t for t in str(s).split(",") if t)


def point_feature_dim(track_types):
    return 3 + len(track_types)
```

Change `sparse_track_arrays_to_points` signature and one-hot to take `n_track_types`:
```python
def sparse_track_arrays_to_points(flat_idx, values, track_type_id, n_track_types,
                                  image_size=512, log_track_pt=True):
    idx = np.asarray(flat_idx, dtype=np.int64)
    pt = np.asarray(values, dtype=np.float32)
    dim = 3 + n_track_types
    if idx.size == 0:
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.float32)

    pt = np.clip(pt, 0.0, None)
    x = ((idx % image_size).astype(np.float32) + 0.5) / float(image_size)
    y = ((idx // image_size).astype(np.float32) + 0.5) / float(image_size)
    x = 2.0 * x - 1.0
    y = 2.0 * y - 1.0

    pt_feature = np.log1p(pt) if log_track_pt else pt
    onehot = np.zeros((idx.size, n_track_types), dtype=np.float32)
    onehot[:, track_type_id] = 1.0
    points = np.concatenate(
        [x[:, None], y[:, None], pt_feature[:, None].astype(np.float32), onehot],
        axis=1,
    ).astype(np.float32)
    return points, pt
```

Change `build_track_point_cloud` to take `track_types`:
```python
def build_track_point_cloud(raw_handle, detector, obj_idx, max_points, track_types,
                            log_track_pt=True):
    prefix = "EB" if detector == "eb" else "EE"
    n_types = len(track_types)
    dim = 3 + n_types
    point_parts = []
    counts = np.zeros(n_types, dtype=np.float32)
    sum_pt = np.zeros(n_types, dtype=np.float32)

    for type_id, track_type in enumerate(track_types):
        idx_key = f"{prefix}_track_pt_{track_type}_idx"
        val_key = f"{prefix}_track_pt_{track_type}_val"
        points, raw_pt = sparse_track_arrays_to_points(
            raw_handle[idx_key][obj_idx], raw_handle[val_key][obj_idx],
            track_type_id=type_id, n_track_types=n_types, log_track_pt=log_track_pt)
        counts[type_id] = float(raw_pt.size)
        sum_pt[type_id] = float(raw_pt.sum())
        if points.size > 0:
            point_parts.append(points)

    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        if max_points is not None and points.shape[0] > max_points:
            order = np.argsort(-points[:, 2], kind="stable")
            points = points[order[:max_points]]
    else:
        points = np.zeros((0, dim), dtype=np.float32)
    return points, counts, sum_pt
```

Make `collate_track_point_cloud_batch` infer the feature dim from items (replace
the `POINT_FEATURE_DIM` use):
```python
def collate_track_point_cloud_batch(batch):
    batch_size = len(batch)
    feat_dim = batch[0]["points"].shape[1]
    max_points_in_batch = max(1, max(item["points"].shape[0] for item in batch))
    points = torch.zeros(batch_size, max_points_in_batch, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_points_in_batch, dtype=torch.bool)
    ...  # rest unchanged
```

Add a `track_types` parameter to `TrackPointCloudDataset.__init__` (store
`self.track_types = tuple(track_types)`) and pass it in `__getitem__`'s
`build_track_point_cloud(..., track_types=self.track_types, ...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_parametrization.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add track_point_transformer.py tests/test_parametrization.py
git commit -m "feat: parametrize track point cloud by track types"
```

---

## Task 2: Track CSV columns + trainer `--track-types`

**Files:**
- Modify: `track_point_transformer.py` (`save_track_predictions_csv`)
- Modify: `train_point_transformer_track_classifier.py`
- Test: `tests/test_parametrization.py`

- [ ] **Step 1: Write the failing test for dynamic CSV header**

Append to `tests/test_parametrization.py`:

```python
def test_track_csv_header_for_track_types(tmp_path):
    import track_point_transformer as tpt
    hdr1 = tpt.track_csv_columns(("GenTrk",))
    assert "n_gentrk" in hdr1 and "sum_pt_gentrk" in hdr1
    assert "n_pf" not in hdr1
    hdr3 = tpt.track_csv_columns(("GSF", "PF", "Lost"))
    assert hdr3 == ["n_gsf", "n_pf", "n_lost", "sum_pt_gsf", "sum_pt_pf", "sum_pt_lost"]
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_parametrization.py -k track_csv_header -v`
Expected: FAIL (`no attribute 'track_csv_columns'`).

- [ ] **Step 3: Implement dynamic columns**

In `track_point_transformer.py` add:
```python
def track_csv_columns(track_types):
    lows = [t.lower() for t in track_types]
    return [f"n_{t}" for t in lows] + [f"sum_pt_{t}" for t in lows]
```

Rewrite `save_track_predictions_csv` to take `track_types` and emit dynamic
columns (the fixed prefix stays the same):
```python
def save_track_predictions_csv(output_path, pack, file_entries, code_to_sample, track_types):
    extra = track_csv_columns(track_types)
    header = "score,label,weight,pt,file_idx,object_idx,event_idx,sample_id,raw_file," + ",".join(extra) + "\n"
    n_types = len(track_types)
    with output_path.open("w") as handle:
        handle.write(header)
        for i in range(len(pack["label"])):
            file_idx = int(pack["file_idx"][i])
            sample_id = code_to_sample[int(pack["sample_code"][i])]
            raw_file = file_entries[file_idx].raw_path
            counts = pack["track_counts"][i]
            sum_pt = pack["track_sum_pt"][i]
            vals = [f"{float(counts[j]):.0f}" for j in range(n_types)] + \
                   [f"{float(sum_pt[j]):.8f}" for j in range(n_types)]
            handle.write(
                f"{float(pack['score'][i]):.8f},{int(pack['label'][i])},"
                f"{float(pack['weight'][i]):.8e},{float(pack['pt'][i]):.8f},"
                f"{file_idx},{int(pack['object_idx'][i])},{int(pack['event_idx'][i])},"
                f"{sample_id},{raw_file}," + ",".join(vals) + "\n"
            )
```

In `train_point_transformer_track_classifier.py`:
- Import `parse_track_types`, `point_feature_dim` from `track_point_transformer`.
- Add arg: `parser.add_argument("--track-types", default="GSF,PF,Lost")`.
- After `args = parse_args()`: `track_types = parse_track_types(args.track_types)`.
- Pass `track_types=track_types` to every `TrackPointCloudDataset(...)`.
- Build the model with `input_dim=point_feature_dim(track_types)`.
- In the checkpoint `torch.save({...})` dict, set `"input_dim": point_feature_dim(track_types)`
  and add `"track_types": list(track_types)`.
- Update both `save_track_predictions_csv(...)` calls to pass `track_types`.
- Add `"track_types": list(track_types)` to the run_config JSON the trainer writes.

- [ ] **Step 4: Run to verify pass + import-check the trainer**

Run: `python3 -m pytest tests/test_parametrization.py -k track_csv_header -v`
Expected: PASS.
Run: `python3 -c "import train_point_transformer_track_classifier"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add track_point_transformer.py train_point_transformer_track_classifier.py tests/test_parametrization.py
git commit -m "feat: track-types-driven CSV columns and track trainer arg"
```

---

## Task 3: Image channel builder from (detector, track_types, ES)

**Files:**
- Modify: `train_resnet_image_classifier.py`
- Test: `tests/test_parametrization.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_parametrization.py`:

```python
import train_resnet_image_classifier as tri


def test_image_channel_keys():
    # EB, MiniAOD 3 track types -> SC + 3 tracks = 4
    eb = tri.image_channel_keys("eb", ("Lost", "PF", "GSF"), include_es=False)
    assert [k for k, _ in eb] == ["calo", "track", "track", "track"]
    assert len(eb) == 4
    # EB AOD GenTrk -> SC + GenTrk = 2
    eb_aod = tri.image_channel_keys("eb", ("GenTrk",), include_es=False)
    assert len(eb_aod) == 2
    # EE with ES, 3 tracks -> EE_seed + es1 + es2 + 3 = 6
    ee = tri.image_channel_keys("ee", ("Lost", "PF", "GSF"), include_es=True)
    assert len(ee) == 6
    # EE no ES, AOD GenTrk -> EE_seed + GenTrk = 2
    ee_noes = tri.image_channel_keys("ee", ("GenTrk",), include_es=False)
    assert len(ee_noes) == 2


def test_image_in_channels():
    assert tri.image_in_channels("eb", ("Lost", "PF", "GSF"), True) == 4
    assert tri.image_in_channels("ee", ("Lost", "PF", "GSF"), True) == 6
    assert tri.image_in_channels("ee", ("GenTrk",), False) == 2
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_parametrization.py -k "image_channel_keys or image_in_channels" -v`
Expected: FAIL (`no attribute 'image_channel_keys'`).

- [ ] **Step 3: Implement channel spec + use it in the dataset**

In `train_resnet_image_classifier.py` add module-level helpers:
```python
def image_channel_keys(detector, track_types, include_es):
    """Ordered list of (kind, h5key) channels. kind in {calo, es, track}."""
    prefix = "EB" if detector == "eb" else "EE"
    keys = []
    if detector == "eb":
        keys.append(("calo", "SC_energy"))
    else:
        keys.append(("calo", "EE_seed_energy"))
        if include_es:
            keys.append(("es", "ES_seed_plane1_energy"))
            keys.append(("es", "ES_seed_plane2_energy"))
    for t in track_types:
        keys.append(("track", f"{prefix}_track_pt_{t}"))
    return keys


def image_in_channels(detector, track_types, include_es):
    return len(image_channel_keys(detector, track_types, include_es))
```

Replace `_build_eb_image` and `_build_ee_image` with one generic builder driven
by `self.channel_keys` (set in `__init__`):
```python
    def _build_image(self, raw, obj_idx):
        channels = []
        for kind, key in self.channel_keys:
            if kind in ("calo", "es"):
                channels.append(repeat_to_512(raw[key][obj_idx]))
            else:  # track: key is the "{PREFIX}_track_pt_{TYPE}" stem
                channels.append(sparse_to_dense(raw[f"{key}_idx"][obj_idx],
                                                 raw[f"{key}_val"][obj_idx]))
        channels = [preprocess_channel(ch, self.log_scale) for ch in channels]
        return np.stack(channels, axis=0).astype(np.float32)
```

Update `DetectorObjectDataset.__init__` to accept `track_types` and `include_es`
and store `self.channel_keys = image_channel_keys(detector, track_types, include_es)`.
Update `__getitem__` to call `self._build_image(raw, obj_idx)` for both detectors.

In `main()`:
- Add args: `parser.add_argument("--track-types", default="Lost,PF,GSF")` and
  `parser.add_argument("--no-es", action="store_true", help="Drop ES channels (EE).")`.
- Compute `track_types = tuple(t for t in args.track_types.split(",") if t)` and
  `include_es = not args.no_es`.
- Pass `track_types=track_types, include_es=include_es` to all three
  `DetectorObjectDataset(...)` constructions.
- Replace `in_channels = 4 if detector == "eb" else 6` with
  `in_channels = image_in_channels(detector, track_types, include_es)`.
- In the checkpoint dict add `"track_types": list(track_types)` and
  `"include_es": include_es` (alongside the existing `"in_channels"`).
- Add `"track_types": list(track_types)` and `"include_es": include_es` to the
  run_config JSON.

- [ ] **Step 4: Run to verify pass + import-check**

Run: `python3 -m pytest tests/test_parametrization.py -k "image_channel_keys or image_in_channels" -v`
Expected: PASS.
Run: `python3 -c "import train_resnet_image_classifier"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add train_resnet_image_classifier.py tests/test_parametrization.py
git commit -m "feat: tier/ES-aware image channel builder"
```

---

## Task 4: Fusion — track-type-driven features and CSV

**Files:**
- Modify: `train_fusion_ensemble.py`
- Test: `tests/test_parametrization.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_parametrization.py`:

```python
import numpy as np
import train_fusion_ensemble as tfe


def test_fusion_feature_names_dynamic():
    names1 = tfe.fusion_feature_names(("GenTrk",))
    assert names1 == ["image_logit", "track_logit", "log1p_n_gentrk", "log1p_sum_pt_gentrk"]
    names3 = tfe.fusion_feature_names(("GSF", "PF", "Lost"))
    assert names3 == [
        "image_logit", "track_logit",
        "log1p_n_gsf", "log1p_n_pf", "log1p_n_lost",
        "log1p_sum_pt_gsf", "log1p_sum_pt_pf", "log1p_sum_pt_lost",
    ]
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_parametrization.py -k fusion_feature_names -v`
Expected: FAIL (`no attribute 'fusion_feature_names'`).

- [ ] **Step 3: Implement dynamic fusion features**

In `train_fusion_ensemble.py`:

Add:
```python
def fusion_feature_names(track_types):
    lows = [t.lower() for t in track_types]
    return (["image_logit", "track_logit"]
            + [f"log1p_n_{t}" for t in lows]
            + [f"log1p_sum_pt_{t}" for t in lows])
```

Rewrite `build_fusion_features` to take `track_types` and build columns from the
actual track-type count:
```python
def build_fusion_features(image_pack, track_pack, track_types):
    counts = np.log1p(np.clip(track_pack["track_counts"], 0.0, None))
    sum_pt = np.log1p(np.clip(track_pack["track_sum_pt"], 0.0, None))
    cols = [score_to_logit(image_pack["score"]), score_to_logit(track_pack["score"])]
    n = len(track_types)
    cols += [counts[:, j] for j in range(n)]
    cols += [sum_pt[:, j] for j in range(n)]
    features = np.column_stack(cols).astype(np.float64)
    return features, fusion_feature_names(track_types)
```

In `load_prediction_pack_from_csv` (the `has_track_columns` branch), read the
columns by the track-type-derived names instead of the fixed `n_gsf...`. Change
its signature to `load_prediction_pack_from_csv(csv_path, track_types=None)` and,
when `track_types` is provided, build `track_counts`/`track_sum_pt` from
`n_<type>`/`sum_pt_<type>` columns:
```python
    if track_types is not None:
        lows = [t.lower() for t in track_types]
        pack["track_counts"] = np.asarray(
            [[float(row[f"n_{t}"]) for t in lows] for row in rows], dtype=np.float64)
        pack["track_sum_pt"] = np.asarray(
            [[float(row[f"sum_pt_{t}"]) for t in lows] for row in rows], dtype=np.float64)
```
Update the call sites: image packs call with `track_types=None`; track packs
call with the track run's `track_types` (read below).

In `main()`:
- Read `track_types` from the track checkpoint or track run_config:
  `track_types = tuple(track_cfg.get("track_types") or ("GSF", "PF", "Lost"))`.
- Pass `track_types` to `load_prediction_pack_from_csv` for the track val/test
  packs, and to `build_fusion_features(...)`.
- When rebuilding the track model in the rescore path, use
  `input_dim=int(track_checkpoint.get("input_dim", len(track_types) + 3))` and
  pass `track_types` into `TrackPointCloudDataset` / `build_track_point_cloud`.
- Rebuild the image model with `in_channels` from the image checkpoint
  (already stored) — unchanged.
- Update `save_fusion_predictions_csv` header/rows to emit the dynamic
  `n_<type>`/`sum_pt_<type>` columns (mirror `track_csv_columns`).

- [ ] **Step 4: Run to verify pass + import-check**

Run: `python3 -m pytest tests/test_parametrization.py -k fusion_feature_names -v`
Expected: PASS.
Run: `python3 -c "import train_fusion_ensemble"`
Expected: no error.

- [ ] **Step 5: Run the full parametrization test module**

Run: `python3 -m pytest tests/test_parametrization.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add train_fusion_ensemble.py tests/test_parametrization.py
git commit -m "feat: track-type-driven fusion features and CSV"
```

---

## Task 5: End-to-end debug smoke (one AOD-style category)

**Files:** none (verification only). Uses the synthetic raw fixtures from
`tests/conftest.py` plus the existing `--debug` paths.

- [ ] **Step 1: Verify track + image accept `--track-types`/`--no-es` without crashing**

This is a guard that the new args thread through argparse and model construction.
Run a minimal import + argparse check:
```bash
python3 train_resnet_image_classifier.py --help | grep -E "track-types|no-es"
python3 train_point_transformer_track_classifier.py --help | grep -E "track-types"
```
Expected: each flag appears in the help output.

- [ ] **Step 2: Confirm full test suite green**

Run: `python3 -m pytest tests/ -q`
Expected: all tests pass (Task-1..4 plus the existing category-weight tests).

---

## Self-Review notes

- **Coverage:** track-types knob (Tasks 1-4), ES toggle (Task 3), dynamic track
  CSV (Task 2) + fusion CSV/features (Task 4), checkpoint/run_config persistence
  (Tasks 2-3), fusion reads persisted track_types (Task 4).
- **Dropped (justified):** hasADD manifest filter and ES validate handling — both
  unnecessary (weight-file encoding; ES absent from compared keys).
- **Type consistency:** `parse_track_types`/`point_feature_dim` live in
  `track_point_transformer.py` and are imported by the trainer and fusion;
  `image_channel_keys`/`image_in_channels` in the image script; track CSV column
  helper `track_csv_columns` shared by trainer and fusion writer.
- **Risk:** the fusion CSV loader previously hardcoded `n_gsf,n_pf,n_lost`; after
  Task 4 it reads columns by the track run's persisted track_types, so old track
  CSVs (written before Task 2) remain readable only for the 3-type MiniAOD case.
  Regenerate track predictions after this change for AOD categories.
