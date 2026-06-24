# Per-Category Weight Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build object-level balanced event-weight files for each `tier × hasADD`
category (with EB+EE inside each file), applying a per-object `hasAdditionalTrk`
filter, per-category signal mass-point exclusion, and ~1M background event
resampling — so the existing training scripts consume them unchanged via
`--weight-h5` + `--weight-key {EB,EE}_weight_split`.

**Architecture:** A new `build_category_weights.py` reuses the pure helpers from
`build_event_level_weights.py` (file scan, pT binning, background resampling) but
balances on a **per-object** basis (because `hasAdditionalTrk` is per-object)
instead of per-event. Selection is encoded as *which objects carry weight > 0*:
the training code already gates on `weight_key > 0` and re-normalizes each class
to sum=1 at train time (`rebalance_manifest_class_weights`), so the weight file
only needs correct *relative* weights (equal sum per non-empty pT bin, signal
split equally across included mass points) plus zeros for excluded objects.

**Tech Stack:** Python 3.13 (cvmfs `LCG_109_cuda` el9 view), h5py, numpy, pytest.

**Contract the output must satisfy** (verified from `train_resnet_image_classifier.py`):
- `parse_file_entries` reads `files/{split}/{stem}` group attrs: `input_file`,
  `sample_id`, `process_name`, `n_events`.
- `build_event_registry` / `build_object_manifest` read
  `weight_h5["files"][split][stem]["event_pt"]` (per event, len `n_events`) and
  `[weight_key]` (per detector object, e.g. `EB_weight_split`), keeping objects
  with `weight > 0` and mapping object→event via raw `EB_ele_event_idx` /
  `EE_ele_event_idx`.

**Environment note:** every Python command below assumes the LCG view is sourced:
```bash
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u
```
System `/usr/bin/python3` is too old (no `int | None`); do not use it.

---

## File Structure

- Create: `build_category_weights.py` — new object-based builder (CLI + library funcs).
- Create: `tests/__init__.py` — make `tests` a package (empty).
- Create: `tests/conftest.py` — synthetic HDF5 fixture factory.
- Create: `tests/test_category_weights.py` — unit + integration tests.
- Create: `run_build_category_weights.sh` — driver looping `tier × hasADD` (6 runs).
- Reuse (import, do not modify): `build_event_level_weights.py`
  (`PT_BIN_EDGES_WITH_OVERFLOW`, `find_bin_indices`, `derive_event_pt`,
  `scan_file_metadata`, `select_background_global_event_indices`,
  `get_background_local_event_indices`).

Category constants (single source of truth) live at the top of
`build_category_weights.py`:

```python
HASADD_MODES = ("all", "eq0", "eq1")
DETECTORS = ("eb", "ee")
HASADD_KEYS = {"eb": "hasAdditionalTrk_EB", "ee": "hasAdditionalTrk_EE"}
DET_EVENT_IDX_KEYS = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}
WEIGHT_KEYS = {"eb": "EB_weight_split", "ee": "EE_weight_split"}
```

---

## Task 1: Category selection rules (pure functions)

**Files:**
- Create: `build_category_weights.py`
- Test: `tests/test_category_weights.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_category_weights.py`:

```python
import numpy as np
import build_category_weights as bcw


def test_hasadd_object_mask_modes():
    v = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    assert bcw.hasadd_object_mask(v, "all").tolist() == [True, True, True, True, True]
    assert bcw.hasadd_object_mask(v, "eq0").tolist() == [True, False, True, False, True]
    assert bcw.hasadd_object_mask(v, "eq1").tolist() == [False, True, False, True, False]


def test_is_excluded_sample():
    # eq1 excludes A0p4 and A1 (any H); keeps A2/A5/A10 and background
    assert bcw.is_excluded_sample("signal_H250_A0p4", "eq1") is True
    assert bcw.is_excluded_sample("signal_H2000_A1", "eq1") is True
    assert bcw.is_excluded_sample("signal_H250_A10", "eq1") is False
    assert bcw.is_excluded_sample("signal_H750_A2", "eq1") is False
    assert bcw.is_excluded_sample("background", "eq1") is False
    # eq0 excludes only H250_A10
    assert bcw.is_excluded_sample("signal_H250_A10", "eq0") is True
    assert bcw.is_excluded_sample("signal_H250_A1", "eq0") is False
    assert bcw.is_excluded_sample("signal_H2000_A10", "eq0") is False
    # all excludes nothing
    assert bcw.is_excluded_sample("signal_H250_A10", "all") is False
    assert bcw.is_excluded_sample("signal_H250_A0p4", "all") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_category_weights.py -k "hasadd_object_mask or is_excluded" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_category_weights'`.

- [ ] **Step 3: Write minimal implementation**

Create `build_category_weights.py`:

```python
#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    PT_BIN_EDGES_WITH_OVERFLOW,
    find_bin_indices,
    derive_event_pt,
    scan_file_metadata,
    select_background_global_event_indices,
    get_background_local_event_indices,
)

HASADD_MODES = ("all", "eq0", "eq1")
DETECTORS = ("eb", "ee")
HASADD_KEYS = {"eb": "hasAdditionalTrk_EB", "ee": "hasAdditionalTrk_EE"}
DET_EVENT_IDX_KEYS = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}
WEIGHT_KEYS = {"eb": "EB_weight_split", "ee": "EE_weight_split"}
N_BINS = len(PT_BIN_EDGES_WITH_OVERFLOW) - 1


def hasadd_object_mask(hasadd_values, mode):
    v = np.asarray(hasadd_values)
    if mode == "eq1":
        return v == 1
    if mode == "eq0":
        return v == 0
    if mode == "all":
        return np.ones(v.shape, dtype=bool)
    raise ValueError(f"Unknown hasADD mode: {mode}")


def is_excluded_sample(sample_id, mode):
    if sample_id == "background":
        return False
    if mode == "eq1":
        return sample_id.endswith("_A0p4") or sample_id.endswith("_A1")
    if mode == "eq0":
        return sample_id == "signal_H250_A10"
    if mode == "all":
        return False
    raise ValueError(f"Unknown hasADD mode: {mode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_category_weights.py -k "hasadd_object_mask or is_excluded" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add build_category_weights.py tests/__init__.py tests/test_category_weights.py
git commit -m "feat: category selection rules for per-category weights"
```

---

## Task 2: Object-based per-bin weight lookup (pure function)

**Files:**
- Modify: `build_category_weights.py`
- Test: `tests/test_category_weights.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_category_weights.py`:

```python
def test_object_weight_lookup_balances_bins_and_classes():
    # 2 bins. background present in both bins; one signal sample in 1 bin.
    sample_bin_counts = {
        "background": np.array([4, 0, 0] + [0] * (bcw.N_BINS - 3), dtype=np.int64),
        "signal_H250_A2": np.array([0, 2, 0] + [0] * (bcw.N_BINS - 3), dtype=np.int64),
        "signal_H250_A5": np.array([0, 0, 5] + [0] * (bcw.N_BINS - 3), dtype=np.int64),
    }
    targets = {"background": 1.0, "signal_H250_A2": 0.5, "signal_H250_A5": 0.5}
    lookup = bcw.make_object_weight_lookup(sample_bin_counts, targets)

    # background: 1 non-empty bin, total 1.0 -> bin sum 1.0 over 4 objects = 0.25 each
    assert np.isclose(lookup["background"][0], 0.25)
    # per-class object-weight * count sums to the target
    bg_sum = (lookup["background"] * sample_bin_counts["background"]).sum()
    sig_sum = sum(
        (lookup[s] * sample_bin_counts[s]).sum() for s in ("signal_H250_A2", "signal_H250_A5")
    )
    assert np.isclose(bg_sum, 1.0)
    assert np.isclose(sig_sum, 1.0)


def test_object_weight_lookup_empty_sample_is_all_zero():
    counts = {"background": np.zeros(bcw.N_BINS, dtype=np.int64)}
    lookup = bcw.make_object_weight_lookup(counts, {"background": 1.0})
    assert np.all(lookup["background"] == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_category_weights.py -k "object_weight_lookup" -v`
Expected: FAIL with `AttributeError: module 'build_category_weights' has no attribute 'make_object_weight_lookup'`.

- [ ] **Step 3: Write minimal implementation**

Append to `build_category_weights.py`:

```python
def make_object_weight_lookup(sample_bin_counts, sample_total_target):
    """Per-sample, per-bin object weight so each non-empty pT bin contributes an
    equal share of the sample's target total, split evenly over its objects."""
    lookup = {}
    for sample_id, counts in sample_bin_counts.items():
        counts = np.asarray(counts, dtype=np.int64)
        nonempty = counts > 0
        n_nonempty = int(nonempty.sum())
        weights = np.zeros(counts.shape, dtype=np.float64)
        if n_nonempty > 0:
            target_bin_sum = float(sample_total_target[sample_id]) / n_nonempty
            weights[nonempty] = target_bin_sum / counts[nonempty]
        lookup[sample_id] = weights
    return lookup


def build_sample_targets(sample_ids):
    """background -> 1.0; signal total 1.0 split equally over included signal ids."""
    signal_ids = [s for s in sample_ids if s != "background"]
    if "background" not in sample_ids:
        raise ValueError("No background sample present.")
    if not signal_ids:
        raise ValueError("No included signal samples present.")
    per_signal = 1.0 / len(signal_ids)
    targets = {"background": 1.0}
    for s in signal_ids:
        targets[s] = per_signal
    return targets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_category_weights.py -k "object_weight_lookup" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add build_category_weights.py tests/test_category_weights.py
git commit -m "feat: object-based per-bin weight lookup"
```

---

## Task 3: Synthetic HDF5 fixture factory

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_category_weights.py`

- [ ] **Step 1: Write the fixture factory**

Create `tests/conftest.py`. It writes tiny raw-style HDF5 files matching the
keys the builder reads (`eventId`, `A_pT`, `A_event_idx`, `EB_ele_event_idx`,
`EE_ele_event_idx`, `SC_energy`, `EE_seed_energy`, `hasAdditionalTrk_EB/EE`,
and the `source_root_files` attr) into `W/ Z/ signal/` under a tier dir.

```python
import ast
import numpy as np
import h5py
import pytest

SIGNAL_SRC = {
    "H250_A2": "/store/user/yeo/HToAATo4L_H250A2_TuneCP5_13p6TeV-pythia8_v1/signal_x/0/0/o.root",
    "H250_A10": "/store/user/yeo/HToAATo4L_H250A10_TuneCP5_13p6TeV-pythia8_v1/signal_x/0/0/o.root",
    "H750_A0p4": "/store/user/yeo/HToAATo4L_H750A0p4_TuneCP5_13p6TeV-pythia8_v1/signal_x/0/0/o.root",
}
BKG_SRC = "/store/user/yeo/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/W_x/0/0/o.root"


def _write_file(path, n_events, eb_events, ee_events, eb_hasadd, ee_hasadd, source, a_pt):
    """eb_events/ee_events: per-object event index arrays. *_hasadd: per-object 0/1."""
    with h5py.File(path, "w") as f:
        f.attrs["source_root_files"] = repr([source])
        f["eventId"] = np.arange(n_events, dtype=np.int64)
        # one A entry per event with the given pT (so derive_event_pt 'max' works)
        f["A_pT"] = np.asarray(a_pt, dtype=np.float32)
        f["A_event_idx"] = np.arange(n_events, dtype=np.int64)
        f["EB_ele_event_idx"] = np.asarray(eb_events, dtype=np.int64)
        f["EE_ele_event_idx"] = np.asarray(ee_events, dtype=np.int64)
        f["SC_energy"] = np.zeros((len(eb_events), 4, 4), dtype=np.float32)
        f["EE_seed_energy"] = np.zeros((len(ee_events), 4, 4), dtype=np.float32)
        f["hasAdditionalTrk_EB"] = np.asarray(eb_hasadd, dtype=np.float32)
        f["hasAdditionalTrk_EE"] = np.asarray(ee_hasadd, dtype=np.int32)


@pytest.fixture
def tier_dir(tmp_path):
    """Builds a tiny tier with W, Z and 3 signal mass points.

    Each file: 4 events, pT all = 25 (lands in a single pT bin). EB objects: one
    per event. EE objects: one per event. hasADD chosen so each mode keeps some.
    """
    root = tmp_path / "MiniAOD"
    for sub in ("W", "Z", "signal"):
        (root / sub).mkdir(parents=True)

    pt = [25.0, 25.0, 25.0, 25.0]
    # Background: mostly hasADD==0, one ==1.
    for split in ("W", "Z"):
        _write_file(
            root / split / f"{split}_0.h5",
            n_events=4, eb_events=[0, 1, 2, 3], ee_events=[0, 1, 2, 3],
            eb_hasadd=[0, 0, 0, 1], ee_hasadd=[0, 0, 0, 1],
            source=BKG_SRC, a_pt=pt,
        )
    # Signal mass points: mix of hasADD 0/1.
    for mass, fname in (("H250_A2", "signal_0"), ("H250_A10", "signal_1"), ("H750_A0p4", "signal_2")):
        _write_file(
            root / "signal" / f"{fname}.h5",
            n_events=4, eb_events=[0, 1, 2, 3], ee_events=[0, 1, 2, 3],
            eb_hasadd=[1, 1, 0, 0], ee_hasadd=[1, 1, 0, 0],
            source=SIGNAL_SRC[mass], a_pt=pt,
        )
    return root
```

- [ ] **Step 2: Add a smoke test that the fixture is readable**

Append to `tests/test_category_weights.py`:

```python
import glob
import h5py


def test_fixture_files_present(tier_dir):
    files = glob.glob(str(tier_dir / "*" / "*.h5"))
    assert len(files) == 5  # W, Z, 3 signal
    with h5py.File(str(tier_dir / "signal" / "signal_0.h5"), "r") as f:
        assert f["hasAdditionalTrk_EB"].shape == (4,)
        assert "source_root_files" in f.attrs
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python3 -m pytest tests/test_category_weights.py -k "fixture_files_present" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_category_weights.py
git commit -m "test: synthetic HDF5 fixture factory for category weights"
```

---

## Task 4: Main build pipeline (scan → count → write)

**Files:**
- Modify: `build_category_weights.py`
- Test: `tests/test_category_weights.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_category_weights.py`:

```python
def _class_sums(out_h5, weight_key):
    """Sum weight_key over all objects, split into background vs signal."""
    bg = sig = 0.0
    seen_samples = set()
    with h5py.File(out_h5, "r") as f:
        for split in f["files"]:
            for stem in f["files"][split]:
                g = f["files"][split][stem]
                w = g[weight_key][:]
                sid = str(g.attrs["sample_id"])
                if sid != "background" and float(w.sum()) > 0:
                    seen_samples.add(sid)
                if sid == "background":
                    bg += float(w.sum())
                else:
                    sig += float(w.sum())
    return bg, sig, seen_samples


def test_build_all_mode_balances_each_class(tier_dir, tmp_path):
    out = tmp_path / "all" / "category_weights.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=out,
                               hasadd_mode="all", background_max_events=None, seed=1)
    for det, wk in bcw.WEIGHT_KEYS.items():
        bg, sig, samples = _class_sums(out, wk)
        assert np.isclose(bg, 1.0), (det, bg)
        assert np.isclose(sig, 1.0), (det, sig)
        # all 3 signal mass points included
        assert samples == {"signal_H250_A2", "signal_H250_A10", "signal_H750_A0p4"}


def test_build_eq1_excludes_a0p4_and_a1(tier_dir, tmp_path):
    out = tmp_path / "eq1" / "category_weights.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=out,
                               hasadd_mode="eq1", background_max_events=None, seed=1)
    bg, sig, samples = _class_sums(out, "EB_weight_split")
    # H750_A0p4 excluded by eq1; remaining signal still sums to 1.0
    assert "signal_H750_A0p4" not in samples
    assert np.isclose(sig, 1.0)
    assert np.isclose(bg, 1.0)


def test_build_eq1_only_keeps_hasadd1_objects(tier_dir, tmp_path):
    out = tmp_path / "eq1b" / "category_weights.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=out,
                               hasadd_mode="eq1", background_max_events=None, seed=1)
    # In the fixture, signal objects 0,1 are hasADD==1 and 2,3 are ==0.
    with h5py.File(out, "r") as f:
        g = f["files"]["signal"]["signal_0"]   # H250_A2, kept by eq1
        w = g["EB_weight_split"][:]
        assert w[0] > 0 and w[1] > 0
        assert w[2] == 0 and w[3] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_category_weights.py -k "build_" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'build_category_weights'`.

- [ ] **Step 3: Write minimal implementation**

Append to `build_category_weights.py`:

```python
def _object_selection(raw, detector, sample_id, hasadd_mode, n_events, bg_local_events):
    """Return (det_event_idx, obj_bin, selected_mask) for one file+detector."""
    det_event_idx = raw[DET_EVENT_IDX_KEYS[detector]][:].astype(np.int64)
    hasadd = raw[HASADD_KEYS[detector]][:]
    mask = hasadd_object_mask(hasadd, hasadd_mode)
    if is_excluded_sample(sample_id, hasadd_mode):
        mask = np.zeros(mask.shape, dtype=bool)
    if sample_id == "background" and bg_local_events is not None:
        ev_sel = np.zeros(n_events, dtype=bool)
        ev_sel[bg_local_events] = True
        mask = mask & ev_sel[det_event_idx]
    return det_event_idx, mask


def build_category_weights(input_dir, output_path, hasadd_mode,
                           background_max_events=None, seed=1234, event_pt_mode="max"):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_infos, n_bkg = scan_file_metadata(input_dir)
    bg_global, bg_selected = select_background_global_event_indices(
        n_bkg, background_max_events, seed)

    # ----- count pass: per detector, per sample, per pT bin (selected objects) -----
    counts = {det: {} for det in DETECTORS}
    for info in file_infos:
        bg_local = get_background_local_event_indices(bg_global, info) if info.sample_id == "background" else None
        with h5py.File(info.path, "r") as raw:
            a_pt = raw["A_pT"][:].astype(np.float64)
            a_event_idx = raw["A_event_idx"][:].astype(np.int64)
            event_pt = derive_event_pt(info.n_events, a_pt, a_event_idx, mode=event_pt_mode)
            for det in DETECTORS:
                det_event_idx, mask = _object_selection(
                    raw, det, info.sample_id, hasadd_mode, info.n_events, bg_local)
                if not np.any(mask):
                    continue
                obj_bin = find_bin_indices(event_pt[det_event_idx][mask])
                c = counts[det].setdefault(info.sample_id, np.zeros(N_BINS, dtype=np.int64))
                c += np.bincount(obj_bin, minlength=N_BINS)

    # ----- per-detector weight lookups -----
    lookups = {}
    for det in DETECTORS:
        targets = build_sample_targets(sorted(counts[det].keys()))
        lookups[det] = make_object_weight_lookup(counts[det], targets)

    # ----- write pass -----
    with h5py.File(output_path, "w") as out:
        out.attrs["input_dir"] = str(input_dir)
        out.attrs["hasadd_mode"] = hasadd_mode
        out.attrs["background_max_events"] = -1 if background_max_events is None else int(background_max_events)
        out.attrs["background_selected_events"] = int(bg_selected)
        out.attrs["seed"] = int(seed)
        out.attrs["weight_basis"] = "object"
        files_group = out.create_group("files")
        for info in file_infos:
            bg_local = get_background_local_event_indices(bg_global, info) if info.sample_id == "background" else None
            with h5py.File(info.path, "r") as raw:
                a_pt = raw["A_pT"][:].astype(np.float64)
                a_event_idx = raw["A_event_idx"][:].astype(np.int64)
                event_pt = derive_event_pt(info.n_events, a_pt, a_event_idx, mode=event_pt_mode)
                grp = files_group.require_group(info.split).create_group(info.stem)
                grp.attrs["input_file"] = str(info.path)
                grp.attrs["split"] = info.split
                grp.attrs["process_name"] = info.process_name
                grp.attrs["sample_id"] = info.sample_id
                grp.attrs["n_events"] = info.n_events
                grp.create_dataset("event_pt", data=event_pt, compression="gzip")
                for det in DETECTORS:
                    det_event_idx, mask = _object_selection(
                        raw, det, info.sample_id, hasadd_mode, info.n_events, bg_local)
                    w = np.zeros(det_event_idx.shape, dtype=np.float64)
                    if np.any(mask):
                        obj_bin = find_bin_indices(event_pt[det_event_idx][mask])
                        lut = lookups[det].get(info.sample_id)
                        if lut is not None:
                            w[mask] = lut[obj_bin]
                    grp.create_dataset(WEIGHT_KEYS[det], data=w, compression="gzip")
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_category_weights.py -k "build_" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full test module**

Run: `python3 -m pytest tests/test_category_weights.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add build_category_weights.py tests/test_category_weights.py
git commit -m "feat: object-based per-category weight build pipeline"
```

---

## Task 5: CLI entry point

**Files:**
- Modify: `build_category_weights.py`
- Test: manual CLI smoke (synthetic dir)

- [ ] **Step 1: Add `parse_args` and `main`**

Append to `build_category_weights.py`:

```python
def parse_args():
    p = argparse.ArgumentParser(
        description="Object-based per-category (tier x hasADD) event weights with "
                    "hasAdditionalTrk filtering and signal mass-point exclusion.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Tier directory containing W/, Z/, signal/.")
    p.add_argument("--output-path", type=Path, required=True,
                   help="Output HDF5 path (category_weights.h5).")
    p.add_argument("--hasadd-mode", choices=HASADD_MODES, required=True)
    p.add_argument("--background-max-events", type=int, default=1_000_000,
                   help="Cap on background events (resampling target). -1 = all.")
    p.add_argument("--event-pt-mode", choices=("max", "leading", "mean"), default="max")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    bmax = None if args.background_max_events is not None and args.background_max_events < 0 else args.background_max_events
    out = build_category_weights(
        input_dir=args.input_dir, output_path=args.output_path,
        hasadd_mode=args.hasadd_mode, background_max_events=bmax,
        seed=args.seed, event_pt_mode=args.event_pt_mode)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: CLI smoke test on a synthetic dir**

Run (creates a tiny tier via an inline python snippet, then runs the CLI):

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, "tests")
import conftest, pathlib, tempfile, h5py, numpy as np
d = pathlib.Path(tempfile.mkdtemp())
# reuse the fixture body by calling the helper directly
root = d / "MiniAOD"
for sub in ("W","Z","signal"): (root/sub).mkdir(parents=True)
pt=[25.0]*4
for s in ("W","Z"):
    conftest._write_file(root/s/f"{s}_0.h5",4,[0,1,2,3],[0,1,2,3],[0,0,0,1],[0,0,0,1],conftest.BKG_SRC,pt)
for mass,fn in (("H250_A2","signal_0"),("H250_A10","signal_1"),("H750_A0p4","signal_2")):
    conftest._write_file(root/"signal"/f"{fn}.h5",4,[0,1,2,3],[0,1,2,3],[1,1,0,0],[1,1,0,0],conftest.SIGNAL_SRC[mass],pt)
print(root)
PY
```

Then run the CLI against the printed dir:
```bash
python3 build_category_weights.py --input-dir <printed_root> \
  --output-path /tmp/yeo/cat_smoke/all/category_weights.h5 \
  --hasadd-mode all --background-max-events -1
```
Expected: prints `[saved] /tmp/yeo/cat_smoke/all/category_weights.h5`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add build_category_weights.py
git commit -m "feat: CLI for build_category_weights"
```

---

## Task 6: Driver script (6 runs: tier × hasADD)

**Files:**
- Create: `run_build_category_weights.sh`

- [ ] **Step 1: Write the driver**

Create `run_build_category_weights.sh`:

```bash
#!/usr/bin/env bash
set -eo pipefail

# Build all 6 per-category weight files (tier x hasADD) on the login node.
# EB and EE live inside each file. Source the cvmfs LCG view before running.

usage() {
  cat <<'EOF'
Usage: ./run_build_category_weights.sh [--data-root DIR] [--output-root DIR]
                                       [--python BIN] [--background-max-events N]
                                       [--tier aod|mini|both] [--hasadd all|eq0|eq1|all-modes]
Defaults: data-root=/eos/user/y/yeo/4l/data  output-root=/eos/user/y/yeo/4l/weights/categories
          python=python3  background-max-events=1000000  tier=both  hasadd=all-modes
EOF
}

DATA_ROOT="/eos/user/y/yeo/4l/data"
OUTPUT_ROOT="/eos/user/y/yeo/4l/weights/categories"
PYTHON_BIN="python3"
BG_MAX="1000000"
TIER="both"
HASADD="all-modes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --background-max-events) BG_MAX="$2"; shift 2;;
    --tier) TIER="$2"; shift 2;;
    --hasadd) HASADD="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${TIER}" in
  aod) TIERS=("AOD");; mini) TIERS=("MiniAOD");; both) TIERS=("AOD" "MiniAOD");;
  *) echo "--tier must be aod|mini|both" >&2; exit 1;;
esac
case "${HASADD}" in
  all-modes) MODES=("all" "eq0" "eq1");; all|eq0|eq1) MODES=("${HASADD}");;
  *) echo "--hasadd must be all|eq0|eq1|all-modes" >&2; exit 1;;
esac

for tier in "${TIERS[@]}"; do
  for mode in "${MODES[@]}"; do
    out="${OUTPUT_ROOT}/${tier}_${mode}/category_weights.h5"
    echo "=== ${tier} ${mode} -> ${out} ==="
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_category_weights.py" \
      --input-dir "${DATA_ROOT}/${tier}" \
      --output-path "${out}" \
      --hasadd-mode "${mode}" \
      --background-max-events "${BG_MAX}"
  done
done
echo "done."
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x run_build_category_weights.sh
bash -n run_build_category_weights.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add run_build_category_weights.sh
git commit -m "feat: driver to build all 6 per-category weight files"
```

---

## Task 7: Full run on the login node (manual, not in CI)

**Files:** none (operational step).

- [ ] **Step 1: Run all 6 categories in the background**

```bash
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u
cd /afs/cern.ch/user/y/yeo/4l/26.06.20/MergedID_SCAnalyzer
./run_build_category_weights.sh --python python3 > /tmp/yeo/cat_weights.log 2>&1 &
```

- [ ] **Step 2: Verify each output balances to 1.0 per class per detector**

For each of the 6 files, confirm `background` and `signal` weight-key sums ≈ 1.0
for both `EB_weight_split` and `EE_weight_split`, and that excluded mass points
carry zero weight (eq1: no `_A0p4`/`_A1`; eq0: no `signal_H250_A10`). Use the
`_class_sums` helper logic from `tests/test_category_weights.py` against each
`/eos/user/y/yeo/4l/weights/categories/<tier>_<mode>/category_weights.h5`.

Expected: 6 files, each `bg≈1.0 sig≈1.0` for EB and EE; correct exclusions.

---

## Self-Review notes

- **Spec coverage:** hasADD filter (Task 1/4), mass-point exclusion per mode
  (Task 1, tested Task 4), object-based balancing with signal split over included
  mass points (Task 2/4), 1M background resampling (Task 4 via
  `select_background_global_event_indices`, CLI default in Task 5), 6 files
  tier×hasADD (Task 6), training-contract-compatible output groups (Task 4).
- **ES-independence:** weights carry no ES axis — satisfied (no ES handling here).
- **Open risk:** EE `eq1` mass points with very low stats (e.g. `H2000_A2`≈91
  objects) are kept per decision; balancing still gives each included mass point
  an equal share, so those few objects get large weights — train-time rebalance
  only fixes class totals, not this. Acceptable per approach (b); revisit if a
  single-mass-point gradient dominates.
