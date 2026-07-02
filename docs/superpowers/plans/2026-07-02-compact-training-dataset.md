# Compact Training Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build, once per `(tier, detector, hasADD)`, one gzip-compressed HDF5
holding only the per-object arrays the models need (for weight-selected objects)
plus the full manifest, so training reads a single (local/staged) file instead of
opening ~1.5M scattered raw files over EOS-FUSE per epoch.

**Architecture:** The compact file stores datasets **named exactly like the raw
keys** the existing datasets read (`SC_energy`, `EE_seed_energy`,
`ES_seed_plane1/2_energy`, `{DET}_track_pt_{TYPE}_idx/val`), but indexed by a
dense compact-row 0..N-1, plus manifest arrays (`label,weight,pt,file_idx,
object_idx,event_idx,sample_code,split,sample_id,raw_file`). Because the names
match, the existing `DetectorObjectDataset._build_image` and
`build_track_point_cloud` work **unchanged** when handed the compact file as the
"raw" handle with `obj_idx = compact row`. Training with `--compact` builds its
manifest from the compact file (filtered by the stored `split`), sets
`file_entries = [the compact file]` and every object's `file_idx = 0`,
`object_idx = row`, and reuses the existing Datasets — no change to their
internals.

**Tech Stack:** Python 3.13 (cvmfs LCG_109_cuda el9), h5py, numpy, torch, pytest.

**Env:** `set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u`
(system python3 is too old; pytest only runs under the LCG view / on an el9 node.)

**Reuses (unchanged):** `parse_file_entries`, `build_event_registry`,
`build_split_event_maps`, `build_object_manifest`, `DetectorObjectDataset`,
`image_channel_keys`, `repeat_to_512`, `sparse_to_dense`, `preprocess_channel`
(train_resnet_image_classifier.py); `TrackPointCloudDataset`,
`build_track_point_cloud`, `parse_track_types` (track_point_transformer.py).

---

## File Structure
- Create: `build_compact_dataset.py` — builder (library + CLI).
- Create: `run_build_compact.sh` — driver over `(tier,det,hasADD)`.
- Create: `pipeline/compact_manifest.py` — `load_compact_manifest(path, split)` helper.
- Modify: `train_resnet_image_classifier.py` — `--compact` path in `main()`.
- Modify: `train_point_transformer_track_classifier.py` — `--compact` path in `main()`.
- Modify: `condor/run_job.sh` — stage the compact file to local scratch (if `--compact` present).
- Test: `tests/test_compact_dataset.py`.

Compact keys (per `(tier,det,hasADD)` file):
- calo: `SC_energy` (eb) or `EE_seed_energy` (ee), shape `(N,32,32)`.
- ES (ee only): `ES_seed_plane1_energy (N,16,512)`, `ES_seed_plane2_energy (N,512,16)`.
- tracks: for each `TYPE` in track_types, `{PREFIX}_track_pt_{TYPE}_idx` and
  `..._val` as **vlen** rows (PREFIX = EB/EE).
- manifest: `label(i8) weight(f4) pt(f4) file_idx(i4) object_idx(i4)
  event_idx(i4) sample_code(i2) split(i1)` + `sample_id`, `raw_file` (vlen str).
- attrs: `detector, tier, hasadd, track_types(csv), seed, train_frac, val_frac,
  effective_pt_bin_edges`.

---

## Task 1: Compact schema constants + row-packing helper

**Files:** Create `build_compact_dataset.py`; Test `tests/test_compact_dataset.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compact_dataset.py`:

```python
import numpy as np
import build_compact_dataset as bcd


def test_calo_key_and_channel_source_keys():
    assert bcd.calo_key("eb") == "SC_energy"
    assert bcd.calo_key("ee") == "EE_seed_energy"
    # datasets we must copy per object for a detector + track types
    eb = bcd.raw_source_keys("eb", ("Lost", "PF", "GSF"))
    assert "SC_energy" in eb["calo"]
    assert eb["track"] == ["EB_track_pt_Lost", "EB_track_pt_PF", "EB_track_pt_GSF"]
    assert eb["es"] == []
    ee = bcd.raw_source_keys("ee", ("GenTrk",))
    assert ee["calo"] == ["EE_seed_energy"]
    assert ee["es"] == ["ES_seed_plane1_energy", "ES_seed_plane2_energy"]
    assert ee["track"] == ["EE_track_pt_GenTrk"]
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_compact_dataset.py -k calo_key -v`
Expected: FAIL (`No module named 'build_compact_dataset'`).

- [ ] **Step 3: Implement**

Create `build_compact_dataset.py`:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np

from train_resnet_image_classifier import (
    parse_file_entries,
    build_event_registry,
    build_split_event_maps,
    build_object_manifest,
)
from build_event_level_weights import PT_BIN_EDGES, PT_BIN_EDGES_WITH_OVERFLOW

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
VLEN_F32 = h5py.special_dtype(vlen=np.float32)
VLEN_I64 = h5py.special_dtype(vlen=np.int64)
VLEN_STR = h5py.special_dtype(vlen=str)


def calo_key(detector):
    return "SC_energy" if detector == "eb" else "EE_seed_energy"


def raw_source_keys(detector, track_types):
    prefix = "EB" if detector == "eb" else "EE"
    es = [] if detector == "eb" else ["ES_seed_plane1_energy", "ES_seed_plane2_energy"]
    return {
        "calo": [calo_key(detector)],
        "es": es,
        "track": [f"{prefix}_track_pt_{t}" for t in track_types],
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_compact_dataset.py -k calo_key -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add build_compact_dataset.py tests/test_compact_dataset.py
git commit -m "feat: compact dataset schema keys"
```

---

## Task 2: Build the combined manifest (train/val/test) with split labels

**Files:** Modify `build_compact_dataset.py`; Test `tests/test_compact_dataset.py`.

- [ ] **Step 1: Write the failing test** (uses the existing fixture + category weights)

Append to `tests/test_compact_dataset.py`:

```python
import build_category_weights as bcw


def test_combined_manifest_has_all_splits(tier_dir, tmp_path):
    wpath = tmp_path / "w.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=wpath,
                               hasadd_mode="all", background_max_events=None, seed=1)
    # weight builder must also store event_idx (added earlier) for registry to work
    m = bcd.build_combined_manifest(weight_h5_path=wpath, detector="eb",
                                    weight_key="EB_weight_split",
                                    train_frac=0.6, val_frac=0.2, seed=1)
    assert set(np.unique(m["split"])).issubset({0, 1, 2})
    for k in ("label", "weight", "pt", "file_idx", "object_idx", "event_idx",
              "sample_code", "split"):
        assert len(m[k]) == len(m["label"])
    assert (m["weight"] > 0).all()
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_compact_dataset.py -k combined_manifest -v`
Expected: FAIL (`no attribute 'build_combined_manifest'`).

- [ ] **Step 3: Implement**

Append to `build_compact_dataset.py`:

```python
def build_combined_manifest(weight_h5_path, detector, weight_key,
                            train_frac, val_frac, seed,
                            debug=False, debug_max_events_per_sample=12):
    """Reuse the existing event-level split + object manifest, tag each object
    with its split, and concatenate train+val+test into one flat manifest."""
    weight_h5_path = Path(weight_h5_path)
    file_entries = parse_file_entries(weight_h5_path)
    ev_file, ev_local, ev_code, sample_to_code, code_to_sample = build_event_registry(
        file_entries=file_entries, weight_h5_path=weight_h5_path, detector=detector,
        weight_key=weight_key, debug=debug,
        debug_max_events_per_sample=debug_max_events_per_sample, seed=seed)
    split_maps = build_split_event_maps(ev_file, ev_local, ev_code,
                                        train_frac=train_frac, val_frac=val_frac, seed=seed)
    parts = []
    for split_name, split_map in split_maps.items():
        man = build_object_manifest(file_entries, weight_h5_path, detector,
                                    weight_key, split_map, sample_to_code)
        man = dict(man)
        man["split"] = np.full(len(man["label"]), SPLIT_CODE[split_name], dtype=np.int8)
        parts.append(man)
    combined = {}
    for k in parts[0]:
        combined[k] = np.concatenate([p[k] for p in parts])
    combined["_file_entries"] = file_entries
    combined["_code_to_sample"] = code_to_sample
    return combined
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_compact_dataset.py -k combined_manifest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add build_compact_dataset.py tests/test_compact_dataset.py
git commit -m "feat: combined split-tagged manifest for compact builder"
```

---

## Task 3: Pack objects into the compact HDF5

**Files:** Modify `build_compact_dataset.py`; Test `tests/test_compact_dataset.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compact_dataset.py`:

```python
import h5py


def test_build_compact_roundtrip(tier_dir, tmp_path):
    wpath = tmp_path / "w.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=wpath,
                               hasadd_mode="all", background_max_events=None, seed=1)
    out = tmp_path / "compact_eb.h5"
    n = bcd.build_compact_dataset(weight_h5_path=wpath, detector="eb",
                                  track_types=("Lost", "PF", "GSF"), output_path=out,
                                  train_frac=0.6, val_frac=0.2, seed=1)
    with h5py.File(out, "r") as f:
        assert f["SC_energy"].shape == (n, 32, 32)
        for t in ("Lost", "PF", "GSF"):
            assert f[f"EB_track_pt_{t}_idx"].shape == (n,)   # vlen
        assert len(f["label"]) == n and len(f["split"]) == n
        assert f.attrs["detector"] == "eb"
        assert f.attrs["track_types"] == "Lost,PF,GSF"
        # first row's calo equals the raw file's array for that object
        raw_file = f["raw_file"][0]; oidx = int(f["object_idx"][0])
        with h5py.File(raw_file, "r") as r:
            assert np.array_equal(f["SC_energy"][0], r["SC_energy"][oidx])
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_compact_dataset.py -k build_compact_roundtrip -v`
Expected: FAIL (`no attribute 'build_compact_dataset'`).

- [ ] **Step 3: Implement**

Append to `build_compact_dataset.py`:

```python
def build_compact_dataset(weight_h5_path, detector, track_types, output_path,
                          train_frac=0.8, val_frac=0.1, seed=42,
                          debug=False, debug_max_events_per_sample=12):
    weight_key = "EB_weight_split" if detector == "eb" else "EE_weight_split"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    man = build_combined_manifest(weight_h5_path, detector, weight_key,
                                  train_frac, val_frac, seed, debug,
                                  debug_max_events_per_sample)
    file_entries = man["_file_entries"]
    code_to_sample = man["_code_to_sample"]
    n = len(man["label"])
    src = raw_source_keys(detector, track_types)
    calo_k = src["calo"][0]

    with h5py.File(output_path, "w") as out:
        out.attrs["detector"] = detector
        out.attrs["hasadd_mode"] = str(h5py.File(weight_h5_path, "r").attrs.get("hasadd_mode", ""))
        out.attrs["track_types"] = ",".join(track_types)
        out.attrs["seed"] = int(seed)
        out.attrs["train_frac"] = float(train_frac)
        out.attrs["val_frac"] = float(val_frac)
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES

        # manifest datasets
        for k, dt in (("label", "i8"), ("weight", "f4"), ("pt", "f4"),
                      ("file_idx", "i4"), ("object_idx", "i4"),
                      ("event_idx", "i4"), ("sample_code", "i2"), ("split", "i1")):
            out.create_dataset(k, data=man[k].astype(dt), compression="gzip")
        sample_ids = np.array([code_to_sample[int(c)] for c in man["sample_code"]], dtype=object)
        raw_files = np.array([file_entries[int(fi)].raw_path for fi in man["file_idx"]], dtype=object)
        out.create_dataset("sample_id", data=sample_ids, dtype=VLEN_STR, compression="gzip")
        out.create_dataset("raw_file", data=raw_files, dtype=VLEN_STR, compression="gzip")

        # calo (N,32,32) + ES (ee)
        calo_ds = out.create_dataset(calo_k, shape=(n, 32, 32), dtype="f4",
                                     compression="gzip", chunks=(1, 32, 32))
        es_ds = {}
        for k in src["es"]:
            with h5py.File(file_entries[int(man["file_idx"][0])].raw_path, "r") as r0:
                shp = r0[k].shape[1:]
            es_ds[k] = out.create_dataset(k, shape=(n,) + shp, dtype="f4",
                                          compression="gzip", chunks=(1,) + shp)
        # track vlen
        trk_idx = {t: out.create_dataset(f"{t}_idx", shape=(n,), dtype=VLEN_I64, compression="gzip")
                   for t in src["track"]}
        trk_val = {t: out.create_dataset(f"{t}_val", shape=(n,), dtype=VLEN_F32, compression="gzip")
                   for t in src["track"]}

        # fill, grouping rows by source raw file to open each raw once
        order = np.argsort(man["file_idx"], kind="stable")
        cur_fi, raw = -1, None
        try:
            for row in order:
                fi = int(man["file_idx"][row]); oidx = int(man["object_idx"][row])
                if fi != cur_fi:
                    if raw is not None:
                        raw.close()
                    raw = h5py.File(file_entries[fi].raw_path, "r")
                    cur_fi = fi
                calo_ds[row] = raw[calo_k][oidx]
                for k in src["es"]:
                    es_ds[k][row] = raw[k][oidx]
                for t in src["track"]:
                    trk_idx[t][row] = np.asarray(raw[f"{t}_idx"][oidx], dtype=np.int64)
                    trk_val[t][row] = np.asarray(raw[f"{t}_val"][oidx], dtype=np.float32)
        finally:
            if raw is not None:
                raw.close()
    return n
```

- [ ] **Step 4: Run to verify pass + full module**

Run: `python3 -m pytest tests/test_compact_dataset.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add build_compact_dataset.py tests/test_compact_dataset.py
git commit -m "feat: pack selected objects into compact HDF5"
```

---

## Task 4: CLI + driver

**Files:** Modify `build_compact_dataset.py`; Create `run_build_compact.sh`.

- [ ] **Step 1: Add CLI**

Append to `build_compact_dataset.py`:

```python
def parse_args():
    p = argparse.ArgumentParser(description="Build a compact per-category training HDF5.")
    p.add_argument("--weight-h5", type=Path, required=True)
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--track-types", required=True, help="e.g. 'Lost,PF,GSF' or 'GenTrk'")
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    tt = tuple(t for t in a.track_types.split(",") if t)
    n = build_compact_dataset(a.weight_h5, a.detector, tt, a.output_path,
                              a.train_frac, a.val_frac, a.seed)
    print(f"[compact] wrote {n} objects -> {a.output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `run_build_compact.sh`**

```bash
#!/usr/bin/env bash
set -eo pipefail
# Build all 12 compact files (tier x detector x hasADD) on the login node.
WROOT="/eos/user/y/yeo/4l/weights/categories"
OUT="/eos/user/y/yeo/4l/compact"
PY="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A TT=( [AOD]="GenTrk" [MiniAOD]="Lost,PF,GSF" )
for tier in AOD MiniAOD; do
  for det in eb ee; do
    for mode in all eq0 eq1; do
      w="${WROOT}/${tier}_${mode}/category_weights.h5"
      o="${OUT}/${tier}_${det}_${mode}.h5"
      echo "=== ${tier} ${det} ${mode} -> ${o} ==="
      "${PY}" "${SCRIPT_DIR}/build_compact_dataset.py" --weight-h5 "${w}" \
        --detector "${det}" --track-types "${TT[$tier]}" --output-path "${o}"
    done
  done
done
echo done.
```

- [ ] **Step 3: Make executable + syntax + CLI help**

```bash
chmod +x run_build_compact.sh
bash -n run_build_compact.sh && echo "syntax OK"
python3 build_compact_dataset.py --help | grep -E "weight-h5|detector|track-types"
```
Expected: `syntax OK` and the three flags listed.

- [ ] **Step 4: Commit**

```bash
git add build_compact_dataset.py run_build_compact.sh
git commit -m "feat: compact builder CLI + 12-file driver"
```

---

## Task 5: Training reads the compact file (`--compact`)

**Files:** Create `pipeline/compact_manifest.py`; Modify both training scripts;
Test `tests/test_compact_dataset.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compact_dataset.py`:

```python
import pipeline.compact_manifest as cm


def test_load_compact_manifest_split(tier_dir, tmp_path):
    wpath = tmp_path / "w.h5"; out = tmp_path / "c_eb.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=wpath,
                               hasadd_mode="all", background_max_events=None, seed=1)
    bcd.build_compact_dataset(wpath, "eb", ("Lost", "PF", "GSF"), out,
                              train_frac=0.6, val_frac=0.2, seed=1)
    fe, manifests = cm.load_compact_manifest(out)
    assert len(fe) == 1 and fe[0].raw_path == str(out)
    for split in ("train", "val", "test"):
        m = manifests[split]
        # every object points at the single compact file, object_idx == row
        assert (m["file_idx"] == 0).all()
        assert m["object_idx"].max() < len(cm._h5_len(out))
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_compact_dataset.py -k load_compact_manifest -v`
Expected: FAIL (`No module named 'pipeline.compact_manifest'`).

- [ ] **Step 3: Implement the loader**

Create `pipeline/compact_manifest.py`:

```python
"""Load a compact dataset file into (file_entries, per-split manifests) so the
existing DetectorObjectDataset / TrackPointCloudDataset can read it unchanged:
each object's file_idx=0 and object_idx=its compact row."""
import h5py
import numpy as np

from train_resnet_image_classifier import FileEntry

SPLIT_NAME = {0: "train", 1: "val", 2: "test"}


def _h5_len(path):
    with h5py.File(path, "r") as f:
        return f["label"][:]


def load_compact_manifest(compact_path):
    compact_path = str(compact_path)
    with h5py.File(compact_path, "r") as f:
        split = f["split"][:]
        base = {k: f[k][:] for k in ("label", "weight", "pt", "event_idx", "sample_code")}
        n = len(split)
    rows = np.arange(n, dtype=np.int32)
    fe = [FileEntry(split="compact", stem="compact", raw_path=compact_path,
                    sample_id="compact", process_name="compact", n_events=n)]
    manifests = {}
    for code, name in SPLIT_NAME.items():
        m = split == code
        manifests[name] = {
            "file_idx": np.zeros(int(m.sum()), dtype=np.int32),
            "object_idx": rows[m].astype(np.int32),
            "event_idx": base["event_idx"][m].astype(np.int32),
            "label": base["label"][m].astype(np.int64),
            "weight": base["weight"][m].astype(np.float32),
            "pt": base["pt"][m].astype(np.float32),
            "sample_code": base["sample_code"][m].astype(np.int16),
        }
    return fe, manifests
```

- [ ] **Step 4: Wire `--compact` into the image script**

In `train_resnet_image_classifier.py` `main()`, add arg
`parser.add_argument("--compact", type=Path, default=None)` and, right after the
split/manifest section, branch:

```python
    if args.compact is not None:
        from pipeline.compact_manifest import load_compact_manifest
        file_entries, split_manifests = load_compact_manifest(args.compact)
        code_to_sample = None  # sample_id read from compact if needed
        class_balance_stats = {}
        for s in ("train", "val", "test"):
            split_manifests[s], class_balance_stats[s] = rebalance_manifest_class_weights(split_manifests[s])
    else:
        ... existing parse_file_entries/build_event_registry/... path ...
```

`DetectorObjectDataset(detector, file_entries, split_manifests[...], log_scale,
track_types, include_es)` then opens the compact file (raw_path) and
`_build_image` reads `SC_energy[row]` etc. from it **unchanged** (keys match).

- [ ] **Step 5: Wire `--compact` into the track script**

Same pattern in `train_point_transformer_track_classifier.py` `main()`:
`--compact` → `load_compact_manifest` → `TrackPointCloudDataset(detector,
file_entries, split_manifests[...], max_points, track_types, log_track_pt)`
reads `{PREFIX}_track_pt_{T}_idx/val[row]` from the compact file unchanged.

- [ ] **Step 6: Run tests + import checks**

Run: `python3 -m pytest tests/test_compact_dataset.py -v`
Expected: PASS.
Run: `python3 -c "import train_resnet_image_classifier, train_point_transformer_track_classifier"`
Expected: no error.
Run: `python3 train_resnet_image_classifier.py --help | grep compact`
Expected: `--compact` shown.

- [ ] **Step 7: Commit**

```bash
git add pipeline/compact_manifest.py train_resnet_image_classifier.py train_point_transformer_track_classifier.py tests/test_compact_dataset.py
git commit -m "feat: training reads compact dataset via --compact"
```

---

## Task 6: Stage compact to local scratch in run_job.sh

**Files:** Modify `condor/run_job.sh`.

- [ ] **Step 1: Add compact stage-in**

After sourcing the env and before running python, if a `--compact <eospath>`
appears in the args, xrdcp it to local scratch and rewrite the arg to the local
path (single small file → fast, no per-object FUSE):

```bash
# stage-in compact dataset (single file) to local scratch
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--compact" ]]; then
    eos_compact="${args[$((i+1))]}"
    local_compact="${scratch}/$(basename "${eos_compact}")"
    echo "[stage-in] ${eos_compact} -> ${local_compact}"
    xrdcp -f "root://eosuser.cern.ch/${eos_compact}" "${local_compact}"
    args[$((i+1))]="${local_compact}"
  fi
done
```
(Place this next to the existing `--output-dir` rewrite loop, before
`python3 "${args[@]}"`.)

- [ ] **Step 2: Syntax check + commit**

```bash
bash -n condor/run_job.sh && echo OK
git add condor/run_job.sh
git commit -m "feat: stage compact dataset to local scratch in run_job"
```

---

## Task 7: Build one compact + 1-epoch canary (measure)

**Files:** none (operational).

- [ ] **Step 1: Build MiniAOD_eb_all compact on the login node**

```bash
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u
python3 build_compact_dataset.py \
  --weight-h5 /eos/user/y/yeo/4l/weights/categories/MiniAOD_all/category_weights.h5 \
  --detector eb --track-types Lost,PF,GSF \
  --output-path /eos/user/y/yeo/4l/compact/MiniAOD_eb_all.h5
```
Expected: `[compact] wrote ~1.35M objects -> ...`; on-disk size ~0.4 GB.

- [ ] **Step 2: Canary — 1 epoch reading the compact, on a high-end GPU**

Submit a 1-job canary (image, epochs 1, `--compact .../MiniAOD_eb_all.h5`,
`--num-workers 8`, `require_gpus (Capability>=8.0)`, `max_retries 3`, stage-out).
Reuse the `condor/canary.*` pattern with the compact arg.

- [ ] **Step 3: Verify (evidence)**

Confirm from the worker/logs: reaches `[start training]` fast, the epoch
**completes** with a measured wall time, `xrdcp` stage-out succeeds, and
`runs/canary/image_mini_eb_all/` has `best_model.pt`, `val/test_predictions.csv`,
`test_metrics.json`. Record the 1-epoch wall time. Only scale to the full
MiniAOD sweep after this passes.

---

## Self-Review notes
- **Spec coverage:** 12-file granularity (Task 4 driver), gzip compression
  (Task 3 `compression="gzip"`), manifest+split stored (Task 2/3), image+track
  share one file via matching key names (Task 5 reuse), build-once on login node
  (Task 4/7), path `/eos/.../compact/` (Task 4), stage single file (Task 6),
  canary measurement (Task 7).
- **Reuse correctness:** compact datasets are named exactly like the raw keys
  (`SC_energy`, `{PREFIX}_track_pt_{T}_idx/val`, `ES_seed_plane*`) so
  `_build_image` / `build_track_point_cloud` read them unchanged with
  `obj_idx = row`; `file_entries=[compact]`, `file_idx=0` for all objects.
- **Risk:** requires `EB/EE_ele_event_idx` present in the category weight file
  (added earlier) for `build_event_registry`; MiniAOD_all is already patched,
  the rest need the patch or a rebuild before their compacts are built.
- **Env risk:** pytest needs the LCG view (el9); run tests on an el9 node.
