# Sub-Project 3: Condor Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Submit the 18 image + 12 track + 18 fusion category trainings to
HTCondor (GPU) with stage-by-stage submit scripts, all driven by one category
source of truth.

**Architecture:** `pipeline/categories.py` enumerates every job (name, the exact
python argv, output dir, and for fusion its image+track run dirs).
`pipeline/make_submit.py` turns that into per-stage HTCondor submit files using
`queue arguments from <listfile>` (one job per line). A thin `condor/run_job.sh`
executable sources the cvmfs LCG view and runs `python3 "$@"`. Inputs (raw HDF5,
category weight files) are read from EOS; run outputs are written to EOS; the
five repo `.py` modules are shipped via `transfer_input_files` (AFS is absent on
workers). Stage 1 submits track+image; stage 2 submits fusion after stage 1
finishes.

**Tech Stack:** Python 3.13 (cvmfs LCG_109_cuda el9), HTCondor, pytest.

**Key facts (verified):**
- image/track scripts take `--weight-h5`; fusion takes `--image-run-dir`,
  `--track-run-dir`, `--prediction-source`.
- Local module deps to transfer: `train_resnet_image_classifier.py`,
  `track_point_transformer.py`, `train_point_transformer_track_classifier.py`,
  `train_fusion_ensemble.py`, `build_event_level_weights.py`.
- Category weight files: `/eos/user/y/yeo/4l/weights/categories/{AOD,MiniAOD}_{all,eq0,eq1}/category_weights.h5`
  (consumable as `--weight-h5`; `weight_key` defaults to `EB_weight_split`/`EE_weight_split` by detector — no need to pass).
- track_types: AOD=`GenTrk`, MiniAOD=`Lost,PF,GSF`.
- ES affects only image (`--no-es` for EE no-ES); fusion reads track_types from
  the track run_config; `validate_compatible_runs` passes because image+track of
  a category share the same weight_h5/detector/seed/splits and ES is not compared.
- Output runs root: `/eos/user/y/yeo/4l/runs/{image,track,fusion}_<name>`.

**Env:** `set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh; set -u`

---

## Naming

- track (12): `{ts}_{det}_{mode}` → e.g. `aod_eb_all`, `mini_ee_eq1`
- image (18): EB `{ts}_eb_{mode}`; EE `{ts}_ee_{es}_{mode}` with `es` in `es|noes`
- fusion (18): same suffix as its image; depends on track `{ts}_{det}_{mode}`

where `ts` in `aod|mini`, `mode` in `all|eq0|eq1`, `det` in `eb|ee`.

---

## File Structure

- Create: `pipeline/__init__.py` (empty)
- Create: `pipeline/categories.py` — single source of truth (pure data + argv).
- Create: `pipeline/make_submit.py` — emit submit files + arg lists + manifest.
- Create: `condor/run_job.sh` — executable wrapper (source env → `python3 "$@"`).
- Create: `pipeline/submit_train.sh`, `pipeline/submit_fusion.sh` — stage drivers.
- Create: `pipeline/collect_results.py` — gather fusion metrics.
- Test: `tests/test_categories.py`.

---

## Task 1: Category source of truth

**Files:**
- Create: `pipeline/__init__.py`, `pipeline/categories.py`
- Test: `tests/test_categories.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_categories.py`:

```python
import pipeline.categories as cat


def test_counts():
    assert len(cat.track_categories()) == 12
    assert len(cat.image_categories()) == 18
    assert len(cat.fusion_categories()) == 18


def test_track_types_by_tier():
    by_name = {c["name"]: c for c in cat.track_categories()}
    assert by_name["aod_eb_all"]["track_types"] == "GenTrk"
    assert by_name["mini_eb_all"]["track_types"] == "Lost,PF,GSF"


def test_image_argv_has_flags():
    by_name = {c["name"]: c for c in cat.image_categories()}
    # EE no-ES AOD eq1
    c = by_name["aod_ee_noes_eq1"]
    argv = c["argv"]
    assert "train_resnet_image_classifier.py" in argv[0]
    assert "--no-es" in argv
    assert "--track-types" in argv and "GenTrk" in argv
    assert "--detector" in argv and "ee" in argv
    # weight file path is the category file
    i = argv.index("--weight-h5")
    assert argv[i + 1].endswith("AOD_eq1/category_weights.h5")
    # EE with-ES must NOT carry --no-es
    assert "--no-es" not in by_name["aod_ee_es_eq1"]["argv"]


def test_fusion_points_to_shared_track():
    by_name = {c["name"]: c for c in cat.fusion_categories()}
    # both EE es/noes fuse with the same track run dir
    es = by_name["mini_ee_es_all"]
    noes = by_name["mini_ee_noes_all"]
    assert es["track_run"] == noes["track_run"]
    assert es["track_run"].endswith("track_mini_ee_all")
    assert es["image_run"].endswith("image_mini_ee_es_all")
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_categories.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'pipeline'`).

- [ ] **Step 3: Implement**

Create `pipeline/__init__.py` (empty) and `pipeline/categories.py`:

```python
"""Single source of truth for the 18/12/18 training categories."""

WEIGHTS_ROOT = "/eos/user/y/yeo/4l/weights/categories"
RUNS_ROOT = "/eos/user/y/yeo/4l/runs"

# (short, weight-dir tier, track_types)
TIERS = [("aod", "AOD", "GenTrk"), ("mini", "MiniAOD", "Lost,PF,GSF")]
DETECTORS = ("eb", "ee")
HASADD = ("all", "eq0", "eq1")
ES_OPTS = ("es", "noes")

# Training hyperparameters (edit here; one knob for the whole sweep).
EPOCHS = 20
IMAGE = {"model": "resnet18", "batch_size": "16", "lr": "1e-4"}
TRACK = {"batch_size": "64", "lr": "1e-4", "max_points": "16",
         "embed_dim": "128", "depth": "4", "num_heads": "4",
         "mlp_ratio": "4.0", "dropout": "0.1"}


def weight_h5(tier_dir, mode):
    return f"{WEIGHTS_ROOT}/{tier_dir}_{mode}/category_weights.h5"


def track_categories():
    out = []
    for ts, tdir, tt in TIERS:
        for det in DETECTORS:
            for mode in HASADD:
                name = f"{ts}_{det}_{mode}"
                outdir = f"{RUNS_ROOT}/track_{name}"
                argv = [
                    "train_point_transformer_track_classifier.py",
                    "--detector", det,
                    "--weight-h5", weight_h5(tdir, mode),
                    "--output-dir", outdir,
                    "--track-types", tt,
                    "--epochs", str(EPOCHS),
                    "--batch-size", TRACK["batch_size"],
                    "--lr", TRACK["lr"],
                    "--max-points", TRACK["max_points"],
                    "--embed-dim", TRACK["embed_dim"],
                    "--depth", TRACK["depth"],
                    "--num-heads", TRACK["num_heads"],
                    "--mlp-ratio", TRACK["mlp_ratio"],
                    "--dropout", TRACK["dropout"],
                ]
                out.append({"name": name, "tier": ts, "detector": det,
                            "hasadd": mode, "track_types": tt,
                            "outdir": outdir, "argv": argv})
    return out


def _image_variants():
    """Yield (name, det, mode, track_types, tier_dir, include_es)."""
    for ts, tdir, tt in TIERS:
        for mode in HASADD:
            yield (f"{ts}_eb_{mode}", "eb", mode, tt, tdir, True)
            for es in ES_OPTS:
                yield (f"{ts}_ee_{es}_{mode}", "ee", mode, tt, tdir, es == "es")


def image_categories():
    out = []
    for name, det, mode, tt, tdir, include_es in _image_variants():
        outdir = f"{RUNS_ROOT}/image_{name}"
        argv = [
            "train_resnet_image_classifier.py",
            "--detector", det,
            "--weight-h5", weight_h5(tdir, mode),
            "--output-dir", outdir,
            "--track-types", tt,
            "--model", IMAGE["model"],
            "--epochs", str(EPOCHS),
            "--batch-size", IMAGE["batch_size"],
            "--lr", IMAGE["lr"],
        ]
        if det == "ee" and not include_es:
            argv.append("--no-es")
        out.append({"name": name, "detector": det, "hasadd": mode,
                    "track_types": tt, "include_es": include_es,
                    "outdir": outdir, "argv": argv})
    return out


def fusion_categories():
    out = []
    for img in image_categories():
        name = img["name"]
        # shared track name: strip the es/noes token for EE
        parts = name.split("_")
        ts, det, mode = parts[0], parts[1], parts[-1]
        track_name = f"{ts}_{det}_{mode}"
        image_run = f"{RUNS_ROOT}/image_{name}"
        track_run = f"{RUNS_ROOT}/track_{track_name}"
        outdir = f"{RUNS_ROOT}/fusion_{name}"
        argv = [
            "train_fusion_ensemble.py",
            "--image-run-dir", image_run,
            "--track-run-dir", track_run,
            "--output-dir", outdir,
            "--prediction-source", "csv",
        ]
        out.append({"name": name, "image_run": image_run,
                    "track_run": track_run, "outdir": outdir, "argv": argv})
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_categories.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py pipeline/categories.py tests/test_categories.py
git commit -m "feat: category source of truth for condor sweep"
```

---

## Task 2: Condor executable wrapper

**Files:**
- Create: `condor/run_job.sh`

- [ ] **Step 1: Write the wrapper**

Create `condor/run_job.sh`:

```bash
#!/usr/bin/env bash
# Condor executable: source the cvmfs LCG view, then run the python job.
# arguments = "<script.py> <args...>" (passed verbatim as argv).
set -eo pipefail
echo "[host] $(hostname)"
echo "[args] $*"
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export MPLCONFIGDIR=/tmp/mplconfig_${USER:-job}
python3 -c "import torch; print('[gpu]', torch.cuda.is_available(), torch.cuda.device_count())" || true
exec python3 "$@"
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x condor/run_job.sh
bash -n condor/run_job.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add condor/run_job.sh
git commit -m "feat: condor job executable wrapper"
```

---

## Task 3: Submit-file generator

**Files:**
- Create: `pipeline/make_submit.py`
- Test: `tests/test_categories.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_categories.py`:

```python
import pipeline.make_submit as ms


def test_arglist_lines_roundtrip(tmp_path):
    cats = cat.track_categories()
    listfile = tmp_path / "track.txt"
    ms.write_arglist(listfile, cats)
    lines = [ln for ln in listfile.read_text().splitlines() if ln.strip()]
    assert len(lines) == 12
    # each line is the argv joined by spaces; track-types comma token intact
    assert any("train_point_transformer_track_classifier.py --detector eb" in ln for ln in lines)
    assert any("--track-types Lost,PF,GSF" in ln for ln in lines)


def test_submit_file_has_gpu_and_transfers(tmp_path):
    sub = tmp_path / "train_track.sub"
    ms.write_submit(sub, "train_track", tmp_path / "track.txt", request_gpus=1)
    text = sub.read_text()
    assert "request_gpus = 1" in text
    assert "queue arguments from" in text
    for mod in ("train_resnet_image_classifier.py", "track_point_transformer.py",
                "train_point_transformer_track_classifier.py",
                "train_fusion_ensemble.py", "build_event_level_weights.py"):
        assert mod in text
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_categories.py -k "arglist or submit_file" -v`
Expected: FAIL (`No module named 'pipeline.make_submit'`).

- [ ] **Step 3: Implement**

Create `pipeline/make_submit.py`:

```python
"""Generate HTCondor submit files + per-job arg lists from categories.py."""
import argparse
from pathlib import Path

from pipeline import categories as cat

TRANSFER_MODULES = [
    "train_resnet_image_classifier.py",
    "track_point_transformer.py",
    "train_point_transformer_track_classifier.py",
    "train_fusion_ensemble.py",
    "build_event_level_weights.py",
]


def write_arglist(path, cats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in cats:
            fh.write(" ".join(c["argv"]) + "\n")
    return path


SUBMIT_TEMPLATE = """\
universe                = vanilla
executable              = condor/run_job.sh
arguments               = $(arguments)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {transfers}
output = condor/logs/{stage}.$(Cluster).$(Process).out
error  = condor/logs/{stage}.$(Cluster).$(Process).err
log    = condor/logs/{stage}.$(Cluster).$(Process).log
request_cpus   = {cpus}
request_memory = {mem}
request_disk   = {disk}
{gpu_line}+JobFlavour = "{flavour}"

queue arguments from {listfile}
"""


def write_submit(path, stage, listfile, request_gpus=1, cpus=2, mem="8 GB",
                 disk="8 GB", flavour="tomorrow"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_line = f"request_gpus   = {request_gpus}\n" if request_gpus else ""
    text = SUBMIT_TEMPLATE.format(
        transfers=", ".join(TRANSFER_MODULES),
        stage=stage, cpus=cpus, mem=mem, disk=disk,
        gpu_line=gpu_line, flavour=flavour, listfile=listfile)
    path.write_text(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="condor", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    specs = [
        ("train_track", cat.track_categories()),
        ("train_image", cat.image_categories()),
        ("fusion", cat.fusion_categories()),
    ]
    for stage, cats in specs:
        listfile = args.out_dir / f"{stage}.txt"
        write_arglist(listfile, cats)
        write_submit(args.out_dir / f"{stage}.sub", stage, listfile)
        print(f"[{stage}] {len(cats)} jobs -> {args.out_dir / (stage + '.sub')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass + generate the real files**

Run: `python3 -m pytest tests/test_categories.py -k "arglist or submit_file" -v`
Expected: PASS.
Run: `python3 -m pipeline.make_submit`
Expected: prints `[train_track] 12 jobs ...`, `[train_image] 18 ...`, `[fusion] 18 ...`.

- [ ] **Step 5: Commit (generated condor/*.sub + *.txt + .py)**

```bash
git add pipeline/make_submit.py tests/test_categories.py condor/train_track.sub condor/train_image.sub condor/fusion.sub condor/train_track.txt condor/train_image.txt condor/fusion.txt
git commit -m "feat: condor submit-file generator + generated sweep files"
```

---

## Task 4: Stage driver scripts

**Files:**
- Create: `pipeline/submit_train.sh`, `pipeline/submit_fusion.sh`

- [ ] **Step 1: Write the drivers**

Create `pipeline/submit_train.sh`:

```bash
#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
python3 -m pipeline.make_submit
mkdir -p condor/logs
echo "[submit] track (12)"; condor_submit condor/train_track.sub
echo "[submit] image (18)"; condor_submit condor/train_image.sub
echo "Track + image submitted. Run pipeline/submit_fusion.sh after they finish."
```

Create `pipeline/submit_fusion.sh`:

```bash
#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
python3 -m pipeline.make_submit
echo "[submit] fusion (18)"; condor_submit condor/fusion.sub
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x pipeline/submit_train.sh pipeline/submit_fusion.sh
bash -n pipeline/submit_train.sh && bash -n pipeline/submit_fusion.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add pipeline/submit_train.sh pipeline/submit_fusion.sh
git commit -m "feat: stage-by-stage condor submit drivers"
```

---

## Task 5: Results collector

**Files:**
- Create: `pipeline/collect_results.py`
- Test: `tests/test_categories.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_categories.py`:

```python
import json
import pipeline.collect_results as cr


def test_collect_reads_fusion_metrics(tmp_path):
    runs = tmp_path / "runs"
    (runs / "fusion_aod_eb_all").mkdir(parents=True)
    (runs / "fusion_aod_eb_all" / "test_metrics.json").write_text(
        json.dumps({"weighted_auc": 0.9, "weighted_f1": 0.8}))
    rows = cr.collect(runs_root=runs)
    assert rows and rows[0]["name"] == "aod_eb_all"
    assert rows[0]["weighted_auc"] == 0.9
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/test_categories.py -k collect -v`
Expected: FAIL (`No module named 'pipeline.collect_results'`).

- [ ] **Step 3: Implement**

Create `pipeline/collect_results.py`:

```python
"""Collect fusion test metrics across all categories into one table."""
import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_RUNS_ROOT = "/eos/user/y/yeo/4l/runs"


def collect(runs_root):
    runs_root = Path(runs_root)
    rows = []
    for d in sorted(runs_root.glob("fusion_*")):
        metrics_path = d / "test_metrics.json"
        if not metrics_path.is_file():
            continue
        m = json.loads(metrics_path.read_text())
        rows.append({"name": d.name[len("fusion_"):],
                     "weighted_auc": m.get("weighted_auc"),
                     "weighted_f1": m.get("weighted_f1")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    args = ap.parse_args()
    rows = collect(args.runs_root)
    if not rows:
        print("No fusion metrics found yet.", file=sys.stderr)
        return
    w = csv.DictWriter(sys.stdout, fieldnames=["name", "weighted_auc", "weighted_f1"])
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `python3 -m pytest tests/test_categories.py -v`
Expected: PASS (all).
Run: `python3 -m pytest tests/ -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/collect_results.py tests/test_categories.py
git commit -m "feat: fusion results collector"
```

---

## Task 6: Pre-flight check (no full submit)

**Files:** none.

- [ ] **Step 1: Verify the metrics key names match the trainer output**

Confirm `train_fusion_ensemble.py` writes `test_metrics.json` with the keys
`collect_results.py` reads. Grep the fusion script for the metrics dump and
reconcile the key names (`weighted_auc`/`weighted_f1`); fix `collect_results.py`
to match the actual keys if they differ.

- [ ] **Step 2: Inspect generated submit + one arg line**

```bash
python3 -m pipeline.make_submit
sed -n '1,5p' condor/train_image.txt
cat condor/fusion.sub
```
Expected: 18 image lines; submit file references `condor/run_job.sh`,
`request_gpus = 1`, and `queue arguments from condor/fusion.txt`.

- [ ] **Step 3: (Operational, user-run) submit stage 1**

`./pipeline/submit_train.sh` → wait for completion (`condor_q`) → then
`./pipeline/submit_fusion.sh` → `python3 -m pipeline.collect_results`.

---

## Self-Review notes

- **Coverage:** categories source (Task 1), GPU/transfer submit (Task 3),
  stage drivers (Task 4), collector (Task 5), key-name reconciliation (Task 6).
- **Shared track:** fusion strips the es/noes token so both EE image variants
  point at the one `track_{ts}_ee_{mode}` run (tested).
- **Commas in args:** `--track-types Lost,PF,GSF` is a single space-delimited
  token, so `queue arguments from` keeps it intact (tested).
- **Risk:** `test_metrics.json` key names are assumed `weighted_auc`/`weighted_f1`;
  Task 6 reconciles against the real fusion output before any reliance on the
  collected table.
