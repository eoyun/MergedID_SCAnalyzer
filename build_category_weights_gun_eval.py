#!/usr/bin/env python3
"""Gun-heavy EVAL sample selection (3-class, 20 GeV inclusive) — NEW, existing untouched.

Common evaluation benchmark for the gunheavy_v1 models:
  gun        = particle_gun_heavy   (label 1, continuous a_mass)
  haa        = signal (H->AA)       (label 1, discrete a_mass)
  background = W, Z                 (label 0)

Selection = per-object reco SC ET >= --pt-min (default 20), i.e. a 20 GeV-cut INCLUSIVE
sample (NOT the 50/100/200 training cut). Uniform weight=1 on every selected object
(eval metrics are unweighted). Optional per-class random cap to bound the compact size.

On-disk layout mirrors build_category_weights_gun.py so build_compact_dataset_gun.py reads
it unchanged (adds a_mass for gun+haa, 0 for background). The gun/haa/background split is
carried by each group's `sample_id` attr -> compact `sample_id` column.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    PT_BIN_EDGES, PT_BIN_EDGES_WITH_OVERFLOW, robust_h5_open, SkippableFileError)

CLASS_OF = {"W": "background", "Z": "background",
            "particle_gun_heavy": "gun", "signal": "haa"}
CLASSES = ("gun", "haa", "background")
DETECTORS = ("eb", "ee")
DET_EVENT_IDX_KEYS = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}
SC_ET_KEYS = {"eb": "SC_energyT_sum", "ee": "EE_seed_energyT_sum"}
WEIGHT_KEYS = {"eb": "EB_weight_split", "ee": "EE_weight_split"}
OBJ_PT_KEYS = {"eb": "EB_obj_pt", "ee": "EE_obj_pt"}


def _read_meta(task):
    split, path = task
    try:
        with robust_h5_open(path) as r:
            ne = int(r["eventId"].shape[0])
            out = {"split": split, "path": str(path), "stem": Path(path).stem,
                   "sample_id": CLASS_OF[split], "n_events": ne, "ok": True}
            for det in DETECTORS:
                ei = r[DET_EVENT_IDX_KEYS[det]][:].astype(np.int64)
                pt = r[SC_ET_KEYS[det]][:].astype(np.float32)
                out[det] = (ei, pt)
    except SkippableFileError as e:
        return {"path": str(path), "ok": False, "err": str(e)}
    return out


def scan(input_dir, workers):
    tasks = []
    for split in CLASS_OF:
        for p in sorted((Path(input_dir) / split).glob("*.h5")):
            tasks.append((split, str(p)))
    print(f"[scan] {len(tasks)} files across {tuple(CLASS_OF)}", flush=True)
    metas = []
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(workers) as pool:
        for m in pool.imap_unordered(_read_meta, tasks):
            done += 1
            if not m.get("ok"):
                print(f"[skip] {m['path']}: {m.get('err')}", flush=True)
                continue
            metas.append(m)
            if done % 400 == 0:
                print(f"[scan] {done}/{len(tasks)}", flush=True)
    metas.sort(key=lambda m: (m["split"], m["stem"]))
    return metas


def class_masks(metas, det, pt_min, cls, cap, seed):
    """ET>=pt_min objects for class cls; random cap total if given."""
    per_file, counts = [], []
    for m in metas:
        if m["sample_id"] != cls:
            continue
        _ei, pt = m[det]
        elig = pt >= pt_min
        per_file.append((m["stem"], elig))
        counts.append(int(elig.sum()))
    n = int(sum(counts))
    if cap is None or n <= cap:
        return {s: e.copy() for s, e in per_file}, n, n
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n, size=cap, replace=False))
    masks, off = {}, 0
    for (s, e), c in zip(per_file, counts):
        lo = np.searchsorted(chosen, off, "left")
        hi = np.searchsorted(chosen, off + c, "left")
        pos = chosen[lo:hi] - off
        mm = np.zeros(e.shape, dtype=bool)
        if pos.size:
            mm[np.nonzero(e)[0][pos]] = True
        masks[s] = mm
        off += c
    return masks, cap, n


def build(input_dir, output_path, pt_min, caps, seed, workers):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metas = scan(input_dir, workers)
    if not metas:
        raise RuntimeError("no readable files")

    masks = {det: {} for det in DETECTORS}
    sel = {det: {} for det in DETECTORS}
    for det in DETECTORS:
        for cls in CLASSES:
            mk, nsel, nelig = class_masks(metas, det, pt_min, cls, caps.get(cls), seed)
            masks[det][cls] = mk
            sel[det][cls] = (nsel, nelig)
            print(f"[{det}] {cls:11s}: eligible(ET>={pt_min})={nelig:8d} selected={nsel}", flush=True)

    with h5py.File(output_path, "w") as out:
        out.attrs["input_dir"] = str(input_dir)
        out.attrs["hasadd_mode"] = "all"
        out.attrs["pt_variable"] = "reco_sc_energyT_sum"
        out.attrs["pt_min"] = float(pt_min)
        out.attrs["weight_basis"] = "eval_uniform"
        out.attrs["sample_kind"] = "gunheavy_eval_3class"
        out.attrs["seed"] = int(seed)
        for det in DETECTORS:
            for cls in CLASSES:
                out.attrs[f"selected_{cls}_{det}"] = int(sel[det][cls][0])
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        fg = out.create_group("files")
        for m in metas:
            cls = m["sample_id"]
            grp = fg.require_group(m["split"]).create_group(m["stem"])
            grp.attrs["input_file"] = m["path"]
            grp.attrs["split"] = m["split"]
            grp.attrs["process_name"] = m["split"]
            grp.attrs["sample_id"] = cls
            grp.attrs["n_events"] = m["n_events"]
            event_pt = np.zeros(m["n_events"], dtype=np.float64)
            for det in DETECTORS:
                ei, pt = m[det]
                if ei.size:
                    np.maximum.at(event_pt, ei, pt.astype(np.float64))
            grp.create_dataset("event_pt", data=event_pt, compression="gzip")
            for det in DETECTORS:
                ei, pt = m[det]
                mm = masks[det][cls].get(m["stem"], np.zeros(pt.shape, dtype=bool))
                w = np.where(mm, 1.0, 0.0)
                grp.create_dataset(WEIGHT_KEYS[det], data=w, compression="gzip")
                grp.create_dataset(DET_EVENT_IDX_KEYS[det], data=ei.astype(np.int64), compression="gzip")
                grp.create_dataset(OBJ_PT_KEYS[det], data=pt.astype(np.float32), compression="gzip")
    print(f"[saved] {output_path}", flush=True)
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Gun-heavy 3-class eval selection (SC ET>=pt_min inclusive).")
    p.add_argument("--input-dir", type=Path, default=Path("/eos/user/y/yeo/4l/data/MiniAOD"))
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--pt-min", type=float, default=20.0)
    p.add_argument("--gun-cap", type=int, default=300000, help="-1 = all")
    p.add_argument("--haa-cap", type=int, default=-1, help="-1 = all (signal/ full)")
    p.add_argument("--background-cap", type=int, default=400000, help="-1 = all")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main():
    a = parse_args()
    caps = {"gun": None if a.gun_cap < 0 else a.gun_cap,
            "haa": None if a.haa_cap < 0 else a.haa_cap,
            "background": None if a.background_cap < 0 else a.background_cap}
    build(a.input_dir, a.output_path, a.pt_min, caps, a.seed, a.workers)


if __name__ == "__main__":
    main()
