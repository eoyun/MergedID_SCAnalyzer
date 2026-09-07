#!/usr/bin/env python3
"""Build a model-readable compact from the particle_gun sample for MASS-REGRESSION
evaluation of the (already-trained) mreg models. The gun is all-signal (A->ee) with a
CONTINUOUS per-object A mass, so the compact stores an extra `a_mass` dataset (the
regression truth) alongside the usual per-object arrays the models read.

Same on-disk layout as compact_pt50 (so pipeline.compact_manifest + the datasets read
it unchanged): label/weight/pt/event_idx/sample_code/sample_id/split/object_idx/raw_file
+ calo image(s) + track vlen arrays, PLUS `a_mass`. All objects are signal (label=1,
split=test). Subsampled to --max-objects to keep it small (EOS is tight)."""
import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import derive_event_pt, robust_h5_open, PT_BIN_EDGES_WITH_OVERFLOW, PT_BIN_EDGES
from build_compact_dataset import raw_source_keys, calo_key, _read_file_objects, VLEN_F32, VLEN_I64, VLEN_STR

GUN_DIR = "/eos/user/y/yeo/4l/data/MiniAOD/particle_gun"
DET_EV = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}


def derive_event_mass(n_events, a_mass, a_event_idx):
    """Per-event A mass = mean of the A masses in that event (gun ~1 A/event)."""
    sums = np.bincount(a_event_idx, weights=a_mass, minlength=n_events).astype(np.float64)
    cnts = np.bincount(a_event_idx, minlength=n_events).astype(np.int64)
    out = np.zeros(n_events, dtype=np.float64)
    nz = cnts > 0
    out[nz] = sums[nz] / cnts[nz]
    return out


def build(detector, track_types, output_path, max_objects, seed):
    files = sorted(glob.glob(GUN_DIR + "/*.h5"))
    ev_key = DET_EV[detector]
    # ---- pass 1: per-file object counts + per-object (event_idx, pt, a_mass) ----
    per_file = []          # (path, n_obj, det_event_idx, obj_pt, obj_mass)
    total = 0
    for p in files:
        try:
            with robust_h5_open(p) as r:
                dei = r[ev_key][:].astype(np.int64)
                ne = int(r["eventId"].shape[0])
                apt = r["A_pT"][:].astype(np.float64)
                am = r["A_mass"][:].astype(np.float64)
                aei = r["A_event_idx"][:].astype(np.int64)
        except Exception as e:
            print(f"[skip] {os.path.basename(p)}: {e}", flush=True)
            continue
        ept = derive_event_pt(ne, apt, aei, mode="max")
        emass = derive_event_mass(ne, am, aei)
        per_file.append((p, len(dei), dei, ept[dei], emass[dei]))
        total += len(dei)
        if max_objects and total >= max_objects * 1.2:   # stop early (gun files are iid)
            break
    print(f"[scan] {len(per_file)} files read, {total} {detector} objects collected", flush=True)

    # ---- global random subsample to max_objects ----
    rng = np.random.default_rng(seed)
    keep = min(max_objects, total) if max_objects else total
    chosen = np.sort(rng.choice(total, size=keep, replace=False)) if keep < total else np.arange(total)
    n = len(chosen)
    src = raw_source_keys(detector, track_types)
    calo_k = src["calo"][0]

    # per-file: which local object indices are kept, and their global rows
    tasks, meta_rows, meta = [], [], {"pt": np.zeros(n, "f4"), "a_mass": np.zeros(n, "f4"),
                                       "event_idx": np.zeros(n, "i4"), "raw_file": np.empty(n, object)}
    offset = 0
    row = 0
    for (p, cnt, dei, opt, omass) in per_file:
        lo = np.searchsorted(chosen, offset, "left")
        hi = np.searchsorted(chosen, offset + cnt, "left")
        pos = chosen[lo:hi] - offset          # local object indices in this file
        if pos.size:
            rows = np.arange(row, row + pos.size, dtype=np.int64)
            tasks.append((p, calo_k, src["es"], src["track"], rows, pos.astype(np.int64)))
            meta["pt"][rows] = opt[pos]
            meta["a_mass"][rows] = omass[pos]
            meta["event_idx"][rows] = dei[pos]
            for rr in rows:
                meta["raw_file"][rr] = p
            row += pos.size
        offset += cnt

    # ---- shapes from first file ----
    with robust_h5_open(per_file[0][0]) as r0:
        calo_shape = r0[calo_k].shape[1:]
        es_shapes = {k: r0[k].shape[1:] for k in src["es"]}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        out.attrs["detector"] = detector
        out.attrs["track_types"] = ",".join(track_types)
        out.attrs["sample_kind"] = "particle_gun_eval"
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        # bookkeeping columns (all signal, single split=test=2)
        out.create_dataset("label", data=np.ones(n, "i8"), compression="gzip")
        out.create_dataset("weight", data=np.ones(n, "f4"), compression="gzip")
        out.create_dataset("pt", data=meta["pt"], compression="gzip")
        out.create_dataset("a_mass", data=meta["a_mass"], compression="gzip")   # <-- regression truth
        out.create_dataset("event_idx", data=meta["event_idx"], compression="gzip")
        out.create_dataset("object_idx", data=np.arange(n, dtype="i4"), compression="gzip")
        out.create_dataset("file_idx", data=np.zeros(n, "i4"), compression="gzip")
        out.create_dataset("sample_code", data=np.zeros(n, "i2"), compression="gzip")
        out.create_dataset("split", data=np.full(n, 2, "i1"), compression="gzip")   # 2=test
        out.create_dataset("sample_id", data=np.array(["particle_gun"] * n, object), dtype=VLEN_STR, compression="gzip")
        out.create_dataset("raw_file", data=meta["raw_file"], dtype=VLEN_STR, compression="gzip")
        calo_ds = out.create_dataset(calo_k, shape=(n,) + calo_shape, dtype="f4",
                                     compression="gzip", chunks=(1,) + calo_shape)
        es_ds = {k: out.create_dataset(k, shape=(n,) + es_shapes[k], dtype="f4",
                                       compression="gzip", chunks=(1,) + es_shapes[k]) for k in src["es"]}
        trk_idx = {t: out.create_dataset(f"{t}_idx", shape=(n,), dtype=VLEN_I64, compression="gzip") for t in src["track"]}
        trk_val = {t: out.create_dataset(f"{t}_val", shape=(n,), dtype=VLEN_F32, compression="gzip") for t in src["track"]}
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(8) as pool:
            for res in pool.imap_unordered(_read_file_objects, tasks):
                rws = res["rows"]; wo = np.argsort(rws, kind="stable"); rs = rws[wo]
                calo_ds[rs] = res["calo"][wo]
                for k in src["es"]:
                    es_ds[k][rs] = res["es"][k][wo]
                for t in src["track"]:
                    for j, rr in zip(wo.tolist(), rs.tolist()):
                        trk_idx[t][int(rr)] = res["track_idx"][t][j]
                        trk_val[t][int(rr)] = res["track_val"][t][j]
    print(f"[saved] {output_path}  n={n}  a_mass[min={meta['a_mass'].min():.3f} max={meta['a_mass'].max():.3f}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=("eb", "ee"), required=True)
    ap.add_argument("--track-types", default="Lost,PF,GSF")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--max-objects", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    build(a.detector, tuple(t for t in a.track_types.split(",") if t), a.output_path, a.max_objects, a.seed)


if __name__ == "__main__":
    main()
