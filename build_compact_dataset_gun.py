#!/usr/bin/env python3
"""Gun-heavy compact builder (NEW; existing build_compact_dataset.py untouched).

Reads a gun-heavy category weight file (build_category_weights_gun.py) and writes a
model-readable compact, IDENTICAL on-disk layout to build_compact_dataset.py so the
datasets/trainers read it unchanged, PLUS two per-object columns needed for the
gun mass-regression campaign:

  pt      = per-object RECO supercluster ET (SC_energyT_sum / EE_seed_energyT_sum)
            -- data-applicable per-object pT (NOT the gen-A_pT broadcast that
            build_object_manifest fills; that is overridden here).
  a_mass  = continuous regression truth. For SIGNAL (particle_gun_heavy) it is the
            event's A mass (both A's in a gun event have identical mass, verified,
            so per-event mass = per-electron target; no A<->electron matching needed).
            For BACKGROUND (W/Z) it is 0 and is masked out in the regression loss
            (is_sig = label==1).

Split/registry/weights are reused from the existing manifest helpers.
"""
import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from build_compact_dataset import (
    SPLIT_CODE, VLEN_F32, VLEN_I64, VLEN_STR,
    calo_key, raw_source_keys, build_combined_manifest,
)
from build_pgun_compact import derive_event_mass
from build_event_level_weights import (
    PT_BIN_EDGES, PT_BIN_EDGES_WITH_OVERFLOW, robust_h5_open)

DET_EVENT_IDX_KEYS = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}
SC_ET_KEYS = {"eb": "SC_energyT_sum", "ee": "EE_seed_energyT_sum"}


def _read_file_objects_gun(task):
    """Worker: like build_compact_dataset._read_file_objects, plus per-object SC ET
    (pt) and continuous a_mass (event A mass for signal, 0 for background)."""
    (raw_path, calo_k, es_keys, track_stems, rows, obj_idx, det, is_signal) = task
    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    order = np.argsort(obj_idx, kind="stable")
    sorted_idx = obj_idx[order]
    inv = np.argsort(order, kind="stable")
    out = {"rows": np.asarray(rows, dtype=np.int64)}
    with robust_h5_open(raw_path) as r:
        out["calo"] = r[calo_k][sorted_idx][inv]
        out["es"] = {k: r[k][sorted_idx][inv] for k in es_keys}
        idx_d, val_d = {}, {}
        for stem in track_stems:
            ia = r[f"{stem}_idx"][:]
            va = r[f"{stem}_val"][:]
            idx_d[stem] = [np.asarray(ia[j], dtype=np.int64) for j in obj_idx]
            val_d[stem] = [np.asarray(va[j], dtype=np.float32) for j in obj_idx]
        out["track_idx"] = idx_d
        out["track_val"] = val_d
        # per-object reco SC ET (pt)
        et = r[SC_ET_KEYS[det]][:].astype(np.float32)
        out["pt"] = et[obj_idx]
        # per-object continuous a_mass
        if is_signal:
            am = r["A_mass"][:].astype(np.float64)
            aei = r["A_event_idx"][:].astype(np.int64)
            ne = int(r["eventId"].shape[0])
            emass = derive_event_mass(ne, am, aei)           # per-event A mass
            dev = r[DET_EVENT_IDX_KEYS[det]][:].astype(np.int64)  # object -> event
            out["a_mass"] = emass[dev[obj_idx]].astype(np.float32)
        else:
            out["a_mass"] = np.zeros(obj_idx.shape, dtype=np.float32)
    return out


def build(weight_h5_path, detector, track_types, output_path,
          train_frac=0.8, val_frac=0.1, seed=42, workers=16):
    weight_key = "EB_weight_split" if detector == "eb" else "EE_weight_split"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    man = build_combined_manifest(weight_h5_path, detector, weight_key,
                                  train_frac, val_frac, seed)
    file_entries = man["_file_entries"]
    code_to_sample = man["_code_to_sample"]
    n = len(man["label"])
    src = raw_source_keys(detector, track_types)
    calo_k = src["calo"][0]

    with robust_h5_open(weight_h5_path) as wf:
        hasadd_mode = str(wf.attrs.get("hasadd_mode", "all"))

    with h5py.File(output_path, "w") as out:
        out.attrs["detector"] = detector
        out.attrs["hasadd_mode"] = hasadd_mode
        out.attrs["track_types"] = ",".join(track_types)
        out.attrs["seed"] = int(seed)
        out.attrs["train_frac"] = float(train_frac)
        out.attrs["val_frac"] = float(val_frac)
        out.attrs["pt_variable"] = "reco_sc_energyT_sum"
        out.attrs["sample_kind"] = "gunheavy"
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES

        # bookkeeping columns from the manifest (NOTE: man["pt"] is the gen broadcast
        # placeholder and is intentionally NOT used; pt is filled per-object below).
        for k, dt in (("label", "i8"), ("weight", "f4"),
                      ("file_idx", "i4"), ("object_idx", "i4"),
                      ("event_idx", "i4"), ("sample_code", "i2"), ("split", "i1")):
            out.create_dataset(k, data=man[k].astype(dt), compression="gzip")
        sample_ids = np.array([code_to_sample[int(c)] for c in man["sample_code"]], dtype=object)
        raw_files = np.array([file_entries[int(fi)].raw_path for fi in man["file_idx"]], dtype=object)
        out.create_dataset("sample_id", data=sample_ids, dtype=VLEN_STR, compression="gzip")
        out.create_dataset("raw_file", data=raw_files, dtype=VLEN_STR, compression="gzip")
        # per-object columns filled from the workers
        pt_ds = out.create_dataset("pt", shape=(n,), dtype="f4", compression="gzip")
        amass_ds = out.create_dataset("a_mass", shape=(n,), dtype="f4", compression="gzip")

        r0_path = file_entries[int(man["file_idx"][0])].raw_path
        with robust_h5_open(r0_path) as r0:
            calo_shape = r0[calo_k].shape[1:]
            es_shapes = {k: r0[k].shape[1:] for k in src["es"]}
        calo_ds = out.create_dataset(calo_k, shape=(n,) + calo_shape, dtype="f4",
                                     compression="gzip", chunks=(1,) + calo_shape)
        es_ds = {k: out.create_dataset(k, shape=(n,) + es_shapes[k], dtype="f4",
                                       compression="gzip", chunks=(1,) + es_shapes[k]) for k in src["es"]}
        trk_idx = {t: out.create_dataset(f"{t}_idx", shape=(n,), dtype=VLEN_I64, compression="gzip")
                   for t in src["track"]}
        trk_val = {t: out.create_dataset(f"{t}_val", shape=(n,), dtype=VLEN_F32, compression="gzip")
                   for t in src["track"]}

        byfile = defaultdict(lambda: ([], []))
        for row in range(n):
            fi = int(man["file_idx"][row])
            byfile[fi][0].append(row)
            byfile[fi][1].append(int(man["object_idx"][row]))
        tasks = []
        for fi, (rws, oix) in byfile.items():
            is_sig = file_entries[fi].sample_id != "background"
            tasks.append((file_entries[fi].raw_path, calo_k, src["es"], src["track"],
                          np.asarray(rws, dtype=np.int64), np.asarray(oix, dtype=np.int64),
                          detector, is_sig))

        def _write(res):
            rws = res["rows"]
            wo = np.argsort(rws, kind="stable")
            rs = rws[wo]
            calo_ds[rs] = res["calo"][wo]
            pt_ds[rs] = res["pt"][wo]
            amass_ds[rs] = res["a_mass"][wo]
            for k in src["es"]:
                es_ds[k][rs] = res["es"][k][wo]
            for t in src["track"]:
                for j, row in zip(wo.tolist(), rs.tolist()):
                    trk_idx[t][int(row)] = res["track_idx"][t][j]
                    trk_val[t][int(row)] = res["track_val"][t][j]

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(_read_file_objects_gun, tasks):
                _write(res)
    return n


def parse_args():
    p = argparse.ArgumentParser(description="Build a gun-heavy compact (pt=reco SC ET, a_mass=continuous).")
    p.add_argument("--weight-h5", type=Path, required=True)
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--track-types", required=True, help="e.g. 'Lost,PF,GSF'")
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main():
    a = parse_args()
    tt = tuple(t for t in a.track_types.split(",") if t)
    n = build(a.weight_h5, a.detector, tt, a.output_path,
              a.train_frac, a.val_frac, a.seed, a.workers)
    print(f"[compact-gun] wrote {n} objects -> {a.output_path}", flush=True)


if __name__ == "__main__":
    main()
