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

    with h5py.File(weight_h5_path, "r") as wf:
        hasadd_mode = str(wf.attrs.get("hasadd_mode", ""))

    with h5py.File(output_path, "w") as out:
        out.attrs["detector"] = detector
        out.attrs["hasadd_mode"] = hasadd_mode
        out.attrs["track_types"] = ",".join(track_types)
        out.attrs["seed"] = int(seed)
        out.attrs["train_frac"] = float(train_frac)
        out.attrs["val_frac"] = float(val_frac)
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES

        for k, dt in (("label", "i8"), ("weight", "f4"), ("pt", "f4"),
                      ("file_idx", "i4"), ("object_idx", "i4"),
                      ("event_idx", "i4"), ("sample_code", "i2"), ("split", "i1")):
            out.create_dataset(k, data=man[k].astype(dt), compression="gzip")
        sample_ids = np.array([code_to_sample[int(c)] for c in man["sample_code"]], dtype=object)
        raw_files = np.array([file_entries[int(fi)].raw_path for fi in man["file_idx"]], dtype=object)
        out.create_dataset("sample_id", data=sample_ids, dtype=VLEN_STR, compression="gzip")
        out.create_dataset("raw_file", data=raw_files, dtype=VLEN_STR, compression="gzip")

        # derive per-object image shapes from a raw file (do not hardcode 32x32)
        r0_path = file_entries[int(man["file_idx"][0])].raw_path
        with h5py.File(r0_path, "r") as r0:
            calo_shape = r0[calo_k].shape[1:]
            es_shapes = {k: r0[k].shape[1:] for k in src["es"]}
        calo_ds = out.create_dataset(calo_k, shape=(n,) + calo_shape, dtype="f4",
                                     compression="gzip", chunks=(1,) + calo_shape)
        es_ds = {}
        for k in src["es"]:
            shp = es_shapes[k]
            es_ds[k] = out.create_dataset(k, shape=(n,) + shp, dtype="f4",
                                          compression="gzip", chunks=(1,) + shp)
        trk_idx = {t: out.create_dataset(f"{t}_idx", shape=(n,), dtype=VLEN_I64, compression="gzip")
                   for t in src["track"]}
        trk_val = {t: out.create_dataset(f"{t}_val", shape=(n,), dtype=VLEN_F32, compression="gzip")
                   for t in src["track"]}

        order = np.argsort(man["file_idx"], kind="stable")
        cur_fi, raw = -1, None
        try:
            for row in order:
                row = int(row)
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
