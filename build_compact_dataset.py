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
