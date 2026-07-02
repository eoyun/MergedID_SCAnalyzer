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
