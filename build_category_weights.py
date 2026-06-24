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
