#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    PT_BIN_EDGES,
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


def _object_selection(raw, detector, sample_id, hasadd_mode, n_events, bg_local_events):
    """Return (det_event_idx, selected_mask) for one file+detector."""
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
        # pT bin edges used for binning; downstream efficiency-vs-pT reads
        # effective_pt_bin_edges to reproduce the exact same bins.
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
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
                    # Persist the object->event map so training does NOT need to
                    # open raw files at startup (that serial EOS-fuse scan was
                    # the dominant stall). Same length as the weight dataset.
                    grp.create_dataset(DET_EVENT_IDX_KEYS[det],
                                       data=det_event_idx.astype(np.int64),
                                       compression="gzip")
    return output_path


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
