#!/usr/bin/env python3
"""Build a low-cut (pT>=20 GeV) EVALUATION sample selection, in the same file
format as build_category_weights.py, so build_compact_dataset.py can turn it into
an eval compact with zero changes.

This is an EVAL sample, not a training sample:
  * signal      = 20% random of ALL signal objects (NO pT cut), per (detector,mode),
                  respecting hasADD mode + the same mass-point exclusions as training.
  * background  = from objects with event pT >= 20 GeV, take up to `bg_max` (1e6)
                  at random, then keep `bg_frac` (20%) of that (i.e. 0.2*min(1e6,N)).
                  If fewer than bg_max are eligible, keep 20% of all eligible.

Because downstream evaluation is UNWEIGHTED, the stored weight is simply 1.0 for a
selected object and 0.0 otherwise (0 = not in the sample). Object selection is what
matters; the split grouping is preserved only so build_compact reads it unchanged.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    PT_BIN_EDGES,
    PT_BIN_EDGES_WITH_OVERFLOW,
    derive_event_pt,
    scan_file_metadata,
)
from build_category_weights import (
    DETECTORS,
    HASADD_KEYS,
    DET_EVENT_IDX_KEYS,
    WEIGHT_KEYS,
    hasadd_object_mask,
    is_excluded_sample,
)


def _global_subsample(per_file_eligible, target, seed):
    """per_file_eligible = list of (stem, bool_mask). Randomly keep `target` of the
    True positions globally across files (or all if target>=N or target is None).
    Returns {stem: selected_bool_mask}, n_selected."""
    counts = [int(m.sum()) for _, m in per_file_eligible]
    n_elig = int(sum(counts))
    if target is None or target >= n_elig:
        return {stem: m.copy() for stem, m in per_file_eligible}, n_elig
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_elig, size=int(target), replace=False))
    out, offset = {}, 0
    for (stem, elig), cnt in zip(per_file_eligible, counts):
        lo = np.searchsorted(chosen, offset, "left")
        hi = np.searchsorted(chosen, offset + cnt, "left")
        pos = chosen[lo:hi] - offset
        m = np.zeros(elig.shape, dtype=bool)
        if pos.size:
            m[np.nonzero(elig)[0][pos]] = True
        out[stem] = m
        offset += cnt
    return out, int(target)


def _eligible_masks(file_infos, detector, hasadd_mode, is_background, pt_min,
                    event_pt_mode):
    """Per-file eligibility for the given class. Signal: hasADD-mode & not-excluded,
    NO pT cut. Background: hasADD-mode & event pT >= pt_min."""
    ha_key = HASADD_KEYS[detector]
    ev_key = DET_EVENT_IDX_KEYS[detector]
    per_file = []
    for info in file_infos:
        if (info.sample_id == "background") != is_background:
            continue
        with h5py.File(info.path, "r") as raw:
            elig = hasadd_object_mask(np.asarray(raw[ha_key][:]), hasadd_mode)
            if is_background:
                if pt_min and pt_min > 0.0:
                    dei = raw[ev_key][:].astype(np.int64)
                    ept = derive_event_pt(info.n_events, raw["A_pT"][:].astype(np.float64),
                                          raw["A_event_idx"][:].astype(np.int64), event_pt_mode)
                    elig = elig & (ept[dei] >= pt_min)
            else:
                if is_excluded_sample(info.sample_id, hasadd_mode):
                    elig = np.zeros(elig.shape, dtype=bool)
        per_file.append((info.stem, elig))
    return per_file


def build_eval_sample(input_dir, output_path, hasadd_mode, signal_frac=0.20,
                      bg_frac=0.20, bg_max=1_000_000, bg_pt_min=20.0, seed=1234,
                      event_pt_mode="max"):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_infos, _ = scan_file_metadata(input_dir)

    # ----- per-detector eval selection masks -----
    sig_masks, bg_masks, n_sig_sel, n_bg_sel = {}, {}, {}, {}
    for det in DETECTORS:
        sig_elig = _eligible_masks(file_infos, det, hasadd_mode, False, 0.0, event_pt_mode)
        n_sig = int(sum(int(m.sum()) for _, m in sig_elig))
        sig_target = int(round(signal_frac * n_sig))
        sig_masks[det], n_sig_sel[det] = _global_subsample(sig_elig, sig_target, seed + 1)

        bg_elig = _eligible_masks(file_infos, det, hasadd_mode, True, bg_pt_min, event_pt_mode)
        n_bg = int(sum(int(m.sum()) for _, m in bg_elig))
        bg_target = int(round(bg_frac * min(bg_max, n_bg)))
        bg_masks[det], n_bg_sel[det] = _global_subsample(bg_elig, bg_target, seed + 2)
        print(f"[{det}] signal eligible={n_sig} -> selected={n_sig_sel[det]} "
              f"({signal_frac:.0%}); bkg eligible(pT>={bg_pt_min:g})={n_bg} "
              f"-> selected={n_bg_sel[det]} ({bg_frac:.0%} of min({bg_max},N))")

    sig_by_stem = {det: sig_masks[det] for det in DETECTORS}
    bg_by_stem = {det: bg_masks[det] for det in DETECTORS}

    with h5py.File(output_path, "w") as out:
        out.attrs["input_dir"] = str(input_dir)
        out.attrs["hasadd_mode"] = hasadd_mode
        out.attrs["sample_kind"] = "eval_low_cut"
        out.attrs["signal_frac"] = float(signal_frac)
        out.attrs["background_frac"] = float(bg_frac)
        out.attrs["background_max_objects"] = int(bg_max)
        out.attrs["background_pt_min"] = float(bg_pt_min)
        out.attrs["signal_pt_min"] = 0.0
        out.attrs["signal_selected_objects_eb"] = int(n_sig_sel["eb"])
        out.attrs["signal_selected_objects_ee"] = int(n_sig_sel["ee"])
        out.attrs["background_selected_objects_eb"] = int(n_bg_sel["eb"])
        out.attrs["background_selected_objects_ee"] = int(n_bg_sel["ee"])
        out.attrs["seed"] = int(seed)
        out.attrs["weight_basis"] = "object_unweighted_eval"
        out.attrs["pt_min"] = 0.0
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        files_group = out.create_group("files")
        for info in file_infos:
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
                    det_event_idx = raw[DET_EVENT_IDX_KEYS[det]][:].astype(np.int64)
                    if info.sample_id == "background":
                        sel = bg_by_stem[det].get(info.stem)
                    else:
                        sel = sig_by_stem[det].get(info.stem)
                    w = np.zeros(det_event_idx.shape, dtype=np.float64)
                    if sel is not None:
                        w[sel] = 1.0
                    grp.create_dataset(WEIGHT_KEYS[det], data=w, compression="gzip")
                    grp.create_dataset(DET_EVENT_IDX_KEYS[det],
                                       data=det_event_idx.astype(np.int64),
                                       compression="gzip")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Build a low-cut (pT>=20) EVAL sample selection.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--hasadd-mode", choices=("all", "eq0", "eq1"), required=True)
    p.add_argument("--signal-frac", type=float, default=0.20)
    p.add_argument("--background-frac", type=float, default=0.20)
    p.add_argument("--background-max-objects", type=int, default=1_000_000)
    p.add_argument("--background-pt-min", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--event-pt-mode", choices=("max", "leading", "mean"), default="max")
    return p.parse_args()


def main():
    a = parse_args()
    out = build_eval_sample(
        input_dir=a.input_dir, output_path=a.output_path, hasadd_mode=a.hasadd_mode,
        signal_frac=a.signal_frac, bg_frac=a.background_frac,
        bg_max=a.background_max_objects, bg_pt_min=a.background_pt_min,
        seed=a.seed, event_pt_mode=a.event_pt_mode)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
