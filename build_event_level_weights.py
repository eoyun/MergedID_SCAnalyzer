#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


PT_BIN_EDGES = np.array(
    [
        0,
        20,
        30,
        50,
        60,
        70,
        80,
        100,
        125,
        150,
        200,
        250,
        300,
        350,
        400,
        500,
        600,
        800,
        1000,
        1200,
        1500,
        2000,
    ],
    dtype=np.float64,
)
PT_BIN_EDGES_WITH_OVERFLOW = np.concatenate([PT_BIN_EDGES, [np.inf]])

SOURCE_PROCESS_RE = re.compile(r"/store/user/yeo/([^/]+)/")
SIGNAL_PROCESS_RE = re.compile(
    r"HToAATo4L_H(?P<hmass>\d+)A(?P<amass>[^_]+)_TuneCP5_13p6TeV-pythia8"
)

CASE_ORDER = ("event", "eb", "ee")
CASE_LABELS = {
    "event": "event",
    "eb": "EB",
    "ee": "EE",
}
DETECTOR_CASES = ("eb", "ee")
DETECTOR_EVENT_IDX_KEYS = {
    "eb": "EB_ele_event_idx",
    "ee": "EE_ele_event_idx",
}
DETECTOR_PREFIX = {
    "eb": "EB",
    "ee": "EE",
}


@dataclass(frozen=True)
class FileInfo:
    path: Path
    split: str
    stem: str
    process_name: str
    sample_id: str
    n_events: int
    n_a: int
    n_eb: int
    n_ee: int
    background_event_offset_start: int = -1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build sample-balanced event weights with detector-separated EB/EE cases. "
            "Within each sample, each non-empty pT bin gets equal total weight sum and "
            "the total weight sum is 1 for the overall event case, the EB case, and "
            "the EE case."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/eoyun/data/260616_v2_Mini"),
        help="Dataset directory containing W/, Z/, and signal/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/260616_v2_Mini_event_weights"),
        help="Directory for the sidecar HDF5 and summaries.",
    )
    parser.add_argument(
        "--background-max-events",
        type=int,
        default=None,
        help=(
            "Maximum number of background events to use after combining W and Z. "
            "If omitted, all background events are used."
        ),
    )
    parser.add_argument(
        "--event-pt-mode",
        choices=("max", "leading", "mean"),
        default="max",
        help="How to derive one representative pT per event from A_pT entries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for background event subsampling.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_h5_files(base_dir: Path):
    for split in ("W", "Z", "signal"):
        split_dir = base_dir / split
        for path in sorted(split_dir.glob("*.h5")):
            yield split, path


def extract_source_path(raw_attr):
    if isinstance(raw_attr, bytes):
        raw_attr = raw_attr.decode("utf-8")
    if isinstance(raw_attr, str):
        value = ast.literal_eval(raw_attr)
    else:
        value = raw_attr
    if not value:
        raise ValueError("source_root_files is empty.")
    return str(value[0])


def extract_process_name(source_path: str):
    match = SOURCE_PROCESS_RE.search(source_path)
    if not match:
        raise ValueError(f"Could not parse process name from source path: {source_path}")
    return match.group(1)


def classify_sample(split: str, process_name: str):
    if split in {"W", "Z"}:
        return "background"

    match = SIGNAL_PROCESS_RE.search(process_name)
    if not match:
        raise ValueError(f"Could not parse signal mass point from process: {process_name}")

    hmass = match.group("hmass")
    amass = match.group("amass")
    return f"signal_H{hmass}_A{amass}"


def scan_file_metadata(base_dir: Path):
    file_infos = []
    n_background_events = 0

    for file_index, (split, path) in enumerate(iter_h5_files(base_dir), start=1):
        with h5py.File(path, "r") as handle:
            source_path = extract_source_path(handle.attrs["source_root_files"])
            process_name = extract_process_name(source_path)
            sample_id = classify_sample(split, process_name)
            n_events = int(handle["eventId"].shape[0])
            n_a = int(handle["A_pT"].shape[0])
            n_eb = int(handle["SC_energy"].shape[0])
            n_ee = int(handle["EE_seed_energy"].shape[0])

        background_event_offset_start = n_background_events if sample_id == "background" else -1
        file_infos.append(
            FileInfo(
                path=path,
                split=split,
                stem=path.stem,
                process_name=process_name,
                sample_id=sample_id,
                n_events=n_events,
                n_a=n_a,
                n_eb=n_eb,
                n_ee=n_ee,
                background_event_offset_start=background_event_offset_start,
            )
        )
        if sample_id == "background":
            n_background_events += n_events

        if file_index % 250 == 0:
            print(f"[scan] files={file_index}")

    return file_infos, n_background_events


def select_background_global_event_indices(n_background_events: int, max_events: int | None, seed: int):
    if max_events is None or max_events >= n_background_events:
        return None, n_background_events

    if max_events <= 0:
        raise ValueError("--background-max-events must be positive when provided.")

    rng = np.random.default_rng(seed)
    selected = rng.choice(n_background_events, size=max_events, replace=False)
    selected.sort()
    return selected.astype(np.int64), max_events


def get_background_local_event_indices(selected_background_global_idx, info: FileInfo):
    if selected_background_global_idx is None:
        return None

    start = info.background_event_offset_start
    stop = start + info.n_events
    lo = np.searchsorted(selected_background_global_idx, start, side="left")
    hi = np.searchsorted(selected_background_global_idx, stop, side="left")
    return selected_background_global_idx[lo:hi] - start


def derive_event_pt(n_events: int, a_pt: np.ndarray, a_event_idx: np.ndarray, mode: str):
    if mode == "max":
        event_pt = np.full(n_events, -np.inf, dtype=np.float64)
        np.maximum.at(event_pt, a_event_idx, a_pt)
    elif mode == "mean":
        sums = np.bincount(a_event_idx, weights=a_pt, minlength=n_events).astype(np.float64)
        counts = np.bincount(a_event_idx, minlength=n_events).astype(np.int64)
        event_pt = np.full(n_events, np.nan, dtype=np.float64)
        nonzero = counts > 0
        event_pt[nonzero] = sums[nonzero] / counts[nonzero]
    elif mode == "leading":
        event_pt = np.full(n_events, np.nan, dtype=np.float64)
        unique_events, first_positions = np.unique(a_event_idx, return_index=True)
        event_pt[unique_events] = a_pt[first_positions]
    else:
        raise ValueError(f"Unsupported event pt mode: {mode}")

    missing = ~np.isfinite(event_pt)
    if np.any(missing):
        raise ValueError(f"Found {int(missing.sum())} events without a valid representative pT.")

    return event_pt


def find_bin_indices(pt_values: np.ndarray):
    bin_idx = np.searchsorted(PT_BIN_EDGES_WITH_OVERFLOW, pt_values, side="right") - 1
    return np.clip(bin_idx, 0, len(PT_BIN_EDGES_WITH_OVERFLOW) - 2).astype(np.int16)


def build_sample_case_statistics(file_infos, selected_background_global_idx, event_pt_mode):
    case_bin_counts = {
        case: defaultdict(lambda: np.zeros(len(PT_BIN_EDGES_WITH_OVERFLOW) - 1, dtype=np.int64))
        for case in CASE_ORDER
    }
    case_total_events = {
        case: defaultdict(int)
        for case in CASE_ORDER
    }
    selected_events_by_file = {
        case: {}
        for case in CASE_ORDER
    }

    for file_index, info in enumerate(file_infos, start=1):
        with h5py.File(info.path, "r") as handle:
            a_pt = handle["A_pT"][:].astype(np.float64)
            a_event_idx = handle["A_event_idx"][:].astype(np.int64)
            detector_event_idx = {
                case: handle[key][:].astype(np.int64)
                for case, key in DETECTOR_EVENT_IDX_KEYS.items()
            }
        event_pt = derive_event_pt(info.n_events, a_pt, a_event_idx, mode=event_pt_mode)

        event_selected_mask = np.ones(info.n_events, dtype=bool)
        if info.sample_id == "background":
            local_selected_idx = get_background_local_event_indices(selected_background_global_idx, info)
            if local_selected_idx is not None:
                event_selected_mask[:] = False
                event_selected_mask[local_selected_idx] = True

        selected_events_by_file["event"][str(info.path)] = int(event_selected_mask.sum())
        selected_event_pt = event_pt[event_selected_mask]
        selected_event_bins = find_bin_indices(selected_event_pt)
        case_bin_counts["event"][info.sample_id] += np.bincount(
            selected_event_bins,
            minlength=len(PT_BIN_EDGES_WITH_OVERFLOW) - 1,
        )
        case_total_events["event"][info.sample_id] += int(len(selected_event_pt))

        for case in DETECTOR_CASES:
            event_presence = np.bincount(
                detector_event_idx[case],
                minlength=info.n_events,
            ).astype(np.int64) > 0
            detector_selected_mask = event_selected_mask & event_presence
            selected_events_by_file[case][str(info.path)] = int(detector_selected_mask.sum())

            if np.any(detector_selected_mask):
                detector_selected_pt = event_pt[detector_selected_mask]
                detector_bins = find_bin_indices(detector_selected_pt)
                case_bin_counts[case][info.sample_id] += np.bincount(
                    detector_bins,
                    minlength=len(PT_BIN_EDGES_WITH_OVERFLOW) - 1,
                )
                case_total_events[case][info.sample_id] += int(len(detector_selected_pt))

        if file_index % 250 == 0:
            print(f"[count] files={file_index}")

    return case_bin_counts, case_total_events, selected_events_by_file


def build_sample_total_target_weights(sample_ids):
    background_ids = [sample_id for sample_id in sample_ids if sample_id == "background"]
    signal_ids = [sample_id for sample_id in sample_ids if sample_id != "background"]

    if not background_ids:
        raise ValueError("No background sample found.")
    if not signal_ids:
        raise ValueError("No signal samples found.")

    class_total_target_weights = {
        "background": 1.0,
        "signal": 1.0,
    }
    sample_total_target_weights = {}
    background_per_sample = class_total_target_weights["background"] / len(background_ids)
    signal_per_sample = class_total_target_weights["signal"] / len(signal_ids)

    for sample_id in background_ids:
        sample_total_target_weights[sample_id] = background_per_sample
    for sample_id in signal_ids:
        sample_total_target_weights[sample_id] = signal_per_sample

    return sample_total_target_weights, class_total_target_weights


def make_sample_case_weight_maps(sample_ids, sample_bin_counts, case_label, sample_total_target_weights):
    sample_bin_weight_sums = {}
    sample_event_weight_lookup = {}

    for sample_id in sample_ids:
        counts = np.asarray(sample_bin_counts[sample_id], dtype=np.int64)
        nonempty = counts > 0
        n_nonempty = int(nonempty.sum())
        if n_nonempty == 0:
            raise ValueError(f"Sample {sample_id} has no selected {case_label} entries.")

        target_sample_sum = float(sample_total_target_weights[sample_id])
        target_bin_sum = target_sample_sum / n_nonempty
        bin_weight_sums = np.zeros_like(counts, dtype=np.float64)
        event_weight_lookup = np.zeros_like(counts, dtype=np.float64)

        bin_weight_sums[nonempty] = target_bin_sum
        event_weight_lookup[nonempty] = target_bin_sum / counts[nonempty]

        sample_bin_weight_sums[sample_id] = bin_weight_sums
        sample_event_weight_lookup[sample_id] = event_weight_lookup

    return sample_bin_weight_sums, sample_event_weight_lookup


def map_event_weights_to_objects(event_weight, event_idx, split: bool):
    mapped = event_weight[event_idx].astype(np.float64, copy=True)
    if not split:
        return mapped

    counts = np.bincount(event_idx, minlength=len(event_weight)).astype(np.float64)
    valid = counts[event_idx] > 0
    mapped[valid] /= counts[event_idx][valid]
    return mapped


def build_outputs(
    input_dir: Path,
    output_dir: Path,
    file_infos,
    sample_ids,
    case_bin_counts,
    case_bin_weight_sums,
    case_event_weight_lookup,
    selected_background_global_idx,
    selected_events_by_file,
    background_max_events,
    background_selected_events,
    sample_total_target_weights,
    class_total_target_weights,
    event_pt_mode,
    seed,
):
    out_h5_path = output_dir / "event_level_weights.h5"
    sample_csv_path = output_dir / "sample_event_bin_summary.csv"
    file_csv_path = output_dir / "file_event_summary.csv"
    config_json_path = output_dir / "config.json"

    verification_event_totals = {
        case: defaultdict(float)
        for case in CASE_ORDER
    }
    verification_event_bin_sums = {
        case: defaultdict(lambda: np.zeros(len(PT_BIN_EDGES_WITH_OVERFLOW) - 1, dtype=np.float64))
        for case in CASE_ORDER
    }

    with h5py.File(out_h5_path, "w") as out_h5:
        out_h5.attrs["input_dir"] = str(input_dir)
        out_h5.attrs["weight_basis"] = "event"
        out_h5.attrs["event_pt_mode"] = event_pt_mode
        out_h5.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        out_h5.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        out_h5.attrs["overflow_bin"] = "[2000, inf)"
        out_h5.attrs["background_max_events"] = -1 if background_max_events is None else background_max_events
        out_h5.attrs["background_selected_events"] = background_selected_events
        out_h5.attrs["seed"] = seed
        out_h5.attrs["n_files"] = len(file_infos)
        out_h5.attrs["n_samples"] = len(sample_ids)
        out_h5.attrs["background_class_total_target_weight"] = class_total_target_weights["background"]
        out_h5.attrs["signal_class_total_target_weight"] = class_total_target_weights["signal"]
        out_h5.attrs["detector_cases"] = np.array(["event", "EB", "EE"], dtype="S8")

        summary_group = out_h5.create_group("sample_summary")
        for case in CASE_ORDER:
            case_group = summary_group.create_group(case)
            for sample_id in sample_ids:
                group = case_group.create_group(sample_id)
                group.create_dataset(
                    "event_bin_counts",
                    data=case_bin_counts[case][sample_id],
                    compression="gzip",
                )
                group.create_dataset(
                    "event_bin_weight_sums",
                    data=case_bin_weight_sums[case][sample_id],
                    compression="gzip",
                )
                group.create_dataset(
                    "event_weight_lookup",
                    data=case_event_weight_lookup[case][sample_id],
                    compression="gzip",
                )
                group.attrs["total_event_weight_sum"] = float(case_bin_weight_sums[case][sample_id].sum())
                group.attrs["n_nonempty_bins"] = int(np.sum(case_bin_counts[case][sample_id] > 0))

        files_group = out_h5.create_group("files")
        for file_index, info in enumerate(file_infos, start=1):
            with h5py.File(info.path, "r") as handle:
                a_pt = handle["A_pT"][:].astype(np.float64)
                a_event_idx = handle["A_event_idx"][:].astype(np.int64)
                detector_event_idx = {
                    case: handle[key][:].astype(np.int64)
                    for case, key in DETECTOR_EVENT_IDX_KEYS.items()
                }

            event_pt = derive_event_pt(info.n_events, a_pt, a_event_idx, mode=event_pt_mode)
            event_bin = find_bin_indices(event_pt)

            event_selected_mask = np.ones(info.n_events, dtype=bool)
            event_weight = np.zeros(info.n_events, dtype=np.float64)

            if info.sample_id == "background":
                local_selected_idx = get_background_local_event_indices(selected_background_global_idx, info)
                if local_selected_idx is None:
                    event_weight = case_event_weight_lookup["event"][info.sample_id][event_bin]
                else:
                    event_selected_mask[:] = False
                    event_selected_mask[local_selected_idx] = True
                    event_weight[local_selected_idx] = case_event_weight_lookup["event"][info.sample_id][
                        event_bin[local_selected_idx]
                    ]
            else:
                event_weight = case_event_weight_lookup["event"][info.sample_id][event_bin]

            selected_event_bins = event_bin[event_selected_mask]
            verification_event_totals["event"][info.sample_id] += float(event_weight.sum())
            verification_event_bin_sums["event"][info.sample_id] += np.bincount(
                selected_event_bins,
                weights=event_weight[event_selected_mask],
                minlength=len(PT_BIN_EDGES_WITH_OVERFLOW) - 1,
            ).astype(np.float64)

            a_weight_duplicate = map_event_weights_to_objects(event_weight, a_event_idx, split=False)
            a_weight_split = map_event_weights_to_objects(event_weight, a_event_idx, split=True)

            detector_case_outputs = {}
            for case in DETECTOR_CASES:
                prefix = DETECTOR_PREFIX[case]
                event_presence = np.bincount(
                    detector_event_idx[case],
                    minlength=info.n_events,
                ).astype(np.int64) > 0
                detector_selected_mask = event_selected_mask & event_presence
                detector_event_weight = np.zeros(info.n_events, dtype=np.float64)
                detector_event_weight[detector_selected_mask] = case_event_weight_lookup[case][info.sample_id][
                    event_bin[detector_selected_mask]
                ]

                verification_event_totals[case][info.sample_id] += float(detector_event_weight.sum())
                verification_event_bin_sums[case][info.sample_id] += np.bincount(
                    event_bin[detector_selected_mask],
                    weights=detector_event_weight[detector_selected_mask],
                    minlength=len(PT_BIN_EDGES_WITH_OVERFLOW) - 1,
                ).astype(np.float64)

                detector_case_outputs[case] = {
                    "event_selected_mask": detector_selected_mask,
                    "event_weight": detector_event_weight,
                    "weight_duplicate": map_event_weights_to_objects(
                        detector_event_weight,
                        detector_event_idx[case],
                        split=False,
                    ),
                    "weight_split": map_event_weights_to_objects(
                        detector_event_weight,
                        detector_event_idx[case],
                        split=True,
                    ),
                    "prefix": prefix,
                }

            group = files_group.require_group(info.split).create_group(info.stem)
            group.create_dataset("event_pt", data=event_pt, compression="gzip")
            group.create_dataset("event_pt_bin", data=event_bin, compression="gzip")
            group.create_dataset("event_selected_mask", data=event_selected_mask, compression="gzip")
            group.create_dataset("event_weight", data=event_weight, compression="gzip")
            group.create_dataset("A_weight_duplicate", data=a_weight_duplicate, compression="gzip")
            group.create_dataset("A_weight_split", data=a_weight_split, compression="gzip")
            for case in DETECTOR_CASES:
                outputs = detector_case_outputs[case]
                prefix = outputs["prefix"]
                group.create_dataset(
                    f"{prefix}_event_selected_mask",
                    data=outputs["event_selected_mask"],
                    compression="gzip",
                )
                group.create_dataset(
                    f"{prefix}_event_weight",
                    data=outputs["event_weight"],
                    compression="gzip",
                )
                group.create_dataset(
                    f"{prefix}_weight_duplicate",
                    data=outputs["weight_duplicate"],
                    compression="gzip",
                )
                group.create_dataset(
                    f"{prefix}_weight_split",
                    data=outputs["weight_split"],
                    compression="gzip",
                )
            group.attrs["input_file"] = str(info.path)
            group.attrs["split"] = info.split
            group.attrs["process_name"] = info.process_name
            group.attrs["sample_id"] = info.sample_id
            group.attrs["n_events"] = info.n_events
            group.attrs["selected_events"] = selected_events_by_file["event"][str(info.path)]
            group.attrs["selected_eb_events"] = selected_events_by_file["eb"][str(info.path)]
            group.attrs["selected_ee_events"] = selected_events_by_file["ee"][str(info.path)]

            if file_index % 250 == 0:
                print(f"[write] files={file_index}")

    with sample_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "sample_id",
                "bin_index",
                "pt_low",
                "pt_high",
                "event_count",
                "event_bin_weight_sum",
                "per_event_weight",
            ],
        )
        writer.writeheader()
        for case in CASE_ORDER:
            for sample_id in sample_ids:
                counts = case_bin_counts[case][sample_id]
                bin_weight_sums = case_bin_weight_sums[case][sample_id]
                event_weight_lookup = case_event_weight_lookup[case][sample_id]
                for ibin in range(len(PT_BIN_EDGES_WITH_OVERFLOW) - 1):
                    high = PT_BIN_EDGES_WITH_OVERFLOW[ibin + 1]
                    writer.writerow(
                        {
                            "case": CASE_LABELS[case],
                            "sample_id": sample_id,
                            "bin_index": ibin,
                            "pt_low": PT_BIN_EDGES_WITH_OVERFLOW[ibin],
                            "pt_high": "inf" if np.isinf(high) else float(high),
                            "event_count": int(counts[ibin]),
                            "event_bin_weight_sum": float(bin_weight_sums[ibin]),
                            "per_event_weight": float(event_weight_lookup[ibin]),
                        }
                    )

    with file_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_file",
                "split",
                "process_name",
                "sample_id",
                "n_events",
                "selected_events",
                "selected_eb_events",
                "selected_ee_events",
                "n_a",
                "n_eb",
                "n_ee",
            ],
        )
        writer.writeheader()
        for info in file_infos:
            writer.writerow(
                {
                    "input_file": str(info.path),
                    "split": info.split,
                    "process_name": info.process_name,
                    "sample_id": info.sample_id,
                    "n_events": info.n_events,
                    "selected_events": selected_events_by_file["event"][str(info.path)],
                    "selected_eb_events": selected_events_by_file["eb"][str(info.path)],
                    "selected_ee_events": selected_events_by_file["ee"][str(info.path)],
                    "n_a": info.n_a,
                    "n_eb": info.n_eb,
                    "n_ee": info.n_ee,
                }
            )

    config = {
        "input_dir": str(input_dir),
        "weight_basis": "event",
        "event_pt_mode": event_pt_mode,
        "requested_pt_bin_edges": PT_BIN_EDGES.tolist(),
        "effective_pt_bin_edges": [
            "inf" if np.isinf(v) else float(v) for v in PT_BIN_EDGES_WITH_OVERFLOW.tolist()
        ],
        "background_max_events": background_max_events,
        "background_selected_events": background_selected_events,
        "seed": seed,
        "sample_ids": sample_ids,
        "sample_total_target_weights": {
            sample_id: float(sample_total_target_weights[sample_id])
            for sample_id in sample_ids
        },
        "class_total_target_weights": {
            key: float(value)
            for key, value in class_total_target_weights.items()
        },
        "balanced_cases": [CASE_LABELS[case] for case in CASE_ORDER],
        "mapped_weight_variants": [
            "event_weight",
            "A_weight_duplicate",
            "A_weight_split",
            "EB_event_weight",
            "EB_weight_duplicate",
            "EB_weight_split",
            "EE_event_weight",
            "EE_weight_duplicate",
            "EE_weight_split",
        ],
        "default_classifier_weight_keys": {
            "eb": "EB_weight_split",
            "ee": "EE_weight_split",
        },
    }
    with config_json_path.open("w") as handle:
        json.dump(config, handle, indent=2)

    return out_h5_path, sample_csv_path, file_csv_path, config_json_path, verification_event_totals, verification_event_bin_sums


def print_verification(sample_ids, case_bin_counts, verification_event_totals, verification_event_bin_sums):
    for case in CASE_ORDER:
        print(f"\n[verification: {CASE_LABELS[case]}]")
        for sample_id in sample_ids:
            counts = case_bin_counts[case][sample_id]
            nonempty = counts > 0
            bin_sums = verification_event_bin_sums[case][sample_id]
            print(
                f"{sample_id:20s} total={verification_event_totals[case][sample_id]:.12f} "
                f"nonempty_bins={int(nonempty.sum())}"
            )
            nonempty_values = bin_sums[nonempty]
            if len(nonempty_values) > 0:
                print(
                    f"  bin_sum[min,max]=({nonempty_values.min():.12f}, "
                    f"{nonempty_values.max():.12f})"
                )
        background_total = sum(
            verification_event_totals[case][sample_id]
            for sample_id in sample_ids
            if sample_id == "background"
        )
        signal_total = sum(
            verification_event_totals[case][sample_id]
            for sample_id in sample_ids
            if sample_id != "background"
        )
        print(
            f"class_totals background={background_total:.12f} "
            f"signal={signal_total:.12f}"
        )


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    print(f"[input]  {input_dir}")
    print(f"[output] {output_dir}")

    file_infos, n_background_events = scan_file_metadata(input_dir)
    sample_ids = sorted({info.sample_id for info in file_infos})
    selected_background_global_idx, background_selected_events = select_background_global_event_indices(
        n_background_events=n_background_events,
        max_events=args.background_max_events,
        seed=args.seed,
    )
    print(
        f"[background] total_events={n_background_events} "
        f"selected_events={background_selected_events}"
    )

    case_bin_counts, case_total_events, selected_events_by_file = build_sample_case_statistics(
        file_infos=file_infos,
        selected_background_global_idx=selected_background_global_idx,
        event_pt_mode=args.event_pt_mode,
    )
    sample_total_target_weights, class_total_target_weights = build_sample_total_target_weights(sample_ids)
    case_bin_weight_sums = {}
    case_event_weight_lookup = {}
    for case in CASE_ORDER:
        bin_weight_sums, event_weight_lookup = make_sample_case_weight_maps(
            sample_ids=sample_ids,
            sample_bin_counts=case_bin_counts[case],
            case_label=CASE_LABELS[case],
            sample_total_target_weights=sample_total_target_weights,
        )
        case_bin_weight_sums[case] = bin_weight_sums
        case_event_weight_lookup[case] = event_weight_lookup

    for case in CASE_ORDER:
        print(f"\n[sample counts: {CASE_LABELS[case]}]")
        for sample_id in sample_ids:
            counts = case_bin_counts[case][sample_id]
            print(
                f"{sample_id:20s} events={case_total_events[case][sample_id]:9d} "
                f"nonempty_bins={int(np.sum(counts > 0)):2d} "
                f"target_sum={sample_total_target_weights[sample_id]:.12f}"
            )

    (
        out_h5_path,
        sample_csv_path,
        file_csv_path,
        config_json_path,
        verification_event_totals,
        verification_event_bin_sums,
    ) = build_outputs(
        input_dir=input_dir,
        output_dir=output_dir,
        file_infos=file_infos,
        sample_ids=sample_ids,
        case_bin_counts=case_bin_counts,
        case_bin_weight_sums=case_bin_weight_sums,
        case_event_weight_lookup=case_event_weight_lookup,
        selected_background_global_idx=selected_background_global_idx,
        selected_events_by_file=selected_events_by_file,
        background_max_events=args.background_max_events,
        background_selected_events=background_selected_events,
        sample_total_target_weights=sample_total_target_weights,
        class_total_target_weights=class_total_target_weights,
        event_pt_mode=args.event_pt_mode,
        seed=args.seed,
    )

    print_verification(sample_ids, case_bin_counts, verification_event_totals, verification_event_bin_sums)

    print("\n[saved]")
    print(out_h5_path)
    print(sample_csv_path)
    print(file_csv_path)
    print(config_json_path)


if __name__ == "__main__":
    main()
