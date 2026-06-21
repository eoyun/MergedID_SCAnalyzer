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


@dataclass(frozen=True)
class FileInfo:
    path: Path
    split: str
    stem: str
    process_name: str
    sample_id: str
    n_entries: int
    background_offset_start: int = -1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build per-A_pT-entry weights so that each sample has total weight 1 "
            "and each non-empty pT bin inside that sample has equal weight sum."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/eoyun/data/260616_v2_Mini"),
        help="Input dataset directory containing W/, Z/, and signal/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/260616_v2_Mini_ApT_weights"),
        help="Directory for sidecar weight files and summaries.",
    )
    parser.add_argument(
        "--background-max-entries",
        type=int,
        default=None,
        help=(
            "Maximum number of background A_pT entries to use globally after "
            "combining W and Z. If omitted, use all background entries."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for background subsampling.",
    )
    return parser.parse_args()


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
        raise ValueError(f"Could not parse process from source path: {source_path}")
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


def find_bin_indices(pt_values: np.ndarray):
    bin_idx = np.searchsorted(PT_BIN_EDGES_WITH_OVERFLOW, pt_values, side="right") - 1
    return np.clip(bin_idx, 0, len(PT_BIN_EDGES_WITH_OVERFLOW) - 2).astype(np.int16)


def summarize_file(path: Path, split: str):
    with h5py.File(path, "r") as handle:
        pt = handle["A_pT"][:].astype(np.float64)
        source_path = extract_source_path(handle.attrs["source_root_files"])
        process_name = extract_process_name(source_path)
        sample_id = classify_sample(split, process_name)

    return FileInfo(
        path=path,
        split=split,
        stem=path.stem,
        process_name=process_name,
        sample_id=sample_id,
        n_entries=len(pt),
    ), pt


def scan_file_metadata(base_dir: Path):
    file_infos = []
    n_background_entries = 0

    for file_index, (split, path) in enumerate(iter_h5_files(base_dir), start=1):
        info, _ = summarize_file(path, split)
        background_offset_start = n_background_entries if info.sample_id == "background" else -1
        info = FileInfo(
            path=info.path,
            split=info.split,
            stem=info.stem,
            process_name=info.process_name,
            sample_id=info.sample_id,
            n_entries=info.n_entries,
            background_offset_start=background_offset_start,
        )
        file_infos.append(info)
        if info.sample_id == "background":
            n_background_entries += info.n_entries

        if file_index % 250 == 0:
            print(f"[scan] files={file_index}")

    return file_infos, n_background_entries


def select_background_global_indices(n_background_entries: int, max_entries: int | None, seed: int):
    if max_entries is None or max_entries >= n_background_entries:
        return None, n_background_entries

    if max_entries <= 0:
        raise ValueError("--background-max-entries must be positive when provided.")

    rng = np.random.default_rng(seed)
    selected = rng.choice(n_background_entries, size=max_entries, replace=False)
    selected.sort()
    return selected.astype(np.int64), max_entries


def get_background_local_indices(selected_background_global_idx, info: FileInfo):
    if selected_background_global_idx is None:
        return None

    start = info.background_offset_start
    stop = start + info.n_entries
    lo = np.searchsorted(selected_background_global_idx, start, side="left")
    hi = np.searchsorted(selected_background_global_idx, stop, side="left")
    return selected_background_global_idx[lo:hi] - start


def build_sample_bin_statistics(file_infos, selected_background_global_idx):
    sample_bin_counts = defaultdict(
        lambda: np.zeros(len(PT_BIN_EDGES_WITH_OVERFLOW) - 1, dtype=np.int64)
    )
    sample_total_counts = defaultdict(int)
    selected_entries_by_file = {}

    for file_index, info in enumerate(file_infos, start=1):
        with h5py.File(info.path, "r") as handle:
            pt = handle["A_pT"][:].astype(np.float64)

        if info.sample_id == "background":
            local_idx = get_background_local_indices(selected_background_global_idx, info)
            if local_idx is None:
                pt_for_weight = pt
                selected_entries_by_file[str(info.path)] = info.n_entries
            else:
                pt_for_weight = pt[local_idx]
                selected_entries_by_file[str(info.path)] = int(len(local_idx))
        else:
            pt_for_weight = pt
            selected_entries_by_file[str(info.path)] = info.n_entries

        bin_idx = find_bin_indices(pt_for_weight)
        counts = np.bincount(bin_idx, minlength=len(PT_BIN_EDGES_WITH_OVERFLOW) - 1)
        sample_bin_counts[info.sample_id] += counts
        sample_total_counts[info.sample_id] += len(pt_for_weight)

        if file_index % 250 == 0:
            print(f"[count] files={file_index}")

    return sample_bin_counts, sample_total_counts, selected_entries_by_file


def make_sample_weight_maps(sample_bin_counts):
    sample_bin_weight_sums = {}
    sample_per_entry_weights = {}
    sample_nonempty_bins = {}

    for sample_id, counts in sorted(sample_bin_counts.items()):
        nonempty = counts > 0
        n_nonempty = int(nonempty.sum())
        if n_nonempty == 0:
            raise ValueError(f"Sample {sample_id} has no entries.")

        target_bin_sum = 1.0 / n_nonempty

        bin_weight_sums = np.zeros_like(counts, dtype=np.float64)
        bin_weight_sums[nonempty] = target_bin_sum

        per_entry_weights = np.zeros_like(counts, dtype=np.float64)
        per_entry_weights[nonempty] = target_bin_sum / counts[nonempty]

        sample_bin_weight_sums[sample_id] = bin_weight_sums
        sample_per_entry_weights[sample_id] = per_entry_weights
        sample_nonempty_bins[sample_id] = nonempty

    return sample_bin_weight_sums, sample_per_entry_weights, sample_nonempty_bins


def ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def build_sidecar_outputs(
    input_dir: Path,
    output_dir: Path,
    file_infos,
    sample_bin_counts,
    sample_bin_weight_sums,
    sample_per_entry_weights,
    selected_background_global_idx,
    selected_entries_by_file,
    background_max_entries,
    background_selected_entries,
    seed,
):
    out_h5_path = output_dir / "A_pT_weights.h5"
    sample_csv_path = output_dir / "sample_bin_summary.csv"
    file_csv_path = output_dir / "file_summary.csv"
    config_json_path = output_dir / "config.json"

    verification_totals = defaultdict(float)
    verification_bin_sums = defaultdict(
        lambda: np.zeros(len(PT_BIN_EDGES_WITH_OVERFLOW) - 1, dtype=np.float64)
    )

    with h5py.File(out_h5_path, "w") as out_h5:
        out_h5.attrs["input_dir"] = str(input_dir)
        out_h5.attrs["weight_basis"] = "A_pT entries"
        out_h5.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        out_h5.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        out_h5.attrs["overflow_bin"] = "[2000, inf)"
        out_h5.attrs["n_files"] = len(file_infos)
        out_h5.attrs["n_samples"] = len(sample_bin_counts)
        out_h5.attrs["background_max_entries"] = -1 if background_max_entries is None else background_max_entries
        out_h5.attrs["background_selected_entries"] = background_selected_entries
        out_h5.attrs["seed"] = seed

        summary_group = out_h5.create_group("sample_summary")
        for sample_id in sorted(sample_bin_counts):
            group = summary_group.create_group(sample_id)
            group.create_dataset("bin_counts", data=sample_bin_counts[sample_id], compression="gzip")
            group.create_dataset(
                "bin_weight_sums",
                data=sample_bin_weight_sums[sample_id],
                compression="gzip",
            )
            group.create_dataset(
                "per_entry_weights",
                data=sample_per_entry_weights[sample_id],
                compression="gzip",
            )
            group.attrs["total_weight_sum"] = float(sample_bin_weight_sums[sample_id].sum())
            group.attrs["n_nonempty_bins"] = int(np.sum(sample_bin_counts[sample_id] > 0))

        files_group = out_h5.create_group("files")
        for file_index, info in enumerate(file_infos, start=1):
            with h5py.File(info.path, "r") as in_h5:
                pt = in_h5["A_pT"][:].astype(np.float64)

            bin_idx = find_bin_indices(pt)
            selected_mask = np.ones(len(pt), dtype=bool)
            weights = np.zeros(len(pt), dtype=np.float64)

            if info.sample_id == "background":
                local_idx = get_background_local_indices(selected_background_global_idx, info)
                if local_idx is None:
                    selected_pt_bins = bin_idx
                    weights = sample_per_entry_weights[info.sample_id][bin_idx]
                else:
                    selected_mask[:] = False
                    selected_mask[local_idx] = True
                    selected_pt_bins = bin_idx[local_idx]
                    weights[local_idx] = sample_per_entry_weights[info.sample_id][selected_pt_bins]
            else:
                selected_pt_bins = bin_idx
                weights = sample_per_entry_weights[info.sample_id][bin_idx]

            verification_totals[info.sample_id] += float(weights.sum())
            verification_bin_sums[info.sample_id] += np.bincount(
                selected_pt_bins,
                weights=weights[selected_mask],
                minlength=len(PT_BIN_EDGES_WITH_OVERFLOW) - 1,
            ).astype(np.float64)

            group = files_group.require_group(info.split).create_group(info.stem)
            group.create_dataset("weight", data=weights, compression="gzip")
            group.create_dataset("pt_bin", data=bin_idx, compression="gzip")
            group.create_dataset("selected_mask", data=selected_mask, compression="gzip")
            group.attrs["input_file"] = str(info.path)
            group.attrs["split"] = info.split
            group.attrs["process_name"] = info.process_name
            group.attrs["sample_id"] = info.sample_id
            group.attrs["n_entries"] = info.n_entries
            group.attrs["selected_entries"] = selected_entries_by_file[str(info.path)]

            if file_index % 250 == 0:
                print(f"[write] files={file_index}")

    sample_rows = []
    for sample_id in sorted(sample_bin_counts):
        counts = sample_bin_counts[sample_id]
        bin_weight_sums = sample_bin_weight_sums[sample_id]
        per_entry_weights = sample_per_entry_weights[sample_id]

        for ibin in range(len(PT_BIN_EDGES_WITH_OVERFLOW) - 1):
            high = PT_BIN_EDGES_WITH_OVERFLOW[ibin + 1]
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "bin_index": ibin,
                    "pt_low": PT_BIN_EDGES_WITH_OVERFLOW[ibin],
                    "pt_high": "inf" if np.isinf(high) else float(high),
                    "count": int(counts[ibin]),
                    "bin_weight_sum": float(bin_weight_sums[ibin]),
                    "per_entry_weight": float(per_entry_weights[ibin]),
                }
            )

    with sample_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "bin_index",
                "pt_low",
                "pt_high",
                "count",
                "bin_weight_sum",
                "per_entry_weight",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with file_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_file",
                "split",
                "process_name",
                "sample_id",
                "n_entries",
                "selected_entries",
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
                    "n_entries": info.n_entries,
                    "selected_entries": selected_entries_by_file[str(info.path)],
                }
            )

    config = {
        "input_dir": str(input_dir),
        "weight_basis": "A_pT entries",
        "requested_pt_bin_edges": PT_BIN_EDGES.tolist(),
        "effective_pt_bin_edges": [
            "inf" if np.isinf(v) else float(v) for v in PT_BIN_EDGES_WITH_OVERFLOW.tolist()
        ],
        "sample_ids": sorted(sample_bin_counts.keys()),
        "background_max_entries": background_max_entries,
        "background_selected_entries": background_selected_entries,
        "seed": seed,
    }
    with config_json_path.open("w") as handle:
        json.dump(config, handle, indent=2)

    return out_h5_path, sample_csv_path, file_csv_path, config_json_path, verification_totals, verification_bin_sums


def print_verification(sample_bin_counts, verification_totals, verification_bin_sums):
    print("\n[verification]")
    for sample_id in sorted(sample_bin_counts):
        counts = sample_bin_counts[sample_id]
        nonempty = counts > 0
        bin_sums = verification_bin_sums[sample_id]

        print(
            f"{sample_id:20s} total={verification_totals[sample_id]:.12f} "
            f"nonempty_bins={int(nonempty.sum())}"
        )

        nonempty_values = bin_sums[nonempty]
        if len(nonempty_values) > 0:
            print(
                f"  bin_sum[min,max]=({nonempty_values.min():.12f}, "
                f"{nonempty_values.max():.12f})"
            )


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    ensure_output_dir(output_dir)

    print(f"[input]  {input_dir}")
    print(f"[output] {output_dir}")

    file_infos, n_background_entries = scan_file_metadata(input_dir)
    selected_background_global_idx, background_selected_entries = select_background_global_indices(
        n_background_entries=n_background_entries,
        max_entries=args.background_max_entries,
        seed=args.seed,
    )
    print(
        f"[background] total={n_background_entries} "
        f"selected={background_selected_entries}"
    )
    sample_bin_counts, sample_total_counts, selected_entries_by_file = build_sample_bin_statistics(
        file_infos=file_infos,
        selected_background_global_idx=selected_background_global_idx,
    )
    sample_bin_weight_sums, sample_per_entry_weights, _ = make_sample_weight_maps(
        sample_bin_counts
    )

    print("\n[sample counts]")
    for sample_id in sorted(sample_total_counts):
        counts = sample_bin_counts[sample_id]
        print(
            f"{sample_id:20s} entries={sample_total_counts[sample_id]:9d} "
            f"nonempty_bins={int(np.sum(counts > 0)):2d}"
        )

    (
        out_h5_path,
        sample_csv_path,
        file_csv_path,
        config_json_path,
        verification_totals,
        verification_bin_sums,
    ) = build_sidecar_outputs(
        input_dir=input_dir,
        output_dir=output_dir,
        file_infos=file_infos,
        sample_bin_counts=sample_bin_counts,
        sample_bin_weight_sums=sample_bin_weight_sums,
        sample_per_entry_weights=sample_per_entry_weights,
        selected_background_global_idx=selected_background_global_idx,
        selected_entries_by_file=selected_entries_by_file,
        background_max_entries=args.background_max_entries,
        background_selected_entries=background_selected_entries,
        seed=args.seed,
    )

    print_verification(sample_bin_counts, verification_totals, verification_bin_sums)

    print("\n[saved]")
    print(out_h5_path)
    print(sample_csv_path)
    print(file_csv_path)
    print(config_json_path)


if __name__ == "__main__":
    main()
