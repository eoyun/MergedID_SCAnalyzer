#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from track_point_transformer import (
    PointCloudTransformerClassifier,
    TrackPointCloudDataset,
    collate_track_point_cloud_batch,
    run_point_transformer_epoch,
)
from train_resnet_image_classifier import (
    DEFAULT_MAX_OPEN_RAW_FILES,
    DEFAULT_PREFETCH_FACTOR,
    DetectorObjectDataset,
    build_event_registry,
    build_object_manifest,
    build_resnet,
    build_split_event_maps,
    choose_threshold,
    compute_efficiency_by_pt,
    configure_torch_multiprocessing,
    parse_file_entries,
    print_split_weight_sums,
    rebalance_manifest_class_weights,
    resolve_num_workers,
    run_one_epoch,
    safe_roc_auc,
    save_efficiency_plot,
    save_manifest_summary,
    save_roc_plot,
    save_score_distribution,
    set_seed,
    summarize_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-score a saved image model and a saved track point-transformer on "
            "the same validation/test splits, then train a stacking ensemble."
        )
    )
    parser.add_argument(
        "--image-run-dir",
        type=Path,
        required=True,
        help="Directory that contains the trained image model run_config.json and best_model.pt.",
    )
    parser.add_argument(
        "--track-run-dir",
        type=Path,
        required=True,
        help="Directory that contains the trained track model run_config.json and best_model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ensemble outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for inference.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=None,
        help="Optional override for image inference batch size.",
    )
    parser.add_argument(
        "--track-batch-size",
        type=int,
        default=None,
        help="Optional override for track inference batch size.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("max-f1", "youden"),
        default="max-f1",
        help="Validation rule used to choose the operating threshold on the fused score.",
    )
    return parser.parse_args()


def load_run_config(run_dir):
    run_dir = run_dir.resolve()
    with (run_dir / "run_config.json").open() as handle:
        return json.load(handle)


def normalize_config_value(key, value):
    if key == "weight_h5":
        return str(Path(value).resolve())
    return value


def validate_compatible_runs(image_cfg, track_cfg):
    required_same = (
        "detector",
        "weight_h5",
        "weight_key",
        "seed",
        "train_frac",
        "val_frac",
        "debug",
        "debug_max_events_per_sample",
    )
    mismatches = []
    for key in required_same:
        image_value = normalize_config_value(key, image_cfg.get(key))
        track_value = normalize_config_value(key, track_cfg.get(key))
        if image_value != track_value:
            mismatches.append((key, image_value, track_value))

    if mismatches:
        details = "; ".join(f"{key}: image={image_value} track={track_value}" for key, image_value, track_value in mismatches)
        raise RuntimeError(f"Image and track runs are not compatible: {details}")


def attach_metrics(pack):
    pack = dict(pack)
    pack["auc"] = safe_roc_auc(pack["label"], pack["score"], pack["weight"])
    pack["f1@0.5"] = f1_score(
        pack["label"],
        (pack["score"] >= 0.5).astype(np.int64),
        average="binary",
        sample_weight=pack["weight"],
        zero_division=0,
    )
    return pack


def reorder_pack(pack, order):
    reordered = {}
    for key, value in pack.items():
        if isinstance(value, np.ndarray) and value.shape[0] == len(order):
            reordered[key] = value[order]
        else:
            reordered[key] = value
    return reordered


def align_packs(image_pack, track_pack):
    dtype = np.dtype([("file_idx", np.int32), ("object_idx", np.int32), ("event_idx", np.int32)])

    def build_keys(pack):
        keys = np.empty(len(pack["label"]), dtype=dtype)
        keys["file_idx"] = pack["file_idx"].astype(np.int32)
        keys["object_idx"] = pack["object_idx"].astype(np.int32)
        keys["event_idx"] = pack["event_idx"].astype(np.int32)
        return keys

    image_keys = build_keys(image_pack)
    track_keys = build_keys(track_pack)
    image_order = np.argsort(image_keys, kind="mergesort")
    track_order = np.argsort(track_keys, kind="mergesort")

    image_pack = reorder_pack(image_pack, image_order)
    track_pack = reorder_pack(track_pack, track_order)
    image_keys = image_keys[image_order]
    track_keys = track_keys[track_order]

    if not np.array_equal(image_keys, track_keys):
        raise RuntimeError("Image and track predictions do not cover the same object set.")

    for key in ("label", "event_idx", "sample_code"):
        if not np.array_equal(image_pack[key], track_pack[key]):
            raise RuntimeError(f"Aligned packs disagree on {key}.")
    for key in ("weight", "pt"):
        if not np.allclose(image_pack[key], track_pack[key], rtol=1e-6, atol=1e-8):
            raise RuntimeError(f"Aligned packs disagree on {key}.")

    return image_pack, track_pack


def score_to_logit(score):
    score = np.clip(score, 1e-6, 1.0 - 1e-6)
    return np.log(score / (1.0 - score))


def build_fusion_features(image_pack, track_pack):
    counts = np.log1p(np.clip(track_pack["track_counts"], 0.0, None))
    sum_pt = np.log1p(np.clip(track_pack["track_sum_pt"], 0.0, None))
    features = np.column_stack(
        [
            score_to_logit(image_pack["score"]),
            score_to_logit(track_pack["score"]),
            counts[:, 0],
            counts[:, 1],
            counts[:, 2],
            sum_pt[:, 0],
            sum_pt[:, 1],
            sum_pt[:, 2],
        ]
    ).astype(np.float64)
    feature_names = [
        "image_logit",
        "track_logit",
        "log1p_n_gsf",
        "log1p_n_pf",
        "log1p_n_lost",
        "log1p_sum_pt_gsf",
        "log1p_sum_pt_pf",
        "log1p_sum_pt_lost",
    ]
    return features, feature_names


def save_fusion_predictions_csv(output_path, fused_score, image_pack, track_pack, file_entries, code_to_sample):
    header = (
        "fused_score,image_score,track_score,label,weight,pt,file_idx,object_idx,event_idx,sample_id,raw_file,"
        "n_gsf,n_pf,n_lost,sum_pt_gsf,sum_pt_pf,sum_pt_lost\n"
    )
    with output_path.open("w") as handle:
        handle.write(header)
        for i in range(len(fused_score)):
            file_idx = int(image_pack["file_idx"][i])
            sample_id = code_to_sample[int(image_pack["sample_code"][i])]
            raw_file = file_entries[file_idx].raw_path
            counts = track_pack["track_counts"][i]
            sum_pt = track_pack["track_sum_pt"][i]
            handle.write(
                f"{float(fused_score[i]):.8f},{float(image_pack['score'][i]):.8f},{float(track_pack['score'][i]):.8f},"
                f"{int(image_pack['label'][i])},{float(image_pack['weight'][i]):.8e},{float(image_pack['pt'][i]):.8f},"
                f"{file_idx},{int(image_pack['object_idx'][i])},{int(image_pack['event_idx'][i])},{sample_id},{raw_file},"
                f"{float(counts[0]):.0f},{float(counts[1]):.0f},{float(counts[2]):.0f},"
                f"{float(sum_pt[0]):.8f},{float(sum_pt[1]):.8f},{float(sum_pt[2]):.8f}\n"
            )


def build_loader_common(batch_size, effective_num_workers, device):
    loader_common = {
        "batch_size": batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if effective_num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    return loader_common


def main():
    args = parse_args()

    image_run_dir = args.image_run_dir.resolve()
    track_run_dir = args.track_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_cfg = load_run_config(image_run_dir)
    track_cfg = load_run_config(track_run_dir)
    validate_compatible_runs(image_cfg, track_cfg)

    set_seed(int(image_cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_num_workers = resolve_num_workers(args.num_workers)
    sharing_strategy = configure_torch_multiprocessing()

    detector = image_cfg["detector"]
    weight_key = image_cfg["weight_key"]
    weight_h5_path = Path(image_cfg["weight_h5"]).resolve()

    print(f"[detector] {detector}")
    print(f"[weight_h5] {weight_h5_path}")
    print(f"[image_run_dir] {image_run_dir}")
    print(f"[track_run_dir] {track_run_dir}")
    print(f"[output] {output_dir}")
    print(f"[device] {device}")
    print(f"[num_workers] requested={args.num_workers} effective={effective_num_workers}")
    if sharing_strategy is not None:
        print(f"[torch_sharing_strategy] {sharing_strategy}")

    file_entries = parse_file_entries(weight_h5_path)
    event_file_idx, event_local_idx, event_sample_code, sample_to_code, code_to_sample = build_event_registry(
        file_entries=file_entries,
        weight_h5_path=weight_h5_path,
        detector=detector,
        weight_key=weight_key,
        debug=bool(image_cfg["debug"]),
        debug_max_events_per_sample=int(image_cfg["debug_max_events_per_sample"]),
        seed=int(image_cfg["seed"]),
    )
    split_event_maps = build_split_event_maps(
        event_file_idx=event_file_idx,
        event_local_idx=event_local_idx,
        event_sample_code=event_sample_code,
        train_frac=float(image_cfg["train_frac"]),
        val_frac=float(image_cfg["val_frac"]),
        seed=int(image_cfg["seed"]),
    )

    split_manifests = {
        split_name: build_object_manifest(
            file_entries=file_entries,
            weight_h5_path=weight_h5_path,
            detector=detector,
            weight_key=weight_key,
            split_event_map=split_map,
            sample_to_code=sample_to_code,
        )
        for split_name, split_map in split_event_maps.items()
    }

    class_balance_stats = {}
    for split_name, manifest in split_manifests.items():
        split_manifests[split_name], class_balance_stats[split_name] = rebalance_manifest_class_weights(manifest)

    for split_name, manifest in split_manifests.items():
        summarize_manifest(split_name, manifest, code_to_sample)
    save_manifest_summary(
        output_dir / "manifest_summary.json",
        split_manifests,
        code_to_sample,
        class_balance_stats,
    )
    print_split_weight_sums(split_manifests, class_balance_stats)

    image_batch_size = args.image_batch_size or int(image_cfg["batch_size"])
    track_batch_size = args.track_batch_size or int(track_cfg["batch_size"])

    image_loader_common = build_loader_common(image_batch_size, effective_num_workers, device)
    track_loader_common = build_loader_common(track_batch_size, effective_num_workers, device)
    track_loader_common["collate_fn"] = collate_track_point_cloud_batch

    image_val_loader = DataLoader(
        DetectorObjectDataset(
            detector=detector,
            file_entries=file_entries,
            manifest=split_manifests["val"],
            log_scale=bool(image_cfg.get("log_scale", True)),
            max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
        ),
        shuffle=False,
        **image_loader_common,
    )
    image_test_loader = DataLoader(
        DetectorObjectDataset(
            detector=detector,
            file_entries=file_entries,
            manifest=split_manifests["test"],
            log_scale=bool(image_cfg.get("log_scale", True)),
            max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
        ),
        shuffle=False,
        **image_loader_common,
    )
    track_val_loader = DataLoader(
        TrackPointCloudDataset(
            detector=detector,
            file_entries=file_entries,
            manifest=split_manifests["val"],
            max_points=int(track_cfg["max_points"]),
            log_track_pt=bool(track_cfg["log_track_pt"]),
            max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
        ),
        shuffle=False,
        **track_loader_common,
    )
    track_test_loader = DataLoader(
        TrackPointCloudDataset(
            detector=detector,
            file_entries=file_entries,
            manifest=split_manifests["test"],
            max_points=int(track_cfg["max_points"]),
            log_track_pt=bool(track_cfg["log_track_pt"]),
            max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
        ),
        shuffle=False,
        **track_loader_common,
    )

    image_checkpoint = torch.load(image_run_dir / "best_model.pt", map_location=device)
    image_model = build_resnet(str(image_cfg["model"]), int(image_checkpoint.get("in_channels", 4 if detector == "eb" else 6))).to(device)
    image_model.load_state_dict(image_checkpoint["model_state_dict"])

    track_checkpoint = torch.load(track_run_dir / "best_model.pt", map_location=device)
    track_model = PointCloudTransformerClassifier(
        input_dim=int(track_checkpoint.get("input_dim", 6)),
        embed_dim=int(track_checkpoint["embed_dim"]),
        depth=int(track_checkpoint["depth"]),
        num_heads=int(track_checkpoint["num_heads"]),
        mlp_ratio=float(track_checkpoint["mlp_ratio"]),
        dropout=float(track_checkpoint["dropout"]),
    ).to(device)
    track_model.load_state_dict(track_checkpoint["model_state_dict"])

    image_val_pack = run_one_epoch(image_model, image_val_loader, device, optimizer=None, scaler=None)
    image_test_pack = run_one_epoch(image_model, image_test_loader, device, optimizer=None, scaler=None)
    track_val_pack = attach_metrics(run_point_transformer_epoch(track_model, track_val_loader, device, optimizer=None, scaler=None))
    track_test_pack = attach_metrics(run_point_transformer_epoch(track_model, track_test_loader, device, optimizer=None, scaler=None))

    image_val_pack, track_val_pack = align_packs(image_val_pack, track_val_pack)
    image_test_pack, track_test_pack = align_packs(image_test_pack, track_test_pack)

    x_val, feature_names = build_fusion_features(image_val_pack, track_val_pack)
    x_test, _ = build_fusion_features(image_test_pack, track_test_pack)
    y_val = image_val_pack["label"]
    y_test = image_test_pack["label"]
    w_val = image_val_pack["weight"]
    w_test = image_test_pack["weight"]

    if len(np.unique(y_val)) < 2:
        raise RuntimeError("Validation split does not contain both classes, cannot train the stacking model.")

    scaler = StandardScaler()
    x_val_scaled = scaler.fit_transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    fusion_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    fusion_model.fit(x_val_scaled, y_val, sample_weight=w_val)

    val_fused_score = fusion_model.predict_proba(x_val_scaled)[:, 1]
    test_fused_score = fusion_model.predict_proba(x_test_scaled)[:, 1]
    threshold = choose_threshold(y_val, val_fused_score, w_val, mode=args.threshold_mode)

    test_pred = (test_fused_score >= threshold).astype(np.int64)
    test_auc = safe_roc_auc(y_test, test_fused_score, w_test)
    test_f1 = f1_score(
        y_test,
        test_pred,
        average="binary",
        sample_weight=w_test,
        zero_division=0,
    )

    component_metrics = {
        "image_val_auc": image_val_pack["auc"],
        "image_test_auc": image_test_pack["auc"],
        "track_val_auc": track_val_pack["auc"],
        "track_test_auc": track_test_pack["auc"],
        "fusion_val_auc": safe_roc_auc(y_val, val_fused_score, w_val),
        "fusion_test_auc": test_auc,
    }

    print("\n[test summary]")
    print(f"  threshold      = {threshold:.6f}")
    print(f"  image_test_auc = {image_test_pack['auc']:.6f}")
    print(f"  track_test_auc = {track_test_pack['auc']:.6f}")
    print(f"  fused_test_auc = {test_auc:.6f}")
    print(f"  fused_test_f1  = {test_f1:.6f}")

    save_roc_plot(y_test, test_fused_score, w_test, output_dir / "roc_test.png")
    save_score_distribution(y_test, test_fused_score, w_test, output_dir / "score_distribution_test.png")

    with h5py.File(weight_h5_path, "r") as weight_h5:
        pt_edges = np.asarray(weight_h5.attrs["effective_pt_bin_edges"], dtype=np.float64)
    eff_rows = compute_efficiency_by_pt(
        y_true=y_test,
        y_score=test_fused_score,
        weights=w_test,
        pt=image_test_pack["pt"],
        threshold=threshold,
        pt_edges=pt_edges,
    )
    save_efficiency_plot(eff_rows, output_dir / "efficiency_vs_pt_test.png", threshold)

    with (output_dir / "efficiency_vs_pt_test.json").open("w") as handle:
        json.dump(eff_rows, handle, indent=2)

    save_fusion_predictions_csv(
        output_dir / "val_predictions.csv",
        val_fused_score,
        image_val_pack,
        track_val_pack,
        file_entries,
        code_to_sample,
    )
    save_fusion_predictions_csv(
        output_dir / "test_predictions.csv",
        test_fused_score,
        image_test_pack,
        track_test_pack,
        file_entries,
        code_to_sample,
    )

    with (output_dir / "stacking_coefficients.json").open("w") as handle:
        json.dump(
            {
                "feature_names": feature_names,
                "coefficients": [float(x) for x in fusion_model.coef_[0]],
                "intercept": float(fusion_model.intercept_[0]),
                "scaler_mean": [float(x) for x in scaler.mean_],
                "scaler_scale": [float(x) for x in scaler.scale_],
            },
            handle,
            indent=2,
        )

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "image_run_dir": str(image_run_dir),
                "track_run_dir": str(track_run_dir),
                "output_dir": str(output_dir),
                "detector": detector,
                "weight_h5": str(weight_h5_path),
                "weight_key": weight_key,
                "num_workers_requested": args.num_workers,
                "num_workers_effective": effective_num_workers,
                "torch_sharing_strategy": sharing_strategy,
                "image_batch_size": image_batch_size,
                "track_batch_size": track_batch_size,
                "threshold_mode": args.threshold_mode,
                "feature_names": feature_names,
            },
            handle,
            indent=2,
        )

    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(
            {
                "detector": detector,
                "threshold": threshold,
                "test_auc": test_auc,
                "test_f1": test_f1,
                "component_metrics": component_metrics,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
