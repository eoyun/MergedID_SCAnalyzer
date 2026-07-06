#!/usr/bin/env python3

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from track_point_transformer import (
    POINT_FEATURE_DIM,
    PointCloudTransformerClassifier,
    TrackPointCloudDataset,
    collate_track_point_cloud_batch,
    parse_track_types,
    point_feature_dim,
    run_point_transformer_epoch,
    save_track_predictions_csv,
)
from train_resnet_image_classifier import (
    DEFAULT_MAX_OPEN_RAW_FILES,
    DEFAULT_PREFETCH_FACTOR,
    build_event_registry,
    build_object_manifest,
    build_split_event_maps,
    choose_threshold,
    compute_efficiency_by_pt,
    configure_torch_multiprocessing,
    parse_file_entries,
    print_split_weight_sums,
    rebalance_manifest_class_weights,
    resolve_num_workers,
    safe_roc_auc,
    save_efficiency_plot,
    save_history_plot,
    save_manifest_summary,
    save_roc_plot,
    save_score_distribution,
    set_seed,
    summarize_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary point-cloud transformer on EB or EE track points "
            "using the same event-level weighting sidecar as the image model."
        )
    )
    parser.add_argument(
        "--detector",
        choices=("eb", "ee"),
        required=True,
        help="Train on EB objects or EE objects.",
    )
    parser.add_argument(
        "--weight-h5",
        type=Path,
        required=True,
        help="Path to event_level_weights.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, metrics, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--track-types",
        type=str,
        default="GSF,PF,Lost",
        help="Comma-separated track collections (e.g. 'GSF,PF,Lost' for MiniAOD, 'GenTrk' for AOD).",
    )
    parser.add_argument(
        "--compact",
        type=Path,
        default=None,
        help="Read a prebuilt compact dataset HDF5 (skips the raw file scan).",
    )
    parser.add_argument(
        "--weight-key",
        type=str,
        default=None,
        help="Override the detector-specific weight dataset name in the sidecar file.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("max-f1", "youden"),
        default="max-f1",
        help="Validation rule used to choose the operating threshold.",
    )
    parser.add_argument(
        "--monitor",
        choices=("auc", "f1"),
        default="auc",
        help="Validation metric for early-stopping checkpoint selection.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (also drives LR-scheduler patience).",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Train the full --epochs (disable early stopping; LR scheduler still runs).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=16,
        help="Maximum number of track points kept per object.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Transformer embedding dimension.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of transformer encoder blocks.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="Feed-forward expansion ratio inside the transformer blocks.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability inside the transformer.",
    )
    parser.add_argument(
        "--disable-log-track-pt",
        action="store_true",
        help="Use raw track pT instead of log1p(pT).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Small smoke-test split and a short training run.",
    )
    parser.add_argument(
        "--debug-max-events-per-sample",
        type=int,
        default=12,
        help="Per-sample event cap when --debug is enabled.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    set_seed(args.seed)

    detector = args.detector
    weight_key = args.weight_key or ("EB_weight_split" if detector == "eb" else "EE_weight_split")
    log_track_pt = not args.disable_log_track_pt
    track_types = parse_track_types(args.track_types)

    weight_h5_path = args.weight_h5.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_num_workers = resolve_num_workers(args.num_workers)
    sharing_strategy = configure_torch_multiprocessing()

    print(f"[detector] {detector}")
    print(f"[weight_h5] {weight_h5_path}")
    print(f"[weight_key] {weight_key}")
    print(f"[output] {output_dir}")
    print(f"[device] {device}")
    print(f"[num_workers] requested={args.num_workers} effective={effective_num_workers}")
    if sharing_strategy is not None:
        print(f"[torch_sharing_strategy] {sharing_strategy}")

    if args.compact is not None:
        from pipeline.compact_manifest import load_compact_manifest
        print(f"[compact] {args.compact}")
        file_entries, split_manifests, code_to_sample = load_compact_manifest(args.compact)
        class_balance_stats = {}
        for split_name in ("train", "val", "test"):
            split_manifests[split_name], class_balance_stats[split_name] = \
                rebalance_manifest_class_weights(split_manifests[split_name])
    else:
        file_entries = parse_file_entries(weight_h5_path)

        event_file_idx, event_local_idx, event_sample_code, sample_to_code, code_to_sample = build_event_registry(
            file_entries=file_entries,
            weight_h5_path=weight_h5_path,
            detector=detector,
            weight_key=weight_key,
            debug=args.debug,
            debug_max_events_per_sample=args.debug_max_events_per_sample,
            seed=args.seed,
        )

        split_event_maps = build_split_event_maps(
            event_file_idx=event_file_idx,
            event_local_idx=event_local_idx,
            event_sample_code=event_sample_code,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
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

    train_dataset = TrackPointCloudDataset(
        detector=detector,
        file_entries=file_entries,
        manifest=split_manifests["train"],
        max_points=args.max_points,
        track_types=track_types,
        log_track_pt=log_track_pt,
        max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
    )
    val_dataset = TrackPointCloudDataset(
        detector=detector,
        file_entries=file_entries,
        manifest=split_manifests["val"],
        max_points=args.max_points,
        track_types=track_types,
        log_track_pt=log_track_pt,
        max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
    )
    test_dataset = TrackPointCloudDataset(
        detector=detector,
        file_entries=file_entries,
        manifest=split_manifests["test"],
        max_points=args.max_points,
        track_types=track_types,
        log_track_pt=log_track_pt,
        max_open_files=DEFAULT_MAX_OPEN_RAW_FILES,
    )

    loader_common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_track_point_cloud_batch,
    }
    if effective_num_workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_common_kwargs)

    model = PointCloudTransformerClassifier(
        input_dim=point_feature_dim(track_types),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "train_f1@0.5": [],
        "val_f1@0.5": [],
    }

    best_metric = -np.inf
    epochs_no_improve = 0
    best_model_path = output_dir / "best_model.pt"

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "detector": detector,
                "weight_h5": str(weight_h5_path),
                "weight_key": weight_key,
                "output_dir": str(output_dir),
                "model": "point_transformer",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "num_workers_requested": args.num_workers,
                "num_workers_effective": effective_num_workers,
                "torch_sharing_strategy": sharing_strategy,
                "dataset_max_open_raw_files": DEFAULT_MAX_OPEN_RAW_FILES,
                "loader_prefetch_factor": (
                    DEFAULT_PREFETCH_FACTOR if effective_num_workers > 0 else None
                ),
                "seed": args.seed,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "debug": args.debug,
                "debug_max_events_per_sample": args.debug_max_events_per_sample,
                "log_track_pt": log_track_pt,
                "track_types": list(track_types),
                "max_points": args.max_points,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
                "class_rebalancing": {
                    "enabled": True,
                    "target_weight_sum_per_class_per_split": 1.0,
                },
                "monitor": args.monitor,
                "threshold_mode": args.threshold_mode,
            },
            handle,
            indent=2,
        )

    print_split_weight_sums(split_manifests, class_balance_stats)

    print("\n[start training]\n")
    for epoch in range(1, args.epochs + 1):
        train_pack = attach_metrics(
            run_point_transformer_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler)
        )
        val_pack = attach_metrics(
            run_point_transformer_epoch(model, val_loader, device, optimizer=None, scaler=None)
        )

        history["train_loss"].append(train_pack["loss"])
        history["val_loss"].append(val_pack["loss"])
        history["train_auc"].append(train_pack["auc"])
        history["val_auc"].append(val_pack["auc"])
        history["train_f1@0.5"].append(train_pack["f1@0.5"])
        history["val_f1@0.5"].append(val_pack["f1@0.5"])

        monitor_value = val_pack["auc"] if args.monitor == "auc" else val_pack["f1@0.5"]
        scheduler.step(monitor_value if np.isfinite(monitor_value) else -1.0)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_pack['loss']:.6f} train_auc={train_pack['auc']:.4f} train_f1@0.5={train_pack['f1@0.5']:.4f} | "
            f"val_loss={val_pack['loss']:.6f} val_auc={val_pack['auc']:.4f} val_f1@0.5={val_pack['f1@0.5']:.4f}"
        )

        if np.isfinite(monitor_value) and monitor_value > best_metric:
            best_metric = monitor_value
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "monitor": args.monitor,
                    "detector": detector,
                    "input_dim": point_feature_dim(track_types),
                    "track_types": list(track_types),
                    "embed_dim": args.embed_dim,
                    "depth": args.depth,
                    "num_heads": args.num_heads,
                    "mlp_ratio": args.mlp_ratio,
                    "dropout": args.dropout,
                    "max_points": args.max_points,
                    "log_track_pt": log_track_pt,
                },
                best_model_path,
            )
        else:
            epochs_no_improve += 1
            if not args.no_early_stopping and epochs_no_improve >= args.patience:
                print(f"\n[early stopping] epoch={epoch}")
                break

    with (output_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    save_history_plot(history, output_dir / "training_history.png")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pack = attach_metrics(run_point_transformer_epoch(model, val_loader, device, optimizer=None, scaler=None))
    test_pack = attach_metrics(run_point_transformer_epoch(model, test_loader, device, optimizer=None, scaler=None))

    threshold = choose_threshold(
        y_true=val_pack["label"],
        y_score=val_pack["score"],
        sample_weight=val_pack["weight"],
        mode=args.threshold_mode,
    )

    test_pred = (test_pack["score"] >= threshold).astype(np.int64)
    test_auc = safe_roc_auc(test_pack["label"], test_pack["score"], test_pack["weight"])
    test_f1 = f1_score(
        test_pack["label"],
        test_pred,
        average="binary",
        sample_weight=test_pack["weight"],
        zero_division=0,
    )

    print("\n[test summary]")
    print(f"  threshold = {threshold:.6f}")
    print(f"  test_auc  = {test_auc:.6f}")
    print(f"  test_f1   = {test_f1:.6f}")

    save_roc_plot(test_pack["label"], test_pack["score"], test_pack["weight"], output_dir / "roc_test.png")
    save_score_distribution(
        test_pack["label"],
        test_pack["score"],
        test_pack["weight"],
        output_dir / "score_distribution_test.png",
    )

    with h5py.File(weight_h5_path, "r") as weight_h5:
        pt_edges = np.asarray(weight_h5.attrs["effective_pt_bin_edges"], dtype=np.float64)
    eff_rows = compute_efficiency_by_pt(
        y_true=test_pack["label"],
        y_score=test_pack["score"],
        weights=test_pack["weight"],
        pt=test_pack["pt"],
        threshold=threshold,
        pt_edges=pt_edges,
    )
    save_efficiency_plot(eff_rows, output_dir / "efficiency_vs_pt_test.png", threshold)

    with (output_dir / "efficiency_vs_pt_test.json").open("w") as handle:
        json.dump(eff_rows, handle, indent=2)

    save_track_predictions_csv(output_dir / "val_predictions.csv", val_pack, file_entries, code_to_sample, track_types)
    save_track_predictions_csv(output_dir / "test_predictions.csv", test_pack, file_entries, code_to_sample, track_types)

    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(
            {
                "detector": detector,
                "threshold": threshold,
                "test_auc": test_auc,
                "test_f1": test_f1,
                "best_monitor_value": best_metric,
                "monitor": args.monitor,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
