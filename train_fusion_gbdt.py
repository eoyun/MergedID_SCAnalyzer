#!/usr/bin/env python3
"""A3 fusion: nonlinear gradient-boosted stacking of the image + track experts.

Same recipe as train_fusion_ensemble.py (late fusion on the two experts'
per-object predictions), but the linear LogisticRegression meta-model is
replaced by a HistGradientBoostingClassifier so the combiner can learn
nonlinear interactions between image_logit, track_logit and the track
multiplicity / sum-pT features.

CSV-only: it reads the already-saved val/test_predictions.csv from the image and
track run dirs (no network re-inference). The existing train_fusion_ensemble.py
is left untouched; all shared logic is imported from it.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

# Reuse the LR-fusion plumbing verbatim (no duplication, no edits to that file).
from train_fusion_ensemble import (
    load_run_config,
    validate_compatible_runs,
    attach_metrics,
    load_prediction_pack_from_csv,
    align_packs,
    build_fusion_features,
    save_fusion_predictions_csv,
    csv_prediction_paths,
)
from train_resnet_image_classifier import (
    choose_threshold,
    safe_roc_auc,
    compute_efficiency_by_pt,
    save_efficiency_plot,
    save_roc_plot,
    save_score_distribution,
)


def parse_args():
    p = argparse.ArgumentParser(description="Gradient-boosted stacking fusion (A3).")
    p.add_argument("--image-run-dir", type=Path, required=True)
    p.add_argument("--track-run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold-mode", choices=("max-f1", "youden"), default="max-f1")
    # GBDT hyperparameters (kept modest: few features, small splits like eq1).
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-leaf-nodes", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--l2-regularization", type=float, default=1.0)
    p.add_argument("--no-early-stopping", action="store_true",
                   help="Disable GBDT internal early stopping.")
    p.add_argument("--seed", type=int, default=0)
    # Accepted for run_job.sh compatibility; fusion is fast, no resume needed.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume-eos-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    image_run_dir = args.image_run_dir.resolve()
    track_run_dir = args.track_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_cfg = load_run_config(image_run_dir)
    track_cfg = load_run_config(track_run_dir)
    validate_compatible_runs(image_cfg, track_cfg)

    detector = image_cfg["detector"]
    weight_h5_path = Path(image_cfg["weight_h5"]).resolve()
    track_types = tuple(track_cfg.get("track_types") or ("GSF", "PF", "Lost"))

    csv_paths = csv_prediction_paths(image_run_dir, track_run_dir)
    missing = [str(p) for p in csv_paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError("Missing prediction CSVs: " + ", ".join(missing))

    print(f"[detector] {detector}")
    print(f"[image_run_dir] {image_run_dir}")
    print(f"[track_run_dir] {track_run_dir}")
    print(f"[output] {output_dir}")

    image_val_pack = attach_metrics(load_prediction_pack_from_csv(csv_paths["image_val"], track_types=None))
    image_test_pack = attach_metrics(load_prediction_pack_from_csv(csv_paths["image_test"], track_types=None))
    track_val_pack = attach_metrics(load_prediction_pack_from_csv(csv_paths["track_val"], track_types=track_types))
    track_test_pack = attach_metrics(load_prediction_pack_from_csv(csv_paths["track_test"], track_types=track_types))

    image_val_pack, track_val_pack = align_packs(image_val_pack, track_val_pack)
    image_test_pack, track_test_pack = align_packs(image_test_pack, track_test_pack)

    x_val, feature_names = build_fusion_features(image_val_pack, track_val_pack, track_types)
    x_test, _ = build_fusion_features(image_test_pack, track_test_pack, track_types)
    y_val = image_val_pack["label"]
    y_test = image_test_pack["label"]
    w_val = image_val_pack["weight"]
    w_test = image_test_pack["weight"]

    if len(np.unique(y_val)) < 2:
        raise RuntimeError("Validation split has one class; cannot train the stacking model.")

    # No feature scaling: gradient boosting is invariant to monotone rescaling.
    fusion_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        early_stopping=not args.no_early_stopping,
        validation_fraction=0.15 if not args.no_early_stopping else None,
        random_state=args.seed,
    )
    fusion_model.fit(x_val, y_val, sample_weight=w_val)

    val_fused_score = fusion_model.predict_proba(x_val)[:, 1]
    test_fused_score = fusion_model.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_val, val_fused_score, w_val, mode=args.threshold_mode)

    test_pred = (test_fused_score >= threshold).astype(np.int64)
    test_auc = safe_roc_auc(y_test, test_fused_score, w_test)
    test_f1 = f1_score(y_test, test_pred, average="binary", sample_weight=w_test, zero_division=0)

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
        y_true=y_test, y_score=test_fused_score, weights=w_test,
        pt=image_test_pack["pt"], threshold=threshold, pt_edges=pt_edges,
    )
    save_efficiency_plot(eff_rows, output_dir / "efficiency_vs_pt_test.png", threshold)
    with (output_dir / "efficiency_vs_pt_test.json").open("w") as handle:
        json.dump(eff_rows, handle, indent=2)

    save_fusion_predictions_csv(output_dir / "val_predictions.csv", val_fused_score,
                                image_val_pack, track_val_pack, track_types, None, None)
    save_fusion_predictions_csv(output_dir / "test_predictions.csv", test_fused_score,
                                image_test_pack, track_test_pack, track_types, None, None)

    # Permutation importance is overkill here; store the model's native gains proxy
    # via feature names + the chosen hyperparameters for reproducibility.
    with (output_dir / "gbdt_model.json").open("w") as handle:
        json.dump(
            {
                "meta_model": "HistGradientBoostingClassifier",
                "feature_names": feature_names,
                "n_iter": int(fusion_model.n_iter_),
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "max_iter": args.max_iter,
                    "max_leaf_nodes": args.max_leaf_nodes,
                    "min_samples_leaf": args.min_samples_leaf,
                    "l2_regularization": args.l2_regularization,
                    "early_stopping": not args.no_early_stopping,
                },
            },
            handle, indent=2,
        )

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "image_run_dir": str(image_run_dir),
                "track_run_dir": str(track_run_dir),
                "output_dir": str(output_dir),
                "detector": detector,
                "weight_h5": str(weight_h5_path),
                "prediction_source": "csv",
                "meta_model": "gbdt",
                "threshold_mode": args.threshold_mode,
                "feature_names": feature_names,
                "csv_paths": {k: str(v) for k, v in csv_paths.items()},
            },
            handle, indent=2,
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
            handle, indent=2,
        )


if __name__ == "__main__":
    main()
