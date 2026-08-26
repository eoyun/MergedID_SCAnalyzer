#!/usr/bin/env python3
"""Cross-attention multi-task trainer: CrossAttentionFusion backbone (final head ->
Identity) + classification head (BCE) + regression head (MAE on log m_A, signal-only).
Reuses datasets/backbone from train_fusion_cross_attention.py (NOT modified)."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import multitask_common as mc
from train_fusion_cross_attention import (
    CrossAttentionFusion, CombinedDataset, collate_combined, _RESNET_FEAT_CHANNELS,
)
from train_resnet_image_classifier import (
    DetectorObjectDataset, image_in_channels, set_seed, resolve_num_workers,
    save_roc_plot, safe_roc_auc, rebalance_manifest_class_weights,
    save_unweighted_test_plots, DEFAULT_PREFETCH_FACTOR,
)
from track_point_transformer import TrackPointCloudDataset, point_feature_dim, parse_track_types

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_epoch(model, loader, device, code_to_sample, lam, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot = 0.0
    n = 0
    S_score, S_label, S_mpred, S_mtrue, S_sig, S_pt = [], [], [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        logm_np, mask_np = mc.mass_targets_from_codes(batch["sample_code"].numpy(), code_to_sample)
        mass = torch.from_numpy(logm_np).to(device)
        sig = torch.from_numpy(mask_np).to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            cls, reg = model(image, points, mask)
            loss, bce, mae = mc.multitask_loss(cls, reg, label, mass, sig, weight, lam)
            if is_train:
                loss.backward()
                optimizer.step()
        tot += float(loss.detach().cpu())
        n += 1
        S_score.append(torch.sigmoid(cls).detach().cpu().numpy())
        S_label.append(batch["label"].numpy())
        S_mpred.append(mc.inv_transform(reg.detach().cpu().numpy()))
        S_mtrue.append(mc.inv_transform(logm_np))
        S_sig.append(mask_np)
        S_pt.append(batch["pt"].numpy())
    label_all = np.concatenate(S_label)
    pack = {"loss": tot / max(n, 1),
            "score": np.concatenate(S_score), "label": label_all,
            "weight": np.ones(len(label_all)), "pt": np.concatenate(S_pt),
            "m_pred": np.concatenate(S_mpred), "m_true": np.concatenate(S_mtrue),
            "is_sig": np.concatenate(S_sig).astype(bool)}
    pack["auc"] = safe_roc_auc(pack["label"], pack["score"], pack["weight"])
    return pack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--compact", required=True)
    p.add_argument("--weight-h5", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--track-types", default="GenTrk")
    p.add_argument("--no-es", action="store_true")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--track-depth", type=int, default=2)
    p.add_argument("--max-points", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--resume-eos-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    detector = args.detector
    track_types = parse_track_types(args.track_types)
    include_es = not args.no_es
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nworkers = resolve_num_workers(args.num_workers)

    from pipeline.compact_manifest import load_compact_manifest
    file_entries, split_manifests, code_to_sample = load_compact_manifest(args.compact)
    for s in ("train", "val", "test"):
        split_manifests[s], _ = rebalance_manifest_class_weights(split_manifests[s])

    def ds(split):
        img = DetectorObjectDataset(detector, file_entries, split_manifests[split],
                                    track_types=track_types, include_es=include_es,
                                    include_track=False)
        trk = TrackPointCloudDataset(detector=detector, file_entries=file_entries,
                                     manifest=split_manifests[split],
                                     max_points=args.max_points, track_types=track_types)
        return CombinedDataset(img, trk)
    kw = dict(batch_size=args.batch_size, num_workers=nworkers,
              pin_memory=(device.type == "cuda"), collate_fn=collate_combined)
    if nworkers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    train_loader = DataLoader(ds("train"), shuffle=True, **kw)
    val_loader = DataLoader(ds("val"), shuffle=False, **kw)
    test_loader = DataLoader(ds("test"), shuffle=False, **kw)

    in_ch = image_in_channels(detector, track_types, include_es, include_track=False)
    backbone = CrossAttentionFusion(
        in_channels=in_ch, point_feat_dim=point_feature_dim(track_types),
        backbone=args.backbone, d_model=args.d_model, num_heads=args.num_heads,
        depth=args.depth, track_depth=args.track_depth, dropout=args.dropout)
    backbone.head = nn.Identity()
    model = mc.MultiTaskModel(backbone, 2 * args.d_model).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best = -np.inf
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last.pt"
    hist = []
    start_epoch = 1
    if args.resume_eos_dir:
        from pipeline import resume as _r
        _r.stage_in(args.resume_eos_dir, output_dir, ["best_model.pt", "last.pt"])
        if last_path.exists():
            rck = torch.load(last_path, map_location=device)
            model.load_state_dict(rck["model_state_dict"])
            if "optimizer_state_dict" in rck:
                opt.load_state_dict(rck["optimizer_state_dict"])
            if rck.get("scheduler_state_dict") is not None:
                sched.load_state_dict(rck["scheduler_state_dict"])
            start_epoch = int(rck.get("epoch", 0)) + 1
            best = float(rck.get("best_metric", best))
            hist = rck.get("history", hist)
            print(f"[resume] from epoch {rck.get('epoch')} -> start {start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, code_to_sample, args.lam, opt)
        va = run_epoch(model, val_loader, device, code_to_sample, args.lam, None)
        vm = mc.regression_metrics(va["m_true"][va["is_sig"]], va["m_pred"][va["is_sig"]])
        sched.step(va["auc"] if np.isfinite(va["auc"]) else -1.0)
        hist.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                     "val_auc": va["auc"], "val_mass_mae": vm["mae"]})
        print(f"[epoch {epoch}] val_auc={va['auc']:.4f} val_mass_mae={vm['mae']:.4f}", flush=True)
        ck = {"epoch": epoch, "model_state_dict": model.state_dict(),
              "optimizer_state_dict": opt.state_dict(),
              "scheduler_state_dict": sched.state_dict(),
              "best_metric": best, "history": hist}
        torch.save(ck, last_path)
        if np.isfinite(va["auc"]) and va["auc"] > best:
            best = va["auc"]
            ck["best_metric"] = best
            torch.save(ck, best_path)
        if args.resume_eos_dir:
            from pipeline import resume as _r
            _r.push(args.resume_eos_dir, last_path)
            if best_path.exists():
                _r.push(args.resume_eos_dir, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val = run_epoch(model, val_loader, device, code_to_sample, args.lam, None)
    test = run_epoch(model, test_loader, device, code_to_sample, args.lam, None)
    sig = test["is_sig"]

    save_roc_plot(test["label"], test["score"], test["weight"], output_dir / "roc_test.png")
    with h5py.File(args.compact, "r") as cf:
        pt_edges = np.asarray(cf.attrs["effective_pt_bin_edges"], dtype=np.float64)
    save_unweighted_test_plots(output_dir, val["label"], val["score"],
                               test["label"], test["score"], test["pt"], pt_edges)
    mc.save_mass_regression_plots(str(output_dir), test["m_true"][sig], test["m_pred"][sig], suffix="_test")

    metrics = {"test_auc": test["auc"],
               **{f"mass_{k}": v for k, v in
                  mc.regression_metrics(test["m_true"][sig], test["m_pred"][sig]).items()}}
    metrics["mass_per_point"] = mc.regression_metrics_per_point(test["m_true"][sig], test["m_pred"][sig])
    with (output_dir / "test_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    with (output_dir / "history.json").open("w") as fh:
        json.dump(hist, fh, indent=2)

    def _save_csv(path, pk):
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["score", "label", "pt", "m_true", "m_pred", "is_sig"])
            for i in range(len(pk["score"])):
                w.writerow([pk["score"][i], int(pk["label"][i]), pk["pt"][i],
                            pk["m_true"][i], pk["m_pred"][i], int(pk["is_sig"][i])])
    _save_csv(output_dir / "val_predictions.csv", val)
    _save_csv(output_dir / "test_predictions.csv", test)
    print("[done] wrote plots + metrics to", output_dir, flush=True)


if __name__ == "__main__":
    main()
