#!/usr/bin/env python3
"""B2 fusion: bidirectional cross-attention between the calo image and the track
point cloud, trained end-to-end.

Unlike the late-fusion stackers (train_fusion_ensemble.py / train_fusion_gbdt.py)
which only see the two experts' scalar scores, this model fuses their internal
token representations:

  calo image --ResNet backbone--> feature-map tokens [B, Ni, d]
  tracks     --point embedder ---> point tokens        [B, Nt, d]
  cross-attention: image tokens attend to track tokens and vice versa (L layers)
  masked-mean pool each modality -> concat -> MLP head -> logit

The image encoder here sees calo(+ES) ONLY (include_track=False) so the two
towers are disjoint views; cross-attention is what mixes them.

Reads a compact HDF5 (same as the single-modality trainers) so image and track
are index-aligned per object. Checkpoint/resume via pipeline.resume.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from train_resnet_image_classifier import (
    build_resnet,
    image_in_channels,
    DetectorObjectDataset,
    weighted_bce_loss,
    choose_threshold,
    safe_roc_auc,
    compute_efficiency_by_pt,
    save_efficiency_plot,
    save_roc_plot,
    save_score_distribution,
    save_history_plot,
    set_seed,
)
from track_point_transformer import (
    TrackPointCloudDataset,
    parse_track_types,
    point_feature_dim,
)
from sklearn.metrics import f1_score

_RESNET_FEAT_CHANNELS = {"resnet18": 512, "resnet34": 512,
                         "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}


# --------------------------------------------------------------------------- #
# Dataset: image + point cloud for the same object (index-aligned manifests)
# --------------------------------------------------------------------------- #
class CombinedDataset(Dataset):
    def __init__(self, image_ds, track_ds):
        assert len(image_ds) == len(track_ds)
        self.image_ds = image_ds
        self.track_ds = track_ds

    def __len__(self):
        return len(self.image_ds)

    def __getitem__(self, i):
        img = self.image_ds[i]
        trk = self.track_ds[i]
        # both come from the same manifest row -> must be the same object
        assert int(img["object_idx"]) == int(trk["object_idx"])
        return {
            "image": img["image"],
            "points": trk["points"],
            "label": img["label"],
            "weight": img["weight"],
            "pt": img["pt"],
            "file_idx": img["file_idx"],
            "object_idx": img["object_idx"],
            "event_idx": img["event_idx"],
            "sample_code": img["sample_code"],
        }


def collate_combined(batch):
    images = torch.stack([b["image"] for b in batch])
    max_n = max(1, max(b["points"].shape[0] for b in batch))
    feat_dim = batch[0]["points"].shape[1]
    points = torch.zeros(len(batch), max_n, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["points"].shape[0]
        if n > 0:
            points[i, :n] = b["points"]
            mask[i, :n] = True
    out = {"image": images, "points": points, "mask": mask}
    for k in ("label", "weight", "pt", "file_idx", "object_idx", "event_idx", "sample_code"):
        out[k] = torch.stack([b[k] for b in batch])
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CrossAttentionBlock(nn.Module):
    """Pre-LN block: query tokens attend to key/value tokens, then FFN."""

    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, q, kv, kv_key_padding_mask=None):
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_key_padding_mask,
                                need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, point_feat_dim, backbone="resnet18",
                 d_model=256, num_heads=4, depth=2, track_depth=2,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        base = build_resnet(backbone, in_channels=in_channels)
        # everything except avgpool + fc -> spatial feature map [B, C, h, w]
        self.image_backbone = nn.Sequential(*list(base.children())[:-2])
        feat_c = _RESNET_FEAT_CHANNELS[backbone]
        self.img_proj = nn.Conv2d(feat_c, d_model, kernel_size=1)

        # point embedder + a couple of self-attention layers for context
        self.trk_in = nn.Sequential(
            nn.Linear(point_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, int(d_model * mlp_ratio), dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.trk_encoder = nn.TransformerEncoder(enc_layer, track_depth)

        # a always-valid "null" track token so an object with 0 tracks never has
        # an all-masked key set (which would make attention softmax NaN)
        self.trk_null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.img_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trk_type = nn.Parameter(torch.zeros(1, 1, d_model))

        self.img2trk = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.trk2img = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(depth)])

        self.out_norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for t in (self.trk_null, self.img_type, self.trk_type):
            nn.init.trunc_normal_(t, std=0.02)

    def forward(self, image, points, mask):
        B = image.size(0)

        fmap = self.image_backbone(image)                       # [B, C, h, w]
        img = self.img_proj(fmap).flatten(2).transpose(1, 2)    # [B, Ni, d]
        img = img + self.img_type

        trk = self.trk_in(points)                               # [B, Nt, d]
        null = self.trk_null.expand(B, -1, -1)                  # [B, 1, d]
        trk = torch.cat([null, trk], dim=1)                     # [B, 1+Nt, d]
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device),
                           mask], dim=1)                        # [B, 1+Nt]
        kpm = ~valid                                            # True = padding
        trk = self.trk_encoder(trk, src_key_padding_mask=kpm)
        trk = trk + self.trk_type

        for a, b in zip(self.img2trk, self.trk2img):
            new_img = a(img, trk, kv_key_padding_mask=kpm)      # image <- track
            new_trk = b(trk, img, kv_key_padding_mask=None)     # track <- image
            img, trk = new_img, new_trk

        img_pool = img.mean(dim=1)
        m = valid.unsqueeze(-1).float()
        trk_pool = (trk * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        feat = torch.cat([img_pool, trk_pool], dim=1)
        return self.head(self.out_norm(feat))


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    num_tot = den_tot = 0.0
    scores, labels, weights, pts = [], [], [], []
    fidx, oidx, eidx, scode = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        w = batch["weight"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(image, points, mask).squeeze(1)
                loss, loss_num, loss_den = weighted_bce_loss(logits, y, w)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        num_tot += float(loss_num.detach().cpu())
        den_tot += float(loss_den.detach().cpu())
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(batch["label"].numpy().astype(np.int64))
        weights.append(batch["weight"].numpy())
        pts.append(batch["pt"].numpy())
        fidx.append(batch["file_idx"].numpy())
        oidx.append(batch["object_idx"].numpy())
        eidx.append(batch["event_idx"].numpy())
        scode.append(batch["sample_code"].numpy())
    pack = {
        "loss": num_tot / max(den_tot, 1e-12),
        "score": np.concatenate(scores), "label": np.concatenate(labels),
        "weight": np.concatenate(weights), "pt": np.concatenate(pts),
        "file_idx": np.concatenate(fidx), "object_idx": np.concatenate(oidx),
        "event_idx": np.concatenate(eidx), "sample_code": np.concatenate(scode),
    }
    pack["auc"] = safe_roc_auc(pack["label"], pack["score"], pack["weight"])
    pred = (pack["score"] >= 0.5).astype(np.int64)
    pack["f1@0.5"] = f1_score(pack["label"], pred, average="binary",
                              sample_weight=pack["weight"], zero_division=0)
    return pack


def save_predictions_csv(path, pack, code_to_sample):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "label", "weight", "pt", "file_idx", "object_idx",
                    "event_idx", "sample_id"])
        for i in range(len(pack["label"])):
            sid = code_to_sample.get(int(pack["sample_code"][i]), "") if code_to_sample else ""
            w.writerow([f"{pack['score'][i]:.8f}", int(pack["label"][i]),
                        f"{pack['weight'][i]:.8e}", f"{pack['pt'][i]:.8f}",
                        int(pack["file_idx"][i]), int(pack["object_idx"][i]),
                        int(pack["event_idx"][i]), sid])


def parse_args():
    p = argparse.ArgumentParser(description="Cross-attention image+track fusion (B2).")
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--weight-h5", required=True)
    p.add_argument("--compact", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--track-types", default="GSF,PF,Lost")
    p.add_argument("--no-es", action="store_true")
    p.add_argument("--backbone", default="resnet18", choices=list(_RESNET_FEAT_CHANNELS))
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2, help="cross-attention layers")
    p.add_argument("--track-depth", type=int, default=2, help="track self-attn layers")
    p.add_argument("--max-points", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--monitor", choices=("auc", "f1"), default="auc")
    p.add_argument("--threshold-mode", choices=("max-f1", "youden"), default="max-f1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-eos-dir", default=None)
    return p.parse_args()


def main():
    from pipeline.compact_manifest import load_compact_manifest

    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = args.detector
    track_types = parse_track_types(args.track_types)
    include_es = not args.no_es

    file_entries, split_manifests, code_to_sample = load_compact_manifest(args.compact)

    def make_loader(split, shuffle):
        img_ds = DetectorObjectDataset(detector, file_entries, split_manifests[split],
                                       track_types=track_types, include_es=include_es,
                                       include_track=False)
        trk_ds = TrackPointCloudDataset(detector=detector, file_entries=file_entries,
                                        manifest=split_manifests[split],
                                        max_points=args.max_points, track_types=track_types)
        ds = CombinedDataset(img_ds, trk_ds)
        kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  collate_fn=collate_combined, pin_memory=(device.type == "cuda"))
        if args.num_workers > 0:
            kw["persistent_workers"] = True
        return DataLoader(ds, shuffle=shuffle, **kw)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    in_channels = image_in_channels(detector, track_types, include_es, include_track=False)
    model = CrossAttentionFusion(
        in_channels=in_channels, point_feat_dim=point_feature_dim(track_types),
        backbone=args.backbone, d_model=args.d_model, num_heads=args.num_heads,
        depth=args.depth, track_depth=args.track_depth, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 2))

    history = {k: [] for k in ("train_loss", "val_loss", "train_auc", "val_auc",
                               "train_f1@0.5", "val_f1@0.5")}
    best_metric = -np.inf
    epochs_no_improve = 0
    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last.pt"

    start_epoch = 1
    if args.resume_eos_dir:
        from pipeline import resume as _resume
        got = _resume.stage_in(args.resume_eos_dir, output_dir, ["last.pt", "best_model.pt"])
        if "last.pt" in got and last_model_path.exists():
            ck = torch.load(last_model_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            if ck.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch = int(ck["epoch"]) + 1
            best_metric = float(ck.get("best_metric", best_metric))
            epochs_no_improve = int(ck.get("epochs_no_improve", 0))
            history = ck.get("history", history)
            print(f"[resume] @epoch={ck['epoch']} best={best_metric:.4f} -> start {start_epoch}")
        else:
            print("[resume] no checkpoint on EOS; starting fresh")

    with (output_dir / "run_config.json").open("w") as fh:
        json.dump({"model": "cross_attention_fusion", "detector": detector,
                   "weight_h5": str(args.weight_h5), "track_types": list(track_types),
                   "include_es": include_es, "backbone": args.backbone,
                   "d_model": args.d_model, "depth": args.depth,
                   "track_depth": args.track_depth, "max_points": args.max_points,
                   "epochs": args.epochs, "threshold_mode": args.threshold_mode}, fh, indent=2)

    print("\n[start training]\n")
    for epoch in range(start_epoch, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, scaler)
        va = run_epoch(model, val_loader, device)
        for k, src in (("loss", "loss"), ("auc", "auc"), ("f1@0.5", "f1@0.5")):
            history[f"train_{k}"].append(tr[src])
            history[f"val_{k}"].append(va[src])
        monitor_value = va["auc"] if args.monitor == "auc" else va["f1@0.5"]
        scheduler.step(monitor_value if np.isfinite(monitor_value) else -1.0)
        print(f"Epoch {epoch:03d} | train_loss={tr['loss']:.5f} train_auc={tr['auc']:.4f} | "
              f"val_loss={va['loss']:.5f} val_auc={va['auc']:.4f} val_f1={va['f1@0.5']:.4f}")

        improved = np.isfinite(monitor_value) and monitor_value > best_metric
        if improved:
            best_metric = monitor_value
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric, "monitor": args.monitor,
                        "detector": detector, "track_types": list(track_types),
                        "include_es": include_es, "backbone": args.backbone,
                        "d_model": args.d_model}, best_model_path)
        else:
            epochs_no_improve += 1

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric": best_metric, "epochs_no_improve": epochs_no_improve,
                    "history": history}, last_model_path)
        if args.resume_eos_dir:
            from pipeline import resume as _resume
            _resume.push(args.resume_eos_dir, last_model_path)
            if improved:
                _resume.push(args.resume_eos_dir, best_model_path)

        if not args.no_early_stopping and epochs_no_improve >= args.patience:
            print(f"\n[early stopping] epoch={epoch}")
            break

    with (output_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)
    save_history_plot(history, output_dir / "training_history.png")

    ck = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    val_pack = run_epoch(model, val_loader, device)
    test_pack = run_epoch(model, test_loader, device)

    threshold = choose_threshold(val_pack["label"], val_pack["score"],
                                 val_pack["weight"], mode=args.threshold_mode)
    test_pred = (test_pack["score"] >= threshold).astype(np.int64)
    test_auc = safe_roc_auc(test_pack["label"], test_pack["score"], test_pack["weight"])
    test_f1 = f1_score(test_pack["label"], test_pred, average="binary",
                       sample_weight=test_pack["weight"], zero_division=0)
    print(f"\n[test] threshold={threshold:.4f} auc={test_auc:.4f} f1={test_f1:.4f}")

    save_roc_plot(test_pack["label"], test_pack["score"], test_pack["weight"],
                  output_dir / "roc_test.png")
    save_score_distribution(test_pack["label"], test_pack["score"], test_pack["weight"],
                            output_dir / "score_distribution_test.png")
    import h5py
    with h5py.File(args.weight_h5, "r") as wf:
        pt_edges = np.asarray(wf.attrs["effective_pt_bin_edges"], dtype=np.float64)
    eff_rows = compute_efficiency_by_pt(test_pack["label"], test_pack["score"],
                                        test_pack["weight"], test_pack["pt"],
                                        threshold, pt_edges)
    save_efficiency_plot(eff_rows, output_dir / "efficiency_vs_pt_test.png", threshold)
    with (output_dir / "efficiency_vs_pt_test.json").open("w") as fh:
        json.dump(eff_rows, fh, indent=2)
    save_predictions_csv(output_dir / "val_predictions.csv", val_pack, code_to_sample)
    save_predictions_csv(output_dir / "test_predictions.csv", test_pack, code_to_sample)
    with (output_dir / "test_metrics.json").open("w") as fh:
        json.dump({"detector": detector, "threshold": threshold,
                   "test_auc": test_auc, "test_f1": test_f1}, fh, indent=2)


if __name__ == "__main__":
    main()
