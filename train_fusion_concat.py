#!/usr/bin/env python3
"""B1 fusion: concatenate the two experts' penultimate EMBEDDINGS, train an MLP head.

Unlike the score-level stackers (train_fusion_ensemble.py / train_fusion_gbdt.py),
which see only the scalar image_logit / track_logit, this fuses the full internal
representations:

  image ResNet  -> features before fc  -> image embedding [B, C]   (512 or 2048)
  track PCT     -> CLS state (final_norm) -> track embedding [B, E]  (e.g. 128)
  concat([img_emb, trk_emb]) -> MLP head -> logit

The two encoders are loaded from their trained best_model.pt and FROZEN; only the
head is trained. Embeddings are extracted once per split (a single encoder pass)
and cached, so head training is fast. Reads a compact HDF5 so image and track are
index-aligned per object.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_resnet_image_classifier import (
    build_resnet,
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
    PointCloudTransformerClassifier,
    point_feature_dim,
)
from train_fusion_cross_attention import CombinedDataset, collate_combined, _RESNET_FEAT_CHANNELS
from sklearn.metrics import f1_score


# --------------------------------------------------------------------------- #
# Encoders -> embeddings
# --------------------------------------------------------------------------- #
def load_image_encoder(ckpt, device):
    model = build_resnet(ckpt["model_name"], in_channels=int(ckpt["in_channels"]))
    model.load_state_dict(ckpt["model_state_dict"])
    # drop the final fc; keep through avgpool -> [B, C, 1, 1]
    feat = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    feat_dim = _RESNET_FEAT_CHANNELS[ckpt["model_name"]]
    return feat, feat_dim


def load_track_encoder(ckpt, device):
    model = PointCloudTransformerClassifier(
        input_dim=int(ckpt["input_dim"]), embed_dim=int(ckpt["embed_dim"]),
        depth=int(ckpt["depth"]), num_heads=int(ckpt["num_heads"]),
        mlp_ratio=float(ckpt["mlp_ratio"]), dropout=float(ckpt["dropout"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), int(ckpt["embed_dim"])


def track_cls_embedding(model, points, mask):
    """Replicate PointCloudTransformerClassifier.forward up to the CLS state."""
    x = model.input_proj(points)
    cls = model.cls_token.expand(x.size(0), -1, -1)
    x = torch.cat([cls, x], dim=1)
    attn_mask = torch.cat(
        [torch.ones(x.size(0), 1, dtype=torch.bool, device=mask.device), mask], dim=1)
    x = model.encoder(x, src_key_padding_mask=~attn_mask)
    return model.final_norm(x[:, 0])


@torch.no_grad()
def extract_embeddings(img_feat, trk_model, loader, device):
    embs, labels, weights, pts = [], [], [], []
    fidx, oidx, eidx, scode = [], [], [], []
    for b in loader:
        image = b["image"].to(device, non_blocking=True)
        points = b["points"].to(device, non_blocking=True)
        mask = b["mask"].to(device, non_blocking=True)
        ie = img_feat(image).flatten(1)                    # [B, C]
        te = track_cls_embedding(trk_model, points, mask)  # [B, E]
        embs.append(torch.cat([ie, te], dim=1).cpu())
        labels.append(b["label"]); weights.append(b["weight"]); pts.append(b["pt"])
        fidx.append(b["file_idx"]); oidx.append(b["object_idx"])
        eidx.append(b["event_idx"]); scode.append(b["sample_code"])
    return {
        "emb": torch.cat(embs).float(),
        "label": torch.cat(labels).float(), "weight": torch.cat(weights).float(),
        "pt": torch.cat(pts).numpy(), "file_idx": torch.cat(fidx).numpy(),
        "object_idx": torch.cat(oidx).numpy(), "event_idx": torch.cat(eidx).numpy(),
        "sample_code": torch.cat(scode).numpy(),
    }


# --------------------------------------------------------------------------- #
# MLP head over concatenated embeddings
# --------------------------------------------------------------------------- #
class ConcatHead(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def run_head_epoch(head, loader, device, optimizer=None):
    is_train = optimizer is not None
    head.train() if is_train else head.eval()
    num = den = 0.0
    scores, labels, weights = [], [], []
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logit = head(x).squeeze(1)
            loss, ln, ld = weighted_bce_loss(logit, y, w)
            if is_train:
                loss.backward(); optimizer.step()
        num += float(ln.detach().cpu()); den += float(ld.detach().cpu())
        scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        labels.append(y.cpu().numpy()); weights.append(w.cpu().numpy())
    s = np.concatenate(scores); l = np.concatenate(labels).astype(np.int64); wt = np.concatenate(weights)
    return {"loss": num / max(den, 1e-12), "score": s, "label": l, "weight": wt,
            "auc": safe_roc_auc(l, s, wt),
            "f1@0.5": f1_score(l, (s >= 0.5).astype(np.int64), average="binary",
                               sample_weight=wt, zero_division=0)}


def emb_loader(pack, batch, shuffle):
    ds = TensorDataset(pack["emb"], pack["label"], pack["weight"])
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def save_predictions_csv(path, pack, scores, code_to_sample):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "label", "weight", "pt", "file_idx", "object_idx", "event_idx", "sample_id"])
        for i in range(len(scores)):
            sid = code_to_sample.get(int(pack["sample_code"][i]), "") if code_to_sample else ""
            w.writerow([f"{scores[i]:.8f}", int(pack['label'][i].item()),
                        f"{float(pack['weight'][i]):.8e}", f"{pack['pt'][i]:.8f}",
                        int(pack['file_idx'][i]), int(pack['object_idx'][i]),
                        int(pack['event_idx'][i]), sid])


def parse_args():
    p = argparse.ArgumentParser(description="Embedding-concat fusion (B1).")
    p.add_argument("--image-ckpt", required=True)
    p.add_argument("--track-ckpt", required=True)
    p.add_argument("--compact", required=True)
    p.add_argument("--weight-h5", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--extract-batch", type=int, default=64)
    p.add_argument("--head-batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--threshold-mode", choices=("max-f1", "youden"), default="max-f1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-eos-dir", default=None)  # accepted for run_job.sh; unused
    return p.parse_args()


def main():
    import h5py
    from pipeline.compact_manifest import load_compact_manifest

    args = parse_args()
    set_seed(args.seed)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_ckpt = torch.load(args.image_ckpt, map_location=device)
    trk_ckpt = torch.load(args.track_ckpt, map_location=device)
    detector = img_ckpt["detector"]
    track_types = tuple(trk_ckpt["track_types"])
    include_es = bool(img_ckpt.get("include_es", True))
    include_track = bool(img_ckpt.get("include_track", True))
    max_points = int(trk_ckpt["max_points"])

    img_feat, img_dim = load_image_encoder(img_ckpt, device)
    trk_model, trk_dim = load_track_encoder(trk_ckpt, device)
    print(f"[encoders] image={img_dim}  track={trk_dim}  concat={img_dim + trk_dim}")

    file_entries, split_manifests, code_to_sample = load_compact_manifest(args.compact)

    def loader(split):
        img_ds = DetectorObjectDataset(detector, file_entries, split_manifests[split],
                                       track_types=track_types, include_es=include_es,
                                       include_track=include_track)
        trk_ds = TrackPointCloudDataset(detector=detector, file_entries=file_entries,
                                        manifest=split_manifests[split],
                                        max_points=max_points, track_types=track_types)
        return DataLoader(CombinedDataset(img_ds, trk_ds), batch_size=args.extract_batch,
                          shuffle=False, num_workers=args.num_workers, collate_fn=collate_combined,
                          pin_memory=(device.type == "cuda"))

    print("[extract] embeddings (frozen encoders)...")
    packs = {s: extract_embeddings(img_feat, trk_model, loader(s), device)
             for s in ("train", "val", "test")}

    head = ConcatHead(img_dim + trk_dim, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr_loader = emb_loader(packs["train"], args.head_batch, True)
    va_loader = emb_loader(packs["val"], args.head_batch, False)

    best_auc = -np.inf; best_state = None; no_improve = 0
    history = {k: [] for k in ("train_loss", "val_loss", "train_auc", "val_auc")}
    print("\n[train head]\n")
    for epoch in range(1, args.epochs + 1):
        tr = run_head_epoch(head, tr_loader, device, opt)
        va = run_head_epoch(head, va_loader, device)
        for k in ("loss", "auc"):
            history[f"train_{k}"].append(tr[k]); history[f"val_{k}"].append(va[k])
        print(f"Epoch {epoch:03d} | train_auc={tr['auc']:.4f} | val_auc={va['auc']:.4f} f1={va['f1@0.5']:.4f}")
        if va["auc"] > best_auc:
            best_auc = va["auc"]; best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stopping] epoch={epoch}"); break
    head.load_state_dict(best_state)

    va = run_head_epoch(head, va_loader, device)
    te = run_head_epoch(head, emb_loader(packs["test"], args.head_batch, False), device)
    threshold = choose_threshold(va["label"], va["score"], va["weight"], mode=args.threshold_mode)
    test_auc = safe_roc_auc(te["label"], te["score"], te["weight"])
    test_f1 = f1_score(te["label"], (te["score"] >= threshold).astype(np.int64),
                       average="binary", sample_weight=te["weight"], zero_division=0)
    print(f"\n[test] threshold={threshold:.4f} auc={test_auc:.4f} f1={test_f1:.4f}")

    torch.save({"head_state_dict": head.state_dict(), "img_dim": img_dim, "trk_dim": trk_dim,
                "hidden_dim": args.hidden_dim}, out / "best_model.pt")
    save_history_plot(history, out / "training_history.png")
    save_roc_plot(te["label"], te["score"], te["weight"], out / "roc_test.png")
    save_score_distribution(te["label"], te["score"], te["weight"], out / "score_distribution_test.png")
    with h5py.File(args.weight_h5, "r") as wf:
        pt_edges = np.asarray(wf.attrs["effective_pt_bin_edges"], dtype=np.float64)
    eff = compute_efficiency_by_pt(te["label"], te["score"], te["weight"], packs["test"]["pt"], threshold, pt_edges)
    save_efficiency_plot(eff, out / "efficiency_vs_pt_test.png", threshold)
    with (out / "efficiency_vs_pt_test.json").open("w") as fh:
        json.dump(eff, fh, indent=2)
    save_predictions_csv(out / "val_predictions.csv", packs["val"], va["score"], code_to_sample)
    save_predictions_csv(out / "test_predictions.csv", packs["test"], te["score"], code_to_sample)
    with (out / "run_config.json").open("w") as fh:
        json.dump({"model": "embedding_concat_fusion", "detector": detector,
                   "image_ckpt": args.image_ckpt, "track_ckpt": args.track_ckpt,
                   "weight_h5": str(args.weight_h5), "img_dim": img_dim, "trk_dim": trk_dim,
                   "hidden_dim": args.hidden_dim, "threshold_mode": args.threshold_mode}, fh, indent=2)
    with (out / "test_metrics.json").open("w") as fh:
        json.dump({"detector": detector, "threshold": threshold,
                   "test_auc": test_auc, "test_f1": test_f1}, fh, indent=2)


if __name__ == "__main__":
    main()
