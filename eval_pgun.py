#!/usr/bin/env python3
"""Evaluate an already-trained mreg model on the particle_gun compact (continuous A
mass), WITHOUT retraining. Loads best_model.pt, scores the gun compact, and writes
mass-regression plots vs the continuous truth (a_mass): pred-vs-true (log + linear)
and resolution-vs-mass (binned). Classification plots are N/A (gun is signal-only).

Usage:
  eval_pgun.py --model-type {image,track,cross} --detector {eb,ee} \
      --run-dir runs/mreg_v1/<run> --compact compact_pgun/pgun_<det>.h5 \
      --track-types Lost,PF,GSF [--no-es] --output-dir <out>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import multitask_common as mc
from pipeline.compact_manifest import load_compact_manifest
from train_resnet_image_classifier import (
    build_resnet, DetectorObjectDataset, image_in_channels, resolve_num_workers)
from train_fusion_cross_attention import (
    _RESNET_FEAT_CHANNELS, CrossAttentionFusion, CombinedDataset, collate_combined)
from track_point_transformer import (
    PointCloudTransformerClassifier, TrackPointCloudDataset,
    collate_track_point_cloud_batch, point_feature_dim, parse_track_types)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_model(mtype, detector, track_types, include_es, args):
    if mtype == "image":
        in_ch = image_in_channels(detector, track_types, include_es, include_track=False)
        bb = build_resnet(args.model, in_channels=in_ch); bb.fc = nn.Identity()
        return mc.MultiTaskModel(bb, _RESNET_FEAT_CHANNELS[args.model])
    if mtype == "track":
        bb = PointCloudTransformerClassifier(input_dim=point_feature_dim(track_types),
             embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
             mlp_ratio=args.mlp_ratio, dropout=args.dropout); bb.head = nn.Identity()
        return mc.MultiTaskModel(bb, args.embed_dim)
    in_ch = image_in_channels(detector, track_types, include_es, include_track=False)
    bb = CrossAttentionFusion(in_channels=in_ch, point_feat_dim=point_feature_dim(track_types),
         backbone=args.backbone, d_model=args.d_model, num_heads=args.num_heads,
         depth=args.cross_depth, track_depth=args.track_depth, dropout=args.dropout)
    bb.head = nn.Identity()
    return mc.MultiTaskModel(bb, 2 * args.d_model)


def make_loader(mtype, detector, fe, manifest, track_types, include_es, args, device):
    kw = dict(batch_size=args.batch_size, num_workers=resolve_num_workers(args.num_workers),
              pin_memory=(device.type == "cuda"))
    if mtype == "image":
        ds = DetectorObjectDataset(detector, fe, manifest, track_types=track_types,
                                   include_es=include_es, include_track=False)
    elif mtype == "track":
        ds = TrackPointCloudDataset(detector=detector, file_entries=fe, manifest=manifest,
                                    max_points=args.max_points, track_types=track_types)
        kw["collate_fn"] = collate_track_point_cloud_batch
    else:
        img = DetectorObjectDataset(detector, fe, manifest, track_types=track_types,
                                    include_es=include_es, include_track=False)
        trk = TrackPointCloudDataset(detector=detector, file_entries=fe, manifest=manifest,
                                     max_points=args.max_points, track_types=track_types)
        ds = CombinedDataset(img, trk); kw["collate_fn"] = collate_combined
    return DataLoader(ds, shuffle=False, **kw)


@torch.no_grad()
def infer(mtype, model, loader, device):
    model.eval()
    reg_all, oidx_all = [], []
    for b in loader:
        if mtype == "image":
            out = model(b["image"].to(device))
        elif mtype == "track":
            out = model(b["points"].to(device), b["mask"].to(device))
        else:
            out = model(b["image"].to(device), b["points"].to(device), b["mask"].to(device))
        reg_all.append(out[1].cpu().numpy())
        oidx_all.append(b["object_idx"].numpy())
    return np.concatenate(reg_all), np.concatenate(oidx_all)


def save_pgun_plots(out_dir, m_true, m_pred, tag):
    os.makedirs(out_dir, exist_ok=True)
    fin = np.isfinite(m_pred) & np.isfinite(m_true) & (m_true > 0)
    m_true, m_pred = m_true[fin], m_pred[fin]
    lims_lin = [0.0, 1.4]
    lims_log = [0.01, 12]
    # linear 2D
    plt.figure(figsize=(6, 6))
    plt.hist2d(np.clip(m_true, *lims_lin), np.clip(m_pred, *lims_lin), bins=50, range=[lims_lin, lims_lin], cmin=1)
    plt.plot(lims_lin, lims_lin, "r--", lw=1); plt.xlim(lims_lin); plt.ylim(lims_lin)
    plt.xlabel("true m_A [GeV] (gun)"); plt.ylabel("predicted m_A [GeV]")
    plt.title(f"pgun pred vs true (linear) - {tag}"); plt.colorbar(label="objects")
    plt.tight_layout(); plt.savefig(f"{out_dir}/pgun_pred_vs_true_linear.png"); plt.close()
    # log 2D (pred can exceed gun range since model trained on 0.4-10)
    plt.figure(figsize=(6, 6))
    plt.hist2d(np.clip(m_true, *lims_log), np.clip(m_pred, *lims_log), bins=50,
               range=[lims_log, lims_log], cmin=1)
    plt.plot(lims_log, lims_log, "r--", lw=1)
    plt.xscale("log"); plt.yscale("log"); plt.xlim(lims_log); plt.ylim(lims_log)
    plt.xlabel("true m_A [GeV] (gun)"); plt.ylabel("predicted m_A [GeV]")
    plt.title(f"pgun pred vs true (log) - {tag}"); plt.colorbar(label="objects")
    plt.tight_layout(); plt.savefig(f"{out_dir}/pgun_pred_vs_true_log.png"); plt.close()
    # resolution & bias vs true mass (log-spaced bins over the gun range)
    edges = np.geomspace(max(m_true.min(), 0.01), m_true.max(), 13)
    cen, res, bias, med = [], [], [], []
    for i in range(len(edges) - 1):
        sel = (m_true >= edges[i]) & (m_true < edges[i + 1])
        if sel.sum() < 20:
            continue
        rel = (m_pred[sel] - m_true[sel]) / m_true[sel]
        cen.append(np.sqrt(edges[i] * edges[i + 1]))
        res.append(rel.std()); bias.append(rel.mean()); med.append(np.median(m_pred[sel]))
    plt.figure(figsize=(7, 5))
    plt.plot(cen, res, "o-", label="resolution std[(pred-true)/true]")
    plt.plot(cen, bias, "s--", label="bias mean[(pred-true)/true]")
    plt.axhline(0, color="gray", lw=0.8); plt.xscale("log")
    plt.xlabel("true m_A [GeV] (gun)"); plt.ylabel("relative residual")
    plt.title(f"pgun resolution/bias vs mass - {tag}"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/pgun_resolution_vs_mass.png"); plt.close()
    return {"n": int(len(m_true)),
            "mae": float(np.abs(m_pred - m_true).mean()),
            "mre": float((np.abs(m_pred - m_true) / m_true).mean())}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=("image", "track", "cross"), required=True)
    p.add_argument("--detector", choices=("eb", "ee"), required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--compact", required=True)
    p.add_argument("--track-types", default="Lost,PF,GSF")
    p.add_argument("--no-es", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="resnet50")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)          # track depth
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--cross-depth", type=int, default=2)
    p.add_argument("--track-depth", type=int, default=2)
    p.add_argument("--max-points", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tt = parse_track_types(a.track_types)
    include_es = not a.no_es
    fe, man, _ = load_compact_manifest(a.compact)
    manifest = man["test"]  # gun compact is all split=test
    model = build_model(a.model_type, a.detector, tt, include_es, a).to(device)
    ck = torch.load(os.path.join(a.run_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    loader = make_loader(a.model_type, a.detector, fe, manifest, tt, include_es, a, device)
    reg_mu, oidx = infer(a.model_type, model, loader, device)
    m_pred = mc.inv_transform(reg_mu)
    import h5py
    with h5py.File(a.compact, "r") as f:
        a_mass = f["a_mass"][:]
    m_true = a_mass[oidx]
    tag = os.path.basename(a.run_dir.rstrip("/"))
    stats = save_pgun_plots(a.output_dir, m_true, m_pred, tag)
    print(f"[pgun-eval] {tag}: n={stats['n']} mae={stats['mae']:.3f} mre={stats['mre']:.3f} -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
