#!/usr/bin/env python3

from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# fp32 training (no AMP); TF32 on Ampere+ for fp32-range, NaN-safe speed.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset

TRACK_TYPES = ("GSF", "PF", "Lost")          # default (MiniAOD) track collections
POINT_FEATURE_DIM = 3 + len(TRACK_TYPES)     # default; real dim derived from track_types


def parse_track_types(s):
    return tuple(t for t in str(s).split(",") if t)


def point_feature_dim(track_types):
    return 3 + len(track_types)


def track_csv_columns(track_types):
    lows = [t.lower() for t in track_types]
    return [f"n_{t}" for t in lows] + [f"sum_pt_{t}" for t in lows]


def weighted_bce_loss(logits, targets, sample_weights):
    losses = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    if sample_weights is None:
        loss_num = losses.sum()
        loss_den = torch.tensor(losses.numel(), device=losses.device, dtype=losses.dtype)
    else:
        weights = sample_weights.to(losses.device).float()
        loss_num = (losses * weights).sum()
        loss_den = weights.sum().clamp_min(1e-12)
    return loss_num / loss_den, loss_num.detach(), loss_den.detach()


def sparse_track_arrays_to_points(flat_idx, values, track_type_id, n_track_types,
                                  image_size=512, log_track_pt=True):
    idx = np.asarray(flat_idx, dtype=np.int64)
    pt = np.asarray(values, dtype=np.float32)
    dim = 3 + n_track_types
    if idx.size == 0:
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.float32)

    pt = np.clip(pt, 0.0, None)
    x = ((idx % image_size).astype(np.float32) + 0.5) / float(image_size)
    y = ((idx // image_size).astype(np.float32) + 0.5) / float(image_size)
    x = 2.0 * x - 1.0
    y = 2.0 * y - 1.0

    pt_feature = np.log1p(pt) if log_track_pt else pt
    onehot = np.zeros((idx.size, n_track_types), dtype=np.float32)
    onehot[:, track_type_id] = 1.0
    points = np.concatenate(
        [
            x[:, None],
            y[:, None],
            pt_feature[:, None].astype(np.float32),
            onehot,
        ],
        axis=1,
    ).astype(np.float32)
    return points, pt


def build_track_point_cloud(raw_handle, detector, obj_idx, max_points, track_types,
                            log_track_pt=True):
    prefix = "EB" if detector == "eb" else "EE"
    n_types = len(track_types)
    dim = 3 + n_types
    point_parts = []
    counts = np.zeros(n_types, dtype=np.float32)
    sum_pt = np.zeros(n_types, dtype=np.float32)

    for type_id, track_type in enumerate(track_types):
        idx_key = f"{prefix}_track_pt_{track_type}_idx"
        val_key = f"{prefix}_track_pt_{track_type}_val"
        points, raw_pt = sparse_track_arrays_to_points(
            raw_handle[idx_key][obj_idx],
            raw_handle[val_key][obj_idx],
            track_type_id=type_id,
            n_track_types=n_types,
            log_track_pt=log_track_pt,
        )
        counts[type_id] = float(raw_pt.size)
        sum_pt[type_id] = float(raw_pt.sum())
        if points.size > 0:
            point_parts.append(points)

    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        # Keep the highest-pT tracks if a cloud ever exceeds the cap.
        if max_points is not None and points.shape[0] > max_points:
            order = np.argsort(-points[:, 2], kind="stable")
            points = points[order[:max_points]]
    else:
        points = np.zeros((0, dim), dtype=np.float32)

    return points, counts, sum_pt


def collate_track_point_cloud_batch(batch):
    batch_size = len(batch)
    feat_dim = batch[0]["points"].shape[1]
    max_points_in_batch = max(1, max(item["points"].shape[0] for item in batch))

    points = torch.zeros(batch_size, max_points_in_batch, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_points_in_batch, dtype=torch.bool)

    for batch_idx, item in enumerate(batch):
        n_points = item["points"].shape[0]
        if n_points > 0:
            points[batch_idx, :n_points] = item["points"]
            mask[batch_idx, :n_points] = True

    out = {
        "points": points,
        "mask": mask,
    }

    tensor_keys = (
        "label",
        "weight",
        "pt",
        "file_idx",
        "object_idx",
        "event_idx",
        "sample_code",
        "track_counts",
        "track_sum_pt",
    )
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    return out


class TrackPointCloudDataset(Dataset):
    def __init__(
        self,
        detector,
        file_entries,
        manifest,
        max_points,
        track_types=TRACK_TYPES,
        log_track_pt=True,
        max_open_files=32,
    ):
        self.detector = detector
        self.file_entries = file_entries
        self.max_points = max(1, int(max_points))
        self.track_types = tuple(track_types)
        self.log_track_pt = log_track_pt
        self.max_open_files = max(1, int(max_open_files))

        self.file_idx = np.asarray(manifest["file_idx"], dtype=np.int32)
        self.object_idx = np.asarray(manifest["object_idx"], dtype=np.int32)
        self.event_idx = np.asarray(manifest["event_idx"], dtype=np.int32)
        self.label = np.asarray(manifest["label"], dtype=np.int64)
        self.weight = np.asarray(manifest["weight"], dtype=np.float32)
        self.pt = np.asarray(manifest["pt"], dtype=np.float32)
        self.sample_code = np.asarray(manifest["sample_code"], dtype=np.int16)

        self._raw_handles = None

    def __len__(self):
        return len(self.label)

    def _close_all_handles(self):
        if self._raw_handles is None:
            return
        while self._raw_handles:
            _, handle = self._raw_handles.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass

    def __del__(self):
        self._close_all_handles()

    def _ensure_open(self, file_idx):
        if self._raw_handles is None:
            self._raw_handles = OrderedDict()

        handle = self._raw_handles.pop(file_idx, None)
        if handle is None:
            if len(self._raw_handles) >= self.max_open_files:
                _, oldest_handle = self._raw_handles.popitem(last=False)
                oldest_handle.close()
            handle = h5py.File(self.file_entries[file_idx].raw_path, "r")

        self._raw_handles[file_idx] = handle
        return handle

    def __getitem__(self, index):
        file_idx = int(self.file_idx[index])
        obj_idx = int(self.object_idx[index])
        raw = self._ensure_open(file_idx)

        points, track_counts, track_sum_pt = build_track_point_cloud(
            raw,
            detector=self.detector,
            obj_idx=obj_idx,
            max_points=self.max_points,
            track_types=self.track_types,
            log_track_pt=self.log_track_pt,
        )

        return {
            "points": torch.from_numpy(points),
            "label": torch.tensor(self.label[index], dtype=torch.float32),
            "weight": torch.tensor(self.weight[index], dtype=torch.float32),
            "pt": torch.tensor(self.pt[index], dtype=torch.float32),
            "file_idx": torch.tensor(file_idx, dtype=torch.int32),
            "object_idx": torch.tensor(obj_idx, dtype=torch.int32),
            "event_idx": torch.tensor(int(self.event_idx[index]), dtype=torch.int32),
            "sample_code": torch.tensor(int(self.sample_code[index]), dtype=torch.int16),
            "track_counts": torch.tensor(track_counts, dtype=torch.float32),
            "track_sum_pt": torch.tensor(track_sum_pt, dtype=torch.float32),
        }


class PointCloudTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim=POINT_FEATURE_DIM,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points, mask):
        x = self.input_proj(points)
        batch_size = x.size(0)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool)
        attn_mask = torch.cat([cls_mask, mask], dim=1)
        key_padding_mask = ~attn_mask

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_state = self.final_norm(x[:, 0])
        return self.head(cls_state)


def run_point_transformer_epoch(model, loader, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_num_total = 0.0
    loss_den_total = 0.0

    all_scores = []
    all_labels = []
    all_weights = []
    all_pts = []
    all_file_idx = []
    all_object_idx = []
    all_event_idx = []
    all_sample_code = []
    all_track_counts = []
    all_track_sum_pt = []

    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(points, mask).squeeze(1)
            loss, loss_num, loss_den = weighted_bce_loss(logits, labels, weights)
            if is_train:
                loss.backward()
                optimizer.step()

        loss_num_total += float(loss_num.detach().cpu().item())
        loss_den_total += float(loss_den.detach().cpu().item())

        scores = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
        all_weights.append(batch["weight"].detach().cpu().numpy())
        all_pts.append(batch["pt"].detach().cpu().numpy())
        all_file_idx.append(batch["file_idx"].detach().cpu().numpy())
        all_object_idx.append(batch["object_idx"].detach().cpu().numpy())
        all_event_idx.append(batch["event_idx"].detach().cpu().numpy())
        all_sample_code.append(batch["sample_code"].detach().cpu().numpy())
        all_track_counts.append(batch["track_counts"].detach().cpu().numpy())
        all_track_sum_pt.append(batch["track_sum_pt"].detach().cpu().numpy())

    return {
        "loss": loss_num_total / max(loss_den_total, 1e-12),
        "score": np.concatenate(all_scores),
        "label": np.concatenate(all_labels),
        "weight": np.concatenate(all_weights),
        "pt": np.concatenate(all_pts),
        "file_idx": np.concatenate(all_file_idx),
        "object_idx": np.concatenate(all_object_idx),
        "event_idx": np.concatenate(all_event_idx),
        "sample_code": np.concatenate(all_sample_code),
        "track_counts": np.concatenate(all_track_counts, axis=0),
        "track_sum_pt": np.concatenate(all_track_sum_pt, axis=0),
    }


def save_track_predictions_csv(output_path, pack, file_entries, code_to_sample, track_types=TRACK_TYPES):
    extra = track_csv_columns(track_types)
    header = (
        "score,label,weight,pt,file_idx,object_idx,event_idx,sample_id,raw_file,"
        + ",".join(extra) + "\n"
    )
    n_types = len(track_types)
    with output_path.open("w") as handle:
        handle.write(header)
        for i in range(len(pack["label"])):
            file_idx = int(pack["file_idx"][i])
            sample_id = code_to_sample[int(pack["sample_code"][i])]
            raw_file = file_entries[file_idx].raw_path
            counts = pack["track_counts"][i]
            sum_pt = pack["track_sum_pt"][i]
            vals = [f"{float(counts[j]):.0f}" for j in range(n_types)] + \
                   [f"{float(sum_pt[j]):.8f}" for j in range(n_types)]
            handle.write(
                f"{float(pack['score'][i]):.8f},{int(pack['label'][i])},{float(pack['weight'][i]):.8e},"
                f"{float(pack['pt'][i]):.8f},{file_idx},{int(pack['object_idx'][i])},"
                f"{int(pack['event_idx'][i])},{sample_id},{raw_file},"
                + ",".join(vals) + "\n"
            )
