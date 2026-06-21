#!/usr/bin/env python3

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50

DEFAULT_MAX_OPEN_RAW_FILES = 32
DEFAULT_PREFETCH_FACTOR = 1


@dataclass(frozen=True)
class FileEntry:
    split: str
    stem: str
    raw_path: str
    sample_id: str
    process_name: str
    n_events: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary ResNet classifier on EB or EE images using the "
            "event-level weight sidecar produced earlier."
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
    parser.add_argument("--model", choices=("resnet18", "resnet34", "resnet50"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
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
    parser.add_argument(
        "--disable-log-scale",
        action="store_true",
        help="Use raw channel values instead of log1p(max(x,0)).",
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
        help="Early stopping patience in epochs.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_workers(requested_num_workers: int):
    if requested_num_workers <= 0:
        return 0

    try:
        probe = mp.get_context().Lock()
        del probe
    except (OSError, PermissionError) as exc:
        print(
            f"[warning] multiprocessing dataloaders are unavailable "
            f"({exc.__class__.__name__}: {exc}). Falling back to num_workers=0."
        )
        return 0

    return requested_num_workers


def configure_torch_multiprocessing():
    try:
        available = torch.multiprocessing.get_all_sharing_strategies()
    except (AttributeError, RuntimeError):
        return None

    if "file_system" in available:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except RuntimeError as exc:
            print(f"[warning] could not set torch sharing strategy to file_system: {exc}")

    try:
        return torch.multiprocessing.get_sharing_strategy()
    except RuntimeError:
        return None


def repeat_to_512(arr: np.ndarray):
    h, w = arr.shape
    if 512 % h != 0 or 512 % w != 0:
        raise ValueError(f"Cannot tile shape {arr.shape} exactly to 512x512.")
    rep_h = 512 // h
    rep_w = 512 // w
    return np.repeat(np.repeat(arr, rep_h, axis=0), rep_w, axis=1)


def sparse_to_dense(flat_idx, values, size=512):
    dense = np.zeros((size, size), dtype=np.float32)
    idx = np.asarray(flat_idx, dtype=np.int64)
    val = np.asarray(values, dtype=np.float32)
    if idx.size == 0:
        return dense
    np.add.at(dense.reshape(-1), idx, val)
    return dense


def preprocess_channel(arr: np.ndarray, log_scale: bool):
    out = np.asarray(arr, dtype=np.float32)
    if log_scale:
        out = np.log1p(np.clip(out, 0.0, None))
    return out


def stratified_split_per_sample(sample_codes, train_frac, val_frac, seed):
    rng = np.random.default_rng(seed)
    sample_codes = np.asarray(sample_codes)

    train_idx = []
    val_idx = []
    test_idx = []

    for code in np.unique(sample_codes):
        idx = np.flatnonzero(sample_codes == code)
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            raise ValueError(f"Need at least 3 events for sample code {code}, found {n}.")

        n_train = int(math.floor(n * train_frac))
        n_val = int(math.floor(n * val_frac))
        n_test = n - n_train - n_val

        if n_train < 1:
            n_train = 1
        if n_val < 1:
            n_val = 1
        n_test = n - n_train - n_val
        if n_test < 1:
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
            n_test = n - n_train - n_val

        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def build_resnet(model_name: str, in_channels: int):
    builders = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }
    model = builders[model_name](weights=None)
    model.conv1 = nn.Conv2d(
        in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


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


def safe_roc_auc(y_true, y_score, sample_weight):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def choose_threshold(y_true, y_score, sample_weight, mode):
    if mode == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
        j = tpr - fpr
        best = int(np.argmax(j))
        return float(thresholds[best])

    thresholds = np.linspace(0.0, 1.0, 501)
    best_thr = 0.5
    best_val = -np.inf
    for thr in thresholds:
        pred = (y_score >= thr).astype(np.int64)
        score = f1_score(y_true, pred, average="binary", sample_weight=sample_weight, zero_division=0)
        if score > best_val:
            best_val = score
            best_thr = float(thr)
    return best_thr


def compute_efficiency_by_pt(y_true, y_score, weights, pt, threshold, pt_edges):
    results = []
    passed = y_score >= threshold
    classes = [(1, "signal"), (0, "background")]

    for cls, name in classes:
        cls_mask = (y_true == cls)
        for ibin in range(len(pt_edges) - 1):
            lo = pt_edges[ibin]
            hi = pt_edges[ibin + 1]
            in_bin = (pt >= lo) & (pt < hi)
            denom_mask = cls_mask & in_bin
            numer_mask = denom_mask & passed

            denom = float(weights[denom_mask].sum())
            numer = float(weights[numer_mask].sum())
            eff = numer / denom if denom > 0 else float("nan")

            results.append(
                {
                    "class_name": name,
                    "bin_index": ibin,
                    "pt_low": float(lo),
                    "pt_high": "inf" if np.isinf(hi) else float(hi),
                    "numer": numer,
                    "denom": denom,
                    "efficiency": eff,
                }
            )
    return results


def save_efficiency_plot(rows, output_path: Path, threshold: float):
    plt.figure(figsize=(8, 6))
    for class_name, color in [("signal", "tab:blue"), ("background", "tab:orange")]:
        subset = [r for r in rows if r["class_name"] == class_name]
        x = []
        y = []
        for row in subset:
            hi = np.inf if row["pt_high"] == "inf" else float(row["pt_high"])
            x.append(row["pt_low"] if np.isinf(hi) else 0.5 * (row["pt_low"] + hi))
            y.append(row["efficiency"])
        plt.plot(x, y, marker="o", label=class_name, color=color)

    plt.xlabel("pT [GeV]")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency vs pT at threshold = {threshold:.4f}")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_roc_plot(y_true, y_score, weights, output_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.title("ROC curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return float(roc_auc)


def save_score_distribution(y_true, y_score, weights, output_path: Path):
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0.0, 1.0, 51)
    for cls, name, color in [(1, "signal", "tab:blue"), (0, "background", "tab:orange")]:
        mask = (y_true == cls)
        plt.hist(
            y_score[mask],
            bins=bins,
            weights=weights[mask],
            histtype="step",
            linewidth=2.0,
            label=name,
            color=color,
        )
    plt.xlabel("Signal score")
    plt.ylabel("Weighted entries")
    plt.title("Score distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_history_plot(history, output_path: Path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_auc"], label="train_auc")
    plt.plot(epochs, history["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class DetectorObjectDataset(Dataset):
    def __init__(self, detector, file_entries, manifest, log_scale=True, max_open_files=DEFAULT_MAX_OPEN_RAW_FILES):
        self.detector = detector
        self.file_entries = file_entries
        self.log_scale = log_scale
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

    def _build_eb_image(self, raw, obj_idx):
        ecal = repeat_to_512(raw["SC_energy"][obj_idx])
        lost = sparse_to_dense(raw["EB_track_pt_Lost_idx"][obj_idx], raw["EB_track_pt_Lost_val"][obj_idx])
        pf = sparse_to_dense(raw["EB_track_pt_PF_idx"][obj_idx], raw["EB_track_pt_PF_val"][obj_idx])
        gsf = sparse_to_dense(raw["EB_track_pt_GSF_idx"][obj_idx], raw["EB_track_pt_GSF_val"][obj_idx])
        channels = [ecal, lost, pf, gsf]
        channels = [preprocess_channel(ch, self.log_scale) for ch in channels]
        return np.stack(channels, axis=0).astype(np.float32)

    def _build_ee_image(self, raw, obj_idx):
        ee = repeat_to_512(raw["EE_seed_energy"][obj_idx])
        es1 = repeat_to_512(raw["ES_seed_plane1_energy"][obj_idx])
        es2 = repeat_to_512(raw["ES_seed_plane2_energy"][obj_idx])
        lost = sparse_to_dense(raw["EE_track_pt_Lost_idx"][obj_idx], raw["EE_track_pt_Lost_val"][obj_idx])
        pf = sparse_to_dense(raw["EE_track_pt_PF_idx"][obj_idx], raw["EE_track_pt_PF_val"][obj_idx])
        gsf = sparse_to_dense(raw["EE_track_pt_GSF_idx"][obj_idx], raw["EE_track_pt_GSF_val"][obj_idx])
        channels = [ee, es1, es2, lost, pf, gsf]
        channels = [preprocess_channel(ch, self.log_scale) for ch in channels]
        return np.stack(channels, axis=0).astype(np.float32)

    def __getitem__(self, index):
        file_idx = int(self.file_idx[index])
        obj_idx = int(self.object_idx[index])
        raw = self._ensure_open(file_idx)

        if self.detector == "eb":
            image = self._build_eb_image(raw, obj_idx)
        else:
            image = self._build_ee_image(raw, obj_idx)

        return {
            "image": torch.from_numpy(image),
            "label": torch.tensor(self.label[index], dtype=torch.float32),
            "weight": torch.tensor(self.weight[index], dtype=torch.float32),
            "pt": torch.tensor(self.pt[index], dtype=torch.float32),
            "file_idx": torch.tensor(file_idx, dtype=torch.int32),
            "object_idx": torch.tensor(obj_idx, dtype=torch.int32),
            "event_idx": torch.tensor(int(self.event_idx[index]), dtype=torch.int32),
            "sample_code": torch.tensor(int(self.sample_code[index]), dtype=torch.int16),
        }


def run_one_epoch(model, loader, device, optimizer=None, scaler=None):
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

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images).squeeze(1)
                loss, loss_num, loss_den = weighted_bce_loss(logits, labels, weights)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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

    out = {
        "loss": loss_num_total / max(loss_den_total, 1e-12),
        "score": np.concatenate(all_scores),
        "label": np.concatenate(all_labels),
        "weight": np.concatenate(all_weights),
        "pt": np.concatenate(all_pts),
        "file_idx": np.concatenate(all_file_idx),
        "object_idx": np.concatenate(all_object_idx),
        "event_idx": np.concatenate(all_event_idx),
        "sample_code": np.concatenate(all_sample_code),
    }
    out["auc"] = safe_roc_auc(out["label"], out["score"], out["weight"])
    out["f1@0.5"] = f1_score(
        out["label"],
        (out["score"] >= 0.5).astype(np.int64),
        average="binary",
        sample_weight=out["weight"],
        zero_division=0,
    )
    return out


def parse_file_entries(weight_h5_path: Path):
    file_entries = []
    with h5py.File(weight_h5_path, "r") as weight_h5:
        files_group = weight_h5["files"]
        for split in sorted(files_group.keys()):
            split_group = files_group[split]
            for stem in sorted(split_group.keys()):
                group = split_group[stem]
                file_entries.append(
                    FileEntry(
                        split=split,
                        stem=stem,
                        raw_path=str(group.attrs["input_file"]),
                        sample_id=str(group.attrs["sample_id"]),
                        process_name=str(group.attrs["process_name"]),
                        n_events=int(group.attrs["n_events"]),
                    )
                )
    return file_entries


def build_event_registry(file_entries, weight_h5_path: Path, detector: str, weight_key: str, debug: bool, debug_max_events_per_sample: int, seed: int):
    event_file_idx = []
    event_local_idx = []
    event_sample_code = []

    sample_to_code = {}
    code_to_sample = {}

    rng = np.random.default_rng(seed)
    pending_by_sample = defaultdict(list)

    with h5py.File(weight_h5_path, "r") as weight_h5:
        for file_idx, entry in enumerate(file_entries):
            weight_group = weight_h5["files"][entry.split][entry.stem]
            with h5py.File(entry.raw_path, "r") as raw:
                if detector == "eb":
                    event_idx = raw["EB_ele_event_idx"][:].astype(np.int64)
                else:
                    event_idx = raw["EE_ele_event_idx"][:].astype(np.int64)
                obj_weights = weight_group[weight_key][:].astype(np.float64)

            positive_events = np.unique(event_idx[obj_weights > 0])
            if positive_events.size == 0:
                continue

            if entry.sample_id not in sample_to_code:
                code = len(sample_to_code)
                sample_to_code[entry.sample_id] = code
                code_to_sample[code] = entry.sample_id

            code = sample_to_code[entry.sample_id]
            for local_event_idx in positive_events:
                pending_by_sample[code].append((file_idx, int(local_event_idx)))

    for code, records in pending_by_sample.items():
        records = np.asarray(records, dtype=np.int64)
        order = rng.permutation(len(records))
        records = records[order]
        if debug:
            records = records[: min(len(records), debug_max_events_per_sample)]
        event_file_idx.append(records[:, 0])
        event_local_idx.append(records[:, 1])
        event_sample_code.append(np.full(len(records), code, dtype=np.int16))

    if not event_file_idx:
        raise RuntimeError(f"No positive-weight {detector.upper()} objects found.")

    return (
        np.concatenate(event_file_idx),
        np.concatenate(event_local_idx),
        np.concatenate(event_sample_code),
        sample_to_code,
        code_to_sample,
    )


def build_split_event_maps(event_file_idx, event_local_idx, event_sample_code, train_frac, val_frac, seed):
    train_sel, val_sel, test_sel = stratified_split_per_sample(
        event_sample_code,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    split_selectors = {
        "train": train_sel,
        "val": val_sel,
        "test": test_sel,
    }

    split_maps = {}
    for split_name, selector in split_selectors.items():
        mapping = defaultdict(list)
        for file_idx, event_idx in zip(event_file_idx[selector], event_local_idx[selector]):
            mapping[int(file_idx)].append(int(event_idx))
        split_maps[split_name] = {
            file_idx: np.asarray(sorted(indices), dtype=np.int64)
            for file_idx, indices in mapping.items()
        }
    return split_maps


def build_object_manifest(file_entries, weight_h5_path: Path, detector: str, weight_key: str, split_event_map, sample_to_code):
    file_idx_all = []
    object_idx_all = []
    event_idx_all = []
    label_all = []
    weight_all = []
    pt_all = []
    sample_code_all = []

    with h5py.File(weight_h5_path, "r") as weight_h5:
        for file_idx, entry in enumerate(file_entries):
            selected_events = split_event_map.get(file_idx)
            if selected_events is None or len(selected_events) == 0:
                continue

            weight_group = weight_h5["files"][entry.split][entry.stem]
            event_pt = weight_group["event_pt"][:].astype(np.float64)
            obj_weights = weight_group[weight_key][:].astype(np.float64)

            with h5py.File(entry.raw_path, "r") as raw:
                if detector == "eb":
                    event_idx = raw["EB_ele_event_idx"][:].astype(np.int64)
                else:
                    event_idx = raw["EE_ele_event_idx"][:].astype(np.int64)

            event_mask = np.zeros(entry.n_events, dtype=bool)
            event_mask[selected_events] = True
            object_mask = (obj_weights > 0) & event_mask[event_idx]
            object_idx = np.flatnonzero(object_mask)
            if object_idx.size == 0:
                continue

            sample_code = sample_to_code[entry.sample_id]
            file_idx_all.append(np.full(object_idx.size, file_idx, dtype=np.int32))
            object_idx_all.append(object_idx.astype(np.int32))
            obj_event_idx = event_idx[object_idx].astype(np.int32)
            event_idx_all.append(obj_event_idx)
            label_all.append(np.full(object_idx.size, 0 if entry.sample_id == "background" else 1, dtype=np.int64))
            weight_all.append(obj_weights[object_idx].astype(np.float32))
            pt_all.append(event_pt[obj_event_idx].astype(np.float32))
            sample_code_all.append(np.full(object_idx.size, sample_code, dtype=np.int16))

    if not file_idx_all:
        raise RuntimeError(f"No {detector.upper()} objects found after split selection.")

    return {
        "file_idx": np.concatenate(file_idx_all),
        "object_idx": np.concatenate(object_idx_all),
        "event_idx": np.concatenate(event_idx_all),
        "label": np.concatenate(label_all),
        "weight": np.concatenate(weight_all),
        "pt": np.concatenate(pt_all),
        "sample_code": np.concatenate(sample_code_all),
    }


def rebalance_manifest_class_weights(manifest, target_sum: float = 1.0):
    labels = manifest["label"]
    weights = manifest["weight"].astype(np.float64, copy=True)

    rebalanced = weights.copy()
    stats = {}
    for label_value, class_name in ((0, "background"), (1, "signal")):
        class_mask = labels == label_value
        if not np.any(class_mask):
            raise RuntimeError(f"Manifest is missing the {class_name} class.")

        original_sum = float(weights[class_mask].sum())
        if original_sum <= 0.0:
            raise RuntimeError(f"Manifest has non-positive total weight for the {class_name} class.")

        scale = target_sum / original_sum
        rebalanced[class_mask] *= scale
        stats[class_name] = {
            "original_weight_sum": original_sum,
            "scale_factor": scale,
            "rebalanced_weight_sum": float(rebalanced[class_mask].sum()),
        }

    out = dict(manifest)
    out["weight"] = rebalanced.astype(np.float32)
    return out, stats


def summarize_manifest(name, manifest, code_to_sample):
    labels = manifest["label"]
    weights = manifest["weight"]
    sample_codes = manifest["sample_code"]
    print(f"\n[{name}] objects={len(labels)} weight_sum={weights.sum():.6f}")
    label_counts = Counter(labels.tolist())
    print(f"  label counts: background={label_counts.get(0, 0)} signal={label_counts.get(1, 0)}")
    print(
        "  label weight sums: "
        f"background={weights[labels == 0].sum():.6f} "
        f"signal={weights[labels == 1].sum():.6f}"
    )
    for code, count in sorted(Counter(sample_codes.tolist()).items()):
        print(f"  {code_to_sample[code]:20s} {count}")


def save_manifest_summary(output_path: Path, split_manifests, code_to_sample, class_balance_stats):
    summary = {}
    for split_name, manifest in split_manifests.items():
        codes, counts = np.unique(manifest["sample_code"], return_counts=True)
        labels = manifest["label"]
        weights = manifest["weight"]
        summary[split_name] = {
            "n_objects": int(len(manifest["label"])),
            "weight_sum": float(np.sum(weights)),
            "background_weight_sum": float(np.sum(weights[labels == 0])),
            "signal_weight_sum": float(np.sum(weights[labels == 1])),
            "class_balance": class_balance_stats[split_name],
            "samples": {
                code_to_sample[int(code)]: int(count)
                for code, count in zip(codes, counts)
            },
        }
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def print_split_weight_sums(split_manifests, class_balance_stats):
    print("\n[weight sums before training]")
    for split_name in ("train", "val", "test"):
        manifest = split_manifests[split_name]
        labels = manifest["label"]
        weights = manifest["weight"].astype(np.float64)
        bg_sum = float(np.sum(weights[labels == 0]))
        sig_sum = float(np.sum(weights[labels == 1]))
        total_sum = float(np.sum(weights))
        bg_original = class_balance_stats[split_name]["background"]["original_weight_sum"]
        sig_original = class_balance_stats[split_name]["signal"]["original_weight_sum"]
        print(
            f"{split_name:5s} "
            f"background={bg_sum:.8f} "
            f"signal={sig_sum:.8f} "
            f"total={total_sum:.8f} "
            f"| before_rebalance "
            f"background={bg_original:.8f} "
            f"signal={sig_original:.8f}"
        )


def save_predictions_csv(output_path: Path, pack, file_entries, code_to_sample):
    header = "score,label,weight,pt,file_idx,object_idx,event_idx,sample_id,raw_file\n"
    with output_path.open("w") as handle:
        handle.write(header)
        for i in range(len(pack["label"])):
            file_idx = int(pack["file_idx"][i])
            sample_id = code_to_sample[int(pack["sample_code"][i])]
            raw_file = file_entries[file_idx].raw_path
            handle.write(
                f"{float(pack['score'][i]):.8f},{int(pack['label'][i])},{float(pack['weight'][i]):.8e},"
                f"{float(pack['pt'][i]):.8f},{file_idx},{int(pack['object_idx'][i])},"
                f"{int(pack['event_idx'][i])},{sample_id},{raw_file}\n"
            )


def main():
    args = parse_args()
    set_seed(args.seed)

    detector = args.detector
    weight_key = args.weight_key or ("EB_weight_split" if detector == "eb" else "EE_weight_split")
    log_scale = not args.disable_log_scale

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

    train_dataset = DetectorObjectDataset(detector, file_entries, split_manifests["train"], log_scale=log_scale)
    val_dataset = DetectorObjectDataset(detector, file_entries, split_manifests["val"], log_scale=log_scale)
    test_dataset = DetectorObjectDataset(detector, file_entries, split_manifests["test"], log_scale=log_scale)

    loader_common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if effective_num_workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_common_kwargs,
    )

    in_channels = 4 if detector == "eb" else 6
    model = build_resnet(args.model, in_channels=in_channels).to(device)
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
                "model": args.model,
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
                "log_scale": log_scale,
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
        train_pack = run_one_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler)
        val_pack = run_one_epoch(model, val_loader, device, optimizer=None, scaler=None)

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
                    "model_name": args.model,
                    "in_channels": in_channels,
                },
                best_model_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n[early stopping] epoch={epoch}")
                break

    with (output_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    save_history_plot(history, output_dir / "training_history.png")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pack = run_one_epoch(model, val_loader, device, optimizer=None, scaler=None)
    test_pack = run_one_epoch(model, test_loader, device, optimizer=None, scaler=None)

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

    save_predictions_csv(output_dir / "val_predictions.csv", val_pack, file_entries, code_to_sample)
    save_predictions_csv(output_dir / "test_predictions.csv", test_pack, file_entries, code_to_sample)

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
