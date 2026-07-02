"""Load a compact dataset file into (file_entries, per-split manifests) so the
existing DetectorObjectDataset / TrackPointCloudDataset can read it unchanged:
each object's file_idx=0 and object_idx=its compact row."""
import h5py
import numpy as np

from train_resnet_image_classifier import FileEntry

SPLIT_NAME = {0: "train", 1: "val", 2: "test"}


def _h5_len(path):
    with h5py.File(str(path), "r") as f:
        return f["label"][:]


def load_compact_manifest(compact_path):
    compact_path = str(compact_path)
    with h5py.File(compact_path, "r") as f:
        split = f["split"][:]
        base = {k: f[k][:] for k in ("label", "weight", "pt", "event_idx", "sample_code")}
        n = len(split)
    rows = np.arange(n, dtype=np.int32)
    fe = [FileEntry(split="compact", stem="compact", raw_path=compact_path,
                    sample_id="compact", process_name="compact", n_events=n)]
    manifests = {}
    for code, name in SPLIT_NAME.items():
        m = split == code
        manifests[name] = {
            "file_idx": np.zeros(int(m.sum()), dtype=np.int32),
            "object_idx": rows[m].astype(np.int32),
            "event_idx": base["event_idx"][m].astype(np.int32),
            "label": base["label"][m].astype(np.int64),
            "weight": base["weight"][m].astype(np.float32),
            "pt": base["pt"][m].astype(np.float32),
            "sample_code": base["sample_code"][m].astype(np.int16),
        }
    return fe, manifests
