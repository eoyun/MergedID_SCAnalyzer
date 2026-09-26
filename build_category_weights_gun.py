#!/usr/bin/env python3
"""Gun-heavy category weights (NEW pipeline; existing scripts untouched).

  signal     = particle_gun_heavy   (continuous A mass 0.1-50 GeV, A->ee)
  background = W, Z

pT variable = per-object RECO supercluster transverse energy, i.e.
  EB: SC_energyT_sum        EE: EE_seed_energyT_sum
This is DATA-APPLICABLE (unlike the gen A_pT used by build_category_weights.py,
which does not exist in data) and is a genuine per-object quantity: the crop is
centered on the seed electron, so SC ET ~ that electron's ET. The --pt-min cut and
the pT binning are BOTH on this per-object SC ET.

Category = 'all' only (no hasADD split).

Output layout mirrors build_category_weights.py so build_compact_dataset.py's
parse_file_entries / build_event_registry / build_object_manifest read it unchanged:
  files/<split>/<stem>/{EB_ele_event_idx, EB_weight_split,
                        EE_ele_event_idx, EE_weight_split, event_pt}
  + group attrs {input_file, split, process_name, sample_id, n_events}
PLUS a per-object EB_obj_pt / EE_obj_pt (the SC ET) so the gun compact builder can
carry the true per-object pT (build_object_manifest's `event_pt` broadcast is only a
placeholder for the gun and is overridden downstream).
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

from build_event_level_weights import (
    PT_BIN_EDGES,
    PT_BIN_EDGES_WITH_OVERFLOW,
    find_bin_indices,
    robust_h5_open,
    SkippableFileError,
)

SIGNAL_DIR = "particle_gun_heavy"
SPLITS = ("W", "Z", SIGNAL_DIR)
DETECTORS = ("eb", "ee")
DET_EVENT_IDX_KEYS = {"eb": "EB_ele_event_idx", "ee": "EE_ele_event_idx"}
SC_ET_KEYS = {"eb": "SC_energyT_sum", "ee": "EE_seed_energyT_sum"}
WEIGHT_KEYS = {"eb": "EB_weight_split", "ee": "EE_weight_split"}
OBJ_PT_KEYS = {"eb": "EB_obj_pt", "ee": "EE_obj_pt"}
N_BINS = len(PT_BIN_EDGES_WITH_OVERFLOW) - 1


def _sample_id(split):
    return "background" if split in ("W", "Z") else "signal"


def _read_meta(task):
    """Worker: read one raw file's per-object (event_idx, SC ET) for both detectors,
    plus n_events. Small arrays only (no images)."""
    split, path = task
    try:
        with robust_h5_open(path) as r:
            ne = int(r["eventId"].shape[0])
            out = {"split": split, "path": str(path), "stem": Path(path).stem,
                   "sample_id": _sample_id(split), "n_events": ne, "ok": True}
            for det in DETECTORS:
                ei = r[DET_EVENT_IDX_KEYS[det]][:].astype(np.int64)
                pt = r[SC_ET_KEYS[det]][:].astype(np.float32)
                out[det] = (ei, pt)
    except SkippableFileError as e:
        return {"path": str(path), "ok": False, "err": str(e)}
    return out


def scan_parallel(input_dir, workers):
    tasks = []
    for split in SPLITS:
        for p in sorted((Path(input_dir) / split).glob("*.h5")):
            tasks.append((split, str(p)))
    print(f"[scan] {len(tasks)} files across {SPLITS}", flush=True)
    metas = []
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(workers) as pool:
        for m in pool.imap_unordered(_read_meta, tasks):
            done += 1
            if not m.get("ok"):
                print(f"[scan][SKIP] {m['path']}: {m.get('err')}", flush=True)
                continue
            metas.append(m)
            if done % 250 == 0:
                print(f"[scan] {done}/{len(tasks)}", flush=True)
    metas.sort(key=lambda m: (m["split"], m["stem"]))
    return metas


def select_background_masks(metas, det, pt_min, max_objects, seed):
    """Per-detector background OBJECT subsample: eligible = SC ET >= pt_min; randomly
    keep up to max_objects across ALL background files (keep all if fewer)."""
    per_file, counts = [], []
    for m in metas:
        if m["sample_id"] != "background":
            continue
        _ei, pt = m[det]
        elig = pt >= pt_min
        per_file.append((m["stem"], elig))
        counts.append(int(elig.sum()))
    n_elig = int(sum(counts))
    if max_objects is None or n_elig <= max_objects:
        return {stem: elig.copy() for stem, elig in per_file}, n_elig, n_elig
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_elig, size=max_objects, replace=False))
    masks, offset = {}, 0
    for (stem, elig), cnt in zip(per_file, counts):
        lo = np.searchsorted(chosen, offset, "left")
        hi = np.searchsorted(chosen, offset + cnt, "left")
        pos = chosen[lo:hi] - offset
        mm = np.zeros(elig.shape, dtype=bool)
        if pos.size:
            mm[np.nonzero(elig)[0][pos]] = True
        masks[stem] = mm
        offset += cnt
    return masks, max_objects, n_elig


def selected_mask(m, det, pt_min, bg_masks):
    """(sc_et, event_idx, selected_mask) for one file+detector."""
    ei, pt = m[det]
    if m["sample_id"] == "background":
        mask = bg_masks[det].get(m["stem"], np.zeros(pt.shape, dtype=bool))
    else:  # signal (gun): keep all objects passing the SC-ET cut
        mask = pt >= pt_min
    return pt, ei, mask


def make_lookup(counts_by_sample):
    """Per-sample, per-bin object weight: bkg total=1, signal total=1, each non-empty
    bin gets an equal share of the sample target, split evenly over its objects."""
    signal_ids = [s for s in counts_by_sample if s != "background"]
    targets = {"background": 1.0}
    for s in signal_ids:
        targets[s] = 1.0 / len(signal_ids)
    lookup = {}
    for s, counts in counts_by_sample.items():
        counts = np.asarray(counts, dtype=np.int64)
        nonempty = counts > 0
        n = int(nonempty.sum())
        w = np.zeros(N_BINS, dtype=np.float64)
        if n > 0:
            per_bin = targets[s] / n
            w[nonempty] = per_bin / counts[nonempty]
        lookup[s] = w
    return lookup


def build(input_dir, output_path, pt_min, background_max_objects, seed, workers):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metas = scan_parallel(input_dir, workers)
    if not metas:
        raise RuntimeError("no readable files")

    bg_masks, bg_selected = {}, {}
    for det in DETECTORS:
        bg_masks[det], bg_selected[det], n_elig = select_background_masks(
            metas, det, pt_min, background_max_objects, seed)
        print(f"[bg-select] {det}: eligible(SC_ET>= {pt_min})={n_elig} selected={bg_selected[det]}", flush=True)

    # ---- count pass: per det, per sample, per pT(SC ET) bin ----
    counts = {det: {} for det in DETECTORS}
    for m in metas:
        for det in DETECTORS:
            pt, ei, mask = selected_mask(m, det, pt_min, bg_masks)
            if not np.any(mask):
                continue
            obj_bin = find_bin_indices(pt[mask].astype(np.float64))
            c = counts[det].setdefault(m["sample_id"], np.zeros(N_BINS, dtype=np.int64))
            c += np.bincount(obj_bin, minlength=N_BINS)
    lookups = {det: make_lookup(counts[det]) for det in DETECTORS}
    for det in DETECTORS:
        for s in sorted(counts[det]):
            tot = int(counts[det][s].sum())
            nb = int((counts[det][s] > 0).sum())
            print(f"[count] {det} {s:12s} objects={tot:9d} nonempty_bins={nb}", flush=True)

    # ---- write pass ----
    with h5py.File(output_path, "w") as out:
        out.attrs["input_dir"] = str(input_dir)
        out.attrs["hasadd_mode"] = "all"
        out.attrs["pt_variable"] = "reco_sc_energyT_sum"     # NOT gen A_pT
        out.attrs["pt_min"] = float(pt_min)
        out.attrs["weight_basis"] = "object"
        out.attrs["background_sampling"] = "per_detector_object_sc_et"
        out.attrs["background_max_objects"] = -1 if background_max_objects is None else int(background_max_objects)
        out.attrs["background_selected_objects_eb"] = int(bg_selected["eb"])
        out.attrs["background_selected_objects_ee"] = int(bg_selected["ee"])
        out.attrs["seed"] = int(seed)
        out.attrs["signal_dir"] = SIGNAL_DIR
        out.attrs["requested_pt_bin_edges"] = PT_BIN_EDGES
        out.attrs["effective_pt_bin_edges"] = PT_BIN_EDGES_WITH_OVERFLOW
        files_group = out.create_group("files")
        for m in metas:
            grp = files_group.require_group(m["split"]).create_group(m["stem"])
            grp.attrs["input_file"] = m["path"]
            grp.attrs["split"] = m["split"]
            grp.attrs["process_name"] = m["split"]
            grp.attrs["sample_id"] = m["sample_id"]
            grp.attrs["n_events"] = m["n_events"]
            # per-event pt placeholder for build_object_manifest = per-event max SC ET
            # over all (EB+EE) objects in the event; the gun compact overrides pt with
            # the true per-object SC ET (EB_obj_pt/EE_obj_pt below).
            event_pt = np.zeros(m["n_events"], dtype=np.float64)
            for det in DETECTORS:
                ei, pt = m[det]
                if ei.size:
                    np.maximum.at(event_pt, ei, pt.astype(np.float64))
            grp.create_dataset("event_pt", data=event_pt, compression="gzip")
            for det in DETECTORS:
                pt, ei, mask = selected_mask(m, det, pt_min, bg_masks)
                w = np.zeros(pt.shape, dtype=np.float64)
                if np.any(mask):
                    obj_bin = find_bin_indices(pt[mask].astype(np.float64))
                    w[mask] = lookups[det][m["sample_id"]][obj_bin]
                grp.create_dataset(WEIGHT_KEYS[det], data=w, compression="gzip")
                grp.create_dataset(DET_EVENT_IDX_KEYS[det], data=ei.astype(np.int64), compression="gzip")
                grp.create_dataset(OBJ_PT_KEYS[det], data=pt.astype(np.float32), compression="gzip")
    print(f"[saved] {output_path}", flush=True)
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Gun-heavy per-object SC-ET category weights (signal=particle_gun_heavy, bkg=W,Z).")
    p.add_argument("--input-dir", type=Path, default=Path("/eos/user/y/yeo/4l/data/MiniAOD"))
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--pt-min", type=float, required=True, help="per-object SC ET cut (GeV)")
    p.add_argument("--background-max-objects", type=int, default=1_000_000, help="-1 = keep all")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main():
    a = parse_args()
    bmax = None if a.background_max_objects is not None and a.background_max_objects < 0 else a.background_max_objects
    build(a.input_dir, a.output_path, a.pt_min, bmax, a.seed, a.workers)


if __name__ == "__main__":
    main()
