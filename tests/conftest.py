import numpy as np
import h5py
import pytest

SIGNAL_SRC = {
    "H250_A2": "/store/user/yeo/HToAATo4L_H250A2_TuneCP5_13p6TeV-pythia8_v1/signal_x/0/0/o.root",
    "H250_A10": "/store/user/yeo/HToAATo4L_H250A10_TuneCP5_13p6TeV-pythia8_v1/signal_x/0/0/o.root",
    "H750_A0p4": "/store/user/yeo/HToAATo4L_H750A0p4_TuneCP5_13p6TeV-pythia8_v1/signal_x/0/0/o.root",
}
BKG_SRC = "/store/user/yeo/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/W_x/0/0/o.root"


def _write_file(path, n_events, eb_events, ee_events, eb_hasadd, ee_hasadd, source, a_pt):
    """eb_events/ee_events: per-object event index arrays. *_hasadd: per-object 0/1."""
    with h5py.File(path, "w") as f:
        f.attrs["source_root_files"] = repr([source])
        f["eventId"] = np.arange(n_events, dtype=np.int64)
        # one A entry per event with the given pT (so derive_event_pt 'max' works)
        f["A_pT"] = np.asarray(a_pt, dtype=np.float32)
        f["A_event_idx"] = np.arange(n_events, dtype=np.int64)
        f["EB_ele_event_idx"] = np.asarray(eb_events, dtype=np.int64)
        f["EE_ele_event_idx"] = np.asarray(ee_events, dtype=np.int64)
        f["SC_energy"] = np.zeros((len(eb_events), 4, 4), dtype=np.float32)
        f["EE_seed_energy"] = np.zeros((len(ee_events), 4, 4), dtype=np.float32)
        f["hasAdditionalTrk_EB"] = np.asarray(eb_hasadd, dtype=np.float32)
        f["hasAdditionalTrk_EE"] = np.asarray(ee_hasadd, dtype=np.int32)
        # minimal per-object sparse track arrays (vlen), for compact builder tests
        vi = h5py.special_dtype(vlen=np.int64)
        vf = h5py.special_dtype(vlen=np.float32)
        for prefix, nobj in (("EB", len(eb_events)), ("EE", len(ee_events))):
            for t in ("Lost", "PF", "GSF"):
                idx = f.create_dataset(f"{prefix}_track_pt_{t}_idx", (nobj,), dtype=vi)
                val = f.create_dataset(f"{prefix}_track_pt_{t}_val", (nobj,), dtype=vf)
                for i in range(nobj):
                    idx[i] = np.array([(i * 7) % 16], dtype=np.int64)
                    val[i] = np.array([5.0 + i], dtype=np.float32)


@pytest.fixture
def tier_dir(tmp_path):
    """Builds a tiny tier with W, Z and 3 signal mass points.

    Each file: 4 events, pT all = 25 (lands in a single pT bin). EB objects: one
    per event. EE objects: one per event. hasADD chosen so each mode keeps some.
    """
    root = tmp_path / "MiniAOD"
    for sub in ("W", "Z", "signal"):
        (root / sub).mkdir(parents=True)

    pt = [25.0, 25.0, 25.0, 25.0]
    # Background: mostly hasADD==0, one ==1.
    for split in ("W", "Z"):
        _write_file(
            root / split / f"{split}_0.h5",
            n_events=4, eb_events=[0, 1, 2, 3], ee_events=[0, 1, 2, 3],
            eb_hasadd=[0, 0, 0, 1], ee_hasadd=[0, 0, 0, 1],
            source=BKG_SRC, a_pt=pt,
        )
    # Signal mass points: mix of hasADD 0/1.
    for mass, fname in (("H250_A2", "signal_0"), ("H250_A10", "signal_1"), ("H750_A0p4", "signal_2")):
        _write_file(
            root / "signal" / f"{fname}.h5",
            n_events=4, eb_events=[0, 1, 2, 3], ee_events=[0, 1, 2, 3],
            eb_hasadd=[1, 1, 0, 0], ee_hasadd=[1, 1, 0, 0],
            source=SIGNAL_SRC[mass], a_pt=pt,
        )
    return root
