import numpy as np

import build_compact_dataset as bcd


def test_calo_key_and_channel_source_keys():
    assert bcd.calo_key("eb") == "SC_energy"
    assert bcd.calo_key("ee") == "EE_seed_energy"
    eb = bcd.raw_source_keys("eb", ("Lost", "PF", "GSF"))
    assert "SC_energy" in eb["calo"]
    assert eb["track"] == ["EB_track_pt_Lost", "EB_track_pt_PF", "EB_track_pt_GSF"]
    assert eb["es"] == []
    ee = bcd.raw_source_keys("ee", ("GenTrk",))
    assert ee["calo"] == ["EE_seed_energy"]
    assert ee["es"] == ["ES_seed_plane1_energy", "ES_seed_plane2_energy"]
    assert ee["track"] == ["EE_track_pt_GenTrk"]


import build_category_weights as bcw


def test_combined_manifest_has_all_splits(tier_dir, tmp_path):
    wpath = tmp_path / "w.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=wpath,
                               hasadd_mode="all", background_max_events=None, seed=1)
    m = bcd.build_combined_manifest(weight_h5_path=wpath, detector="eb",
                                    weight_key="EB_weight_split",
                                    train_frac=0.6, val_frac=0.2, seed=1)
    assert set(np.unique(m["split"])).issubset({0, 1, 2})
    for k in ("label", "weight", "pt", "file_idx", "object_idx", "event_idx",
              "sample_code", "split"):
        assert len(m[k]) == len(m["label"])
    assert (m["weight"] > 0).all()


import h5py


def test_build_compact_roundtrip(tier_dir, tmp_path):
    wpath = tmp_path / "w.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=wpath,
                               hasadd_mode="all", background_max_events=None, seed=1)
    out = tmp_path / "compact_eb.h5"
    n = bcd.build_compact_dataset(weight_h5_path=wpath, detector="eb",
                                  track_types=("Lost", "PF", "GSF"), output_path=out,
                                  train_frac=0.6, val_frac=0.2, seed=1)
    with h5py.File(out, "r") as f:
        assert f["SC_energy"].shape[0] == n and f["SC_energy"].ndim == 3
        for t in ("Lost", "PF", "GSF"):
            assert f[f"EB_track_pt_{t}_idx"].shape == (n,)
        assert len(f["label"]) == n and len(f["split"]) == n
        assert f.attrs["detector"] == "eb"
        assert f.attrs["track_types"] == "Lost,PF,GSF"
        raw_file = f["raw_file"][0]; oidx = int(f["object_idx"][0])
        with h5py.File(raw_file, "r") as r:
            assert np.array_equal(f["SC_energy"][0], r["SC_energy"][oidx])
