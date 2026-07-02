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
