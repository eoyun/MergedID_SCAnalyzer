import numpy as np

import track_point_transformer as tpt


def test_point_feature_dim():
    assert tpt.point_feature_dim(("GSF", "PF", "Lost")) == 6
    assert tpt.point_feature_dim(("GenTrk",)) == 4


def test_sparse_points_onehot_width_matches_track_types():
    idx = np.array([0, 600], dtype=np.int64)
    val = np.array([10.0, 20.0], dtype=np.float32)
    # 1 track type -> dim 4 (x,y,pt,onehot[0])
    pts1, _ = tpt.sparse_track_arrays_to_points(idx, val, track_type_id=0, n_track_types=1)
    assert pts1.shape == (2, 4)
    assert np.allclose(pts1[:, 3], 1.0)
    # 3 track types, id=2 -> dim 6, onehot in column 5
    pts3, _ = tpt.sparse_track_arrays_to_points(idx, val, track_type_id=2, n_track_types=3)
    assert pts3.shape == (2, 6)
    assert np.allclose(pts3[:, 3 + 2], 1.0)
    assert np.all(pts3[:, 3] == 0)


def test_parse_track_types():
    assert tpt.parse_track_types("Lost,PF,GSF") == ("Lost", "PF", "GSF")
    assert tpt.parse_track_types("GenTrk") == ("GenTrk",)


def test_track_csv_columns_for_track_types():
    hdr1 = tpt.track_csv_columns(("GenTrk",))
    assert "n_gentrk" in hdr1 and "sum_pt_gentrk" in hdr1
    assert "n_pf" not in hdr1
    hdr3 = tpt.track_csv_columns(("GSF", "PF", "Lost"))
    assert hdr3 == ["n_gsf", "n_pf", "n_lost", "sum_pt_gsf", "sum_pt_pf", "sum_pt_lost"]


import train_resnet_image_classifier as tri


def test_image_channel_keys():
    eb = tri.image_channel_keys("eb", ("Lost", "PF", "GSF"), include_es=False)
    assert [k for k, _ in eb] == ["calo", "track", "track", "track"]
    assert len(eb) == 4
    eb_aod = tri.image_channel_keys("eb", ("GenTrk",), include_es=False)
    assert len(eb_aod) == 2
    ee = tri.image_channel_keys("ee", ("Lost", "PF", "GSF"), include_es=True)
    assert len(ee) == 6
    ee_noes = tri.image_channel_keys("ee", ("GenTrk",), include_es=False)
    assert len(ee_noes) == 2


def test_image_in_channels():
    assert tri.image_in_channels("eb", ("Lost", "PF", "GSF"), True) == 4
    assert tri.image_in_channels("ee", ("Lost", "PF", "GSF"), True) == 6
    assert tri.image_in_channels("ee", ("GenTrk",), False) == 2


import train_fusion_ensemble as tfe


def test_fusion_feature_names_dynamic():
    names1 = tfe.fusion_feature_names(("GenTrk",))
    assert names1 == ["image_logit", "track_logit", "log1p_n_gentrk", "log1p_sum_pt_gentrk"]
    names3 = tfe.fusion_feature_names(("GSF", "PF", "Lost"))
    assert names3 == [
        "image_logit", "track_logit",
        "log1p_n_gsf", "log1p_n_pf", "log1p_n_lost",
        "log1p_sum_pt_gsf", "log1p_sum_pt_pf", "log1p_sum_pt_lost",
    ]


def test_build_fusion_features_dynamic_shape():
    img = {"score": np.array([0.2, 0.8])}
    trk = {"score": np.array([0.5, 0.5]),
           "track_counts": np.array([[1.0], [2.0]]),
           "track_sum_pt": np.array([[10.0], [20.0]])}
    feats, names = tfe.build_fusion_features(img, trk, ("GenTrk",))
    assert feats.shape == (2, 4)
    assert names == ["image_logit", "track_logit", "log1p_n_gentrk", "log1p_sum_pt_gentrk"]
