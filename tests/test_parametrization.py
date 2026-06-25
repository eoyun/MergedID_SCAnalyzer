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
