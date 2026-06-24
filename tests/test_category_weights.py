import glob

import h5py
import numpy as np

import build_category_weights as bcw


def test_hasadd_object_mask_modes():
    v = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    assert bcw.hasadd_object_mask(v, "all").tolist() == [True, True, True, True, True]
    assert bcw.hasadd_object_mask(v, "eq0").tolist() == [True, False, True, False, True]
    assert bcw.hasadd_object_mask(v, "eq1").tolist() == [False, True, False, True, False]


def test_is_excluded_sample():
    # eq1 excludes A0p4 and A1 (any H); keeps A2/A5/A10 and background
    assert bcw.is_excluded_sample("signal_H250_A0p4", "eq1") is True
    assert bcw.is_excluded_sample("signal_H2000_A1", "eq1") is True
    assert bcw.is_excluded_sample("signal_H250_A10", "eq1") is False
    assert bcw.is_excluded_sample("signal_H750_A2", "eq1") is False
    assert bcw.is_excluded_sample("background", "eq1") is False
    # eq0 excludes only H250_A10
    assert bcw.is_excluded_sample("signal_H250_A10", "eq0") is True
    assert bcw.is_excluded_sample("signal_H250_A1", "eq0") is False
    assert bcw.is_excluded_sample("signal_H2000_A10", "eq0") is False
    # all excludes nothing
    assert bcw.is_excluded_sample("signal_H250_A10", "all") is False
    assert bcw.is_excluded_sample("signal_H250_A0p4", "all") is False


def test_object_weight_lookup_balances_bins_and_classes():
    # 2 bins. background present in both bins; one signal sample in 1 bin.
    sample_bin_counts = {
        "background": np.array([4, 0, 0] + [0] * (bcw.N_BINS - 3), dtype=np.int64),
        "signal_H250_A2": np.array([0, 2, 0] + [0] * (bcw.N_BINS - 3), dtype=np.int64),
        "signal_H250_A5": np.array([0, 0, 5] + [0] * (bcw.N_BINS - 3), dtype=np.int64),
    }
    targets = {"background": 1.0, "signal_H250_A2": 0.5, "signal_H250_A5": 0.5}
    lookup = bcw.make_object_weight_lookup(sample_bin_counts, targets)

    # background: 1 non-empty bin, total 1.0 -> bin sum 1.0 over 4 objects = 0.25 each
    assert np.isclose(lookup["background"][0], 0.25)
    # per-class object-weight * count sums to the target
    bg_sum = (lookup["background"] * sample_bin_counts["background"]).sum()
    sig_sum = sum(
        (lookup[s] * sample_bin_counts[s]).sum() for s in ("signal_H250_A2", "signal_H250_A5")
    )
    assert np.isclose(bg_sum, 1.0)
    assert np.isclose(sig_sum, 1.0)


def test_object_weight_lookup_empty_sample_is_all_zero():
    counts = {"background": np.zeros(bcw.N_BINS, dtype=np.int64)}
    lookup = bcw.make_object_weight_lookup(counts, {"background": 1.0})
    assert np.all(lookup["background"] == 0.0)


def test_fixture_files_present(tier_dir):
    files = glob.glob(str(tier_dir / "*" / "*.h5"))
    assert len(files) == 5  # W, Z, 3 signal
    with h5py.File(str(tier_dir / "signal" / "signal_0.h5"), "r") as f:
        assert f["hasAdditionalTrk_EB"].shape == (4,)
        assert "source_root_files" in f.attrs


def _class_sums(out_h5, weight_key):
    """Sum weight_key over all objects, split into background vs signal."""
    bg = sig = 0.0
    seen_samples = set()
    with h5py.File(out_h5, "r") as f:
        for split in f["files"]:
            for stem in f["files"][split]:
                g = f["files"][split][stem]
                w = g[weight_key][:]
                sid = str(g.attrs["sample_id"])
                if sid != "background" and float(w.sum()) > 0:
                    seen_samples.add(sid)
                if sid == "background":
                    bg += float(w.sum())
                else:
                    sig += float(w.sum())
    return bg, sig, seen_samples


def test_build_all_mode_balances_each_class(tier_dir, tmp_path):
    out = tmp_path / "all" / "category_weights.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=out,
                               hasadd_mode="all", background_max_events=None, seed=1)
    for det, wk in bcw.WEIGHT_KEYS.items():
        bg, sig, samples = _class_sums(out, wk)
        assert np.isclose(bg, 1.0), (det, bg)
        assert np.isclose(sig, 1.0), (det, sig)
        assert samples == {"signal_H250_A2", "signal_H250_A10", "signal_H750_A0p4"}


def test_build_eq1_excludes_a0p4_and_a1(tier_dir, tmp_path):
    out = tmp_path / "eq1" / "category_weights.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=out,
                               hasadd_mode="eq1", background_max_events=None, seed=1)
    bg, sig, samples = _class_sums(out, "EB_weight_split")
    assert "signal_H750_A0p4" not in samples
    assert np.isclose(sig, 1.0)
    assert np.isclose(bg, 1.0)


def test_build_eq1_only_keeps_hasadd1_objects(tier_dir, tmp_path):
    out = tmp_path / "eq1b" / "category_weights.h5"
    bcw.build_category_weights(input_dir=tier_dir, output_path=out,
                               hasadd_mode="eq1", background_max_events=None, seed=1)
    # In the fixture, signal objects 0,1 are hasADD==1 and 2,3 are ==0.
    with h5py.File(out, "r") as f:
        g = f["files"]["signal"]["signal_0"]   # H250_A2, kept by eq1
        w = g["EB_weight_split"][:]
        assert w[0] > 0 and w[1] > 0
        assert w[2] == 0 and w[3] == 0
