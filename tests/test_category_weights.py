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
