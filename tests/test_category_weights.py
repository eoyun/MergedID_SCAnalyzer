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
