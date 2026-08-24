import numpy as np
import torch
import torch.nn as nn

import multitask_common as mc


def test_sample_id_to_mass():
    assert mc.sample_id_to_mass("signal_H2000_A0p4") == 0.4
    assert mc.sample_id_to_mass("signal_H250_A1") == 1.0
    assert mc.sample_id_to_mass("signal_H750_A10") == 10.0
    assert mc.sample_id_to_mass("signal_H2000_A5") == 5.0
    assert mc.sample_id_to_mass("background") is None
    assert mc.sample_id_to_mass(b"signal_H250_A2") == 2.0


def test_mass_transform_roundtrip():
    for m in (0.4, 1.0, 2.0, 5.0, 10.0):
        assert abs(mc.inv_transform(mc.transform_mass(m)) - m) < 1e-6


def test_mass_targets_from_codes():
    code_to_sample = {0: "background", 1: "signal_H2000_A0p4", 2: "signal_H250_A10"}
    logm, mask = mc.mass_targets_from_codes(np.array([0, 1, 2, 1]), code_to_sample)
    assert mask.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert abs(logm[1] - np.log(0.4)) < 1e-6
    assert abs(logm[2] - np.log(10.0)) < 1e-6
    assert logm[0] == 0.0


def test_multitask_loss_masks_background():
    cls = torch.zeros(4)
    reg = torch.tensor([9.0, 0.0, 0.0, 0.0])
    label = torch.tensor([1., 0., 0., 0.])
    logm = torch.tensor([0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([1., 0., 0., 0.])
    total, bce, mae = mc.multitask_loss(cls, reg, label, logm, mask, lam=1.0)
    assert abs(mae.item() - 9.0) < 1e-5
    _, _, mae0 = mc.multitask_loss(cls, reg, label, logm, torch.zeros(4), lam=1.0)
    assert mae0.item() == 0.0


def test_regression_metrics():
    m_true = np.array([1.0, 2.0, 5.0, 10.0])
    m_pred = np.array([1.1, 1.8, 5.5, 9.0])
    d = mc.regression_metrics(m_true, m_pred)
    assert abs(d["mae"] - np.mean([0.1, 0.2, 0.5, 1.0])) < 1e-6
    assert abs(d["mre"] - np.mean([0.1, 0.1, 0.1, 0.1])) < 1e-6


def test_regression_metrics_per_point():
    m_true = np.array([1.0, 1.0, 10.0])
    m_pred = np.array([1.2, 0.8, 9.0])
    per = mc.regression_metrics_per_point(m_true, m_pred)
    assert set(per.keys()) == {1.0, 10.0}
    assert abs(per[1.0]["resolution"] - np.std([0.2, -0.2])) < 1e-6


def test_multitask_wrapper_shapes():
    class DummyBackbone(nn.Module):
        def __init__(self, feat):
            super().__init__()
            self.lin = nn.Linear(8, feat)

        def forward(self, x):
            return self.lin(x)
    model = mc.MultiTaskModel(DummyBackbone(16), feat_dim=16)
    x = torch.randn(5, 8)
    cls, reg = model(x)
    assert cls.shape == (5,) and reg.shape == (5,)
    assert torch.isfinite(cls).all() and torch.isfinite(reg).all()


def test_save_mass_plots(tmp_path):
    rng = np.random.default_rng(0)
    m_true = np.repeat([0.4, 1.0, 2.0, 5.0, 10.0], 50)
    m_pred = m_true * rng.normal(1.0, 0.2, size=m_true.size)
    mc.save_mass_regression_plots(str(tmp_path), m_true, m_pred, suffix="")
    for f in ("mass_pred_vs_true.png", "mass_resolution_vs_mass.png",
              "mass_pred_hist_per_point.png"):
        assert (tmp_path / f).exists()


def test_image_multitask_forward_and_loss():
    from train_fusion_cross_attention import _RESNET_FEAT_CHANNELS
    from train_resnet_image_classifier import build_resnet
    backbone = build_resnet("resnet18", in_channels=1)
    backbone.fc = nn.Identity()
    model = mc.MultiTaskModel(backbone, feat_dim=_RESNET_FEAT_CHANNELS["resnet18"])
    x = torch.randn(4, 1, 512, 512)
    cls, reg = model(x)
    assert cls.shape == (4,) and reg.shape == (4,)
    code_to_sample = {0: "background", 1: "signal_H250_A5"}
    logm, mask = mc.mass_targets_from_codes(np.array([1, 0, 1, 0]), code_to_sample)
    total, bce, mae = mc.multitask_loss(cls, reg, torch.tensor([1., 0, 1, 0]),
                                        torch.tensor(logm), torch.tensor(mask))
    assert torch.isfinite(total)
