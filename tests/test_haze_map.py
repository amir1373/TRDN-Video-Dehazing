import torch

from src.config import TRDNConfig
from src.haze_map import HazeMapEstimator, pseudo_haze_severity
from src.train import build_optimizer, build_temporal_modules, haze_estimator_of


def test_pseudo_severity_recovers_known_transmission():
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 64, 64) * 0.6            # radiance well away from the airlight
    t = torch.linspace(0.2, 0.9, 64).view(1, 1, 1, 64).expand(1, 1, 64, 64).clone()
    t[..., :6] = 0.02                                  # a dense-haze strip, as the airlight estimate assumes
    airlight = torch.tensor([0.95, 0.95, 0.95]).view(1, 3, 1, 1)
    hazy = clean * t + airlight * (1 - t)
    severity, valid = pseudo_haze_severity(hazy, clean, smooth=1)
    ok = valid.bool()
    assert ok.float().mean() > 0.9
    assert (severity[ok] - (1 - t)[ok]).abs().mean() < 0.05


def test_estimator_starts_near_all_ones_and_has_own_optimizer_group():
    est = HazeMapEstimator()
    out = est(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64) and float(out.min()) > 0.97
    config = TRDNConfig(haze_map_conditioning=True, haze_map_learning_rate=1e-4)
    config.train_unet = False
    memory, transformer, selector, adapter = build_temporal_modules(config, 768, "cpu")
    assert haze_estimator_of(adapter) is not None
    opt = build_optimizer(config, None, memory, transformer, selector, adapter)
    assert [g["lr"] for g in opt.param_groups][-1] == 1e-4
    est_ids = {id(p) for p in haze_estimator_of(adapter).parameters()}
    assert all(id(p) not in est_ids for p in opt.param_groups[0]["params"])
