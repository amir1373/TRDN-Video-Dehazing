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


def test_warm_start_leaves_a_new_skipped_submodule_at_its_init(tmp_path):
    from safetensors.torch import save_file

    from src.train import warm_start_from_checkpoint

    plain = TRDNConfig()
    memory, transformer, selector, adapter = build_temporal_modules(plain, 768, "cpu")
    for stem, module in (("model_1", memory), ("model_2", transformer), ("model_3", selector), ("model_4", adapter)):
        save_file({k: v.contiguous() for k, v in module.state_dict().items()}, str(tmp_path / f"{stem}.safetensors"))
    config = TRDNConfig(haze_map_conditioning=True, init_weights_from=str(tmp_path),
                        init_skip_modules="conditioning_adapter.haze_estimator")
    m2, t2, s2, a2 = build_temporal_modules(config, 768, "cpu")
    report = warm_start_from_checkpoint(config, {"unet": None, "temporal_memory": m2, "temporal_transformer": t2,
                                                 "reference_selector": s2, "conditioning_adapter": a2})
    assert report["modules"]["conditioning_adapter"]["loaded_tensors"] > 0
    assert float(haze_estimator_of(a2)(torch.rand(1, 3, 32, 32)).min()) > 0.97
