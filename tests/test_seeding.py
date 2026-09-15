import torch

from src.seeding import derive_generator


def test_derive_generator_is_deterministic():
    gen_a = derive_generator(1234, "clip_001", 5)
    gen_b = derive_generator(1234, "clip_001", 5)
    sample_a = torch.randn(4, 4, generator=gen_a)
    sample_b = torch.randn(4, 4, generator=gen_b)
    assert torch.equal(sample_a, sample_b)


def test_derive_generator_differs_by_parts():
    gen_a = derive_generator(1234, "clip_001", 5)
    gen_b = derive_generator(1234, "clip_001", 6)  # different frame index
    gen_c = derive_generator(1234, "clip_002", 5)  # different clip
    sample_a = torch.randn(4, 4, generator=gen_a)
    sample_b = torch.randn(4, 4, generator=gen_b)
    sample_c = torch.randn(4, 4, generator=gen_c)
    assert not torch.equal(sample_a, sample_b)
    assert not torch.equal(sample_a, sample_c)


def test_derive_generator_differs_by_seed():
    gen_a = derive_generator(1234, "clip_001", 5)
    gen_b = derive_generator(9999, "clip_001", 5)
    sample_a = torch.randn(4, 4, generator=gen_a)
    sample_b = torch.randn(4, 4, generator=gen_b)
    assert not torch.equal(sample_a, sample_b)


def test_eval_script_would_be_reproducible_run_to_run():
    """Simulates the '#4 verification' requirement at the noise-generation level:
    two independent 'runs' (fresh generator derivations) with the same seed and
    sample ids must produce identical noise tensors -- the only randomness
    source in seeded DDIM (eta=0) inference."""
    def run():
        seed = 42
        sample_ids = ["clip_007", 3]
        gen = derive_generator(seed, *sample_ids)
        return torch.randn(1, 4, 8, 8, generator=gen)

    first_run = run()
    second_run = run()
    assert torch.equal(first_run, second_run)
