"""Check deterministic retained-stride sampling on a Slurm CPU node."""

import json
import random
from unittest.mock import patch

from ed_fidelity import require_compute_node


def main():
    require_compute_node()

    import numpy as np
    # Use the evaluator's import order so SciPy loads its C++ runtime first.
    from vmc_eval_fit import sample_observables
    import torch
    from ed_fit import build_model
    from vmc_utils_gpu import GlobalSampler_gpu, Obtain_Sampling_batch_gpu, local_energy_on_the_fly_gpu

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(95107)
    length, cutoff, stride = 6, 6, 8
    model = build_model("FCNet", length, cutoff, 8).eval().requires_grad_(False)
    initial = torch.tensor([[1, 1, 1, 1, 1, 1], [2, 0, 1, 2, 0, 1], [0, 2, 0, 2, 0, 2]])
    proposals = 40

    def sample(**options):
        torch.manual_seed(14701)
        choices = []

        def counted(walkers, **kwargs):
            choices.append(len(walkers))
            return GlobalSampler_gpu(walkers, **kwargs)

        with patch("vmc_utils_gpu.local_energy_on_the_fly_gpu", wraps=local_energy_on_the_fly_gpu) as local:
            result = Obtain_Sampling_batch_gpu(
                model, length, cutoff, initial, proposals, Sampler=counted, evaluation_batch_size=16, **options,
            )
        assert local.call_count == 1
        assert len(local.call_args.args[0]) == len(result[3])
        assert len(choices) == proposals and sum(choices) == proposals * len(initial)
        return result, torch.get_rng_state()

    full, full_rng = sample()
    explicit, explicit_rng = sample(retain_every=1)
    for default, supplied in zip((*full, full_rng), (*explicit, explicit_rng)):
        torch.testing.assert_close(default, supplied, rtol=0, atol=0)
    sparse, sparse_rng = sample(retain_every=stride)
    torch.testing.assert_close(sparse[2], full[2], rtol=0, atol=0)
    torch.testing.assert_close(sparse_rng, full_rng, rtol=0, atol=0)
    index = torch.arange(stride - 1, proposals, stride)[:, None] * len(initial) + torch.arange(len(initial))
    index = index.flatten()
    torch.testing.assert_close(sparse[3], full[3][index], rtol=0, atol=0)
    for observed, expected in zip(sparse[:2], full[:2]):
        torch.testing.assert_close(observed, expected[index], rtol=2e-6, atol=2e-6)
    assert (sparse[3].sum(dim=1) == length).all()
    assert ((sparse[3] >= 0) & (sparse[3] <= cutoff)).all()

    def evaluate(**controls):
        random.seed(51323)
        torch.manual_seed(51323)
        config = dict(length=length, nmax=cutoff, chains=5, burn_in=9, steps=proposals, batch_steps=11,
                      chunk_size=16, distances=[1, 3], number_blocks=[2])
        return sample_observables(model, dict(config, **controls), 3.275)

    full_series, full_moves = evaluate()
    default_series, default_moves = evaluate(retain_every=1)
    sparse_series, _ = evaluate(steps=proposals // stride, batch_steps=3, retain_every=stride)
    for name, series in full_series.items():
        np.testing.assert_array_equal(series, default_series[name])
        np.testing.assert_allclose(series[stride - 1::stride], sparse_series[name], rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(full_moves, default_moves)
    try:
        Obtain_Sampling_batch_gpu(model, length, cutoff, initial, proposals - 1, retain_every=stride)
    except ValueError:
        pass
    else:
        raise AssertionError("a partial retained interval was accepted")
    print(json.dumps({
        "status": "pass", "stride": stride, "proposals_per_chain": proposals,
        "default_one_bitwise_equal": True, "same_final_walkers_and_rng": True,
        "retained_states_per_chain": proposals // stride, "fixed_sector_and_cutoff": True,
        "energy_and_g1_subsequence": True, "local_energy_evaluated_on_retained_states_only": True,
    }), flush=True)


if __name__ == "__main__":
    main()
