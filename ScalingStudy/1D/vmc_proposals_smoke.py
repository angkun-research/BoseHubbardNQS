"""Verify fixed-sector balance and the injected sampler on a Slurm compute node."""

from functools import partial
from itertools import product
import json

from ed_fidelity import require_compute_node
from vmc_train import NONLOCAL_TRANSFER_FRACTION


def main():
    require_compute_node()

    import torch
    from ed_fit import one_hot
    from vmc_proposals import distinct_destination, mixed_proposal, transfer
    from vmc_utils_gpu import Obtain_Sampling_batch_gpu

    torch.set_num_threads(1)
    torch.manual_seed(43107)
    length, particles, cutoff = 4, 4, 2
    occupations = [row for row in product(range(cutoff + 1), repeat=length) if sum(row) == particles]
    states = torch.tensor(occupations)
    count = len(states)
    lookup = {row: index for index, row in enumerate(occupations)}
    state_index = torch.tensor([
        lookup.get(row, -1) for row in product(range(cutoff + 1), repeat=length)
    ])
    radix = (cutoff + 1) ** torch.arange(length - 1, -1, -1)
    fraction = NONLOCAL_TRANSFER_FRACTION

    def indices(walkers):
        assert (walkers.sum(dim=-1) == particles).all()
        assert ((0 <= walkers) & (walkers <= cutoff)).all()
        return state_index[(walkers * radix).sum(dim=-1)]

    expected = torch.zeros((count, count), dtype=torch.float64)
    invalid_pairs = 0
    for source in range(length):
        destinations = distinct_destination(source, torch.arange(length - 1), length)
        assert set(destinations.tolist()) == set(range(length)) - {source}
        for dest in destinations.tolist():
            pair = [torch.full((count,), site) for site in (source, dest)]
            proposed = transfer(states, *pair, cutoff)
            valid = (states[:, source] > 0) & (states[:, dest] < cutoff)
            assert torch.equal(proposed[~valid], states[~valid])
            assert (proposed[valid, source] == states[valid, source] - 1).all()
            assert (proposed[valid, dest] == states[valid, dest] + 1).all()
            invalid_pairs += int((~valid).sum())
            neighbor = (dest - source) % length in (1, length - 1)
            weight = fraction / (length * (length - 1)) + neighbor * (1 - fraction) / (2 * length)
            expected[torch.arange(count), indices(proposed)] += weight
    assert invalid_pairs and (expected.diag() > 0).any()
    torch.testing.assert_close(expected.sum(dim=1), torch.ones(count, dtype=expected.dtype), rtol=0, atol=1e-14)
    torch.testing.assert_close(expected, expected.T, rtol=0, atol=1e-14)

    class ToyAmplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            labels = torch.arange(cutoff + 1, dtype=torch.float64)
            self.log_factor = torch.nn.Parameter(-0.22 * labels * (labels - 1), requires_grad=False)

        def forward(self, inputs):
            return (inputs @ self.log_factor).sum(dim=1)

    model = ToyAmplitude()
    target = (2 * model(one_hot(states, cutoff + 1).double())).softmax(dim=0)
    transition = expected * (target[None, :] / target[:, None]).clamp(max=1)
    transition.diagonal().add_(1 - transition.sum(dim=1))
    flow = target[:, None] * transition
    torch.testing.assert_close(flow, flow.T, rtol=0, atol=1e-14)
    torch.testing.assert_close(target @ transition, target, rtol=0, atol=1e-14)

    proposal = partial(mixed_proposal, fraction=fraction)
    draws = 4096
    proposed = proposal(states.repeat_interleave(draws, dim=0), length, cutoff)
    actual = torch.bincount(
        torch.arange(count).repeat_interleave(draws) * count + indices(proposed),
        minlength=count * count,
    ).reshape(count, count) / draws
    difference = (actual - expected).abs()
    deviations = (expected * (1 - expected) / draws).sqrt()
    assert (difference <= 7 * deviations + 0.002).all()

    starts = states[torch.multinomial(target, 8192, replacement=True)]
    _, _, final, _ = Obtain_Sampling_batch_gpu(
        model, length, cutoff, starts, 12, burnin=True, Sampler=proposal, evaluation_batch_size=512,
    )
    final_probability = torch.bincount(indices(final), minlength=count) / len(final)
    total_variation = 0.5 * (final_probability - target).abs().sum().item()
    assert total_variation < 0.05
    print(json.dumps({
        "status": "pass", "sector_states": count, "nonlocal_fraction": fraction,
        "invalid_ordered_pairs_verified": invalid_pairs,
        "max_proposal_asymmetry": (expected - expected.T).abs().max().item(),
        "max_detailed_balance_residual": (flow - flow.T).abs().max().item(),
        "max_sampled_proposal_frequency_error": difference.max().item(),
        "legacy_sampler_target_total_variation": total_variation,
    }), flush=True)


if __name__ == "__main__":
    main()
