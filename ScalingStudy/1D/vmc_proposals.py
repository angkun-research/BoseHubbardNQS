"""Optional symmetric occupation proposals for the periodic 1D study."""

import torch

from vmc_utils_gpu import GlobalSampler_gpu


def distinct_destination(source, offset, length):
    return (source + 1 + offset) % length


def transfer(states, source, destination, cutoff):
    rows = torch.arange(len(states), device=states.device)
    valid = (states[rows, source] > 0) & (states[rows, destination] < cutoff)
    proposed = states.clone()
    proposed[rows, source] -= valid.long()
    proposed[rows, destination] += valid.long()
    return proposed


def mixed_proposal(states, L, Nmax, pbc=True, *, fraction):
    """Mix legacy bond proposals with uniform ordered pairs of distinct sites.

    Choose the component and the pair independently of occupations. Invalid
    transfers remain self moves, preserving the symmetry of each component.
    """
    if states.ndim != 2 or states.shape[1] != L or L < 2 or not pbc:
        raise ValueError("the mixed proposal requires periodic rings with at least two sites")
    local = GlobalSampler_gpu(states, L=L, Nmax=Nmax, pbc=pbc)
    device = states.device
    source = torch.randint(L, (len(states),), device=device)
    offset = torch.randint(L - 1, (len(states),), device=device)
    nonlocal_move = transfer(states, source, distinct_destination(source, offset, L), Nmax)
    choose_nonlocal = torch.rand(len(states), device=device) < fraction
    return torch.where(choose_nonlocal[:, None], nonlocal_move, local)
