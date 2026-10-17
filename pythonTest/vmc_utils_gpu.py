"""Tensor-resident Bose-Hubbard VMC; models return real log amplitudes.

The model determines device/dtype (the compatibility ``device`` arguments are
ignored). Walkers are long tensors, samples are flattened in step/walker order.
Models must be deterministic while sampling, e.g. disable dropout beforehand.
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap

from vmc_utils import _bonds_1d, generate_initial_state


def states_to_occupancy_tensor(states, L, device):
    states = torch.as_tensor(states, dtype=torch.long, device=device)
    if states.ndim != 2 or states.shape[1] != L:
        raise ValueError(f"states must have shape (batch, {L})")
    return states


def occupancy_tensor_to_onehot(states, Nmax, dtype=torch.float32):
    return F.one_hot(states.long(), num_classes=Nmax + 1).to(dtype=dtype)


def _chunk_size(size, evaluation_batch_size):
    if evaluation_batch_size is not None and evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    return max(size, 1) if evaluation_batch_size is None else evaluation_batch_size


def _log_amplitudes(psi, states, Nmax, evaluation_batch_size=None):
    parameter = next(psi.parameters())
    size = _chunk_size(len(states), evaluation_batch_size)
    if not len(states):
        return parameter.new_empty(0)
    return torch.cat([
        psi(occupancy_tensor_to_onehot(block, Nmax, parameter.dtype)).reshape(-1)
        for block in states.split(size)
    ])


def GlobalSampler_gpu(states, L=None, Nmax=None, pbc=True):
    """Uniform bond/direction proposals, including invalid moves as self moves.

    Choosing only valid hops would break proposal symmetry. Keep duplicate L=2
    periodic bonds exactly as in the scalar sampler.
    """
    if states.ndim != 2 or (L is not None and L != states.shape[1]):
        raise ValueError("states must have shape (n_walkers, L)")
    if Nmax is None or Nmax < 0:
        raise ValueError("Nmax must be nonnegative")
    L = states.shape[1]
    n_bonds = L if pbc and L > 1 else L - 1
    if n_bonds <= 0:
        return states.clone()
    a = torch.randint(n_bonds, (len(states),), device=states.device)
    b = (a + 1) % L
    reverse = torch.randint(2, (len(states),), device=states.device).bool()
    src = torch.where(reverse, b, a)
    dst = torch.where(reverse, a, b)
    rows = torch.arange(len(states), device=states.device)
    valid = (states[rows, src] > 0) & (states[rows, dst] < Nmax)
    proposed = states.clone()
    proposed[rows, src] -= valid.long()
    proposed[rows, dst] += valid.long()
    return proposed


@torch.no_grad()
def local_energy_on_the_fly_gpu(
    states, psi, L, Nmax, t, U, mu=0.0, pbc=True, device=None,
    evaluation_batch_size=None, logpsi=None,
):
    """Detached batched H psi / psi using exp(logpsi_connected - logpsi).

    Chunk both input states and connected-state network evaluations. Only valid
    hops are evaluated. Optional cached logpsi must match these states/parameters.
    One state (L,) returns a scalar; a batch returns (batch,).
    """
    parameter = next(psi.parameters())
    states = torch.as_tensor(states, device=parameter.device, dtype=torch.long)
    single = states.ndim == 1
    if single:
        states = states.unsqueeze(0)
    states = states_to_occupancy_tensor(states, L, parameter.device)
    size = _chunk_size(len(states), evaluation_batch_size)
    cached = None if logpsi is None else torch.as_tensor(
        logpsi, device=parameter.device, dtype=parameter.dtype
    ).detach().reshape(-1)
    if cached is not None and cached.numel() != len(states):
        raise ValueError("logpsi must have one value per state")
    bonds = _bonds_1d(L, pbc)
    if bonds:
        undirected = torch.tensor(bonds, device=parameter.device, dtype=torch.long)
        src, dst = torch.cat((undirected, undirected.flip(1))).unbind(1)
    energies = []
    for start in range(0, len(states), size):
        block = states[start:start + size]
        occ = block.to(parameter.dtype)
        energy = (U * occ * (occ - 1) - mu * occ).sum(1)
        if bonds and t != 0:
            base = (_log_amplitudes(psi, block, Nmax, evaluation_batch_size)
                    if cached is None else cached[start:start + size])
            valid = (block[:, src] > 0) & (block[:, dst] < Nmax)
            rows, moves = valid.nonzero(as_tuple=True)
            source, target = src[moves], dst[moves]
            connected = block[rows].clone()
            indices = torch.arange(len(rows), device=parameter.device)
            connected[indices, source] -= 1
            connected[indices, target] += 1
            coefficients = -t * torch.sqrt(
                block[rows, source].to(parameter.dtype)
                * (block[rows, target].to(parameter.dtype) + 1)
            )
            other = _log_amplitudes(psi, connected, Nmax, evaluation_batch_size)
            energy.index_add_(0, rows, coefficients * torch.exp(other - base[rows]))
        energies.append(energy)
    result = torch.cat(energies) if energies else parameter.new_empty(0)
    return result[0] if single else result


def Obtain_Sampling_batch_gpu(
    psi, L, Nmax, initial_states, n_steps, burnin=False, print_rate=False,
    Sampler=GlobalSampler_gpu, t=1.0, U=2.0, mu=0.0, pbc=True, device=None,
    evaluation_batch_size=None,
):
    """Return (logpsis with gradients, detached energies, walkers, samples).

    Cache current log amplitudes for Metropolis updates, without sampling graphs.
    Burn-in and zero steps return empty values/samples and the final walkers.
    evaluation_batch_size bounds network batches, not the retained sample storage
    or total final autograd graph. The model is never moved by this function.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    _chunk_size(0, evaluation_batch_size)
    parameter = next(psi.parameters())
    states = states_to_occupancy_tensor(initial_states, L, parameter.device).clone()
    if not len(states):
        raise ValueError("at least one walker is required")
    samples = []
    with torch.no_grad():
        accepted = torch.zeros((), device=parameter.device, dtype=torch.long)
        current = _log_amplitudes(psi, states, Nmax, evaluation_batch_size)
        for _ in range(n_steps):
            proposed = Sampler(states, L=L, Nmax=Nmax, pbc=pbc)
            new = _log_amplitudes(psi, proposed, Nmax, evaluation_batch_size)
            # Log-space comparison avoids overflow at large log amplitudes.
            accept = torch.rand(len(states), device=parameter.device).log() < (
                2 * (new - current)
            ).clamp(max=0)
            states = torch.where(accept[:, None], proposed, states)
            current = torch.where(accept, new, current)
            accepted += accept.sum()
            if not burnin:
                samples.append(states.clone())
        if print_rate and n_steps:
            print(f"Acceptance rate (batched GPU): {accepted.item() / (n_steps * len(states)):.4f}")
    if not samples:
        return (parameter.new_empty(0), parameter.new_empty(0), states,
                states.new_empty((0, L)))
    sampled_states = torch.stack(samples).reshape(-1, L)
    with torch.enable_grad():
        logpsis = _log_amplitudes(psi, sampled_states, Nmax, evaluation_batch_size)
    energies = local_energy_on_the_fly_gpu(
        sampled_states, psi, L, Nmax, t, U, mu, pbc,
        evaluation_batch_size=evaluation_batch_size, logpsi=logpsis.detach(),
    )
    return logpsis, energies, states, sampled_states


def compute_log_psi_jacobian_vmap(psi, samples_onehot, device=None,
                                  evaluation_batch_size=None):
    """Jacobian of model log amplitudes; accepts device-resident one-hot inputs."""
    parameter = next(psi.parameters())
    x = torch.as_tensor(samples_onehot, device=parameter.device, dtype=parameter.dtype)
    params = {k: v for k, v in psi.named_parameters() if v.requires_grad}
    if not params or not len(x):
        raise ValueError("SR requires trainable parameters and nonempty samples")
    if parameter.is_cuda:
        # Initialize the backward thread's context before jacrev uses cuBLAS.
        with torch.enable_grad():
            warmup = parameter.new_zeros((), requires_grad=True)
            torch.autograd.grad(warmup.square(), warmup)

    def single(p, sample):
        # NQSpy takes log(abs(output) + eps) here because it returns amplitudes.
        # These models already return log amplitudes: differentiate them directly.
        return functional_call(psi, p, (sample.unsqueeze(0),)).reshape(())

    blocks = []
    for block in x.split(_chunk_size(len(x), evaluation_batch_size)):
        jac = vmap(jacrev(single), in_dims=(None, 0))(params, block)
        blocks.append(torch.cat([jac[k].reshape(len(block), -1) for k in params], 1).detach())
    return torch.cat(blocks)


@torch.no_grad()
def sr_gradient(J, E_loc, tau=1e-3, solve_dtype=torch.float64):
    """NQSpy SR solve: (J.T J + tau I) delta = J.T E_centered.

    No Jacobian centering, sample normalization, or extra factor of two. Solve in
    solve_dtype on J.device; MPS falls back to CPU for float64 only.
    Set solve_dtype=None to use J.dtype, as in NQSpy.
    Dense parameter-space SR still requires O(parameters**2) storage.
    """
    if solve_dtype is None:
        solve_dtype = J.dtype
    solve_device = (torch.device("cpu")
                    if J.device.type == "mps" and solve_dtype == torch.float64 else J.device)
    J_work = J.detach().to(device=solve_device, dtype=solve_dtype)
    E_work = E_loc.detach().to(device=solve_device, dtype=solve_dtype).reshape(-1)
    S = J_work.T @ J_work
    S.diagonal().add_(tau)
    f = J_work.T @ E_work
    delta = torch.linalg.solve(S, f.unsqueeze(-1)).squeeze(-1)
    return delta.to(device=J.device, dtype=J.dtype)


def _sr_direction(psi, sampled_states, E_tensor, L, Nmax, tau, evaluation_batch_size):
    parameter = next(psi.parameters())
    states = states_to_occupancy_tensor(sampled_states, L, parameter.device)
    x = occupancy_tensor_to_onehot(states, Nmax, parameter.dtype)
    J = compute_log_psi_jacobian_vmap(psi, x, evaluation_batch_size=evaluation_batch_size)
    e = E_tensor.detach().to(parameter.device).reshape(-1)
    if len(e) != len(states):
        raise ValueError("energies must have one value per sample")
    return sr_gradient(J, e - e.mean(), tau)


def sr_update(psi, sampled_states, E_tensor, L, Nmax, lr, tau, device=None,
              evaluation_batch_size=None):
    delta = _sr_direction(psi, sampled_states, E_tensor, L, Nmax, tau, evaluation_batch_size)
    with torch.no_grad():
        offset = 0
        for p in psi.parameters():
            if p.requires_grad:
                p.add_(delta[offset:offset + p.numel()].reshape_as(p), alpha=-lr)
                offset += p.numel()


def sr_update_optimizer(psi, sampled_states, E_tensor, L, Nmax, optimizer, tau,
                        device=None, adaptive_lr=False, lr=1e-3,
                        evaluation_batch_size=None):
    delta = _sr_direction(psi, sampled_states, E_tensor, L, Nmax, tau, evaluation_batch_size)
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for p in psi.parameters():
        if p.requires_grad:
            p.grad = delta[offset:offset + p.numel()].reshape_as(p).clone()
            offset += p.numel()
    if adaptive_lr:
        scale = delta.abs().max().item() + 1e-5
        for group in optimizer.param_groups:
            group["lr"] = lr / scale
    optimizer.step()


def native_vmc_update(model, sampled_states, E_tensor, L, Nmax, optimizer,
                      device=None, logpsis=None, evaluation_batch_size=None):
    """Native 2*mean(centered detached energy * logpsi) optimizer step.

    Pass sampling's final logpsis to reuse its graph before parameters change.
    """
    parameter = next(model.parameters())
    if logpsis is None:
        states = states_to_occupancy_tensor(sampled_states, L, parameter.device)
        logpsis = _log_amplitudes(model, states, Nmax, evaluation_batch_size)
    e = E_tensor.detach().to(parameter.device).reshape(-1)
    if not len(e) or logpsis.numel() != len(e):
        raise ValueError("nonempty log amplitudes and energies must have equal lengths")
    loss = 2 * ((e - e.mean()) * logpsis.reshape(-1)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
