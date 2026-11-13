"""Frozen real NQS measurements with explicit (time, walker, ...) sample axes.

rho[i,j] = <b_i^dag b_j>. Its ROW-action local estimator at bra n uses
sqrt(n_i*(n_j+1))*psi(n-e_i+e_j)/psi(n), not the transposed ket-action hop.
All physical indices use i=y*Lx+x. No matrix is symmetrized by this module.
Block errors are conditional estimates, not a claim of equilibration or ESS.
"""

import torch

if __package__:
    from .lattice import _integer, bonds_2d, lattice_shape
else:
    from lattice import _integer, bonds_2d, lattice_shape
if __package__:
    from .vmc import _states, _batch_size, _log_amplitudes, local_energy, physical_parameters, sample
else:
    from vmc import _states, _batch_size, _log_amplitudes, local_energy, physical_parameters, sample


DEFAULT_MAX_OBSERVABLE_BYTES = 512 * 1024**2


def _chain_states(model, chains):
    chains = torch.as_tensor(chains)
    if chains.ndim != 3 or not chains.shape[0] or not chains.shape[1]:
        raise ValueError("occupations must have nonempty (time, walker, site) axes")
    T, C, N = chains.shape
    flat = _states(model, chains.reshape(T * C, N))
    return flat, T, C


def _pairs(pairs, n_sites):
    if pairs is None:
        return [(i, j) for i in range(n_sites) for j in range(n_sites)]
    if isinstance(pairs, torch.Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")
        pairs = pairs.tolist()
    result = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each pair must contain two physical site indices")
        i, j = (_integer(value, "pair site") for value in pair)
        if max(i, j) >= n_sites:
            raise ValueError("pair site is outside the lattice")
        result.append((i, j))
    if len(set(result)) != len(result):
        raise ValueError("selected pairs must be unique")
    return result


def check_observable_budget(n_samples, n_sites, n_pairs, max_bytes=DEFAULT_MAX_OBSERVABLE_BYTES):
    """Conservative float64 storage estimate, excluding model/forward workspace."""
    n_samples = _integer(n_samples, "n_samples", 1)
    n_sites = _integer(n_sites, "n_sites", 1)
    n_pairs = _integer(n_pairs, "n_pairs")
    max_bytes = _integer(max_bytes, "max_observable_bytes", 1)
    estimate = 8 * (3 * n_samples * n_pairs + 6 * n_samples * n_sites + 12 * n_samples + 4 * n_sites**2)
    if estimate > max_bytes:
        raise ValueError(f"estimated observable storage {estimate} exceeds max_observable_bytes={max_bytes}")
    return {"estimated_bytes": estimate, "max_bytes": max_bytes, "samples": n_samples, "pairs": n_pairs}


@torch.no_grad()
def sample_frozen(model, initial_states, n_steps, burnin_steps=0, evaluation_batch_size=256,
                  *, proposal_steps_per_sample=1):
    """Wrap the stable sampler, returning occupations as (time, walker, site)."""
    model.eval()
    flat, walkers, diagnostics = sample(model, initial_states, n_steps, burnin_steps, evaluation_batch_size,
                                        proposal_steps_per_sample=proposal_steps_per_sample)
    return flat.reshape(n_steps, len(walkers), model.n_sites), walkers, diagnostics


@torch.no_grad()
def one_body_estimators(model, chains, pairs=None, evaluation_batch_size=256,
                        max_observable_bytes=DEFAULT_MAX_OBSERVABLE_BYTES):
    """Return raw row-action estimators, without averaging or symmetrization.

    pairs=None explicitly requests full (T,C,N,N) output, intended for tiny
    references. Selected physical pairs give (T,C,K), in the supplied order.
    At most one evaluation batch of connected configurations is constructed.
    """
    size = _batch_size(evaluation_batch_size)
    states, T, C = _chain_states(model, chains)
    N = model.n_sites
    count = N * N if pairs is None else len(pairs)
    check_observable_budget(T * C, N, count, max_observable_bytes)
    selected = _pairs(pairs, N)
    result = torch.empty((T * C, len(selected)), device=states.device, dtype=torch.float64)
    for start in range(0, len(states), size):
        block = states[start:start + size]
        occ = block.double()
        base = None
        for k, (i, j) in enumerate(selected):
            if i == j:
                result[start:start + len(block), k] = occ[:, i]
                continue
            values = result[start:start + len(block), k]
            values.zero_()
            rows = ((block[:, i] > 0) & (block[:, j] < model.Nmax)).nonzero().flatten()
            if not len(rows):
                continue
            if base is None:
                base = _log_amplitudes(model, block, size).double()
            connected = block[rows]
            connected[:, i] -= 1
            connected[:, j] += 1
            other = _log_amplitudes(model, connected, size).double()
            values[rows] = torch.sqrt(occ[rows, i] * (occ[rows, j] + 1)) * (other - base[rows]).exp()
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite one-body estimators")
    shape = (T, C, N, N) if pairs is None else (T, C, len(selected))
    return result.reshape(shape)


@torch.no_grad()
def energy_from_observables(density, onsite_factorial_moment, one_body, Lx, Ly=None,
                            t=1.0, U=16.74, mu=0.0, pbc=True, *, pbc_x=None, pbc_y=None, pairs=None):
    """Linear reconstruction from raw rho and <n(n-1)>; accepts sample axes.

    A selected-pair reconstruction requires both directions of every bond.
    Missing entries raise; they are never filled by assuming Hermiticity.
    """
    Lx, Ly = lattice_shape(Lx, Ly)
    t, U, mu = physical_parameters(t, U, mu)
    density = torch.as_tensor(density, dtype=torch.float64)
    moment = torch.as_tensor(onsite_factorial_moment, device=density.device, dtype=torch.float64)
    rho = torch.as_tensor(one_body, device=density.device, dtype=torch.float64)
    N = Lx * Ly
    if density.shape[-1:] != (N,) or moment.shape != density.shape:
        raise ValueError("density and onsite moment shapes must match the lattice")
    bonds = bonds_2d(Lx, Ly, pbc, pbc_x=pbc_x, pbc_y=pbc_y)
    energy = (0.5 * U * moment - mu * density).sum(-1)
    if pairs is None:
        if rho.shape != (*density.shape[:-1], N, N):
            raise ValueError("full one-body matrix shape must match density")
        for i, j in bonds:
            energy -= t * (rho[..., i, j] + rho[..., j, i])
    else:
        selected = _pairs(pairs, N)
        if rho.shape != (*density.shape[:-1], len(selected)):
            raise ValueError("selected one-body values must match pairs and sample axes")
        lookup = {pair: k for k, pair in enumerate(selected)}
        for i, j in bonds:
            if (i, j) not in lookup or (j, i) not in lookup:
                raise ValueError("selected pairs miss required directed Hamiltonian bonds")
            energy -= t * (rho[..., lookup[i, j]] + rho[..., lookup[j, i]])
    if not torch.isfinite(energy).all():
        raise RuntimeError("non-finite reconstructed energy")
    return energy


@torch.no_grad()
def block_mean_statistics(values, block_length):
    """Nonoverlapping time blocks within each walker; never flatten before blocking.

    The block SEM assumes stationary, sufficiently decorrelated blocks and
    independent walkers. Between-chain SEM is also reported as a diagnostic.
    Neither checks mixing or estimates ESS. Tail steps are explicitly reported;
    the blocked mean/error use complete blocks, with an all-sample mean separate.
    Fewer than two complete blocks per chain yields no block error estimate.
    """
    block_length = _integer(block_length, "block_length", 1)
    values = torch.as_tensor(values).detach().double()
    if values.ndim < 2 or not values.shape[0] or not values.shape[1] or not torch.isfinite(values).all():
        raise ValueError("values must have finite nonempty (time, walker, ...) axes")
    T, C = values.shape[:2]
    blocks = T // block_length
    used = blocks * block_length
    all_mean = values.mean((0, 1))
    if blocks:
        means = values[:used].reshape(blocks, block_length, C, *values.shape[2:]).mean(1)
        mean = means.mean((0, 1))
        chain_means = values[:used].mean(0)
    else:
        means = values.new_empty((0, C, *values.shape[2:]))
        mean, chain_means = all_mean, values.mean(0)
    block_error = means.flatten(0, 1).std(0, unbiased=True) / (blocks * C)**0.5 if blocks >= 2 else None
    chain_error = chain_means.std(0, unbiased=True) / C**0.5 if C >= 2 else None
    return {"mean": mean, "all_samples_mean": all_mean, "block_standard_error": block_error,
            "between_chain_standard_error": chain_error, "block_means": means,
            "time_steps": T, "walkers": C, "block_length": block_length,
            "complete_blocks_per_chain": blocks, "used_time_steps": used, "discarded_tail_steps": T - used,
            "status": "conditional_block_estimate" if blocks >= 2 else "insufficient_complete_time_blocks",
            "assumptions": "stationary chains, independent walkers and negligible correlation between time blocks; test block-length stability; no ESS or convergence claim"}


@torch.no_grad()
def measure_observables(model, chains, t=1.0, U=16.74, mu=0.0, evaluation_batch_size=256,
                        *, pairs=None, block_length=None, max_observable_bytes=DEFAULT_MAX_OBSERVABLE_BYTES):
    """Frozen density, onsite moment, selected/full rho and energy with raw checks."""
    states, T, C = _chain_states(model, chains)
    N = model.n_sites
    count = N * N if pairs is None else len(pairs)
    budget = check_observable_budget(T * C, N, count, max_observable_bytes)
    selected = _pairs(pairs, N)
    rho = one_body_estimators(model, chains, pairs, evaluation_batch_size, max_observable_bytes)
    pair_samples = rho.reshape(T, C, len(selected))
    density = states.reshape(T, C, N).double()
    moment = density * (density - 1)
    energy = local_energy(model, states, t, U, mu, evaluation_batch_size).double().reshape(T, C)
    lookup = {pair: k for k, pair in enumerate(selected)}
    mean_rho = pair_samples.mean((0, 1))
    conjugate_pairs = [(lookup[i, j], lookup[j, i]) for i, j in selected if i < j and (j, i) in lookup]
    raw_hermiticity = max((abs(mean_rho[a] - mean_rho[b]).item() for a, b in conjugate_pairs), default=None)
    diagonal = [(i, lookup[i, i]) for i in range(N) if (i, i) in lookup]
    diagonal_error = max(((pair_samples[..., k] - density[..., i]).abs().max().item() for i, k in diagonal), default=None)
    trace_error = None
    if len(diagonal) == N:
        trace_error = (pair_samples[..., [k for _, k in diagonal]].sum(-1) - states[0].sum()).abs().max().item()
    bonds = bonds_2d(model.Lx, model.Ly, pbc_x=getattr(model, "pbc_x", model.pbc),
                     pbc_y=getattr(model, "pbc_y", model.pbc))
    missing = [(i, j) for a, b in bonds for i, j in ((a, b), (b, a)) if (i, j) not in lookup]
    reconstructed = None
    if not missing:
        reconstructed = energy_from_observables(
            density, moment, pair_samples, model.Lx, model.Ly, t, U, mu,
            pbc_x=getattr(model, "pbc_x", model.pbc), pbc_y=getattr(model, "pbc_y", model.pbc), pairs=selected,
        )
    diagnostics = {"raw_mean_rho_hermiticity_max_abs": raw_hermiticity,
                   "hermiticity_unordered_pairs_checked": len(conjugate_pairs), "symmetrized": False,
                   "rho_diagonal_density_max_abs": diagonal_error, "rho_diagonal_entries_checked": len(diagonal),
                   "rho_trace_minus_Ntot_max_abs": trace_error,
                   "density_sum_minus_Ntot_max_abs": (density.sum(-1) - states[0].sum()).abs().max().item(),
                   "missing_directed_bonds_for_energy_reconstruction": missing,
                   "local_energy_reconstruction_max_abs": (reconstructed - energy).abs().max().item() if reconstructed is not None else None}
    result = {"density_samples": density, "onsite_factorial_moment_samples": moment,
              "one_body_samples": rho, "one_body_pair_samples": pair_samples, "one_body_pairs": selected,
              "local_energy_samples": energy, "reconstructed_energy_samples": reconstructed,
              "density": density.mean((0, 1)), "onsite_factorial_moment": moment.mean((0, 1)),
              "one_body_pair_means": mean_rho, "one_body_matrix": mean_rho.reshape(N, N) if pairs is None else None,
              "diagnostics": diagnostics, "budget": budget, "statistics": {}}
    if block_length is not None:
        arrays = {"density": density, "onsite_factorial_moment": moment,
                  "one_body": pair_samples, "energy": energy}
        if reconstructed is not None:
            arrays["reconstructed_energy"] = reconstructed
        result["statistics"] = {name: block_mean_statistics(values, block_length) for name, values in arrays.items()}
    return result
