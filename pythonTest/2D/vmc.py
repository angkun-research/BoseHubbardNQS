"""Batched fixed-number VMC on CPU or CUDA; models return real log amplitudes.

Sampling never retains autograd graphs. Each sample() call recomputes its initial
log cache, including after optimizer updates. Retained samples cost O(M*sites);
evaluation_batch_size bounds network batches, connected states, and native
backward graphs, not the total sample storage. Uniform bond/direction proposals
retain invalid hops as self proposals, so no Hastings proposal factor is needed.
"""

from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

if __package__:
    from .lattice import _integer, bonds_2d
else:
    from lattice import _integer, bonds_2d
if __package__:
    from .exact_diagonalization import _physical_scalar
else:
    from exact_diagonalization import _physical_scalar


SR_CONVENTION = "centered Jc; S=Jc.T@Jc/M; f=Jc.T@(E-Emean)/M; (S+tau*I)delta=f; no extra 2"
MINSR_CONVENTION = "A=Jc/sqrt(M); e=(E-Emean)/sqrt(M); delta=A.T@g(A@A.T)@e; soft spectral pseudoinverse power6; SGD subtracts lr*delta; tau unused; no extra 2"
DEFAULT_MAX_SR_PARAMETERS = 1_000_000
DEFAULT_MAX_SR_BYTES = 512 * 1024**2
MAX_JACOBIAN_BATCH_SIZE = 64
MAX_VMAP_JACOBIAN_BATCH_SIZE = 128


def physical_parameters(t=1.0, U=16.74, mu=0.0):
    return tuple(_physical_scalar(value, name)
                 for name, value in (("t", t), ("U", U), ("mu", mu)))


def _batch_size(evaluation_batch_size):
    return _integer(evaluation_batch_size, "evaluation_batch_size", 1)


def occupancy_tensor_to_onehot(states, Nmax, dtype=torch.float32):
    return F.one_hot(states, num_classes=Nmax + 1).to(dtype=dtype)


def _states(model, states):
    parameter = next(model.parameters())
    states = torch.as_tensor(states, device=parameter.device)
    if states.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("states must contain integer occupations")
    if states.ndim != 2 or states.shape[1] != model.n_sites or not len(states):
        raise ValueError(f"states must have nonempty shape (batch, {model.n_sites})")
    states = states.long()
    if (states < 0).any() or (states > model.Nmax).any():
        raise ValueError("occupations must be in [0, Nmax]")
    totals = states.sum(1)
    if (totals != totals[0]).any():
        raise ValueError("all states must belong to one fixed-number sector")
    return states


def _log_amplitudes(model, states, evaluation_batch_size, check_finite=True):
    dtype = next(model.parameters()).dtype
    logs = torch.cat([
        model(occupancy_tensor_to_onehot(block, model.Nmax, dtype)).reshape(-1)
        for block in states.split(evaluation_batch_size)
    ])
    if check_finite and not torch.isfinite(logs).all():
        raise RuntimeError("non-finite model log amplitudes")
    return logs


def generate_initial_states(n_sites, Ntot, Nmax=4, n_walkers=64, device="cpu"):
    """Generate feasible integer walkers; initialization is not a target sample."""
    n_sites = _integer(n_sites, "n_sites", 1)
    Ntot = _integer(Ntot, "Ntot")
    Nmax = _integer(Nmax, "Nmax")
    n_walkers = _integer(n_walkers, "n_walkers", 1)
    if Ntot > n_sites * Nmax:
        raise ValueError("Ntot exceeds n_sites * Nmax")
    states = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    remaining = torch.full((n_walkers,), Ntot, dtype=torch.long, device=device)
    for site in range(n_sites):
        lower = (remaining - (n_sites - site - 1) * Nmax).clamp(min=0)
        upper = remaining.clamp(max=Nmax)
        occupation = lower + (torch.rand(n_walkers, device=device) * (upper - lower + 1)).long()
        states[:, site] = occupation
        remaining -= occupation
    # Randomize the sequential initializer's site preference for each walker.
    order = torch.rand(n_walkers, n_sites, device=device).argsort(dim=1)
    return states.gather(1, order)


def _model_bonds(model):
    # Legacy/custom models may still expose only the both-axis pbc attribute.
    legacy = getattr(model, "pbc", True)
    return bonds_2d(model.Lx, model.Ly, pbc_x=getattr(model, "pbc_x", legacy),
                    pbc_y=getattr(model, "pbc_y", legacy))


def _bond_tensor(model, device):
    return torch.tensor(_model_bonds(model),
                        dtype=torch.long, device=device).reshape(-1, 2)


def propose_hops(states, bonds, Nmax):
    """Return (proposals, valid-hop mask); choose all bonds/directions uniformly.

    States and bonds are device-resident long tensors, with unique non-self bonds.
    Invalid hops and the no-bond case return self proposals, never resampled moves.
    """
    if not len(bonds):
        return states.clone(), torch.zeros(len(states), dtype=torch.bool, device=states.device)
    chosen = bonds[torch.randint(len(bonds), (len(states),), device=states.device)]
    reverse = torch.randint(2, (len(states),), device=states.device).bool()
    src = torch.where(reverse, chosen[:, 1], chosen[:, 0])
    dst = torch.where(reverse, chosen[:, 0], chosen[:, 1])
    rows = torch.arange(len(states), device=states.device)
    valid = (states[rows, src] > 0) & (states[rows, dst] < Nmax)
    proposed = states.clone()
    proposed[rows, src] -= valid.long()
    proposed[rows, dst] += valid.long()
    return proposed, valid


@torch.no_grad()
def sample(model, initial_states, n_steps, burnin_steps=0, evaluation_batch_size=256,
           *, proposal_steps_per_sample=1):
    """Return (samples, final walkers, diagnostics), flattened step then walker.

    Burn-in proposals are excluded from diagnostics and samples. Accepted includes
    accepted self proposals; moved counts only accepted occupation changes. Caches
    are local to this call and cannot survive parameter updates.
    n_steps counts RETAINED samples per walker. After burnin_steps proposals,
    advance proposal_steps_per_sample proposals between retained configurations.
    Diagnostics count every post-burn proposal, including discarded intervals.
    The default stride one preserves the previous seeded trajectory exactly.
    """
    n_steps = _integer(n_steps, "n_steps")
    burnin_steps = _integer(burnin_steps, "burnin_steps")
    stride = _integer(proposal_steps_per_sample, "proposal_steps_per_sample", 1)
    size = _batch_size(evaluation_batch_size)
    states = _states(model, initial_states).clone()
    bonds = _bond_tensor(model, states.device)
    current = _log_amplitudes(model, states, size)
    samples = []
    counts = torch.zeros(3, dtype=torch.long, device=states.device)
    for step in range(burnin_steps + n_steps * stride):
        proposed, valid = propose_hops(states, bonds, model.Nmax)
        new = _log_amplitudes(model, proposed, size)
        accept = torch.rand(len(states), device=states.device).log() < (2 * (new - current)).clamp(max=0)
        states = torch.where(accept[:, None], proposed, states)
        current = torch.where(accept, new, current)
        if step >= burnin_steps:
            counts += torch.stack((accept.sum(), (accept & valid).sum(), valid.sum()))
            if (step - burnin_steps + 1) % stride == 0:
                samples.append(states.clone())
    accepted, moved, valid = counts.tolist()
    attempts = n_steps * stride * len(states)
    diagnostics = {"attempts": attempts, "accepted": accepted, "moved": moved,
                   "valid_proposals": valid, "accepted_rate": accepted / max(attempts, 1),
                   "moved_rate": moved / max(attempts, 1), "valid_proposal_rate": valid / max(attempts, 1),
                   "proposal_steps_per_sample": stride, "retained_steps_per_walker": n_steps,
                   "postburn_proposal_steps_per_walker": n_steps * stride,
                   "burnin_proposal_steps_per_walker": burnin_steps,
                   "attempts_unit": "one bond/direction proposal for one walker, not a lattice sweep",
                   "time_axis_unit": "one retained configuration per walker after proposal_steps_per_sample proposals",
                   "diagnostics_scope": "all post-burn proposals, including intervals between retained samples; burn-in excluded"}
    result = torch.stack(samples).reshape(-1, model.n_sites) if samples else states.new_empty((0, model.n_sites))
    return result, states, diagnostics


@torch.no_grad()
def local_energy(model, states, t=1.0, U=16.74, mu=0.0, evaluation_batch_size=256):
    """Detached H psi / psi with U/2, bounded connected-state allocations.

    Traverse directed bonds sequentially inside each sample block. At most one
    evaluation batch of connected occupations exists at a time, rather than a
    batch times coordination-number expansion. Returns one energy per state.
    """
    t, U, mu = physical_parameters(t, U, mu)
    size = _batch_size(evaluation_batch_size)
    states = _states(model, states)
    dtype = next(model.parameters()).dtype
    bonds = _model_bonds(model)
    energies = []
    finite_logs = torch.ones((), dtype=torch.bool, device=states.device)
    for block in states.split(size):
        occ = block.to(dtype=dtype)
        energy = (0.5 * U * occ * (occ - 1) - mu * occ).sum(1)
        if t and bonds:
            base = _log_amplitudes(model, block, size, check_finite=False)
            finite_logs &= torch.isfinite(base).all()
            for i, j in bonds:
                for src, dst in ((i, j), (j, i)):
                    rows = ((block[:, src] > 0) & (block[:, dst] < model.Nmax)).nonzero().flatten()
                    if not len(rows):
                        continue
                    connected = block[rows].clone()
                    connected[:, src] -= 1
                    connected[:, dst] += 1
                    other = _log_amplitudes(model, connected, size, check_finite=False)
                    finite_logs &= torch.isfinite(other).all()
                    coefficient = -t * torch.sqrt(occ[rows, src] * (occ[rows, dst] + 1))
                    energy[rows] += coefficient * torch.exp(other - base[rows])
        energies.append(energy)
    result = torch.cat(energies)
    # Check once on the host, but track every evaluated log on device: a -inf
    # connected log can yield a finite zero ratio and must still be rejected.
    if not (finite_logs & torch.isfinite(result).all()):
        raise RuntimeError("non-finite model log amplitudes or local energies")
    return result


def native_vmc_update(model, states, energies, optimizer, evaluation_batch_size=256):
    """Accumulate 2*mean((E-Emean).detach()*logpsi) gradients per chunk."""
    size = _batch_size(evaluation_batch_size)
    states = _states(model, states)
    parameter = next(model.parameters())
    energies = torch.as_tensor(energies, device=parameter.device, dtype=parameter.dtype).detach().reshape(-1)
    if len(energies) != len(states) or not torch.isfinite(energies).all():
        raise ValueError("finite energies must match the samples")
    centered = energies - energies.mean()
    optimizer.zero_grad(set_to_none=True)
    loss_value = parameter.new_zeros((), dtype=torch.float64)
    for start in range(0, len(states), size):
        logs = _log_amplitudes(model, states[start:start + size], size)
        loss = 2 * (centered[start:start + size] * logs).sum() / len(states)
        loss.backward()
        loss_value += loss.detach()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("non-finite native gradient")
    optimizer.step()
    return loss_value.item()


def _sr_budget(parameters, n_samples, max_sr_parameters, max_sr_bytes, *, solver="auto", jacobian_batch_size=1,
               jacobian_method="vjp", jacobian_input_elements=0, regularizer="ridge"):
    n_samples = _integer(n_samples, "n_samples", 1)
    limit = _integer(max_sr_parameters, "max_sr_parameters", 1)
    byte_limit = _integer(max_sr_bytes, "max_sr_bytes", 1)
    batch = _integer(jacobian_batch_size, "jacobian_batch_size", 1)
    if jacobian_method not in ("vjp", "vmap"):
        raise ValueError("jacobian_method must be vjp or vmap")
    cap = MAX_JACOBIAN_BATCH_SIZE if jacobian_method == "vjp" else MAX_VMAP_JACOBIAN_BATCH_SIZE
    if batch > cap:
        raise ValueError(f"jacobian_batch_size exceeds {cap} for {jacobian_method}")
    batch = min(batch, n_samples)
    if solver not in ("auto", "parameter", "sample"):
        raise ValueError("SR solver must be auto, parameter or sample")
    if regularizer not in ("ridge", "minsr"):
        raise ValueError("SR regularizer must be ridge or minsr")
    selected = ("parameter" if parameters <= n_samples else "sample") if solver == "auto" else solver
    dimension = parameters if selected == "parameter" else n_samples
    vjp_bytes = 8 * (2 * batch * parameters + batch**2) if jacobian_method == "vjp" and jacobian_batch_size > 1 else 0
    vmap_bytes = 8 * (2 * batch * parameters + batch * jacobian_input_elements) if jacobian_method == "vmap" else 0
    estimate = 8 * (3 * n_samples * parameters + 8 * dimension**2 + 8 * parameters + 4 * n_samples) + vjp_bytes + vmap_bytes
    # Additional eigenvectors, matrix copies and dense eigensolver workspace.
    # Keep the original ridge estimate exactly unchanged.
    spectral_bytes = 8 * (8 * dimension**2 + 4 * dimension) if regularizer == "minsr" else 0
    estimate += spectral_bytes
    if not parameters or parameters > limit:
        raise ValueError(f"SR parameter count {parameters} exceeds max_sr_parameters={limit} or is zero")
    if estimate > byte_limit:
        raise ValueError(f"estimated SR workspace {estimate} exceeds max_sr_bytes={byte_limit}")
    return {"parameters": parameters, "samples": n_samples, "solver_requested": solver,
            "solver": selected, "solve_dimension": dimension,
            "jacobian_bytes": 8 * n_samples * parameters, "estimated_bytes": estimate,
            "jacobian_batch_size": jacobian_batch_size, "jacobian_batch_rows": batch,
            "jacobian_method": jacobian_method, "batched_vjp_workspace_bytes": vjp_bytes,
            "vmap_gradient_and_input_bytes": vmap_bytes,
            "regularizer": regularizer, "additional_eigh_workspace_bytes": spectral_bytes}


def check_sr_budget(model, n_samples, max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS,
                    max_sr_bytes=DEFAULT_MAX_SR_BYTES, *, solver="auto", jacobian_batch_size=1, jacobian_method="vjp",
                    regularizer="ridge"):
    """Guard before allocating J or float64 dense solve arrays.

    Budget three MxP float64 Jacobians (input/conversion/centered), eight DxD
    float64 arrays for covariance, regularization, solver copies/workspace, and
    vectors, D=P or M according to solver. Auto selects min(P,M). The explicit
    byte budget guards M*P even for sample-space solves. The generous parameter
    cap is a separate fail-safe, not a substitute for this allocation guard.
    Optional MinSR budgets eight additional DxD arrays plus spectral vectors
    before any Jacobian/eigh allocation. This conservative estimate does not
    bound model activations, all vendor-library memory, or total process memory.
    """
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _sr_budget(parameters, n_samples, max_sr_parameters, max_sr_bytes, solver=solver,
                       jacobian_batch_size=jacobian_batch_size, jacobian_method=jacobian_method,
                       jacobian_input_elements=model.n_sites * (model.Nmax + 1) if jacobian_method == "vmap" else 0,
                       regularizer=regularizer)


def compute_log_psi_jacobian(model, states, max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS,
                            max_sr_bytes=DEFAULT_MAX_SR_BYTES, *, solver="auto", jacobian_batch_size=1, jacobian_method="vjp",
                            regularizer="ridge"):
    """Guarded log-amplitude Jacobian; default one-sample reference is unchanged.

    B>1 evaluates B inputs once and uses is_grads_batched=True with an identity
    cotangent matrix, returning B parameter VJPs. Attention explicitly uses the
    math SDPA backend because fused attention backward lacks batching rules.
    Guards include returned VJP/cotangent arrays, not model activation memory;
    intermediate backward work can scale quadratically in B. Cap B at 64.
    Opt-in vmap(grad(functional_call)) differentiates one configuration at a time,
    vectorized in chunks up to 128. Guards include returned gradient/input blocks;
    model activations and total process memory are not bounded by this estimate.
    """
    states = _states(model, states)
    budget = check_sr_budget(model, len(states), max_sr_parameters, max_sr_bytes, solver=solver,
                              jacobian_batch_size=jacobian_batch_size, jacobian_method=jacobian_method,
                              regularizer=regularizer)
    if jacobian_method == "vmap":
        parameters = {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
        buffers = {name: b.detach() for name, b in model.named_buffers()}
        def single(params, onehot):
            return torch.func.functional_call(model, (params, buffers), (onehot.unsqueeze(0),)).squeeze()
        per_sample = torch.func.vmap(torch.func.grad(single), in_dims=(None, 0), randomness="error")
        attention = any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
        context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if attention else nullcontext()
        with context:
            with torch.no_grad():
                _log_amplitudes(model, states, evaluation_batch_size=budget["jacobian_batch_rows"])
            jacobian = torch.empty((len(states), budget["parameters"]), dtype=torch.float64, device=states.device)
            for start in range(0, len(states), budget["jacobian_batch_rows"]):
                block = states[start:start + budget["jacobian_batch_rows"]]
                onehot = occupancy_tensor_to_onehot(block, model.Nmax, next(model.parameters()).dtype)
                gradients = per_sample(parameters, onehot)
                offset = 0
                for name, parameter in parameters.items():
                    size = parameter.numel()
                    jacobian[start:start + len(block), offset:offset + size] = gradients[name].detach().reshape(len(block), size)
                    offset += size
        if not torch.isfinite(jacobian).all():
            raise RuntimeError("non-finite SR Jacobian")
        return jacobian
    parameters = [p for p in model.parameters() if p.requires_grad]
    jacobian = torch.empty((len(states), budget["parameters"]), dtype=torch.float64, device=states.device)
    if jacobian_batch_size == 1:
        for index in range(len(states)):
            logpsi = _log_amplitudes(model, states[index:index + 1], 1).sum()
            gradients = torch.autograd.grad(logpsi, parameters, allow_unused=True)
            offset = 0
            for parameter, gradient in zip(parameters, gradients):
                size = parameter.numel()
                jacobian[index, offset:offset + size] = 0 if gradient is None else gradient.detach().flatten()
                offset += size
    else:
        attention = any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
        context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if attention else nullcontext()
        with context:
            for start in range(0, len(states), budget["jacobian_batch_rows"]):
                block = states[start:start + budget["jacobian_batch_rows"]]
                logs = _log_amplitudes(model, block, len(block))
                gradients = torch.autograd.grad(
                    logs, parameters, grad_outputs=torch.eye(len(block), device=logs.device, dtype=logs.dtype),
                    is_grads_batched=True, allow_unused=True,
                )
                offset = 0
                for parameter, gradient in zip(parameters, gradients):
                    size = parameter.numel()
                    jacobian[start:start + len(block), offset:offset + size] = (
                        0 if gradient is None else gradient.detach().reshape(len(block), size)
                    )
                    offset += size
    if not torch.isfinite(jacobian).all():
        raise RuntimeError("non-finite SR Jacobian")
    return jacobian


def _sr_regularization(regularizer, tau, rpinv, apinv):
    if regularizer == "ridge":
        tau = _physical_scalar(tau, "tau")
        if tau <= 0:
            raise ValueError("tau must be finite and positive")
    elif regularizer == "minsr":
        rpinv, apinv = _physical_scalar(rpinv, "rpinv"), _physical_scalar(apinv, "apinv")
        if rpinv < 0 or apinv < 0:
            raise ValueError("rpinv and apinv must be finite and nonnegative")
        tau = None  # There is no diagonal ridge or tau schedule in this mode.
    else:
        raise ValueError("SR regularizer must be ridge or minsr")
    return tau, rpinv, apinv


@torch.no_grad()
def _minsr_soft_filter(eigenvalues, rpinv=1e-12, apinv=0.0):
    """PSD soft pseudoinverse; stable below cutoff and exactly zero at zero."""
    if eigenvalues.dtype != torch.float64 or eigenvalues.ndim != 1 or not len(eigenvalues):
        raise ValueError('eigenvalues must be nonempty float64 vector')
    if not torch.isfinite(eigenvalues).all():
        raise ValueError('eigenvalues must be finite')
    if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in (rpinv, apinv)):
        raise ValueError('rpinv and apinv must be finite nonnegative scalars')
    scale = eigenvalues.abs().max().item()
    tolerance = 64 * torch.finfo(torch.float64).eps * len(eigenvalues) * scale
    if eigenvalues.min().item() < -tolerance:
        raise RuntimeError('materially negative PSD Gram eigenvalue')
    cutoff = rpinv * scale + apinv
    if not math.isfinite(cutoff):
        raise ValueError('spectral cutoff overflow')
    weights = torch.zeros_like(eigenvalues)
    positive = eigenvalues > 0
    above = positive & (eigenvalues >= cutoff)
    # Ratios are bounded by one in both branches. Never form (c/lambda)**6
    # when lambda<c, never multiply an infinite reciprocal by a zero weight.
    if cutoff == 0:
        weights[positive] = 1 / eigenvalues[positive]
    else:
        large = eigenvalues[above]
        weights[above] = (1 / large) / (1 + (cutoff / large)**6)
        below = positive & ~above
        ratio = eigenvalues[below] / cutoff
        weights[below] = (ratio**5 / cutoff) / (1 + ratio**6)
    if not torch.isfinite(weights).all():
        raise RuntimeError('non-finite soft pseudoinverse weights')
    diagnostics = {
        'regularizer': 'minsr', 'soft_cutoff_power': 6, 'rpinv': rpinv, 'apinv': apinv,
        'spectral_cutoff': cutoff, 'max_abs_eigenvalue': scale,
        'min_eigenvalue': eigenvalues.min().item(), 'max_eigenvalue': eigenvalues.max().item(),
        'positive_eigenvalues': positive.sum().item(),
        'zero_eigenvalues': (eigenvalues == 0).sum().item(),
        'roundoff_negative_eigenvalues': (eigenvalues < 0).sum().item(),
        'negative_roundoff_tolerance': tolerance,
        'positive_modes_at_or_above_cutoff': above.sum().item(),
        'soft_effective_modes': (weights * eigenvalues.clamp_min(0)).sum().item(),
        'maximum_spectral_gain': weights.max().item(),
        'zero_negative_policy': 'exact zero and roundoff-negative PSD eigenvalues map0; materially negative spectrum raises; no extra positive-eigenvalue truncation',
    }
    return weights, diagnostics


def sr_gradient(jacobian, energies, tau=1e-3,
                max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS, max_sr_bytes=DEFAULT_MAX_SR_BYTES,
                *, solver="auto", return_diagnostics=False, regularizer="ridge", rpinv=1e-12, apinv=0.0):
    """Centered, sample-normalized imaginary-time direction with no extra 2.

    Call check_sr_budget before constructing the input Jacobian. The training API
    does this automatically; the solve also checks before allocating workspace.
    All covariance and solve arithmetic uses float64.
    Sample space: delta=Jc.T @ solve(Jc@Jc.T + M*tau*I, E-Emean).
    It is exactly equivalent to the parameter-space ridge direction, with no
    extra factor two. Auto solves in the smaller space; it still stores J.
    Optional real MinSR uses the Methods20-23 soft spectral pseudoinverse from
    Chen/Heyl (2024), power6, rather than a ridge. Gram formation, eigh and
    backprojection are float64; tau is unused. The equivalent parameter-space
    form applies the same filter to A.T@A and then A.T@e. No extra truncation.
    """
    tau, rpinv, apinv = _sr_regularization(regularizer, tau, rpinv, apinv)
    if jacobian.ndim != 2 or not jacobian.shape[0] or not jacobian.shape[1]:
        raise ValueError("Jacobian must be a nonempty matrix")
    budget = _sr_budget(jacobian.shape[1], jacobian.shape[0], max_sr_parameters, max_sr_bytes, solver=solver, regularizer=regularizer)
    if regularizer == "minsr" and (jacobian.is_complex() or torch.as_tensor(energies).is_complex()):
        raise ValueError("real Jacobian and energies required for MinSR")
    J = jacobian.detach().to(dtype=torch.float64)
    E = torch.as_tensor(energies, device=J.device, dtype=torch.float64).detach().reshape(-1)
    if len(E) != len(J) or not torch.isfinite(E).all() or not torch.isfinite(J).all():
        raise ValueError("finite energies and Jacobian rows must match")
    centered = J - J.mean(0)
    Ec = E - E.mean()
    spectral = {}
    if regularizer == "minsr":
        A, e = centered / math.sqrt(len(J)), Ec / math.sqrt(len(J))
        gram = A @ A.T if budget["solver"] == "sample" else A.T @ A
        if not torch.isfinite(gram).all():
            raise RuntimeError("non-finite MinSR Gram")
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        weights, spectral = _minsr_soft_filter(eigenvalues, rpinv, apinv)
        rhs = e if budget["solver"] == "sample" else A.T @ e
        solution = eigenvectors @ (weights * (eigenvectors.T @ rhs))
        delta = A.T @ solution if budget["solver"] == "sample" else solution
    elif budget["solver"] == "parameter":
        S = centered.T @ centered / len(J)
        S.diagonal().add_(tau)
        force = centered.T @ Ec / len(J)
        delta = torch.linalg.solve(S, force)
    else:
        gram = centered @ centered.T
        gram.diagonal().add_(len(J) * tau)
        delta = centered.T @ torch.linalg.solve(gram, Ec)
    if not torch.isfinite(delta).all():
        raise RuntimeError("non-finite SR direction")
    if return_diagnostics:
        return delta, {**budget, **spectral, "tau": tau, "direction_norm": delta.norm().item()}
    return delta


def sr_update(model, states, energies, optimizer, tau=1e-3,
              max_sr_parameters=DEFAULT_MAX_SR_PARAMETERS, max_sr_bytes=DEFAULT_MAX_SR_BYTES,
              *, solver="auto", return_diagnostics=False, jacobian_batch_size=1, jacobian_method="vjp",
              regularizer="ridge", rpinv=1e-12, apinv=0.0):
    """Assign the guarded imaginary-time direction to optimizer gradients."""
    tau, rpinv, apinv = _sr_regularization(regularizer, tau, rpinv, apinv)
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("SR requires torch.optim.SGD")
    J = compute_log_psi_jacobian(model, states, max_sr_parameters, max_sr_bytes, solver=solver,
                                 jacobian_batch_size=jacobian_batch_size, jacobian_method=jacobian_method, regularizer=regularizer)
    direction, diagnostics = sr_gradient(J, energies, tau, max_sr_parameters, max_sr_bytes,
                                         solver=solver, return_diagnostics=True, regularizer=regularizer, rpinv=rpinv, apinv=apinv)
    diagnostics.update(check_sr_budget(model, len(states), max_sr_parameters, max_sr_bytes,
                                       solver=solver, jacobian_batch_size=jacobian_batch_size, jacobian_method=jacobian_method, regularizer=regularizer))
    diagnostics["jacobian_implementation"] = ("functional_vmap_grad" if jacobian_method == "vmap" else
                                               "per_sample_reference" if jacobian_batch_size == 1 else "batched_vjp_math_attention")
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            size = parameter.numel()
            parameter.grad = direction[offset:offset + size].reshape_as(parameter).to(parameter.dtype).clone()
            offset += size
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("non-finite SR direction after model-dtype conversion")
    optimizer.step()
    if return_diagnostics:
        return direction, diagnostics
    return direction
