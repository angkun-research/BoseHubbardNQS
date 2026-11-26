"""Centered, batch-normalized ridge SR for real log-amplitude NQS."""

import math
import time


def damping_at_update(initial, final, hold_updates, total_updates, step):
    """Keep the initial damping through the hold, then decay in log space."""
    if step <= hold_updates:
        return initial
    if step == total_updates:
        return final
    progress = (step - hold_updates) / (total_updates - hold_updates)
    return initial * (final / initial) ** progress


def workload(samples, parameters, requested_chunk, workspace_mib):
    dimension = min(samples, parameters)
    solver = "sample" if samples < parameters else "parameter"
    # Float64 Jacobian and dense solve workspace plus float32 Jacobian blocks.
    # Model activations and backend allocations are outside this estimate.
    fixed_bytes = 8 * samples * parameters + 32 * dimension**2 + 24 * (samples + parameters)
    bytes_per_chunk_row = 16 * parameters
    limit = workspace_mib * 1024**2
    rows = min(samples, requested_chunk, (limit - fixed_bytes) // bytes_per_chunk_row)
    if rows < 1:
        raise ValueError(f"SR workspace needs at least {fixed_bytes + bytes_per_chunk_row} bytes for one Jacobian row")
    return {
        "solver": solver, "solve_dimension": dimension, "samples": samples, "parameters": parameters,
        "jacobian_chunk_requested": requested_chunk, "jacobian_chunk": rows, "jacobian_memory_throttled": (
            rows < min(samples, requested_chunk)
        ),
        "estimated_workspace_bytes": fixed_bytes + bytes_per_chunk_row * rows,
        "estimated_linear_algebra_flops": 2 * samples * parameters * (dimension + 1) + round(dimension**3 / 3),
    }


def ridge_direction(jacobian, energies, damping, solver="auto", *, overwrite=False):
    """Solve normal equations or their algebraically equivalent ridge MinSR form."""
    import torch

    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("SR damping must be finite and positive")
    if jacobian.ndim != 2 or min(jacobian.shape) == 0 or jacobian.is_complex() or energies.is_complex():
        raise ValueError("SR needs a nonempty real Jacobian and real energies")
    samples, parameters = jacobian.shape
    if solver == "auto":
        solver = "sample" if samples < parameters else "parameter"
    if solver not in ("sample", "parameter"):
        raise ValueError("SR solver must be auto, sample, or parameter")
    with torch.no_grad():
        A = jacobian.detach().to(dtype=torch.float64, copy=not overwrite)
        energy = energies.detach().to(device=A.device, dtype=torch.float64).reshape(-1)
        if energy.numel() != samples or not (torch.isfinite(A).all() & torch.isfinite(energy).all()).item():
            raise ValueError("SR requires finite Jacobian rows and one finite energy per sample")
        A.sub_(A.mean(dim=0)).div_(math.sqrt(samples))
        b = (energy - energy.mean()) * (2 / math.sqrt(samples))
        if solver == "sample":
            gram, rhs = A @ A.T, b
        else:
            gram, rhs = A.T @ A, A.T @ b
        gram.diagonal().add_(damping)
        factor = torch.linalg.cholesky(gram)
        reduced = torch.cholesky_solve(rhs[:, None], factor).flatten()
        direction = A.T @ reduced if solver == "sample" else reduced
        if not torch.isfinite(direction).all().item():
            raise RuntimeError("non-finite centered SR direction")
        residual = torch.linalg.vector_norm(gram @ reduced - rhs).item()
    return direction, residual


def update(model, samples, energies, config, workspace):
    import torch
    from vmc_utils_gpu import compute_log_psi_jacobian_vmap, occupancy_tensor_to_onehot

    def synchronize():
        if samples.is_cuda:
            torch.cuda.synchronize(samples.device)

    params = [p for p in model.parameters() if p.requires_grad]
    synchronize()
    start = time.perf_counter()
    jacobian = torch.empty((len(samples), workspace["parameters"]), dtype=torch.float64, device=samples.device)
    begin = 0
    for block in samples.split(workspace["jacobian_chunk"]):
        inputs = occupancy_tensor_to_onehot(block, config["nmax"], params[0].dtype)
        jacobian[begin:begin + len(block)].copy_(compute_log_psi_jacobian_vmap(model, inputs))
        begin += len(block)
    synchronize()
    solve_started = time.perf_counter()
    direction, residual = ridge_direction(jacobian, energies, config["sr_lambda"], workspace["solver"], overwrite=True)
    synchronize()
    update_started = time.perf_counter()
    with torch.no_grad():
        begin = 0
        for param in params:
            end = begin + param.numel()
            param.add_(direction[begin:end].reshape_as(param).to(param.dtype), alpha=-config["lr"])
            begin = end
    norm = direction.norm().item()
    return dict(workspace, jacobian_seconds=solve_started - start, solve_seconds=update_started - solve_started,
                apply_seconds=time.perf_counter() - update_started, solve_residual_l2=residual, direction_l2=norm,
                jacobian_rows=len(samples))
