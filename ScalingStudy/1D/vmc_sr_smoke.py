"""Slurm-only checks for the study's normalized ridge SR path."""

import argparse
import json
import math
import os

from ed_fidelity import require_compute_node
from vmc_sr import ridge_direction, update, workload


def attention_smoke(device):
    import torch
    from ed_fit import build_model, one_hot
    from vmc_utils_gpu import compute_log_psi_jacobian_vmap, generate_initial_state, local_energy_on_the_fly_gpu

    model = build_model("ConvTransformer", 16, 6, 16).to(device).eval()
    samples = torch.tensor([generate_initial_state(16, 16, 6) for _ in range(8)], device=device)
    inputs = one_hot(samples, 7)
    parameters = tuple(model.parameters())
    count = sum(param.numel() for param in parameters)
    assert count == 6705
    batched = compute_log_psi_jacobian_vmap(model, inputs, evaluation_batch_size=4)
    direct = torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(model(item[None]).sum(), parameters)])
                          for item in inputs])
    torch.testing.assert_close(batched, direct, atol=3e-5, rtol=3e-5)
    energies = local_energy_on_the_fly_gpu(samples, model, 16, 6, t=1.0, U=1.6375, evaluation_batch_size=64)
    before = torch.cat([param.detach().flatten().clone() for param in parameters])
    metrics = update(model, samples, energies, dict(nmax=6, sr_lambda=0.001, lr=0.001), workload(8, count, 4, 64))
    after = torch.cat([param.detach().flatten() for param in parameters])
    assert torch.isfinite(after).all() and not torch.equal(before, after)
    print(json.dumps({"device": str(device), "attention_jacobian_max_error": (batched - direct).abs().max().item(),
                      "parameter_update_norm": (after - before).norm().item(), "sr": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--attention-only", action="store_true", help="check the L=16 ConvTransformer Jacobian")
    args = parser.parse_args()
    if not 0 < args.threads <= require_compute_node():
        raise SystemExit("thread count exceeds the Slurm allocation")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    from ed_fit import FixedSector, build_model, configure_runtime, one_hot
    from vmc_utils_gpu import compute_log_psi_jacobian_vmap, local_energy_on_the_fly_gpu

    device = configure_runtime(dict(device=args.device, threads=args.threads, seed=123))
    if args.attention_only:
        attention_smoke(device)
        return
    damping = 0.001
    errors = {}
    for m, p in ((7, 11), (11, 7)):
        J = torch.randn((m, p), dtype=torch.float64, device=device)
        E = torch.randn(m, dtype=torch.float64, device=device)
        primal, _ = ridge_direction(J, E, damping, "parameter")
        dual, _ = ridge_direction(J, E, damping, "sample")
        torch.testing.assert_close(primal, dual, atol=2e-10, rtol=2e-10)
        A = (J - J.mean(0)) / math.sqrt(m)
        b = (E - E.mean()) * (2 / math.sqrt(m))
        identity = torch.eye(p, dtype=torch.float64, device=device)
        independent = torch.linalg.solve(A.T @ A + damping * identity, A.T @ b)
        torch.testing.assert_close(primal, independent, atol=2e-10, rtol=2e-10)
        shifted, _ = ridge_direction(J + 17, E - 9, damping)
        repeated, _ = ridge_direction(J.repeat((3, 1)), E.repeat(3), damping)
        torch.testing.assert_close(primal, shifted, atol=2e-10, rtol=2e-10)
        torch.testing.assert_close(primal, repeated, atol=2e-10, rtol=2e-10)
        errors[f"{m}x{p}"] = (primal - dual).abs().max().item()
    assert workload(128, 4096, 64, 5)["jacobian_memory_throttled"]
    try:
        workload(128, 4096, 64, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("workspace too small for one row was accepted")

    sector = FixedSector(6, 6, 3.275)
    states = torch.from_numpy(sector.basis).to(device=device, dtype=torch.long)
    input_states = one_hot(states, 7)
    h = torch.from_numpy(sector.hamiltonian.toarray()).to(device=device)
    model = build_model("FCNet", 6, 6, 8).to(device).eval()
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()
    local = local_energy_on_the_fly_gpu(states, model, 6, 6, t=1.0, U=1.6375, evaluation_batch_size=64)
    torch.testing.assert_close(local.double(), h.sum(1), atol=1e-5, rtol=1e-6)
    # At the uniform state, use normalized exp(log amplitude) for the exact
    # derivative. Every configuration then has equal Monte Carlo weight.
    logs = model(input_states).double()
    psi = torch.softmax(2 * logs, dim=0).sqrt()
    energy = psi @ (h @ psi)
    exact_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(energy, tuple(model.parameters()))])
    J = compute_log_psi_jacobian_vmap(model, input_states, evaluation_batch_size=16).double()
    covariance_force = 2 * (J - J.mean(0)).T @ (local.double() - local.double().mean()) / len(local)
    torch.testing.assert_close(covariance_force, exact_gradient.double(), atol=1e-6, rtol=1e-5)
    direction, residual = ridge_direction(J, local, damping)
    assert (exact_gradient.double() @ direction).item() > 0
    print(json.dumps({"device": str(device), "primal_dual_max_error": errors,
                      "physics_force_max_error": (exact_gradient.double() - covariance_force).abs().max().item(),
                      "physics_solve_residual": residual, "physics_energy": energy.item()}), flush=True)


if __name__ == "__main__":
    main()
