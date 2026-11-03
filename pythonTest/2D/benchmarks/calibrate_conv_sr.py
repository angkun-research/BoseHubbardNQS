#!/usr/bin/env python3
"""One-batch CPU float64 SR calibration against full-sector 3x3-cylinder ED.

Run from repo root; paths are relative to pythonTest/2D/ unless absolute:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u \
    pythonTest/2D/benchmarks/calibrate_conv_sr.py \
    --checkpoint outputs/2026-10-31-3x3-cylinder-supervised/Conv2DNet-seed123.pt \
    --output benchmarks/results/conv-sr-calibration

Exactly one MC batch (64 walkers, 16 retained steps, thin32, burn2048) and one
default-API Jacobian are used. Each tau/lr case resets identical weights and
takes ONE fresh plain SGD step. Full-sector evaluations are deterministic;
they do not optimize the model. No checkpoint is overwritten or saved.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from exact_diagonalization import exact_ground_state
from models import load_model
from vmc import (check_sr_budget, compute_log_psi_jacobian, generate_initial_states,
                 local_energy, occupancy_tensor_to_onehot, sample, sr_gradient)


@torch.no_grad()
def exact_metrics(model, basis, H, ed_state, ed_energy):
    logs = torch.cat([model(occupancy_tensor_to_onehot(
        torch.as_tensor(block, dtype=torch.long), model.Nmax, torch.float64)).reshape(-1)
        for block in (basis[start:start + 512] for start in range(0, len(basis), 512))])
    if not torch.isfinite(logs).all():
        raise RuntimeError("nonfinite full-sector log amplitudes")
    psi = torch.exp(logs - .5 * torch.logsumexp(2 * logs, 0)).numpy()
    applied = H @ psi
    energy = float(psi @ applied)
    overlap = float(abs(ed_state @ psi))
    return {"energy": energy, "energy_per_site": energy / model.n_sites,
            "energy_error_to_ed": energy - ed_energy,
            "energy_variance": float(np.linalg.norm(applied - energy * psi)**2),
            "amplitude_overlap_abs_unsquared": overlap, "amplitude_overlap_squared": overlap**2,
            "normalized_sector_norm": float(np.linalg.norm(psi))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory under benchmarks/results/")
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args(argv)
    checkpoint, output = (ROOT / args.checkpoint).resolve(), (ROOT / args.output).resolve()
    if not checkpoint.is_relative_to(ROOT) or not checkpoint.is_file():
        parser.error("checkpoint must exist under 2D/")
    if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
        parser.error("output must be a new directory under benchmarks/results/")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0,2**63)")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    source_names = ["benchmarks/calibrate_conv_sr.py", "models.py", "lattice.py", "exact_diagonalization.py", "vmc.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, metadata = load_model(checkpoint, device="cpu")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("checkpoint changed during loading")
    required = dict(arch="Conv2DNet", Lx=3, Ly=3, Ntot=9, Nmax=4, t=1., U=16.74, mu=0., pbc_x=False, pbc_y=True)
    if any(metadata.get(k) != v for k, v in required.items()) or metadata["architecture_settings"]["hidden_dim"] != 32:
        parser.error("requires the specified Conv32,3x3,N9,Nmax4,t1,U16.74,mu0 cylinder checkpoint")
    model = model.double().eval()
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    budget = check_sr_budget(model, 1024)
    ed_energy, ed_state, basis, H = exact_ground_state(3, 3, 9, 4, 1., 16.74, 0., pbc_x=False, pbc_y=True)
    baseline = exact_metrics(model, basis, H, ed_state, ed_energy)
    initial = generate_initial_states(9, 9, 4, 64, "cpu")
    samples, walkers, diagnostics = sample(model, initial, 16, 2048, 512, proposal_steps_per_sample=32)
    energies = local_energy(model, samples, 1., 16.74, 0., 512).double()
    jac_started = time.perf_counter()
    J = compute_log_psi_jacobian(model, samples)
    jac_seconds = time.perf_counter() - jac_started
    Jc = J - J.mean(0)
    results = []
    for tau in (.05, .01, .005, .001):
        direction, sr = sr_gradient(J, energies, tau, return_diagnostics=True)
        log_direction = Jc @ direction
        rms_direction = float(log_direction.square().mean().sqrt())
        for lr in (.01, .03, .1):
            model.load_state_dict(original)
            optimizer = torch.optim.SGD(parameters, lr=lr)
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                parameter.grad = direction[offset:offset + count].reshape_as(parameter).to(parameter.dtype).clone()
                offset += count
            optimizer.step()
            metrics = exact_metrics(model, basis, H, ed_state, ed_energy)
            result = {"tau": tau, "lr": lr, "sr": sr,
                "rms_centered_linearized_log_step": lr * rms_direction,
                "parameter_step_norm": lr * sr["direction_norm"],
                "sampled_linearized_energy_change": float(-2 * lr * ((energies - energies.mean()) * log_direction).mean()),
                "exact_metrics_after": metrics,
                "exact_energy_change": metrics["energy"] - baseline["energy"],
                "exact_variance_change": metrics["energy_variance"] - baseline["energy_variance"]}
            results.append(result)
            print(f"tau={tau} lr={lr} exact_dE={result['exact_energy_change']:.8g} "
                  f"variance={metrics['energy_variance']:.8g} rms_log_step={lr*rms_direction:.8g}", flush=True)
    model.load_state_dict(original)
    unchanged = exact_metrics(model, basis, H, ed_state, ed_energy)
    if unchanged != baseline:
        raise RuntimeError("restored weights do not reproduce baseline metrics")
    settings = {**required, "hidden_dim": 32, "seed": args.seed, "dtype": "float64", "device": "cpu",
        "walkers": 64, "retained_steps_per_walker": 16, "thin": 32, "burnin_proposals_per_walker": 2048,
        "M": 1024, "evaluation_batch_size": 512, "jacobian_batch_size": "stable API default1",
        "taus": [.05, .01, .005, .001], "learning_rates": [.01, .03, .1],
        "updates_per_case": 1, "optimizer": "fresh torch.optim.SGD, momentum0,weight_decay0",
        "ED_limits": "exact_ground_state default max_dimension20000,max_sparse_bytes512MiB"}
    record = {"status": "completed", "kind": "single_batch_SR_exact_ED_calibration",
        "recorded_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256_at_start": checkpoint_hash,
        "checkpoint_metadata": metadata, "source_sha256_at_start": hashes,
        "source_sha256_at_finish": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version},
        "ED": {"dimension": len(basis), "nnz": H.nnz, "energy": ed_energy,
               "residual_norm": float(np.linalg.norm(H @ ed_state - ed_energy * ed_state))},
        "before": baseline, "sampling": diagnostics, "sampled_energy": float(energies.mean()),
        "sampled_local_energy_variance": float(energies.var(unbiased=False)), "SR_budget": budget,
        "jacobian_shape": list(J.shape), "jacobian_seconds": jac_seconds, "results": results,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": "One frozen MC batch and identical initial weights; one plain SGD update per grid point using one shared Jacobian. Exact full-sector physical metrics use raw ED, no target floor. Results diagnose this batch and checkpoint, not optimal large-system hyperparameters, capacity or convergence.",
        "linearized_log_step_definition": "lr*sqrt(mean((Jc@delta)^2)), Jc=J-mean_samples(J), delta=(S+tau I)^-1 f, S=Jc.T Jc/M,f=Jc.T Ec/M; no extra2 in SR direction"}
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("source checkpoint was replaced during calibration")
    serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "frozen-batch.npz"
    np.savez_compressed(raw, occupations=samples.numpy().reshape(16, 64, 9), local_energy=energies.numpy().reshape(16, 64), final_walkers=walkers.numpy())
    record["frozen_batch_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    record["command"] = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    (output / "report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Saved calibration: {output}; runtime={record['runtime_seconds']:.3f}s", flush=True)
    return record


if __name__ == "__main__":
    main()
