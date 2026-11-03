#!/usr/bin/env python3
"""Bounded deterministic energy SR on the existing 3x3-cylinder Jastrow model.

No MC, GPU, new ansatz, target flooring or production checkpoint overwrite.
CPU float64 fixed table features are verified against model forwards. Exact
probabilities weight S=Jc.T diag(p) Jc and f=Jc.T(p*(Eloc-E)); a native plain
torch.optim.SGD step assigns delta=solve(S+tau I,f), with no extra factor2.

From repository root (paths relative to pythonTest/2D/):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
    pythonTest/2D/benchmarks/check_exact_jastrow_sr.py \
    --output benchmarks/results/exact-jastrow-sr
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
from scipy.special import logsumexp
import torch

from exact_diagonalization import exact_ground_state
from models import load_model


def table_features(model, basis):
    """Actual ordered-table derivatives for displacement or reflection sharing."""
    if list(dict(model.named_parameters())) != ["onsite", "pair"]:
        raise ValueError("unexpected Jastrow parameter order")
    basis = np.asarray(basis, dtype=np.int64)
    channels = model.Nmax + 1
    matrix = np.zeros((len(basis), sum(p.numel() for p in model.parameters())), dtype=np.float64)
    rows = np.arange(len(basis))
    orbit = getattr(model, "pair_sharing", "displacement") == "reflection_orbit"
    onsite_count = model.onsite.numel()
    for site in range(model.n_sites):
        group = int(model.onsite_classes[site]) if orbit else 0
        matrix[rows, group * channels + basis[:, site]] += 1
    for (i, j), cls in zip(model.pair_sites.T.tolist(), model.pair_classes.tolist()):
        a, b = basis[:, i], basis[:, j]
        offset = onsite_count + cls * channels**2
        if orbit and not bool(model.symmetric_classes[cls]):
            matrix[rows, offset + a * channels + b] += 1
        else:
            matrix[rows, offset + a * channels + b] += .5
            matrix[rows, offset + b * channels + a] += .5
    return matrix


def flat_parameters(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).numpy().copy()


def verify_features(model, basis, matrix):
    theta = flat_parameters(model)
    with torch.no_grad():
        logs = torch.cat([model(torch.nn.functional.one_hot(torch.as_tensor(block), model.Nmax + 1).double())
                          for block in np.array_split(np.asarray(basis), max(1, (len(basis) + 511) // 512))]).numpy()
    error = float(abs(logs - matrix @ theta).max())
    if error > 1e-10:
        raise RuntimeError(f"fixed feature matrix disagrees with model: {error}")
    return error


def exact_state(theta, matrix, H, ed):
    logs = matrix @ theta
    psi = np.exp(logs - .5 * logsumexp(2 * logs))
    if not np.isfinite(psi).all() or (psi <= 0).any():
        raise RuntimeError("nonfinite or underflowed amplitudes; no floor applied")
    p = psi**2
    applied = H @ psi
    local = applied / psi
    energy = float(psi @ applied)
    if not np.isfinite(local).all():
        raise RuntimeError("nonfinite exact local energy")
    overlap = float(abs(ed @ psi))
    return p, local, {"energy": energy, "energy_variance": float(np.linalg.norm(applied - energy * psi)**2),
                      "amplitude_overlap_abs_unsquared": overlap, "amplitude_overlap_squared": overlap**2,
                      "normalized_sector_norm": float(np.linalg.norm(psi)), "minimum_amplitude": float(psi.min())}


def weighted_sr(matrix, p, local, energy, tau):
    centered = matrix - p @ matrix
    force = centered.T @ (p * (local - energy))
    covariance = centered.T @ (p[:, None] * centered)
    covariance[np.diag_indices_from(covariance)] += tau
    direction = np.linalg.solve(covariance, force)
    if not np.isfinite(direction).all():
        raise RuntimeError("nonfinite exact SR direction")
    return direction, force, centered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/2026-10-31-jastrow-cylinder-infidelity-followup/JastrowNet-infidelity-best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--tau", type=float, default=.001)
    parser.add_argument("--max-seconds", type=float, default=120., help="stop after completed update; includes setup")
    args = parser.parse_args(argv)
    checkpoint, output = (ROOT / args.checkpoint).resolve(), (ROOT / args.output).resolve()
    if not checkpoint.is_relative_to(ROOT) or not checkpoint.is_file():
        parser.error("checkpoint must exist under 2D/")
    if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
        parser.error("output must be a new directory under benchmarks/results/")
    if not 1 <= args.updates <= 500 or not all(np.isfinite(v) and v > 0 for v in (args.lr, args.tau, args.max_seconds)) or args.max_seconds > 180:
        parser.error("require 1..500 updates, finite positive lr/tau/time and time<=180s")
    started = time.perf_counter()
    torch.set_num_threads(1)
    names = ["models.py", "lattice.py", "exact_diagonalization.py", "benchmarks/check_exact_jastrow_sr.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, metadata = load_model(checkpoint, device="cpu")
    required = dict(arch="JastrowNet", Lx=3, Ly=3, Ntot=9, Nmax=4, t=1., U=16.74, mu=0., pbc_x=False, pbc_y=True)
    if any(metadata[k] != value for k, value in required.items()):
        parser.error("requires existing 3x3,N9,Nmax4,t1,U16.74,mu0 open-x/periodic-y Jastrow")
    model = model.double().eval()
    energy_ed, ed, basis, H = exact_ground_state(3, 3, 9, 4, 1., 16.74, 0., pbc_x=False, pbc_y=True)
    matrix = table_features(model, basis)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_error = verify_features(model, basis, matrix)
    rng = torch.Generator().manual_seed(812)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=rng, dtype=torch.float64) * .1)
    random_error = verify_features(model, basis, matrix)
    model.load_state_dict(original)
    initial_theta = flat_parameters(model)
    p, local, metrics = exact_state(initial_theta, matrix, H, ed)
    direction, force, _ = weighted_sr(matrix, p, local, metrics["energy"], args.tau)
    # Independent sparse Rayleigh/autograd gradient includes normalization derivative.
    x = torch.nn.functional.one_hot(torch.as_tensor(np.asarray(basis)), 5).double()
    logs = model(x)
    psi = (logs - .5 * torch.logsumexp(2 * logs, 0)).exp()
    coo = H.tocoo()
    sparse = torch.sparse_coo_tensor(np.stack([coo.row, coo.col]), coo.data, H.shape, dtype=torch.float64, check_invariants=True)
    rayleigh = psi @ torch.sparse.mm(sparse, psi[:, None])[:, 0]
    grads = torch.autograd.grad(rayleigh, tuple(model.parameters()))
    autograd = torch.cat([g.flatten() for g in grads]).detach().numpy()
    grad_error = float(abs(autograd - 2 * force).max())
    if grad_error > 1e-9:
        raise RuntimeError(f"exact energy gradient mismatch: {grad_error}")
    del x, logs, psi, sparse, rayleigh, grads
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    trajectory = []
    stop = "fixed update budget"
    for update in range(args.updates + 1):
        p, local, metrics = exact_state(flat_parameters(model), matrix, H, ed)
        direction, force, centered = weighted_sr(matrix, p, local, metrics["energy"], args.tau)
        row = {"update": update, **metrics, "energy_error_to_ed": metrics["energy"] - energy_ed,
               "relative_energy_error": (metrics["energy"] - energy_ed) / abs(energy_ed),
               "exact_energy_gradient_norm": float(2 * np.linalg.norm(force)),
               "SR_direction_norm": float(np.linalg.norm(direction)),
               "next_linearized_log_step_RMS": float(args.lr * np.sqrt(p @ (centered @ direction)**2))}
        trajectory.append(row)
        if update % 25 == 0 or update == args.updates:
            print(f"update={update} E={row['energy']:.12f} gap%={100*row['relative_energy_error']:.6f} "
                  f"F={row['amplitude_overlap_abs_unsquared']:.10f} var={row['energy_variance']:.7g} grad={row['exact_energy_gradient_norm']:.7g}", flush=True)
        if update == args.updates:
            break
        if time.perf_counter() - started > args.max_seconds:
            stop = "wall-time guard after completed update"
            break
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.grad = torch.from_numpy(direction[offset:offset + count].copy()).reshape_as(parameter)
            offset += count
        optimizer.step()
    final_error = verify_features(model, basis, matrix)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("source checkpoint changed during diagnostic")
    record = {"status": "completed_bounded_diagnostic", "stop": stop,
        "recorded_at": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "settings": {**required, "pair_sharing": getattr(model, "pair_sharing", "displacement"),
                     "dtype": "float64", "device": "cpu", "updates_requested": args.updates,
                     "updates_completed": trajectory[-1]["update"], "lr": args.lr, "tau": args.tau,
                     "max_seconds": args.max_seconds, "optimizer": "native torch.optim.SGD, momentum0,weight_decay0", "MC_samples": 0},
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256_at_start": checksum, "checkpoint_metadata": metadata,
        "source_sha256_at_start": hashes, "source_sha256_at_finish": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        "ED": {"dimension": len(basis), "nnz": H.nnz, "energy": energy_ed, "residual_norm": float(np.linalg.norm(H @ ed - energy_ed * ed)), "guards": "default dimension20000 and estimated sparse512MiB"},
        "features": {"shape": list(matrix.shape), "sha256": hashlib.sha256(matrix.tobytes()).hexdigest(), "model_forward_max_abs_errors": [initial_error, random_error, final_error], "energy_autograd_vs_2force_max_abs": grad_error},
        "equations": "p=psi^2,psi=exp(logpsi-.5logsumexp(2logpsi)); Jc=J-EpJ,S=Jc.T diag(p) Jc,f=Jc.T(p*(Eloc-E)); (S+tau I)delta=f,SGD theta-=lr*delta; no extra2",
        "trajectory": trajectory, "energy_increase_count": sum(b['energy'] > a['energy'] + 1e-12 for a,b in zip(trajectory, trajectory[1:])),
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": "Local deterministic minimization within the checkpoint's Jastrow coefficient-sharing mode. Energy objective differs from supervised log-MSE/infidelity fits. Exact force removes MC noise; finite local trajectory is not global optimization or capacity proof and does not transfer a quantitative bound to4x4/6x4. Boundary sharing and general pairwise expressiveness remain distinct. No new model variant or production checkpoint created."}
    serialized = json.dumps(record, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    params = output / "parameters.npz"
    np.savez_compressed(params, initial_theta=initial_theta, final_theta=flat_parameters(model))
    record["parameters_sha256"] = hashlib.sha256(params.read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"saved={output}; runtime={record['runtime_seconds']:.3f}s; stop={stop}", flush=True)
    return record


if __name__ == "__main__":
    main()
