#!/usr/bin/env python3
"""Paired CPU FCNet diagnostic: sampled native versus exact weighted gradients.

Fixed 2x3, N=6, cutoff=4, t=1, U=16.74, mu=0, periodic unique bonds.
At most 500 Adam updates; this is one paired seed diagnostic, not a benchmark
ranking or a generalization/accuracy claim. No dense Hamiltonian is allocated.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_diagonalization import exact_ground_state
from models import BOND_CONVENTION, HAMILTONIAN_CONVENTION, build_model, parameter_count, save_model
from vmc import generate_initial_states, local_energy, native_vmc_update, occupancy_tensor_to_onehot, sample


def rayleigh(model, features, H):
    logs = model(features)
    log_norm = 0.5 * torch.logsumexp(2 * logs, dim=0)
    psi = torch.exp(logs - log_norm)
    applied = torch.sparse.mm(H, psi[:, None]).squeeze(1)
    return torch.dot(psi, applied), psi, applied, logs, log_norm


def exact_diagnostics(model, features, H, ground_state, ground_energy):
    energy, psi, applied, _, log_norm = rayleigh(model, features, H)
    gradients = torch.autograd.grad(energy, tuple(model.parameters()))
    gradient_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
    with torch.no_grad():
        norm_log = log_norm.item()
        return {
            "expectation": "exact_full_sector_probability_weighted",
            "energy": energy.item(), "energy_per_site": energy.item() / 6,
            "energy_error_to_ed": energy.item() - ground_energy,
            "energy_variance": (applied - energy * psi).square().sum().item(),
            "amplitude_overlap_abs_unsquared": torch.dot(ground_state, psi).abs().item(),
            "exact_gradient_norm": gradient_norm.item(),
            "normalized_full_basis_norm": psi.norm().item(),
            "unnormalized_full_basis_log_norm": norm_log,
            "unnormalized_full_basis_norm": math.exp(norm_log) if norm_log < 709 else None,
        }


def initial_crosscheck(model, features, states, H):
    energy, psi, applied, logs, _ = rayleigh(model, features, H)
    parameters = tuple(model.parameters())
    direct = torch.autograd.grad(energy, parameters, retain_graph=True)
    local = local_energy(model, states, evaluation_batch_size=1024)
    torch.testing.assert_close(local, (applied / psi).detach(), rtol=1e-11, atol=1e-11)
    covariance_loss = 2 * (psi.detach().square() * (local - energy.detach()) * logs).sum()
    covariance = torch.autograd.grad(covariance_loss, parameters)
    direct_flat = torch.cat([gradient.flatten() for gradient in direct])
    covariance_flat = torch.cat([gradient.flatten() for gradient in covariance])
    torch.testing.assert_close(direct_flat, covariance_flat, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(psi.detach().square().sum(), torch.tensor(1.0), rtol=1e-12, atol=1e-12)
    return {
        "status": "passed", "normalized_full_basis_norm": psi.detach().norm().item(),
        "local_energy_vs_sparse_Hpsi_over_psi_max_abs": (local - (applied / psi).detach()).abs().max().item(),
        "weighted_covariance_vs_direct_rayleigh_gradient_max_abs": (direct_flat - covariance_flat).abs().max().item(),
        "direct_gradient_norm": direct_flat.norm().item(),
        "weighted_covariance_gradient_norm": covariance_flat.norm().item(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=500, help="1..500; use 2 for a smoke")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("paired-training-%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 500:
        parser.error("updates must be in [1, 500]")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("tag must be a simple name without path separators")
    results = ROOT / "benchmarks" / "results" / args.tag
    outputs = ROOT / "outputs" / args.tag
    for path, parent in ((results, ROOT / "benchmarks" / "results"), (outputs, ROOT / "outputs")):
        if path.exists() or not path.resolve().is_relative_to(parent):
            parser.error("run directories must not exist and must stay under their designated roots")
    recorded_at = datetime.now(timezone.utc).isoformat()
    sources = [ROOT / name for name in ("lattice.py", "exact_diagonalization.py", "models.py", "vmc.py", "train.py")]
    sources += sorted((ROOT / "benchmarks").glob("*.py"))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    results.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(123)
    report = {
        "status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "source_hashes_recorded_at": recorded_at, "source_sha256_at_start": hashes,
        "settings": {"Lx": 2, "Ly": 3, "Ntot": 6, "Nmax": 4, "t": 1.0, "U": 16.74, "mu": 0.0,
                     "pbc": True, "arch": "FCNet", "seed": 123, "width": 16, "dtype": "float64", "device": "cpu",
                     "threads": 1, "optimizer": "Adam", "lr": 0.001, "updates": args.updates,
                     "walkers": 64, "retained_steps": 16, "burnin_per_update": 32, "evaluation_batch_size": 1024,
                     "diagnostic_max_dimension": 1000, "max_sparse_bytes": 512 * 1024**2,
                     "tag": args.tag, "save_checkpoints": args.save_checkpoints,
                     "initialization": "float64 FCNet initialization, then deepcopy before creating either optimizer"},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "hamiltonian_convention": HAMILTONIAN_CONVENTION, "bond_convention": BOND_CONVENTION,
        "interpretation": "One paired initialization. MC updates estimate expectations from correlated walkers; exact updates use the full-sector probability distribution. All checkpoint diagnostics use exact expectations. No rankings, confidence intervals or generalization claims.",
        "measurements": [], "checkpoints": {},
    }
    summary_path = results / "summary.json"
    with (results / "run.log").open("w", buffering=1) as log:
        def emit(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        def persist():
            summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

        persist()
        started = time.perf_counter()
        try:
            # The strict diagnostic dimension guard is passed before basis or
            # sparse assembly allocation. No later path converts H to dense.
            ground_energy, ground_state, basis, matrix = exact_ground_state(
                2, 3, Ntot=6, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True,
                max_dimension=1000, max_sparse_bytes=512 * 1024**2,
            )
            if len(basis) > 1000:
                raise RuntimeError("diagnostic dimension guard violated")
            coo = matrix.tocoo()
            indices = torch.from_numpy(np.stack((coo.row, coo.col)).astype(np.int64))
            H = torch.sparse_coo_tensor(
                indices, torch.from_numpy(coo.data.copy()), coo.shape, check_invariants=True
            ).coalesce()
            states = torch.tensor(basis, dtype=torch.long)
            features = occupancy_tensor_to_onehot(states, 4, torch.float64)
            ground_state = torch.from_numpy(ground_state)
            report["ed_reference"] = {"dimension": len(basis), "energy": ground_energy, "energy_per_site": ground_energy / 6,
                                      "nnz": matrix.nnz, "torch_operator": "sparse_coo_float64"}
            sampled_model = build_model("FCNet", 2, 3, Nmax=4, pbc=True, hidden_dim=16).double().eval()
            exact_model = copy.deepcopy(sampled_model)
            for sampled_parameter, exact_parameter in zip(sampled_model.parameters(), exact_model.parameters()):
                torch.testing.assert_close(sampled_parameter, exact_parameter, rtol=0, atol=0)
            report["parameter_count"] = parameter_count(sampled_model)
            report["initial_gradient_crosscheck"] = initial_crosscheck(sampled_model, features, states, H)
            optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in (sampled_model, exact_model)]
            walkers = generate_initial_states(6, 6, 4, 64, "cpu")
            checkpoints = {0, 100, 250, 500, args.updates}
            for update in range(args.updates + 1):
                if update in checkpoints:
                    measurement = {"update": update,
                                   "sampled_native": exact_diagnostics(sampled_model, features, H, ground_state, ground_energy),
                                   "exact_probability_weighted": exact_diagnostics(exact_model, features, H, ground_state, ground_energy),
                                   "elapsed_seconds": time.perf_counter() - started}
                    report["measurements"].append(measurement)
                    persist()
                    for name in ("sampled_native", "exact_probability_weighted"):
                        value = measurement[name]
                        emit(f"update={update} {name}: exact_E={value['energy']:.12f} variance={value['energy_variance']:.8g} "
                             f"overlap={value['amplitude_overlap_abs_unsquared']:.9f} exact_gradient_norm={value['exact_gradient_norm']:.8g}")
                if update == args.updates:
                    break
                samples, walkers, diagnostics = sample(sampled_model, walkers, 16, 32, 1024)
                energies = local_energy(sampled_model, samples, evaluation_batch_size=1024)
                native_vmc_update(sampled_model, samples, energies, optimizers[0], 1024)
                report["last_mc_update_measurement"] = {
                    "used_for_update": update + 1, "timing": "before_update",
                    "expectation": "Monte_Carlo_unweighted_sample_mean", "energy": energies.mean().item(),
                    "local_energy_variance": energies.var(unbiased=False).item(), "sampling": diagnostics,
                }
                optimizers[1].zero_grad(set_to_none=True)
                energy, _, _, _, _ = rayleigh(exact_model, features, H)
                energy.backward()
                if any(not torch.isfinite(p.grad).all() for p in exact_model.parameters()):
                    raise RuntimeError("non-finite exact gradient")
                optimizers[1].step()
                if any(not torch.isfinite(p).all() for model in (sampled_model, exact_model) for p in model.parameters()):
                    raise RuntimeError("non-finite model parameters")
                if (update + 1) % 25 == 0:
                    emit(f"completed_updates={update + 1} elapsed_seconds={time.perf_counter() - started:.3f}")
            if args.save_checkpoints:
                for name, model in (("sampled_native", sampled_model), ("exact_probability_weighted", exact_model)):
                    path = save_model(outputs / f"{name}.pt", model, Ntot=6, t=1.0, U=16.74, mu=0.0)
                    report["checkpoints"][name] = str(path.relative_to(ROOT))
            report["status"] = "completed"
            report["wall_seconds_including_ed_and_diagnostics"] = time.perf_counter() - started
            persist()
        except Exception as error:
            report.update({"status": "failed", "failure": {"type": type(error).__name__, "message": str(error)}})
            persist()
            emit(f"failed: {type(error).__name__}: {error}")
            raise
        emit(f"diagnostic record: {summary_path}")
    return report


if __name__ == "__main__":
    main()
