#!/usr/bin/env python3
"""Bounded float64 CPU/optional-CUDA verification for all five architectures.

This is a correctness smoke, not a throughput or convergence benchmark. CUDA
comparisons use identical weights, inputs and supplied energies, not independent
random Monte Carlo trajectories. Relative JSON output paths start in benchmarks/.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_diagonalization import build_bose_hubbard_hamiltonian
from models import BASELINE_ARCHITECTURES, build_model, parameter_count
from vmc import (
    SR_CONVENTION, check_sr_budget, local_energy, native_vmc_update,
    occupancy_tensor_to_onehot, sr_update,
)


def _compare(actual, expected, rtol, atol):
    actual = actual.detach().cpu()
    expected = torch.as_tensor(expected).detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return (actual - expected).abs().max().item()


def _native_step(model, states, energies, size):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    native_vmc_update(model, states, energies, optimizer, size)
    parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    gradients = torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()])
    return parameters, gradients


def _sr_step(model, states, energies):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    direction = sr_update(model, states, energies, optimizer, tau=0.1)
    parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    return direction.detach().cpu(), parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="skip CUDA even when available")
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--cli-smoke", action="store_true", help="also run tiny native/SR CLI subprocesses")
    parser.add_argument("--output", type=Path, default=Path("results/verify_gpu.json"))
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**63 or any(not np.isfinite(v) or v <= 0 for v in (args.rtol, args.atol)):
        parser.error("seed must be in [0, 2**63); tolerances must be finite and positive")
    output = (ROOT / "benchmarks" / args.output).resolve()
    if not output.is_relative_to(ROOT / "benchmarks" / "results") or output.is_dir():
        parser.error("verification JSON must stay under benchmarks/results/")
    cuda = None
    if not args.cpu_only and torch.cuda.is_available():
        try:
            cuda = torch.device(args.cuda_device)
            if cuda.type != "cuda" or (cuda.index or 0) >= torch.cuda.device_count():
                raise ValueError("unavailable CUDA device")
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
    torch.manual_seed(args.seed)
    # Fixed tiny geometry keeps the full basis, all local-energy checks, and
    # float64 dense SR workspace small even for the attention architecture.
    H, basis = build_bose_hubbard_hamiltonian(2, 2, Ntot=2, Nmax=2)
    states = torch.tensor(basis, dtype=torch.long)
    sr_states = states[:4]
    report = {
        "kind": "correctness_smoke_not_accuracy_or_timing_benchmark",
        "settings": {**vars(args), "output": str(output), "geometry": [2, 2], "Ntot": 2, "Nmax": 2,
                     "t": 1.0, "U": 16.74, "mu": 0.0, "pbc": True, "width": 2,
                     "dtype": "float64", "evaluation_batch_size": 3,
                     "native_optimizer": "SGD", "native_lr": 0.001,
                     "sr_optimizer": "SGD", "sr_lr": 0.001, "sr_tau": 0.1, "sr_samples": 4},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "scipy": scipy.__version__, "cuda_runtime": torch.version.cuda,
                        "threads": torch.get_num_threads(),
                        "cuda_device": torch.cuda.get_device_name(cuda) if cuda is not None else None},
        "cuda_status": "enabled" if cuda is not None else ("skipped_cpu_only" if args.cpu_only else "skipped_unavailable"),
        "sr_convention": SR_CONVENTION, "architectures": [], "cli_smokes": [],
    }
    for arch in BASELINE_ARCHITECTURES:
        settings = {} if arch == "JastrowNet" else {"n_hidden" if arch == "RBMNet" else "hidden_dim": 2}
        model = build_model(arch, 2, 2, Nmax=2, **settings).double().eval()
        # Nonconstant Jastrow weights exercise pair gathers and gradients.
        if arch == "JastrowNet":
            with torch.no_grad():
                model.onsite.normal_(std=0.02)
                model.pair.normal_(std=0.02)
        with torch.no_grad():
            logs = model(occupancy_tensor_to_onehot(states, 2, torch.float64))
        amplitudes = (logs - logs.max()).exp().numpy()
        reference = torch.from_numpy((H @ amplitudes) / amplitudes)
        energies = local_energy(model, states, evaluation_batch_size=3)
        record = {"arch": arch, "parameter_count": parameter_count(model),
                  "architecture_settings": dict(model.architecture_settings),
                  "sr_budget": check_sr_budget(model, len(sr_states)),
                  "cpu_local_energy_vs_ed_max_abs": _compare(energies, reference, args.rtol, args.atol)}
        # Native chunking is checked on CPU even when CUDA is unavailable.
        native_cpu, gradients_cpu = _native_step(copy.deepcopy(model), states, energies, 3)
        native_unchunked, gradients_unchunked = _native_step(copy.deepcopy(model), states, energies, len(states))
        record["native_chunked_vs_unchunked_max_abs"] = _compare(native_cpu, native_unchunked, args.rtol, args.atol)
        record["native_gradients_chunked_vs_unchunked_max_abs"] = _compare(
            gradients_cpu, gradients_unchunked, args.rtol, args.atol
        )
        sr_cpu, sr_parameters_cpu = _sr_step(copy.deepcopy(model), sr_states, energies[:4])
        record["sr_direction_norm"] = sr_cpu.norm().item()
        if cuda is not None:
            gpu_model = copy.deepcopy(model).to(cuda)
            gpu_states = states.to(cuda)
            with torch.no_grad():
                gpu_logs = gpu_model(occupancy_tensor_to_onehot(gpu_states, 2, torch.float64))
            gpu_energies = local_energy(gpu_model, gpu_states, evaluation_batch_size=3)
            record["cuda_forward_vs_cpu_max_abs"] = _compare(gpu_logs, logs, args.rtol, args.atol)
            record["cuda_local_energy_vs_ed_max_abs"] = _compare(gpu_energies, reference, args.rtol, args.atol)
            native_gpu, gradients_gpu = _native_step(copy.deepcopy(gpu_model), gpu_states, energies.to(cuda), 3)
            record["cuda_native_step_vs_cpu_max_abs"] = _compare(native_gpu, native_cpu, args.rtol, args.atol)
            record["cuda_native_gradients_vs_cpu_max_abs"] = _compare(gradients_gpu, gradients_cpu, args.rtol, args.atol)
            sr_gpu, sr_parameters_gpu = _sr_step(copy.deepcopy(gpu_model), gpu_states[:4], energies[:4].to(cuda))
            record["cuda_sr_direction_vs_cpu_max_abs"] = _compare(sr_gpu, sr_cpu, args.rtol, args.atol)
            record["cuda_sr_step_vs_cpu_max_abs"] = _compare(sr_parameters_gpu, sr_parameters_cpu, args.rtol, args.atol)
        report["architectures"].append(record)
        print(f"{arch}: CPU/ED and native/SR checks passed" + ("; CUDA matched" if cuda is not None else "; CUDA skipped"))
    if args.cli_smoke:
        tag = datetime.now(timezone.utc).strftime("verify-%Y%m%dT%H%M%S%fZ")
        devices = ["cpu"] + ([str(cuda)] if cuda is not None else [])
        for device in devices:
            for method in ("native", "sr"):
                name = f"{device.replace(':', '-')}-{method}"
                checkpoint = ROOT / "outputs" / tag / f"{name}.pt"
                command = [sys.executable, str(ROOT / "train.py"), "--arch", "TransformerNet", "--hidden-dim", "2",
                           "--Lx", "2", "--Ly", "1", "--Nmax", "2", "--epochs", "1", "--walkers", "2",
                           "--steps", "2", "--burnin", "1", "--dtype", "float64", "--device", device,
                           "--update-method", method, "--evaluation-batch-size", "2", "--seed", str(args.seed),
                           "--output", str(checkpoint)]
                started = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, timeout=60,
                                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                log = output.parent / f"{tag}-{name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(completed.stdout + completed.stderr)
                report["cli_smokes"].append({"command": command, "returncode": completed.returncode,
                                             "subprocess_wall_seconds": time.perf_counter() - started,
                                             "log": str(log.relative_to(ROOT)), "checkpoint": str(checkpoint.relative_to(ROOT))})
                if completed.returncode:
                    raise RuntimeError(f"CLI smoke failed; see {log}")
    report["status"] = "passed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"verification record: {output}")
    return report


if __name__ == "__main__":
    main()
