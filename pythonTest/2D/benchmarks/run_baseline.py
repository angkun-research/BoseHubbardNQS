#!/usr/bin/env python3
"""Run separate architecture/seed training subprocesses and guarded exact checks.

Defaults: periodic 2x3, U=16.74, t=1, Nmax=4; five architectures, three seeds.
No architecture ranking or confidence claims are inferred. Binary checkpoints
stay in outputs/; reproducible text records and subprocess logs stay in
benchmarks/results/. Use --skip-ed for larger geometries such as 4x4.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_diagonalization import DEFAULT_MAX_DIMENSION, DEFAULT_MAX_SPARSE_BYTES, exact_ground_state
from lattice import lattice_shape, add_boundary_arguments, resolve_cli_boundaries
from models import BASELINE_ARCHITECTURES, ARCH_BUILDERS, BOND_CONVENTION, HAMILTONIAN_CONVENTION, load_model, parameter_count
from vmc import occupancy_tensor_to_onehot, physical_parameters


def _relative(path):
    return str(path.relative_to(ROOT))


def _write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=3, help="default 3; set explicitly for other rectangles")
    parser.add_argument("--Ntot", type=int)
    parser.add_argument("--Nmax", type=int, default=4)
    add_boundary_arguments(parser)
    parser.add_argument("--arches", nargs="+", choices=list(ARCH_BUILDERS), default=list(BASELINE_ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--hidden-layers", "--depth", dest="hidden_layers", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--burnin", type=int, default=32)
    parser.add_argument("--thin", type=int, default=1, help="proposals per retained step; step count is unchanged")
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=16.74)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--skip-ed", action="store_true")
    parser.add_argument("--max-ed-dimension", type=int, default=DEFAULT_MAX_DIMENSION)
    parser.add_argument("--max-ed-bytes", type=int, default=DEFAULT_MAX_SPARSE_BYTES)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("baseline-%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args(argv)
    try:
        args.Lx, args.Ly = lattice_shape(args.Lx, args.Ly)
        resolve_cli_boundaries(args)
        args.Ntot = args.Lx * args.Ly if args.Ntot is None else args.Ntot
        if args.Nmax < 0 or not 0 <= args.Ntot <= args.Lx * args.Ly * args.Nmax:
            raise ValueError("invalid particle sector or cutoff")
        if any(getattr(args, key) <= 0 for key in ("epochs", "width", "hidden_layers", "walkers", "steps", "thin", "evaluation_batch_size",
                                                  "threads", "max_ed_dimension", "max_ed_bytes")) or args.burnin < 0:
            raise ValueError("training counts and resource limits must be positive; burnin must be nonnegative")
        if not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("lr must be finite and positive")
        args.t, args.U, args.mu = physical_parameters(args.t, args.U, args.mu)
        if any(not 0 <= seed < 2**63 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("seeds must be distinct and in [0, 2**63)")
        if len(set(args.arches)) != len(args.arches):
            raise ValueError("architectures must be distinct")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
            raise ValueError("run-name must be a simple name, without path separators")
        if args.device != "auto":
            device = torch.device(args.device)
            if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
                raise ValueError("device must be cpu, cuda[:index], or auto")
            if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
                raise ValueError("requested CUDA device is unavailable")
        for parent in (ROOT / "outputs", ROOT / "benchmarks" / "results"):
            target = (parent / args.run_name).resolve()
            if not target.is_relative_to(parent) or target.exists():
                raise ValueError("run directories must stay in their designated roots and must not already exist")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return args


@torch.no_grad()
def exact_metrics(model, basis, H, ground_energy, ground_state, batch_size):
    """Normalize positive amplitudes over this sector and compute exact moments.

    Evaluate saved rounded weights in float64 on CPU. This avoids Monte Carlo
    noise; it does not remove finite size, cutoff, or optimization errors.
    """
    model = model.cpu().double().eval()
    logs = []
    for start in range(0, len(basis), batch_size):
        states = torch.tensor(basis[start:start + batch_size], dtype=torch.long)
        logs.append(model(occupancy_tensor_to_onehot(states, model.Nmax, torch.float64)))
    logs = torch.cat(logs)
    if not torch.isfinite(logs).all():
        raise RuntimeError("non-finite full-sector log amplitudes")
    amplitudes = (logs - logs.max()).exp().numpy()
    amplitudes /= np.linalg.norm(amplitudes)
    applied = H @ amplitudes
    energy = float(amplitudes @ applied)
    # Residual norm avoids subtracting two large nearly equal second moments.
    variance = float(np.linalg.norm(applied - energy * amplitudes)**2)
    overlap = float(abs(np.vdot(ground_state, amplitudes)))
    return {"kind": "exact_full_sector_saved_weights_evaluated_float64_cpu", "dimension": len(basis),
            "energy": energy, "energy_per_site": energy / model.n_sites,
            "energy_error_to_ed": energy - ground_energy, "amplitude_overlap_abs": overlap,
            "overlap_squared": overlap**2, "energy_variance": variance}


def main(argv=None):
    args = parse_args(argv)
    source_hashes_recorded_at = datetime.now(timezone.utc).isoformat()
    source_paths = [ROOT / name for name in (
        "lattice.py", "exact_diagonalization.py", "models.py", "vmc.py", "train.py"
    )] + sorted((ROOT / "benchmarks").glob("*.py"))
    source_hashes = {_relative(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    torch.set_num_threads(args.threads)
    results_dir = ROOT / "benchmarks" / "results" / args.run_name
    outputs_dir = ROOT / "outputs" / args.run_name
    results_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)
    cuda_available = torch.cuda.is_available()
    environment = {
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": cuda_available,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],
        "threads": args.threads, "subprocess_thread_environment": {"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)},
    }
    report = {"status": "running", "expected_run_count": len(args.arches) * len(args.seeds),
              "source_hashes_recorded_at": source_hashes_recorded_at, "source_sha256_at_start": source_hashes,
              "settings": vars(args), "environment": environment, "runs": [],
              "hamiltonian_convention": HAMILTONIAN_CONVENTION, "bond_convention": BOND_CONVENTION,
              "claims": "Separate seed observations only; no confidence intervals or architecture ranking.",
              "timing": "Subprocess wall includes Python/import/startup and artifact writes. CLI internal timing is recorded separately; neither is steady-state throughput."}
    _write_json(results_dir / "summary.json", report)
    reference = None
    record = None
    stage = "exact_reference"
    try:
        if args.skip_ed:
            report["exact_reference"] = {"status": "skipped", "reason": "--skip-ed requested; no sector enumeration or ED"}
        else:
            started = time.perf_counter()
            try:
                reference = exact_ground_state(args.Lx, args.Ly, args.Ntot, args.Nmax, args.t, args.U, args.mu, args.pbc,
                                               max_dimension=args.max_ed_dimension, max_sparse_bytes=args.max_ed_bytes,
                                               pbc_x=args.pbc_x, pbc_y=args.pbc_y)
            except ValueError as error:
                if not str(error).startswith(("sector dimension ", "estimated sparse memory ",
                                              "sector counting work exceeds ")):
                    raise
                report["exact_reference"] = {"status": "skipped", "reason": f"ED resource preflight: {error}"}
            else:
                energy, ground_state, basis, H = reference
                report["exact_reference"] = {"status": "computed", "dimension": len(basis), "ground_energy": energy,
                                             "ground_energy_per_site": energy / (args.Lx * args.Ly),
                                             "amplitude_overlap_reference": "one normalized ED ground vector; degeneracies can make this overlap nonunique"}
            report["exact_reference"]["wall_seconds"] = time.perf_counter() - started
        print(f"exact evaluation: {report['exact_reference']}")
        _write_json(results_dir / "summary.json", report)
        child_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        for arch in args.arches:
            for seed in args.seeds:
                name = f"{arch}-seed{seed}"
                checkpoint = outputs_dir / f"{name}.pt"
                log = results_dir / f"{name}.log"
                command = [sys.executable, "-u", str(ROOT / "train.py"), "--arch", arch, "--Lx", str(args.Lx), "--Ly", str(args.Ly),
                           "--Ntot", str(args.Ntot), "--Nmax", str(args.Nmax), "--t", str(args.t), "--U", str(args.U), "--mu", str(args.mu),
                           "--hidden-dim", str(args.width), "--epochs", str(args.epochs), "--walkers", str(args.walkers), "--steps", str(args.steps),
                           "--burnin", str(args.burnin), "--thin", str(args.thin), "--seed", str(seed), "--lr", str(args.lr), "--device", args.device,
                           "--dtype", args.dtype, "--evaluation-batch-size", str(args.evaluation_batch_size), "--output", str(checkpoint),
                           "--hidden-layers", str(args.hidden_layers),
                           "--pbc-x" if args.pbc_x else "--no-pbc-x", "--pbc-y" if args.pbc_y else "--no-pbc-y"]
                if not args.pbc:
                    command.append("--no-pbc")
                record = {"status": "running", "arch": arch, "seed": seed, "command": command,
                          "checkpoint": _relative(checkpoint), "cli_sidecar": _relative(Path(str(checkpoint) + ".json")), "log": _relative(log)}
                report["runs"].append(record)
                stage = "training"
                _write_json(results_dir / f"{name}.json", record)
                _write_json(results_dir / "summary.json", report)
                started = time.perf_counter()
                try:
                    with log.open("w") as stream:
                        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=child_environment)
                    record["returncode"] = completed.returncode
                finally:
                    record["subprocess_wall_seconds"] = time.perf_counter() - started
                if completed.returncode:
                    raise RuntimeError(f"training failed; see {log}")
                stage = "artifact_loading"
                cli_report = json.loads(Path(str(checkpoint) + ".json").read_text())
                model, metadata = load_model(checkpoint)
                record.update({"checkpoint_metadata": metadata, "parameter_count": parameter_count(model),
                               "sampled_final_measurement": cli_report["final_measurement"],
                               "cli_internal_training_and_final_measurement_seconds": cli_report["training_and_final_measurement_seconds"],
                               "cli_training_settings": cli_report["settings"],
                               "cuda_peak_allocated_bytes": cli_report.get("cuda_peak_allocated_bytes"),
                               "cuda_peak_allocated_scope": cli_report.get("cuda_peak_allocated_scope")})
                stage = "exact_evaluation"
                if reference is not None:
                    energy, ground_state, basis, H = reference
                    started = time.perf_counter()
                    record["exact_full_sector"] = exact_metrics(model, basis, H, energy, ground_state, args.evaluation_batch_size)
                    record["exact_evaluation_wall_seconds"] = time.perf_counter() - started
                else:
                    record["exact_full_sector"] = {"status": "skipped", "reason": report["exact_reference"]["reason"]}
                record["status"] = "completed"
                _write_json(results_dir / f"{name}.json", record)
                _write_json(results_dir / "summary.json", report)
                print(f"{name}: subprocess {record['subprocess_wall_seconds']:.3f}s; sampled E={record['sampled_final_measurement']['energy']:.8f}; exact={record['exact_full_sector']}")
                record = None
    except Exception as error:
        failure = {"stage": stage, "error_type": type(error).__name__, "message": str(error)}
        report.update({"status": "failed", "failure": failure})
        if record is not None:
            record.update({"status": "failed", "failure": failure})
            _write_json(results_dir / f"{name}.json", record)
        elif stage == "exact_reference":
            report["exact_reference"] = {"status": "failed", "failure": failure}
        _write_json(results_dir / "summary.json", report)
        raise
    report["status"] = "completed"
    _write_json(results_dir / "summary.json", report)
    print(f"baseline record: {results_dir / 'summary.json'}")
    return report


if __name__ == "__main__":
    main()
