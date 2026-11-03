#!/usr/bin/env python3
"""Sequential ED-supervised fits; retain full trainer JSON alongside text logs.

Defaults: 2x3, all five models, seed 123, width 32, 10000 Adam updates at 1e-3,
float64, primary log-MSE (--loss infidelity is experimental). A false physical fit gate is a scientific observation, not an execution
failure. Checkpoints go under outputs/<run-name>/; retained evidence goes under
benchmarks/results/<run-name>/. Existing run directories are refused.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import BASELINE_ARCHITECTURES, ARCH_BUILDERS
from lattice import add_boundary_arguments, resolve_cli_boundaries
from train_ed import DEFAULT_MAX_DIMENSION, DEFAULT_MAX_SPARSE_BYTES, parse_args as validate_trainer_args


def _command(args, arch, seed, checkpoint):
    command = [sys.executable, "-u", str(ROOT / "train_ed.py"), "--arch", arch, "--seed", str(seed),
               "--Lx", str(args.Lx), "--Ly", str(args.Ly), "--Nmax", str(args.Nmax),
               "--epochs", str(args.epochs), "--width", str(args.width), "--lr", str(args.lr), "--loss", args.loss,
               "--dtype", args.dtype, "--device", args.device, "--eval-every", str(args.eval_every),
               "--t", str(args.t), "--U", str(args.U), "--mu", str(args.mu), "--eps", str(args.eps),
               "--max-ed-dimension", str(args.max_ed_dimension), "--max-ed-bytes", str(args.max_ed_bytes),
               "--output", str(checkpoint), "--hidden-layers", str(args.hidden_layers),
               "--pbc-x" if args.pbc_x else "--no-pbc-x", "--pbc-y" if args.pbc_y else "--no-pbc-y"]
    if args.Ntot is not None:
        command.extend(["--Ntot", str(args.Ntot)])
    if not args.pbc:
        command.append("--no-pbc")
    return command


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=3)
    parser.add_argument("--Ntot", type=int)
    parser.add_argument("--Nmax", type=int, default=4)
    add_boundary_arguments(parser)
    parser.add_argument("--arches", nargs="+", choices=list(ARCH_BUILDERS), default=list(BASELINE_ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--width", type=int, default=32, help="unused for JastrowNet, which has no width parameter")
    parser.add_argument("--hidden-layers", "--depth", dest="hidden_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["log_mse", "infidelity"], default="log_mse")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=16.74)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--max-ed-dimension", type=int, default=DEFAULT_MAX_DIMENSION)
    parser.add_argument("--max-ed-bytes", type=int, default=DEFAULT_MAX_SPARSE_BYTES)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("ed-fits-%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args(argv)
    resolve_cli_boundaries(args)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name)
            or args.threads <= 0 or any(not 0 <= seed < 2**63 for seed in args.seeds)
            or len(set(args.seeds)) != len(args.seeds) or len(set(args.arches)) != len(args.arches)):
        parser.error("use a simple run name, positive threads, distinct architectures and distinct seeds in [0, 2**63)")
    for parent in (ROOT / "outputs", ROOT / "benchmarks" / "results"):
        path = parent / args.run_name
        if path.exists() or path.is_symlink() or not path.resolve().is_relative_to(parent):
            parser.error("run directories must be new and stay under outputs/ and benchmarks/results/")
    # Reuse the trainer's scalar, geometry, device and output preflight. This
    # parses settings only; it does not construct a model or an ED dataset.
    example = ROOT / "outputs" / args.run_name / "preflight.pt"
    validated = validate_trainer_args(_command(args, args.arches[0], args.seeds[0], example)[3:])
    args.Lx, args.Ly, args.Ntot = validated.Lx, validated.Ly, validated.Ntot
    return args


def main(argv=None):
    args = parse_args(argv)
    results = ROOT / "benchmarks" / "results" / args.run_name
    outputs = ROOT / "outputs" / args.run_name
    results.mkdir(parents=True)
    outputs.mkdir(parents=True)
    summary = {
        "status": "running", "expected_run_count": len(args.arches) * len(args.seeds),
        "completed_run_count": 0, "failed_run_count": 0, "settings": vars(args), "runs": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "launcher_sha256_at_start": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution_vs_science": "completed means the trainer and report retention succeeded; a false fit gate is not an execution failure",
    }
    summary_path = results / "summary.json"
    _write_json(summary_path, summary)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    for arch in args.arches:
        for seed in args.seeds:
            name = f"{arch}-seed{seed}"
            checkpoint = outputs / f"{name}.pt"
            detailed_report = results / f"{name}-trainer.json"
            log = results / f"{name}.log"
            record_path = results / f"{name}.json"
            command = _command(args, arch, seed, checkpoint)
            record = {
                "execution_status": "running", "arch": arch, "seed": seed, "loss_name": args.loss, "command": command,
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "detailed_report_path": str(detailed_report.relative_to(ROOT)), "log_path": str(log.relative_to(ROOT)),
            }
            summary["runs"].append(record)
            _write_json(record_path, record)
            _write_json(summary_path, summary)
            stage = "training"
            started = time.perf_counter()
            try:
                try:
                    with log.open("w") as stream:
                        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                    record["returncode"] = completed.returncode
                finally:
                    record["subprocess_wall_seconds"] = time.perf_counter() - started
                if completed.returncode:
                    raise RuntimeError(f"trainer exited with {completed.returncode}; see {log}")
                stage = "report_retention"
                # Retain the original report byte-for-byte, including all history,
                # target clamp metadata and trainer source hashes, before summary extraction.
                shutil.copyfile(Path(str(checkpoint) + ".json"), detailed_report)
                trainer = json.loads(detailed_report.read_text())
                if trainer["status"] != "completed":
                    raise RuntimeError("trainer report is not completed")
                record.update({
                    "execution_status": "completed", "best_checkpoint_metrics": trainer["best_checkpoint_metrics"],
                    "final_metrics": trainer["final_metrics"], "parameter_count": trainer["parameter_count"],
                    "best_epoch": trainer["best_epoch"], "best_normalized_log_mse": trainer["best_normalized_log_mse"],
                    "loss_name": trainer.get("loss_name", "log_mse"),
                    "best_training_loss": trainer.get("best_training_loss", trainer["best_normalized_log_mse"]),
                    "selection": trainer["selection"],
                    "best_checkpoint_fit_gate_passed": trainer["best_checkpoint_metrics"]["fit_gate_passed"],
                    "final_fit_gate_passed": trainer["final_metrics"]["fit_gate_passed"],
                    "best_checkpoint_larger_system_fidelity_converged": trainer["best_checkpoint_metrics"].get("larger_system_fidelity_converged"),
                    "final_larger_system_fidelity_converged": trainer["final_metrics"].get("larger_system_fidelity_converged"),
                    "trainer_source_sha256_at_start": trainer["source_sha256_at_start"],
                    "trainer_source_hashes_recorded_at": trainer["source_hashes_recorded_at"],
                    "trainer_wall_seconds_including_ed_fit_and_staging": trainer["wall_seconds_including_ed_fit_and_staging"],
                })
                summary["completed_run_count"] += 1
            except Exception as error:
                failure = {"stage": stage, "type": type(error).__name__, "message": str(error)}
                record.update({"execution_status": "failed", "failure": failure})
                summary.update({"status": "failed", "failure": failure})
                summary["failed_run_count"] += 1
                _write_json(record_path, record)
                _write_json(summary_path, summary)
                raise
            _write_json(record_path, record)
            _write_json(summary_path, summary)
            print(f"{name}: completed; loss={record['loss_name']} best_epoch={record['best_epoch']} "
                  f"strict_joint_gate={record['best_checkpoint_fit_gate_passed']} "
                  f"larger_fidelity={record['best_checkpoint_larger_system_fidelity_converged']} "
                  f"wall={record['subprocess_wall_seconds']:.3f}s", flush=True)
    summary["status"] = "completed"
    _write_json(summary_path, summary)
    print(f"supervised summary: {summary_path}")
    return summary


if __name__ == "__main__":
    main()
