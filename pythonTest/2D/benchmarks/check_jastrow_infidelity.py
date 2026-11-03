#!/usr/bin/env python3
"""Bounded Adam/infidelity diagnostic warm-started from cylinder log-MSE weights.

Uses existing train_model_on_ed without new trainer features. Adam is reset;
calibration models/optimizers are discarded before the actual <=2000-epoch fit.
Log-MSE remains primary. This diagnostic does not change the architecture.
"""

import argparse
import copy
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import load_model, save_model
from train_ed import compare_predictions, make_ed_dataset, train_model_on_ed
from check_jastrow_span import design_matrix, pair_geometry


def numpy_check(model, dataset):
    """Independent float64 logits from the onsite/pair design, not model.forward."""
    pairs, classes = pair_geometry(3, 3, pbc_x=False, pbc_y=True)
    X = design_matrix(dataset["basis"], pairs, classes, 5)
    coefficients = np.concatenate((model.onsite.detach().numpy(), model.pair.detach().numpy().reshape(-1)))
    logs = X @ coefficients
    normalized = logs - 0.5 * logsumexp(2 * logs)
    psi = np.exp(normalized)
    target = dataset["target_log"].numpy()
    target = target - 0.5 * logsumexp(2 * target)
    applied = dataset["H"] @ psi
    energy = float(psi @ applied)
    overlap = float(abs(dataset["raw_ed_state"] @ psi))
    return {"method": "independent NumPy float64 displacement design from saved rounded weights",
            "normalized_log_mse": float(np.mean((normalized - target)**2)),
            "amplitude_overlap_abs_unsquared": overlap, "amplitude_overlap_squared": overlap**2,
            "infidelity": 1 - overlap**2, "energy": energy,
            "energy_error_per_site": (energy - dataset["energy"]) / 9,
            "energy_variance": float(np.linalg.norm(applied - energy * psi)**2),
            "normalized_sector_norm": float(np.linalg.norm(psi)),
            "larger_system_fidelity_converged": bool(overlap >= 0.99)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/2026-10-31-3x3-cylinder-supervised/JastrowNet-seed123.pt")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:1"], default="auto")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("jastrow-cylinder-infidelity-%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 2000 or not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.run_name) or args.run_name in (".", ".."):
        parser.error("epochs must be 1..2000 and run-name a simple directory name")
    checkpoint = args.checkpoint.resolve()
    outputs = ROOT / "outputs" / args.run_name
    results = ROOT / "benchmarks/results" / args.run_name
    for path, parent in ((outputs, ROOT / "outputs"), (results, ROOT / "benchmarks/results")):
        if path.exists() or not path.resolve().is_relative_to(parent):
            parser.error("output/result directories must be new and stay under 2D/")
    initial_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    initial, metadata = load_model(checkpoint, device="cpu")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != initial_hash:
        raise RuntimeError("initial checkpoint changed during load")
    expected = {"arch": "JastrowNet", "Lx": 3, "Ly": 3, "Nmax": 4, "Ntot": 9,
                "pbc_x": False, "pbc_y": True, "t": 1.0, "U": 16.74, "mu": 0.0}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("diagnostic requires the specified 3x3 cylinder Jastrow checkpoint")
    results.mkdir(parents=True)
    outputs.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    sources = [ROOT / name for name in ("lattice.py", "exact_diagonalization.py", "models.py", "train_ed.py")]
    sources += [Path(__file__).resolve(), Path(__file__).with_name("check_jastrow_span.py")]
    report = {"status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
              "source_hashes_recorded_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256_at_start": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": initial_hash,
              "initial_checkpoint_metadata": metadata,
              "settings": {"epochs": args.epochs, "loss": "infidelity", "optimizer": "Adam", "lr": 0.001,
                           "seed": 123, "training_dtype": "float32", "threads": 1, "eval_every": 100},
              "initialization": "warm start from the completed cylinder log-MSE inference checkpoint; no zero reinitialization",
              "optimizer_reset": "new Adam with no restored moments/step state; this is not an exact training resume",
              "calibration": [], "primary_method": "Adam with log-MSE; infidelity is experimental",
              "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version,
                              "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}}
    summary_path = results / "report.json"
    def persist():
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    persist()
    started = time.perf_counter()
    try:
        dataset = make_ed_dataset(3, 3, Ntot=9, Nmax=4, t=1.0, U=16.74, mu=0.0,
                                  pbc_x=False, pbc_y=True, dtype=torch.float64)
        report["dataset"] = dataset["metadata"]
        report["initial_metrics_cpu_float64"] = compare_predictions(copy.deepcopy(initial).double(), dataset, loss="infidelity")
        gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
        if args.device == "cuda:1" and not gpu_available:
            raise RuntimeError("GPU1 unavailable")
        devices = [args.device] if args.device != "auto" else (["cpu", "cuda:1"] if gpu_available else ["cpu"])
        report["gpu1_available"] = gpu_available
        # Short calibration through the same API; discarded weights and fresh
        # optimizer each time. Timings are inclusive, noisy and not throughput claims.
        with (results / "run.log").open("w", buffering=1) as log:
            for device in devices:
                warmup = copy.deepcopy(initial).float().to(device)
                with redirect_stdout(log):
                    train_model_on_ed(warmup, dataset, epochs=2, eval_every=2, device=device, loss="infidelity")
                del warmup
                candidate = copy.deepcopy(initial).float().to(device)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                trial_started = time.perf_counter()
                with redirect_stdout(log):
                    train_model_on_ed(candidate, dataset, epochs=20, eval_every=20, device=device, loss="infidelity")
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - trial_started
                report["calibration"].append({"device": device, "timed_updates": 20, "warmup_updates_discarded": 2,
                                              "inclusive_api_wall_seconds": elapsed, "weights_and_optimizer_discarded": True})
                del candidate
                print(f"short calibration {device}: {elapsed:.3f}s / 20 updates", flush=True)
            selected = min(report["calibration"], key=lambda row: row["inclusive_api_wall_seconds"])["device"]
            report["selected_device"] = selected
            report["device_selection_scope"] = "short inclusive API timing under current shared load, not a steady-state performance claim; GPU1 not timed if unavailable"
            persist()
            print(f"starting {args.epochs} updates on {selected} from original saved weights with fresh Adam", flush=True)
            model = copy.deepcopy(initial).float().to(selected)
            fit_started = time.perf_counter()
            with redirect_stdout(log):
                model, fit, best = train_model_on_ed(model, dataset, epochs=args.epochs, lr=0.001, eval_every=100,
                                                     device=selected, loss="infidelity")
            if selected.startswith("cuda"):
                torch.cuda.synchronize(selected)
            report["fit_wall_seconds"] = time.perf_counter() - fit_started
            model.load_state_dict(best)
            output_checkpoint = outputs / "JastrowNet-infidelity-best.pt"
            save_model(output_checkpoint, model, Ntot=9, t=1.0, U=16.74, mu=0.0)
            saved, saved_metadata = load_model(output_checkpoint, device=selected)
            fit["best_checkpoint_metrics_training_dtype"] = compare_predictions(saved, dataset, loss="infidelity")
            cpu_saved = load_model(output_checkpoint, device="cpu")[0].double().eval()
            check = numpy_check(cpu_saved, dataset)
            cpu_metrics = compare_predictions(cpu_saved, dataset, loss="infidelity")
            for name in ("normalized_log_mse", "amplitude_overlap_abs_unsquared", "energy", "energy_error_per_site", "energy_variance"):
                np.testing.assert_allclose(check[name], cpu_metrics[name], rtol=1e-10, atol=1e-10)
            report.update({"status": "completed", "fit": fit, "checkpoint": str(output_checkpoint.relative_to(ROOT)),
                           "checkpoint_sha256": hashlib.sha256(output_checkpoint.read_bytes()).hexdigest(),
                           "checkpoint_metadata": saved_metadata, "best_metrics_cpu_float64": cpu_metrics,
                           "independent_numpy_float64_check": check,
                           "wall_seconds_including_ed_calibration_and_checks": time.perf_counter() - started,
                           "interpretation": "The analytic log-target span residual is not a bound on achievable overlap. A passing saved candidate demonstrates Fabs>=0.99 for this cylinder/ansatz; a failed bounded fit does not prove inability. Energy/variance and historical strict joint gate remain separate. No torus result is substituted."})
            persist()
            Path(str(output_checkpoint) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        report.update({"status": "failed", "failure": {"type": type(error).__name__, "message": str(error)}})
        persist()
        raise
    print(json.dumps({"best_metrics_cpu_float64": report["best_metrics_cpu_float64"], "report": str(summary_path)}, indent=2))
    return report


if __name__ == "__main__":
    main()
