#!/usr/bin/env python3
"""Tiny CPU Jacobian timing: fixed 4x4 cylinder Conv32, M=512/1024, B=1/8/16.

No optimizer updates or MC campaign. Same frozen weights/inputs across chunks.
Timings are short shared-workstation observations, not a steady-state ranking.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import Conv2DNet, parameter_count
from vmc import check_sr_budget, compute_log_psi_jacobian, generate_initial_states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/conv32-batched-vjp-cpu.json"))
    args = parser.parse_args(argv)
    output = (ROOT / "benchmarks" / args.output).resolve()
    if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
        parser.error("output must be a new JSON file under benchmarks/results/")
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = Conv2DNet(4, 4, Nmax=4, hidden_dim=32, pbc_x=False, pbc_y=True).double().eval()
    states = generate_initial_states(16, 16, 4, 1024, "cpu")
    report = {"status": "running", "kind": "bounded_CPU_Jacobian_timing_no_training",
              "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
              "source_hashes_recorded_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256_at_start": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                         for name in ("models.py", "vmc.py", "lattice.py", "benchmarks/benchmark_batched_vjp.py")},
              "settings": {"Lx": 4, "Ly": 4, "Ntot": 16, "Nmax": 4, "pbc_x": False, "pbc_y": True,
                           "arch": "Conv2DNet", "hidden_dim": 32, "dtype": "float64", "device": "cpu",
                           "seed": 123, "threads": 1, "M": [512, 1024], "chunks": [1, 8, 16],
                           "repeats": 1, "input_distribution": "feasible initializer occupations, not target-distributed MC samples"},
              "environment": {"torch": str(torch.__version__), "python": sys.version, "platform": platform.platform(),
                              "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
              "parameters": parameter_count(model), "timings": [],
              "limitations": "One inclusive Jacobian timing per case after small warmup. No solve time, training, GPU speed, exclusive hardware or convergence claim. Budget excludes model activations. Peak RSS is cumulative for this process, not an isolated case."}
    output.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    persist()
    for chunk in (1, 8, 16):
        compute_log_psi_jacobian(model, states[:min(chunk, 16)], jacobian_batch_size=chunk)
    for M in (512, 1024):
        reference = None
        reference_time = None
        for chunk in (1, 8, 16):
            budget = check_sr_budget(model, M, jacobian_batch_size=chunk)
            started = time.perf_counter()
            J = compute_log_psi_jacobian(model, states[:M], jacobian_batch_size=chunk)
            elapsed = time.perf_counter() - started
            if reference is None:
                reference, reference_time = J, elapsed
            torch.testing.assert_close(J, reference, rtol=1e-10, atol=1e-11)
            maximum_difference = (J - reference).abs().max().item()
            row = {"M": M, "jacobian_batch_size": chunk, "seconds": elapsed,
                   "speedup_vs_reference": reference_time / elapsed, "max_abs_difference_vs_reference": maximum_difference,
                   "budget": budget, "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            report["timings"].append(row)
            persist()
            print(f"M={M} chunk={chunk} seconds={elapsed:.4f} speedup={reference_time/elapsed:.3f} maxdiff={maximum_difference:.3g}", flush=True)
            if chunk != 1:
                del J
        del reference
    report["status"] = "completed"
    report["default_policy"] = "Keep reference chunk=1 default; larger chunks remain opt-in and must be justified by workload timings."
    persist()
    print(f"timing record: {output}")
    return report


if __name__ == "__main__":
    main()
