#!/usr/bin/env python3
"""Measure physical observables on fresh chains for a frozen 1D checkpoint.

Use --dry-run on a login node. Runtime strictly loads an ED or bridge-style
checkpoint with embedded model and phase metadata, including at larger L.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

from ed_fidelity import REPO, print_json, require_compute_node, runtime_options, source_provenance
from ed_protocol import PHASES


def prepare(args):
    if args.length < 4 or not 2 <= args.nmax <= 8:
        raise ValueError("use L >= 4 and an inclusive occupation cutoff from 2 through 8")
    if len(set(args.distances)) != len(args.distances) or any(not 1 <= r <= args.length // 2 for r in args.distances):
        raise ValueError("G1 distances must be distinct integers in [1, L//2]")
    if len(set(args.number_blocks)) != len(args.number_blocks) or any(
        not 1 <= length <= args.length // 2 for length in args.number_blocks
    ):
        raise ValueError("number block lengths must be distinct integers in [1, L//2]")
    if len(set(args.block_sizes)) != len(args.block_sizes) or any(
        size <= 0 or args.steps % size or args.steps // size < 4 for size in args.block_sizes
    ):
        raise ValueError("each time-block size must be distinct, divide steps, and leave at least four blocks per chain")
    if not 0 <= args.seed < 2**32:
        raise ValueError("evaluation seed must be in [0, 2**32)")
    if args.retain_every < 1:
        raise ValueError("retain_every must be a positive number of proposals per snapshot")
    if args.chains < 2 or min(args.burn_in, args.batch_steps, args.threads, args.chunk_size) <= 0:
        raise ValueError("use at least two fresh chains, positive burn-in, batch steps, threads and chunk size")
    if not math.isfinite(args.confidence) or not 0 < args.confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if not args.checkpoint.is_file() or args.output.exists():
        raise ValueError("provide an existing checkpoint file and a new output directory")
    config = vars(args).copy()
    config.pop("dry_run")
    if args.retain_every == 1:
        config.pop("retain_every")
    config.update(checkpoint=str(args.checkpoint.resolve()), output=str(args.output.resolve()),
                  block_sizes=sorted(args.block_sizes))
    return config


def run(args):
    config = prepare(args)
    if args.dry_run:
        print_json({"config": config, "particles": args.length, "u_phys_over_t": PHASES[args.phase],
                    "burn_in_proposals": args.chains * args.burn_in,
                    "measurement_proposals": args.chains * args.steps * args.retain_every,
                    "retained_configurations": args.chains * args.steps,
                    "block_sizes_proposal_steps": [size * args.retain_every for size in config["block_sizes"]],
                    "checkpoint_metadata_validated_at_runtime": True})
        return
    if args.threads > require_compute_node():
        raise ValueError("requested threads exceed Slurm CPUs per task")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    from vmc_eval_fit import evaluate

    started_from = source_provenance()
    for path in ("ScalingStudy/1D/vmc_evaluate.py", "ScalingStudy/1D/vmc_eval_fit.py", "pythonTest/vmc_utils.py",
                 "pythonTest/vmc_utils_gpu.py"):
        started_from["source_sha256"][path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    result, series = evaluate(config)
    np.savez_compressed(output / "series.npz", **series)
    result.update(config=config, provenance=started_from, series="series.npz", schema_version=1)
    partial = output / "result.json.tmp"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(output / "result.json")
    print(f"result={output / 'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--nmax", type=int, default=6)
    parser.add_argument("--distances", nargs="+", type=int, required=True, help="G1 separations along the ring")
    parser.add_argument("--number-blocks", nargs="+", type=int, default=[], help="optional physical block lengths")
    parser.add_argument("--seed", type=int, required=True, help="independent evaluation seed")
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--burn-in", type=int, default=256, help="discarded proposals per chain")
    parser.add_argument("--steps", type=int, default=512, help="retained time steps per chain")
    parser.add_argument("--retain-every", type=int, default=1,
                        help="actual proposals per chain between retained snapshots")
    parser.add_argument("--batch-steps", type=int, default=16, help="retained steps buffered for local measurements")
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64], help="time blocks in retained steps")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True, help="new output directory on Lustre")
    parser.add_argument("--dry-run", action="store_true")
    runtime_options(parser)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
