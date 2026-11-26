#!/usr/bin/env python3
"""Train small-ring VMC or score a frozen ED/VMC checkpoint on Slurm.

Use train --dry-run or score --dry-run for login-node validation. Scoring
loads checkpoints from this bridge or the companion exact-fidelity pilot.
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
from ed_protocol import ATTENTION, PARAMETER_CAP, PHASES, SCREEN_MODELS, model_cost, sector_dimension


NMAX = 6
LENGTHS = (6, 8, 10)


def model_config(settings):
    """Extract shared model and physics settings from CLI or checkpoint metadata."""
    config = {key: settings[key] for key in ("arch", "length", "width", "phase", "nmax", "seed")}
    arch, length, width = config["arch"], config["length"], config["width"]
    if length not in LENGTHS or config["nmax"] != NMAX or config["phase"] not in PHASES:
        raise ValueError("the bridge uses L=6, 8 or 10, inclusive Nmax=6 and a study phase")
    if arch not in SCREEN_MODELS or type(width) is not int or width <= 0:
        raise ValueError("select a supported adjustable model and positive integer width")
    if arch in ATTENTION and width % 4:
        raise ValueError("attention widths must be divisible by four")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("initialization seed must be in [0, 2**32)")
    if model_cost(arch, length, NMAX, width)[0] > PARAMETER_CAP:
        raise ValueError(f"model exceeds the common {PARAMETER_CAP} parameter cap")
    return config


def prepare(args):
    config = vars(args).copy()
    config.pop("dry_run")
    config["output"] = str(args.output.resolve())
    if args.output.exists():
        raise ValueError("choose a new output directory")
    if min(args.threads, args.chunk_size) <= 0:
        raise ValueError("threads and chunk size must be positive")
    if args.command == "score":
        if not args.checkpoint.is_file():
            raise ValueError("checkpoint path must name an existing file")
        config["checkpoint"] = str(args.checkpoint.resolve())
        return config, {"checkpoint_metadata_validated_at_runtime": True}

    config["nmax"] = NMAX
    settings = model_config(config)
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if min(args.steps, args.walkers, args.sample_steps, args.score_every) <= 0 or args.burn_in < 0:
        raise ValueError("steps, walkers, sample steps and score interval must be positive and burn-in nonnegative")
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else (args.seed + 1) % 2**32
    if not 0 <= sampler_seed < 2**32:
        raise ValueError("sampler seed must be in [0, 2**32)")
    config["sampler_seed"] = sampler_seed
    return config, {
        "dimension": sector_dimension(args.length, NMAX),
        "parameters": model_cost(settings["arch"], args.length, NMAX, args.width)[0],
        "u_phys_over_t": PHASES[args.phase], "legacy_u_code": PHASES[args.phase] / 2,
        "proposals": args.steps * args.walkers * (args.burn_in + args.sample_steps),
        "retained_samples": args.steps * args.walkers * args.sample_steps,
    }


def run(args):
    config, planned = prepare(args)
    if args.dry_run:
        print_json({"config": config, "planned": planned})
        return
    if args.threads > require_compute_node():
        raise ValueError("requested threads exceed Slurm CPUs per task")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    from bridge_fit import run_bridge

    checkpoint = None
    settings = config
    if args.command == "score":
        checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
        settings = model_config(checkpoint["config"])

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    for path in ("ScalingStudy/1D/ed_vmc_bridge.py", "ScalingStudy/1D/bridge_fit.py",
                 "pythonTest/vmc_utils.py", "pythonTest/vmc_utils_gpu.py"):
        provenance["source_sha256"][path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps({"config": config, "provenance": provenance}, indent=2) + "\n")
    result = run_bridge(config, settings, checkpoint, output)
    result.update(schema_version=1, config=config, provenance=provenance)
    partial = output / "result.json.tmp"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(output / "result.json")
    print(f"result={output / 'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run sampled Adam and optionally a matched exact-energy arm")
    train.add_argument("--length", type=int, choices=LENGTHS, default=8)
    train.add_argument("--phase", choices=PHASES, required=True)
    train.add_argument("--arch", choices=SCREEN_MODELS, required=True)
    train.add_argument("--width", type=int, required=True)
    train.add_argument("--seed", type=int, required=True, help="reuse the ED trial's initialization seed")
    train.add_argument("--sampler-seed", type=int, help="defaults to the initialization seed plus one modulo 2**32")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--steps", type=int, required=True, help="Adam updates per arm")
    train.add_argument("--walkers", type=int, default=16)
    train.add_argument("--burn-in", type=int, default=8, help="discarded proposals per walker per update")
    train.add_argument("--sample-steps", type=int, default=8, help="retained configurations per walker per update")
    train.add_argument("--score-every", type=int, default=10, help="checkpoint interval in updates")
    train.add_argument("--exact-energy", action="store_true", help="apply the same update budget to exact energy")

    score = commands.add_parser("score", help="score a frozen bridge or ED fidelity checkpoint without updates")
    score.add_argument("--checkpoint", type=Path, required=True)
    for command in (train, score):
        command.add_argument("--output", type=Path, required=True, help="new output directory on Lustre")
        command.add_argument("--dry-run", action="store_true")
        runtime_options(command)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
