#!/usr/bin/env python3
"""Train gated large-ring VMC with Adam or centered SR on a Slurm compute node.

Provide both completed JSON outputs from ed_fidelity.py report-screen.
--dry-run checks qualification and configuration without numerical imports.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

sys.dont_write_bytecode = True

from ed_fidelity import REPO, print_json, require_compute_node, runtime_options, source_provenance
from ed_protocol import ATTENTION, CONFIRM_SEEDS, GATE, PHASES, SCREEN_MODELS, model_cost
from vmc_sr import workload as sr_workload


LENGTHS = (16, 24, 32, 48, 64)
NMAX = 6
NONLOCAL_TRANSFER_FRACTION = 0.25


def qualification(path, length, arch):
    """Validate the current report-screen schema and the selected candidate's evidence."""
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError(f"L={length} requires a completed report-screen JSON object")
    screen = report.get(f"l{length}_screen")
    if not isinstance(screen, dict) or not screen:
        raise ValueError(f"{path} has no L={length} confirmation results")
    expected_tasks = len(screen) * len(PHASES) * len(CONFIRM_SEEDS)
    if report.get("missing_results") != [] or report.get("completed_tasks") != expected_tasks:
        raise ValueError(f"L={length} confirmation report is incomplete")
    entry = screen.get(arch)
    if not isinstance(entry, dict) or entry.get(f"l{length}_all_three_pass") is not True:
        raise ValueError(f"exact model {arch} must pass all three phases at L={length}")
    phases = entry.get("phases")
    if not isinstance(phases, dict) or phases.keys() != PHASES.keys():
        raise ValueError(f"L={length} report must include the three study phases for {arch}")
    expected_seeds = {str(seed) for seed in CONFIRM_SEEDS}
    medians = {}
    for phase, outcome in phases.items():
        scores = outcome.get("seed_infidelities") if isinstance(outcome, dict) else None
        if not isinstance(scores, dict) or scores.keys() != expected_seeds or outcome.get("missing_seeds") != []:
            raise ValueError(f"L={length} {arch} {phase} requires all confirmation seeds")
        if any(type(score) not in (int, float) or not 0 <= score <= 1 for score in scores.values()):
            raise ValueError(f"L={length} {arch} {phase} has an invalid seed infidelity")
        middle = median(scores.values())
        if outcome.get("default_pass") is not True or outcome.get("median_infidelity") != middle or middle > GATE:
            raise ValueError(f"L={length} {arch} {phase} fails the median infidelity gate")
        medians[phase] = middle
    return {"length": length, "arch": arch, "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "completed_tasks": expected_tasks, "median_infidelity": medians}


def prepare(args):
    if args.width <= 0 or (args.arch in ATTENTION and args.width % 4):
        raise ValueError("width must be positive and divisible by four for attention models")
    if min(args.steps, args.walkers, args.sample_steps, args.checkpoint_every, args.threads, args.chunk_size) <= 0:
        raise ValueError("steps, walkers, sample steps, checkpoint interval, threads and chunk size must be positive")
    if args.burn_in < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("burn-in must be nonnegative and learning rate finite and positive")
    if not math.isfinite(args.sr_lambda) or args.sr_lambda <= 0 or min(args.sr_jacobian_chunk, args.sr_workspace_mib) <= 0:
        raise ValueError("SR lambda must be finite and positive, with positive Jacobian chunk and workspace limit")
    scheduled_damping = args.sr_lambda_final is not None or args.sr_lambda_decay_after is not None
    if scheduled_damping and (
        args.optimizer != "sr" or args.sr_lambda_final is None or args.sr_lambda_decay_after is None
        or not math.isfinite(args.sr_lambda_final) or not 0 < args.sr_lambda_final < args.sr_lambda
        or not 1 <= args.sr_lambda_decay_after < args.steps
    ):
        raise ValueError("SR decay requires a smaller positive final lambda and a hold strictly inside the update budget")
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else (args.seed + 1) % 2**32
    if not 0 <= args.seed < 2**32 or not 0 <= sampler_seed < 2**32 or sampler_seed == args.seed:
        raise ValueError("initialization and sampler seeds must be distinct integers in [0, 2**32)")
    if args.output.exists():
        raise ValueError("choose a new output directory")
    reports = [qualification(path, length, args.arch) for path, length in ((args.screen_l8, 8), (args.screen_l10, 10))]
    config = vars(args).copy()
    for key in ("dry_run", "screen_l8", "screen_l10", "mix_nonlocal"):
        config.pop(key)
    if not scheduled_damping:
        config.pop("sr_lambda_final")
        config.pop("sr_lambda_decay_after")
    coupling = PHASES[args.phase]
    config.update(output=str(args.output.resolve()), nmax=NMAX, particles=args.length, t=1.0, mu=0.0,
                  boundary="periodic", pbc=True, u_phys_over_t=coupling, legacy_u_code=coupling / 2,
                  sampler_seed=sampler_seed)
    parameters, forward_flops = model_cost(args.arch, args.length, NMAX, args.width)
    planned = {
        "parameter_count": parameters, "forward_flops_per_amplitude_estimate": forward_flops,
        "burn_in_proposals": args.walkers * args.burn_in * args.steps,
        "measurement_proposals": args.walkers * args.sample_steps * args.steps,
        "retained_samples": args.walkers * args.sample_steps * args.steps,
        "sampler_policy": {
            "initial_walkers": "Seeded fixed-number configurations from the legacy initializer",
            "initial_extra_burn_in": 0, "burn_in_before_each_update": args.burn_in,
            "between_updates": "Persistent walkers across optimizer updates",
        },
        "initialization": "Fresh seeded ED model constructor at target L. Original initializers and tensor shapes. No rescaling or checkpoint transfer.",
        "capacity": "Explicit width at each L. Parameter count follows the unchanged architecture at that L.",
    }
    if args.mix_nonlocal:
        config["nonlocal_transfer_fraction"] = NONLOCAL_TRANSFER_FRACTION
        planned["sampler_policy"]["proposal"] = (
            "75% legacy nearest-neighbor hops and 25% uniform ordered distinct-site transfers. "
            "Invalid transfers are self moves."
        )
    if args.optimizer == "sr":
        planned["sr"] = sr_workload(args.walkers * args.sample_steps, parameters,
                                    args.sr_jacobian_chunk, args.sr_workspace_mib)
        planned["sr_step_policy"] = "Fixed learning rate and ridge. Direct update without clipping or retries. Numerical failure stops the run."
        if scheduled_damping:
            planned["sr_step_policy"] = (
                "Fixed learning rate and scheduled ridge. Direct update without clipping or retries. Numerical failure stops the run."
            )
            planned["sr_damping_schedule"] = {
                "rule": "log-linear after the hold",
                "initial": args.sr_lambda, "hold_through_update": args.sr_lambda_decay_after,
                "final": args.sr_lambda_final, "final_update": args.steps,
            }
    return config, reports, planned


def run(args):
    config, reports, planned = prepare(args)
    if args.dry_run:
        print_json({"config": config, "qualification": reports, "planned": planned})
        return
    if args.threads > require_compute_node():
        raise ValueError("requested threads exceed Slurm CPUs per task")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from vmc_train_fit import train

    provenance = source_provenance()
    paths = ["ScalingStudy/1D/vmc_train.py", "ScalingStudy/1D/vmc_train_fit.py", "ScalingStudy/1D/vmc_sr.py",
             "pythonTest/vmc_utils.py", "pythonTest/vmc_utils_gpu.py"]
    if args.mix_nonlocal:
        paths.append("ScalingStudy/1D/vmc_proposals.py")
    for path in paths:
        provenance["source_sha256"][path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "config": config, "qualification": reports, "planned": planned,
                "provenance": provenance}
    (output / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    result = train(config, output, planned["parameter_count"])
    result.update(metadata)
    partial = output / "result.json.tmp"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(output / "result.json")
    print(f"result={output / 'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-l8", type=Path, required=True, help="completed L=8 report-screen JSON")
    parser.add_argument("--screen-l10", type=Path, required=True, help="completed L=10 report-screen JSON")
    parser.add_argument("--arch", choices=SCREEN_MODELS, required=True, help="exact passing model name")
    parser.add_argument("--length", choices=LENGTHS, type=int, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--width", type=int, required=True, help="original constructor width or RBM hidden-unit count")
    parser.add_argument("--seed", type=int, required=True, help="fresh weight initialization seed")
    parser.add_argument("--sampler-seed", type=int, help="defaults to the initialization seed plus one modulo 2**32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, required=True, help="number of optimizer updates")
    parser.add_argument("--optimizer", choices=("adam", "sr"), default="adam")
    parser.add_argument("--sr-lambda", type=float, default=1e-3, help="initial or fixed positive ridge in the centered SR system")
    parser.add_argument("--sr-lambda-final", type=float, help="optional positive damping at the final SR update")
    parser.add_argument("--sr-lambda-decay-after", type=int, help="last update at the initial damping before log-linear decay")
    parser.add_argument("--sr-jacobian-chunk", type=int, default=32, help="maximum SR Jacobian rows per autodiff batch")
    parser.add_argument("--sr-workspace-mib", type=int, default=4096, help="cap on estimated dense SR workspace")
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=32, help="discarded proposals per walker per update")
    parser.add_argument("--sample-steps", type=int, default=8, help="retained configurations per walker per update")
    parser.add_argument("--mix-nonlocal", action="store_true",
                        help="mix 25%% uniform distinct-site transfers into the study training proposals")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="write checkpoints every this many updates")
    parser.add_argument("--output", type=Path, required=True, help="new output directory on Lustre")
    parser.add_argument("--dry-run", action="store_true")
    runtime_options(parser)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
