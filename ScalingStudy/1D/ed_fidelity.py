#!/usr/bin/env python3
"""Run an exact-fidelity trial or build screen manifests for Slurm arrays.

Use `trial --dry-run` or `task --dry-run` for validation on a login node.
`plan-screen`, `select-screen` and `report-screen` write JSON to stdout.
Only `trial` and `task` without `--dry-run` execute ED or training.
Use Python 3.8 or newer. Compute nodes also need compatible NumPy, SciPy and PyTorch.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.dont_write_bytecode = True

from ed_protocol import (
    ATTENTION, BANDS, MODELS, PARAMETER_CAP, PHASES, SCREEN_MODELS,
    confirmation_report, confirmation_trials, cost_estimate, tuning_choices, tuning_trials, width_for_band,
)


SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {
    "length", "phase", "nmax", "arch", "width", "band", "lr", "seed", "device",
    "chunk_size", "threads", "max_steps", "max_flops", "output",
}


def print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def validate(config):
    if not isinstance(config, dict) or config.keys() != CONFIG_KEYS:
        raise ValueError("trial configuration has missing or unexpected keys")
    length, nmax = config["length"], config["nmax"]
    arch, width, band = config["arch"], config["width"], config["band"]
    if length not in (6, 8, 10) or nmax not in range(2, 9):
        raise ValueError("use L=6, 8 or 10 and an inclusive occupation cutoff from 2 through 8")
    if length in (8, 10) and nmax != 6:
        raise ValueError("L=8 and L=10 trials use the common inclusive Nmax=6")
    if arch not in MODELS or config["phase"] not in PHASES or config["device"] not in ("cpu", "cuda"):
        raise ValueError("unknown model, phase, or compute device")
    if type(width) is not int or width < 1 or (arch in ATTENTION and width % 4):
        raise ValueError("width must be positive and divisible by four for attention models")
    if arch == "JastrowNet" and (width != 1 or band is not None):
        raise ValueError("the optional fixed-capacity Jastrow reference has no width or band choice")
    if band is not None and (band not in BANDS or width != width_for_band(arch, length, nmax, band)):
        raise ValueError("width does not match the requested parameter band")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("learning rate must be finite and positive")
    for name in ("chunk_size", "threads", "max_steps", "max_flops"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    cost = cost_estimate(config)
    if cost["parameters"] > PARAMETER_CAP:
        raise ValueError(f"model has {cost['parameters']} trainable parameters, above the {PARAMETER_CAP} cap")
    if config["max_flops"] < cost["estimated_flops_per_step"]:
        raise ValueError(f"FLOP budget cannot cover one update ({cost['estimated_flops_per_step']} estimated FLOPs)")
    if not Path(config["output"]).is_absolute():
        raise ValueError("trial output directory must be absolute in a manifest")
    return cost


def require_compute_node():
    job_id = os.environ.get("SLURM_JOB_ID")
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not nodelist:
        raise ValueError("ED and training require a Slurm compute node. Use --dry-run on a login node.")
    nodes = subprocess.run(["scontrol", "show", "hostnames", nodelist], check=True, capture_output=True, text=True)
    node_names = {node.split(".", 1)[0] for node in nodes.stdout.splitlines()}
    host = socket.gethostname().split(".", 1)[0]
    if host not in node_names and os.environ.get("SLURMD_NODENAME") not in node_names:
        raise ValueError("this host is outside the Slurm allocation. Launch the trial with sbatch or srun.")
    cpu_limit = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    return int(cpu_limit)


def source_provenance():
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    paths = (
        "ScalingStudy/1D/ed_fidelity.py", "ScalingStudy/1D/ed_fit.py", "ScalingStudy/1D/ed_protocol.py",
        "CriticalBehavior/src/ed.py", "pythonTest/NeuralNetworkClass.py",
    )
    return {"revision": version.stdout.strip() or None, "argv": sys.argv, "host": socket.gethostname(),
            "source_sha256": {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in paths}}


def run_trial(config, dry_run):
    cost = validate(config)
    planned = min(config["max_steps"], config["max_flops"] // cost["estimated_flops_per_step"])
    if dry_run:
        print_json({"config": config, "cost": cost, "maximum_complete_updates": planned,
                    "u_phys_over_t": PHASES[config["phase"]]})
        return
    if config["threads"] > require_compute_node():
        raise ValueError("requested threads exceed Slurm CPUs per task")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    (output / "config.json").write_text(json.dumps({"config": config, "provenance": provenance}, indent=2) + "\n")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["threads"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Keep numerical package imports below the Slurm guard.
    import torch
    from ed_fit import fit

    result, state_dict = fit(config, cost)
    torch.save({"state_dict": state_dict, "config": config, "best": result["best"]}, output / "best.pt")
    result.update(schema_version=SCHEMA_VERSION, config=config, provenance=provenance)
    partial = output / "result.json.tmp"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(output / "result.json")
    print(f"result={output / 'result.json'}", flush=True)


def read_manifest(path, stage=None):
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["stage"] not in ("tuning", "confirmation"):
        raise ValueError("unsupported screening manifest schema")
    if stage is not None and manifest["stage"] != stage:
        raise ValueError(f"this command requires a {stage} manifest")
    trials = manifest["trials"]
    if not trials or len({trial["output"] for trial in trials}) != len(trials):
        raise ValueError("manifest must contain at least one trial and unique output directories")
    return manifest


def completed_results(manifest):
    records, missing = [], []
    for config in manifest["trials"]:
        validate(config)
        path = Path(config["output"]) / "result.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        result = json.loads(path.read_text())
        if result["schema_version"] != SCHEMA_VERSION or result["config"] != config:
            raise ValueError(f"result does not match its manifest task ({path})")
        records.append((config, result))
    return records, missing


def run_command(args):
    if args.command == "trial":
        config = {key: getattr(args, key) for key in CONFIG_KEYS}
        if args.band is not None:
            config["width"] = width_for_band(args.arch, args.length, args.nmax, args.band)
        elif args.width is None:
            config["width"] = 1
        config["output"] = str(args.output.resolve())
        run_trial(config, args.dry_run)
    elif args.command == "task":
        trials = read_manifest(args.manifest)["trials"]
        index = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if not 0 <= index < len(trials):
            raise ValueError(f"provide a zero-based index from 0 through {len(trials) - 1} or use a Slurm array")
        run_trial(trials[index], args.dry_run)
    elif args.command == "plan-screen":
        architectures = list(dict.fromkeys(args.arch or SCREEN_MODELS))
        if args.length == 10:
            previous = read_manifest(args.from_confirmation, "confirmation")
            if {trial["length"] for trial in previous["trials"]} != {8}:
                raise ValueError("L=10 requires an L=8 confirmation manifest")
            records, missing = completed_results(previous)
            if missing:
                raise ValueError(f"waiting for {len(missing)} L=8 confirmations including {missing[0]}")
            outcomes = confirmation_report(records, previous["trials"])
            architectures = [arch for arch in architectures if outcomes.get(arch, {}).get("l8_all_three_pass")]
            if not architectures:
                raise ValueError("none of the selected candidates passed all three phases at L=8")
        options = {"length": args.length, "nmax": 6, "device": args.device, "threads": args.threads,
                   "chunk_size": args.chunk_size}
        trials, unavailable = tuning_trials(architectures, options, args.output_root.resolve())
        for config in trials:
            validate(config)
        print_json({"schema_version": SCHEMA_VERSION, "stage": "tuning", "task_count": len(trials),
                    "unavailable_bands": unavailable, "trials": trials})
    elif args.command == "select-screen":
        manifest = read_manifest(args.manifest, "tuning")
        records, missing = completed_results(manifest)
        if missing:
            raise ValueError(f"waiting for {len(missing)} tuning results including {missing[0]}")
        choices = tuning_choices(records)
        trials = confirmation_trials(choices, args.output_root.resolve())
        for config in trials:
            validate(config)
        ranked = [{"arch": arch, "worst_tuning_infidelity": worst, "band": config["band"],
                   "width": config["width"], "lr": config["lr"]} for arch, worst, config in choices]
        print_json({"schema_version": SCHEMA_VERSION, "stage": "confirmation", "task_count": len(trials),
                    "choices": ranked, "trials": trials})
    else:
        manifest = read_manifest(args.manifest, "confirmation")
        records, missing = completed_results(manifest)
        length = manifest["trials"][0]["length"]
        print_json({"completed_tasks": len(records), "missing_results": missing,
                    f"l{length}_screen": confirmation_report(records, manifest["trials"])})


def runtime_options(parser):
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--chunk-size", type=int, default=256, help="configurations per network forward")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per trial")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    trial = commands.add_parser("trial", help="validate or run one phase, model and seed")
    trial.add_argument("--length", type=int, choices=(6, 8, 10), default=6)
    trial.add_argument("--phase", choices=PHASES, required=True)
    trial.add_argument("--nmax", type=int, default=6)
    trial.add_argument("--arch", choices=MODELS, required=True)
    sizing = trial.add_mutually_exclusive_group()
    sizing.add_argument("--width", type=int)
    sizing.add_argument("--band", type=int, choices=BANDS)
    trial.add_argument("--lr", type=float, default=1e-3)
    trial.add_argument("--seed", type=int, required=True)
    trial.add_argument("--max-steps", type=int, required=True)
    trial.add_argument("--max-flops", type=int, required=True)
    trial.add_argument("--output", type=Path, required=True, help="new directory on Lustre for result and checkpoint")
    trial.add_argument("--dry-run", action="store_true")
    runtime_options(trial)

    task = commands.add_parser("task", help="validate or run one zero-based manifest row")
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--index", type=int, help="defaults to SLURM_ARRAY_TASK_ID")
    task.add_argument("--dry-run", action="store_true")

    screen = commands.add_parser("plan-screen", help="print a tuning manifest for the L=8 or L=10 screen")
    screen.add_argument("--length", type=int, choices=(8, 10), default=8)
    screen.add_argument("--from-confirmation", type=Path, help="L=8 confirmation manifest required for L=10")
    screen.add_argument("--arch", choices=SCREEN_MODELS, action="append", help="defaults to all screening candidates")
    screen.add_argument("--output-root", type=Path, required=True, help="existing or new Lustre root for tuning trials")
    runtime_options(screen)

    select = commands.add_parser("select-screen", help="print a confirmation manifest from completed tuning trials")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--output-root", type=Path, required=True, help="Lustre output root for confirmation trials")

    report = commands.add_parser("report-screen", help="report three-seed medians and all-three-phase status")
    report.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "trial" and args.width is None and args.band is None and args.arch != "JastrowNet":
        parser.error("trial requires --width or --band for adjustable models")
    if args.command == "plan-screen" and (args.length == 10) != (args.from_confirmation is not None):
        parser.error("--from-confirmation is required for L=10 and accepted only for that size")
    try:
        run_command(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
