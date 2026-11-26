#!/usr/bin/env python3
"""Plan, replay, and merge one bounded ED retest per original confirmation.

Pass each original confirmation manifest with --original. plan and merge
print JSON. task --dry-run validates without loading numerical packages.
An optional merged confirmation manifest lets the existing L=10 planner
consume L=8 retest results without any changes to that runner.

Replay requires the original host, package versions, numerical source hashes,
and trial settings. It verifies checkpoint scores within 1e-8 with unchanged
threshold classifications. The original artifacts cannot verify bitwise Adam
state. Replayed prefix scores never replace the official original scores.
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

from ed_fidelity import REPO, SCHEMA_VERSION, print_json, read_manifest, require_compute_node, source_provenance, validate
from ed_protocol import CHECKPOINT_FRACTIONS, CONFIRM_SEEDS, GATE, PHASES, SCREEN_MODELS, TIGHT_TARGET, confirmation_report


REPLAY_ATOL = 1e-8
CORE_SOURCES = ("ScalingStudy/1D/ed_fit.py", "CriticalBehavior/src/ed.py", "pythonTest/NeuralNetworkClass.py")
RUNTIME_KEYS = ("device", "torch", "numpy", "scipy", "cuda", "gpu")
REPLAY_POLICY = ("Same host, numerical packages, numerical source hashes and trial settings. Original checkpoint "
                 "scores must agree within 1e-8 with unchanged target crossings. Bitwise Adam state is unverified.")


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def key(config):
    return config["arch"], config["phase"], config["seed"]


def horizon(config, step_flops):
    return min(config["max_steps"], config["max_flops"] // step_flops)


def schedule(start, stop):
    return sorted({start, *(start + math.ceil(fraction * (stop - start)) for fraction in CHECKPOINT_FRACTIONS)})


def cap_reason(config, limit):
    return "flop_budget" if limit < config["max_steps"] else "step_budget"


def validate_trajectory(config, result, allowed):
    """Check a complete checkpoint prefix and its stopping and best-score metadata."""
    estimate = validate(config)
    flops = estimate["estimated_flops_per_step"]
    if result["schema_version"] != SCHEMA_VERSION or result["config"] != config:
        raise ValueError(f"result configuration does not match its trial {key(config)}")
    if any(result["cost"][field] != value for field, value in estimate.items()):
        raise ValueError(f"FLOP or sector metadata does not match trial {key(config)}")
    rows = result["checkpoints"]
    if not isinstance(rows, list) or not rows or len(rows) > len(allowed):
        raise ValueError(f"invalid checkpoint list for {key(config)}")
    for row, step in zip(rows, allowed):
        score = row["infidelity"]
        if row["step"] != step or row["training_flops_estimate"] != step * flops or (
            type(score) not in (int, float) or not 0 <= score <= 1
        ):
            raise ValueError(f"invalid score, FLOPs, or checkpoint schedule for {key(config)} at step {step}")
    final = rows[-1]
    best = min(rows, key=lambda row: row["infidelity"])
    first_default = next((row["step"] for row in rows if row["infidelity"] <= GATE), None)
    first_tight = next((row["step"] for row in rows if row["infidelity"] <= TIGHT_TARGET), None)
    if first_tight is not None:
        terminal = first_tight == final["step"] and result["termination"] == "tight_target"
    else:
        terminal = final["step"] == allowed[-1] and result["termination"] == cap_reason(config, allowed[-1])
    if not terminal or result["final"] != final or result["best"] != best or (
        result["first_default_crossing_step"] != first_default or result["first_tight_crossing_step"] != first_tight
    ):
        raise ValueError(f"invalid best, crossing, or termination metadata for {key(config)}")


def load_originals(paths):
    entries = {}
    outputs = set()
    for path in paths:
        manifest = read_manifest(path, "confirmation")
        for config in manifest["trials"]:
            estimate = validate(config)
            trial = key(config)
            output = Path(config["output"]).resolve()
            if config["arch"] not in SCREEN_MODELS or trial in entries or output in outputs:
                raise ValueError(f"duplicate or ineligible original confirmation trial {trial}")
            outputs.add(output)
            result_path = output / "result.json"
            result, digest = read_json(result_path)
            if "replay" in result:
                raise ValueError("original confirmation must precede any retest")
            validate_trajectory(config, result, schedule(0, horizon(config, estimate["estimated_flops_per_step"])))
            entries[trial] = (config, result, digest)
    sizes = {entry[0]["length"] for entry in entries.values()}
    caps = {(entry[0]["max_steps"], entry[0]["max_flops"]) for entry in entries.values()}
    if len(sizes) != 1 or not sizes <= {8, 10} or len(caps) != 1:
        raise ValueError("original confirmations must share L=8 or L=10 and common production caps")
    architectures = {trial[0] for trial in entries}
    expected = {(arch, phase, seed) for arch in architectures for phase in PHASES for seed in CONFIRM_SEEDS}
    if entries.keys() != expected:
        raise ValueError("each original candidate requires all three phases and all three confirmation seeds")
    for arch in architectures:
        settings = [{k: v for k, v in config.items() if k not in ("phase", "seed", "output")}
                    for config, _, _ in entries.values() if config["arch"] == arch]
        if any(setting != settings[0] for setting in settings):
            raise ValueError(f"original settings drift across confirmation cells for {arch}")
    configs = [entry[0] for entry in entries.values()]
    return sizes.pop(), entries, confirmation_report([entry[:2] for entry in entries.values()], configs)


def environment_metadata(result):
    provenance = result["provenance"]
    host = provenance["host"]
    source_hashes = {path: provenance["source_sha256"][path] for path in CORE_SOURCES}
    runtime = {name: result["runtime"][name] for name in RUNTIME_KEYS}
    return {"host": host, "source_sha256": source_hashes, "runtime": runtime}


def build_plan(paths, output_root, length, entries, screen):
    trials = []
    original_outputs = {Path(config["output"]).resolve() for config, _, _ in entries.values()}
    for (arch, phase, seed), (config, result, digest) in sorted(entries.items()):
        outcome = screen[arch]
        if not outcome["bounded_retest_eligible"] or outcome["phases"][phase]["default_pass"]:
            continue
        output = (output_root / arch / f"seed-{seed}" / phase).resolve()
        if output in original_outputs:
            raise ValueError("retest outputs must be separate from original confirmation outputs")
        extended = dict(config, max_steps=2 * config["max_steps"], max_flops=2 * config["max_flops"], output=str(output))
        estimate = validate(extended)
        original_limit = horizon(config, estimate["estimated_flops_per_step"])
        extended_limit = horizon(extended, estimate["estimated_flops_per_step"])
        trials.append({"config": extended, "original_result_sha256": digest, "environment": environment_metadata(result),
                       "original_horizon": original_limit, "original_terminal_step": result["final"]["step"],
                       "extended_horizon": extended_limit, "prefix_checkpoint_steps": [row["step"] for row in result["checkpoints"]],
                       "extension_checkpoint_steps": schedule(original_limit, extended_limit)[1:]
                       if result["termination"] != "tight_target" else []})
    if len({row["config"]["output"] for row in trials}) != len(trials):
        raise ValueError("retest output directories must be unique")
    return {"schema_version": SCHEMA_VERSION, "stage": "bounded_retest", "length": length,
            "original_manifests": sorted(str(path.resolve()) for path in paths), "output_root": str(output_root),
            "replay_atol": REPLAY_ATOL, "replay_policy": REPLAY_POLICY, "task_count": len(trials), "trials": trials}


def read_plan(path, paths, original_data):
    manifest, _ = read_json(path)
    expected = build_plan(paths, Path(manifest["output_root"]).resolve(), *original_data)
    if expected != manifest:
        raise ValueError("retest manifest differs from the eligible same-seed plan or original results changed")
    return manifest


def compare_replay(original, replay):
    if len(original) != len(replay):
        raise ValueError("replay did not cover all original checkpoints")
    differences = []
    for expected, actual in zip(original, replay):
        a, b = expected["infidelity"], actual["infidelity"]
        if expected["step"] != actual["step"] or expected["training_flops_estimate"] != actual["training_flops_estimate"] or (
            not 0 <= b <= 1 or abs(a - b) > REPLAY_ATOL or any((a <= cutoff) != (b <= cutoff) for cutoff in (GATE, TIGHT_TARGET))
        ):
            raise ValueError(f"replay differs at step {expected['step']} (original {a:.12g}, replay {b:.12g})")
        differences.append(abs(a - b))
    return max(differences)


def validate_retest(task, original, result):
    config = task["config"]
    prefix = original["checkpoints"]
    validate_trajectory(config, result, task["prefix_checkpoint_steps"] + task["extension_checkpoint_steps"])
    if result["checkpoints"][:len(prefix)] != prefix:
        raise ValueError("retest must retain immutable original scores at every prefix checkpoint")
    replay = result["replay"]
    if replay["original_result_sha256"] != task["original_result_sha256"] or (
        environment_metadata(result) != task["environment"]
    ):
        raise ValueError("retest does not match the actual original result or runtime constraints")
    difference = compare_replay(prefix, replay["checkpoints"])
    if difference != replay["max_abs_score_difference"]:
        raise ValueError("retest score-difference metadata is inconsistent")
    replay_step = task["original_terminal_step"]
    extension_steps = result["final"]["step"] - replay_step
    flops = validate(config)["estimated_flops_per_step"]
    if (replay["updates"], replay["training_flops_estimate"], result["extension"]["updates"],
        result["extension"]["training_flops_estimate"]) != (replay_step, replay_step * flops, extension_steps, extension_steps * flops):
        raise ValueError("replay and extension accounting does not match the completed trajectory")
    if any(not math.isfinite(section["wall_seconds"]) or section["wall_seconds"] < 0
           for section in (replay, result["extension"])):
        raise ValueError("replay and extension wall time must be finite and nonnegative")


def run_task(plan, index, entries, dry_run):
    if not 0 <= index < len(plan["trials"]):
        raise ValueError(f"task index must be in [0, {len(plan['trials'])})")
    task = plan["trials"][index]
    config = task["config"]
    original = entries[key(config)][1]
    if dry_run:
        print_json(task)
        return
    if config["threads"] > require_compute_node():
        raise ValueError("original thread count exceeds Slurm CPUs per task")
    if socket.gethostname() != task["environment"]["host"]:
        raise ValueError(f"replay requires original compute host {task['environment']['host']}")
    provenance = source_provenance()
    if any(provenance["source_sha256"][path] != digest for path, digest in task["environment"]["source_sha256"].items()):
        raise ValueError("replay requires the original numerical source hashes")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["threads"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from ed_retest_fit import replay_and_extend

    for path in ("ScalingStudy/1D/ed_retest.py", "ScalingStudy/1D/ed_retest_fit.py"):
        provenance["source_sha256"][path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    output = Path(config["output"])
    result = replay_and_extend(task, original, validate(config), output)
    result.update(schema_version=SCHEMA_VERSION, config=config, provenance=provenance)
    validate_retest(task, original, result)
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "result.json")
    print(f"result={output / 'result.json'}", flush=True)


def merge(original_data, plan, confirmation_path):
    length, originals, original_screen = original_data
    effective = {trial: entry[:2] for trial, entry in originals.items()}
    finished = {}
    missing = []
    retest_architectures = set()
    pending_architectures = set()
    for task in plan["trials"] if plan is not None else ():
        config = task["config"]
        trial = key(config)
        retest_architectures.add(trial[0])
        path = Path(config["output"]) / "result.json"
        if not path.is_file():
            missing.append(str(path))
            pending_architectures.add(trial[0])
            continue
        result, _ = read_json(path)
        validate_retest(task, originals[trial][1], result)
        finished[trial] = (config, result)
    # Promote a candidate as a unit, after all of its eligible cells finish.
    effective.update({trial: entry for trial, entry in finished.items() if trial[0] not in pending_architectures})
    configs = [config for config, _ in effective.values()]
    screen = confirmation_report(list(effective.values()), configs)
    summary = {}
    pass_flag = f"l{length}_all_three_pass"
    for arch, prior in original_screen.items():
        if arch in pending_architectures:
            status = "retest_pending"
        elif arch in retest_architectures:
            status = "retest_passed" if screen[arch][pass_flag] else "retest_failed"
            screen[arch]["bounded_retest_eligible"] = False
        elif prior[pass_flag]:
            status = "original_pass"
        elif prior["bounded_retest_eligible"]:
            status = "near_miss_eligible"
        else:
            status = "original_failure_outside_retest_band"
        summary[arch] = {"status": status,
                         "original_medians": {phase: row["median_infidelity"] for phase, row in prior["phases"].items()},
                         "original_missed_phases": [phase for phase, row in prior["phases"].items()
                                                    if not row["default_pass"]]}
    if confirmation_path is not None:
        if missing:
            raise ValueError("wait for all planned retest results before exporting a merged confirmation manifest")
        with confirmation_path.open("x") as stream:
            json.dump({"schema_version": SCHEMA_VERSION, "stage": "confirmation", "task_count": len(configs),
                       "trials": configs}, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return {
        "completed_tasks": len(originals) - len(missing), "missing_results": missing, f"l{length}_screen": screen,
        "bounded_retest": {"candidates": summary, "retest_tasks_completed": len(finished), "replay_policy": REPLAY_POLICY,
                           "replay_training_flops_estimate": sum(row[1]["replay"]["training_flops_estimate"]
                                                                 for row in finished.values()),
                           "extension_training_flops_estimate": sum(row[1]["extension"]["training_flops_estimate"]
                                                                     for row in finished.values()),
                           "replay_wall_seconds": sum(row[1]["replay"]["wall_seconds"] for row in finished.values()),
                           "extension_wall_seconds": sum(row[1]["extension"]["wall_seconds"] for row in finished.values())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print the one bounded retest manifest")
    plan.add_argument("--output-root", type=Path, required=True)
    combine = commands.add_parser("merge", help="print a canonical gate report with retest status and costs")
    combine.add_argument("--retest", type=Path, help="bounded retest manifest containing result paths")
    combine.add_argument("--confirmation-manifest", type=Path, help="optional new manifest for the live L=10 planner")
    for command in (plan, combine):
        command.add_argument("--original", action="append", type=Path, required=True,
                             help="original confirmation manifest, repeat for separately screened candidates")
    task = commands.add_parser("task", help="validate or run one zero-based retest row")
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--index", type=int, help="defaults to SLURM_ARRAY_TASK_ID")
    task.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "task":
            raw, _ = read_json(args.manifest)
            paths = [Path(path) for path in raw["original_manifests"]]
            original_data = load_originals(paths)
            plan_data = read_plan(args.manifest, paths, original_data)
            index = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
            run_task(plan_data, index, original_data[1], args.dry_run)
        elif args.command == "plan":
            print_json(build_plan(args.original, args.output_root.resolve(), *load_originals(args.original)))
        else:
            original_data = load_originals(args.original)
            plan_data = read_plan(args.retest, args.original, original_data) if args.retest else None
            print_json(merge(original_data, plan_data, args.confirmation_manifest))
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
