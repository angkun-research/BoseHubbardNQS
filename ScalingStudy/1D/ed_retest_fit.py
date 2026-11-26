"""Replay the original ED optimizer prefix and spend only the permitted extension."""

import math
import os
from pathlib import Path
import time

import numpy as np
import scipy
import torch

from ed_fit import FixedSector, build_model, configure_runtime, fidelity_terms, log_amplitudes, one_hot, translation_error
from ed_protocol import GATE, PHASES, TIGHT_TARGET
from ed_retest import RUNTIME_KEYS, cap_reason, compare_replay


def replay_and_extend(task, original, estimate, output):
    started = time.perf_counter()
    config = task["config"]
    device = configure_runtime(config)
    runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
               "cuda": torch.version.cuda, "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "job_id": os.environ["SLURM_JOB_ID"]}
    if {name: runtime[name] for name in RUNTIME_KEYS} != task["environment"]["runtime"]:
        raise ValueError("replay requires the original device, GPU model and numerical package versions")
    output.mkdir(parents=True, exist_ok=False)
    length, nmax = config["length"], config["nmax"]
    channels = nmax + 1
    model = build_model(config["arch"], length, nmax, config["width"]).to(device).eval()
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != estimate["parameters"]:
        raise RuntimeError("parameter count differs from the original architecture formula")
    sector = FixedSector(length, nmax, PHASES[config["phase"]])
    ground, _ = sector.ground_state()
    if len(ground) != estimate["dimension"]:
        raise RuntimeError("enumerated sector does not match the recorded dimension")
    basis = torch.from_numpy(sector.basis).to(device=device, dtype=torch.long)
    chunks = basis.split(config["chunk_size"])
    logref = torch.from_numpy(np.log(ground)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    translation_error(model, basis, channels)

    prefix = {row["step"]: row for row in original["checkpoints"]}
    old_terminal = task["original_terminal_step"]
    extension_schedule = set(task["extension_checkpoint_steps"])
    limit = task["extended_horizon"] if extension_schedule else old_terminal
    stop = cap_reason(config, limit) if extension_schedule else original["termination"]
    flops = estimate["estimated_flops_per_step"]
    replay, extension = [], []
    best = original["best"]
    replay_started = time.perf_counter()
    for step in range(limit + 1):
        logits = log_amplitudes(model, chunks, channels)
        log_fidelity, gradient, _ = fidelity_terms(logits, logref)
        if step in prefix or step in extension_schedule:
            raw = -torch.expm1(log_fidelity).item()
            if not math.isfinite(raw) or raw < -1e-12:
                raise RuntimeError(f"invalid infidelity at step {step} ({raw})")
            row = {"step": step, "infidelity": max(0.0, raw), "training_flops_estimate": step * flops}
            if step in prefix:
                compare_replay([prefix[step]], [row])
                replay.append(row)
                print(f"replay step={step} infidelity={row['infidelity']:.12g}", flush=True)
            else:
                extension.append(row)
                print(f"extension step={step} infidelity={row['infidelity']:.12g}", flush=True)
                if row["infidelity"] < best["infidelity"]:
                    best = row
                    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    temporary = output / "best.pt.tmp"
                    torch.save({"state_dict": weights, "config": config, "best": row}, temporary)
                    temporary.replace(output / "best.pt")
                if row["infidelity"] <= TIGHT_TARGET:
                    stop = "tight_target"
                    break
        if step == old_terminal:
            extension_started = time.perf_counter()
        if step == limit:
            break
        optimizer.zero_grad(set_to_none=True)
        for states, derivative in zip(chunks, gradient.split(config["chunk_size"])):
            output_logs = model(one_hot(states, channels))
            output_logs.backward(derivative.to(output_logs.dtype))
        optimizer.step()
    ended = time.perf_counter()
    checkpoints = original["checkpoints"] + extension
    extension_updates = checkpoints[-1]["step"] - old_terminal
    original_best = str(Path(original["config"]["output"]) / "best.pt")
    return {
        "checkpoints": checkpoints, "best": best, "final": checkpoints[-1], "termination": stop,
        "best_checkpoint": str(output / "best.pt") if extension and best in extension else original_best,
        "first_default_crossing_step": next((row["step"] for row in checkpoints if row["infidelity"] <= GATE), None),
        "first_tight_crossing_step": next((row["step"] for row in checkpoints if row["infidelity"] <= TIGHT_TARGET), None),
        "cost": estimate, "runtime": runtime, "setup_seconds": replay_started - started,
        "replay": {"original_result_sha256": task["original_result_sha256"], "checkpoints": replay,
                   "max_abs_score_difference": compare_replay(original["checkpoints"], replay),
                   "updates": old_terminal, "training_flops_estimate": old_terminal * flops,
                   "wall_seconds": extension_started - replay_started},
        "extension": {"updates": extension_updates, "training_flops_estimate": extension_updates * flops,
                      "wall_seconds": ended - extension_started if extension_updates else 0.0},
    }
