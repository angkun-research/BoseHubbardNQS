"""Sampled Adam and centered SR for periodic rings with fixed particle number."""

from functools import partial
import json
import math
import os
import random
import time

import torch

from ed_fit import build_model, configure_runtime
from vmc_sr import damping_at_update, update as sr_update, workload as sr_workload
from vmc_utils_gpu import Obtain_Sampling_batch_gpu, generate_initial_state, native_vmc_update


def save_checkpoint(model, config, output, step, elapsed):
    if not all(torch.isfinite(param).all().item() for param in model.parameters()):
        raise RuntimeError(f"non-finite model parameters at checkpoint {step}")
    name = f"vmc-{step:06d}.pt"
    partial = output / f"{name}.tmp"
    weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"schema_version": 1, "config": config, "state_dict": weights,
                "arm": "large_vmc", "step": step}, partial)
    partial.replace(output / name)
    return {"step": step, "path": name, "elapsed_seconds_before_save": elapsed}


def sample_update_batch(model, options, walkers, steps, keep_adam_graph):
    if keep_adam_graph:
        return Obtain_Sampling_batch_gpu(model, **options, initial_states=walkers, n_steps=steps)
    trainable = [param for param in model.parameters() if param.requires_grad]
    try:
        for param in trainable:
            param.requires_grad_(False)
        return Obtain_Sampling_batch_gpu(model, **options, initial_states=walkers, n_steps=steps)
    finally:
        for param in trainable:
            param.requires_grad_(True)


def train(config, output, expected_parameters):
    started = time.perf_counter()
    device = configure_runtime(config)
    length, nmax, chains = config["length"], config["nmax"], config["walkers"]
    model = build_model(config["arch"], length, nmax, config["width"]).to(device).eval()
    if sum(param.numel() for param in model.parameters() if param.requires_grad) != expected_parameters:
        raise RuntimeError("model parameter count differs from the unchanged ED architecture formula")
    use_adam = config["optimizer"] == "adam"
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False) if use_adam else None
    workspace = None if use_adam else sr_workload(chains * config["sample_steps"], expected_parameters,
                                                  config["sr_jacobian_chunk"], config["sr_workspace_mib"])
    random.seed(config["sampler_seed"])
    torch.manual_seed(config["sampler_seed"])
    walkers = [generate_initial_state(length, length, nmax) for _ in range(chains)]
    options = {"L": length, "Nmax": nmax, "t": config["t"], "U": config["legacy_u_code"], "mu": config["mu"],
               "pbc": config["pbc"], "evaluation_batch_size": config["chunk_size"]}
    if "nonlocal_transfer_fraction" in config:
        from vmc_proposals import mixed_proposal

        options["Sampler"] = partial(mixed_proposal, fraction=config["nonlocal_transfer_fraction"])

    # The sampler evaluates one cached current amplitude per walker per call,
    # then one per attempted proposal and one amplitude per
    # retained configuration. All other rows in the retained call are valid
    # connected neighbors evaluated by local energy. It reuses the retained logs.
    forwarded_rows = 0

    def count_rows(_model, inputs):
        nonlocal forwarded_rows
        forwarded_rows += inputs[0].shape[0]

    hook = model.register_forward_pre_hook(count_rows)
    totals = {"attempted_proposals": 0, "burn_in_proposals": 0, "measurement_proposals": 0,
              "retained_samples": 0, "valid_local_energy_neighbor_evaluations": 0,
              "forward_amplitude_rows": 0}
    checkpoints = []
    checkpoint_seconds = 0.0
    update_seconds = 0.0
    try:
        checkpoint_started = time.perf_counter()
        checkpoints.append(save_checkpoint(model, config, output, 0, checkpoint_started - started))
        checkpoint_seconds += time.perf_counter() - checkpoint_started
        with (output / "updates.jsonl").open("w", buffering=1) as history:
            for step in range(1, config["steps"] + 1):
                step_started = time.perf_counter()
                forwarded_rows = 0
                discarded = chains * config["burn_in"]
                if config["burn_in"]:
                    _, _, walkers, _ = Obtain_Sampling_batch_gpu(
                        model, **options, initial_states=walkers, n_steps=config["burn_in"], burnin=True,
                    )
                    if forwarded_rows != chains + discarded:
                        raise RuntimeError("legacy burn-in forward accounting has changed")
                burn_rows = forwarded_rows
                logpsi, energies, walkers, samples = sample_update_batch(
                    model, options, walkers, config["sample_steps"], keep_adam_graph=use_adam)
                measured = chains * config["sample_steps"]
                retained = len(samples)
                if retained != measured or len(energies) != measured or logpsi.numel() != measured:
                    raise RuntimeError("legacy sampler returned unexpected retained counts")
                neighbor_rows = forwarded_rows - burn_rows - chains - measured - retained
                if not 0 <= neighbor_rows <= 2 * length * retained:
                    raise RuntimeError("legacy local-energy forward accounting is inconsistent")
                if not ((samples.sum(dim=1) == length).all() & (samples >= 0).all() & (samples <= nmax).all()).item():
                    raise RuntimeError("sampler left the requested particle sector or cutoff")
                if not (torch.isfinite(energies).all() & torch.isfinite(logpsi).all()).item():
                    raise RuntimeError(f"non-finite sampled wavefunction or energy at step {step}")
                sampling_rows = forwarded_rows
                diagnostics = {}
                if use_adam:
                    loss = native_vmc_update(model, samples, energies, length, nmax, optimizer, logpsis=logpsi)
                    if not math.isfinite(loss):
                        raise RuntimeError(f"non-finite Adam surrogate at step {step}")
                else:
                    if logpsi.requires_grad:
                        raise RuntimeError("SR unexpectedly retained the sampled Adam graph")
                    update_config = config
                    if "sr_lambda_final" in config:
                        actual_damping = damping_at_update(
                            config["sr_lambda"], config["sr_lambda_final"], config["sr_lambda_decay_after"],
                            config["steps"], step,
                        )
                        update_config = dict(config, sr_lambda=actual_damping)
                    diagnostics["sr"] = sr_update(model, samples, energies, update_config, workspace)
                    if "sr_lambda_final" in config:
                        diagnostics["sr"]["damping"] = actual_damping
                mean, std = energies.double().mean().item() / length, energies.double().std(unbiased=False).item() / length
                wall = time.perf_counter() - step_started
                update_seconds += wall
                counts = {"attempted_proposals": discarded + measured, "burn_in_proposals": discarded,
                          "measurement_proposals": measured, "retained_samples": retained,
                          "valid_local_energy_neighbor_evaluations": neighbor_rows,
                          "forward_amplitude_rows": sampling_rows}
                for name, value in counts.items():
                    totals[name] += value
                history.write(json.dumps({"step": step, **counts, "pre_update_energy_per_site_mean": mean,
                                          "pre_update_local_energy_per_site_std": std,
                                          "wall_seconds": wall, **diagnostics}, allow_nan=False) + "\n")
                if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                    checkpoint_started = time.perf_counter()
                    checkpoints.append(save_checkpoint(model, config, output, step, checkpoint_started - started))
                    checkpoint_seconds += time.perf_counter() - checkpoint_started
                    print(f"step={step} pre_update_energy_per_site={mean:.8g} checkpoint={checkpoints[-1]['path']}",
                          flush=True)
    finally:
        hook.remove()
    optimizer_metadata = {"class": "Adam", "lr": config["lr"], "betas": [0.9, 0.999], "eps": 1e-8,
                          "objective": "legacy centered local-energy gradient for total energy"} if use_adam else {
        "class": "centered normalized ridge SR", "lr": config["lr"], "lambda": config["sr_lambda"],
        "solve_dtype": "float64", "workspace": workspace, "update": "direct negative ridge direction",
    }
    if "sr_lambda_final" in config:
        optimizer_metadata.update(lambda_final=config["sr_lambda_final"],
                                  lambda_decay_after=config["sr_lambda_decay_after"])
    return {
        "updates": "updates.jsonl", "checkpoints": checkpoints, "counts": totals,
        "optimizer": optimizer_metadata,
        "timing": {"wall_seconds": time.perf_counter() - started, "sampling_and_update_seconds": update_seconds,
                   "checkpoint_seconds": checkpoint_seconds},
        "runtime": {"device": str(device), "job_id": os.environ["SLURM_JOB_ID"], "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None},
        "counting": "Attempted moves include invalid self moves. Connected rows include duplicates when evaluated. Forward rows count sampler amplitudes. SR Jacobian work is logged separately.",
    }
