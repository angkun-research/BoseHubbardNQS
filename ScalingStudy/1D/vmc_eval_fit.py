"""Frozen-ring observables and chain-preserving Monte Carlo diagnostics."""

import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import t as student_t
import torch

from ed_fit import build_model, configure_runtime
from ed_protocol import ATTENTION, MODELS, PHASES
from ed_vmc_bridge import LENGTHS as BRIDGE_LENGTHS, NMAX as BRIDGE_NMAX, model_config as bridge_model_config
from vmc_utils_gpu import Obtain_Sampling_batch_gpu, _log_amplitudes, generate_initial_state


def load_frozen_model(config, device):
    path = Path(config["checkpoint"])
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    settings = checkpoint["config"]
    for key in ("length", "phase", "nmax"):
        if settings[key] != config[key]:
            raise ValueError(f"checkpoint {key}={settings[key]!r} disagrees with requested {config[key]!r}")
    length, nmax = config["length"], config["nmax"]
    coupling = PHASES[config["phase"]]
    for key, expected in {"particles": length, "Ntot": length, "pbc": True, "boundary": "periodic",
                          "t": 1.0, "mu": 0.0, "u_phys_over_t": coupling, "legacy_u_code": coupling / 2}.items():
        if key in settings and settings[key] != expected:
            raise ValueError(f"checkpoint metadata {key} is inconsistent with the evaluation Hamiltonian")
    arch, width = settings["arch"], settings["width"]
    if arch not in MODELS or type(width) is not int or width <= 0 or (arch in ATTENTION and width % 4):
        raise ValueError("unsupported checkpoint architecture or model width")
    if arch == "JastrowNet" and width != 1:
        raise ValueError("the fixed-capacity Jastrow checkpoint must use width 1")
    # Preserve existing validation for ED/bridge artifacts. Larger checkpoints use
    # the same embedded format without the small-ring screen's parameter cap.
    if ("best" in checkpoint or "exact_score" in checkpoint) and arch != "JastrowNet" and (
        length in BRIDGE_LENGTHS and nmax == BRIDGE_NMAX
    ):
        bridge_model_config(settings)
    if config["seed"] in (settings.get("seed"), settings.get("sampler_seed")):
        raise ValueError("evaluation seed must differ from known checkpoint training seeds")
    model = build_model(arch, length, nmax, width)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    information = {"arch": arch, "width": width, "sha256": digest.hexdigest(), "arm": checkpoint.get("arm"),
                   "step": checkpoint.get("step", checkpoint.get("best", {}).get("step")),
                   "training_seed": settings.get("seed"), "training_sampler_seed": settings.get("sampler_seed")}
    return model, information


def local_g1(model, states, logs, distance, nmax, chunk_size):
    """Local estimator of the site-averaged real one-body correlator."""
    length = states.shape[1]
    dest = torch.arange(length, device=states.device)
    source = (dest + distance) % length
    estimates = []
    for block, base in zip(states.split(chunk_size), logs.split(chunk_size)):
        # Applying b_i^dagger b_{i+r} to the sampled configuration gives the
        # adjoint local estimator. Its expectation equals G1 for real states.
        rows, sites = ((block[:, source] > 0) & (block[:, dest] < nmax)).nonzero(as_tuple=True)
        origins = source[sites]
        connected = block[rows].clone()
        moves = torch.arange(len(rows), device=states.device)
        connected[moves, origins] -= 1
        connected[moves, sites] += 1
        ratios = (_log_amplitudes(model, connected, nmax, chunk_size).double() - base[rows].double()).exp()
        coefficients = (block[rows, origins] * (block[rows, sites] + 1)).double().sqrt()
        values = torch.zeros(len(block), device=states.device, dtype=torch.float64)
        values.index_add_(0, rows, coefficients * ratios)
        estimates.append(values / length)
    return torch.cat(estimates)


def local_number_variance(states, block_lengths):
    """Average (N_block - block_length)^2 over periodic windows in each sample."""
    length = states.shape[1]
    periodic = torch.cat((states, states[:, :max(block_lengths)]), dim=1)
    prefix = torch.nn.functional.pad(periodic.cumsum(dim=1), (1, 0))
    return {f"number_variance_l{size}": (prefix[:, size:size + length] - prefix[:, :length] - size)
            .double().square().mean(dim=1) for size in block_lengths}


@torch.inference_mode()
def sample_observables(model, config, coupling):
    length, chains, nmax = config["length"], config["chains"], config["nmax"]
    total_steps, batch_steps = config["steps"], config["batch_steps"]
    chunk = config["chunk_size"]
    retain_every = config.get("retain_every", 1)
    options = dict(L=length, Nmax=nmax, t=1.0, U=coupling / 2, mu=0.0, pbc=True, evaluation_batch_size=chunk)
    initial = [generate_initial_state(length, length, nmax) for _ in range(chains)]
    _, _, walkers, _ = Obtain_Sampling_batch_gpu(model, **options, initial_states=initial,
                                                n_steps=config["burn_in"], burnin=True)
    history = {"energy_per_site": np.empty((total_steps, chains), dtype=np.float64)}
    for name, values in (("g1_r", config["distances"]), ("number_variance_l", config["number_blocks"])):
        history.update({f"{name}{value}": np.empty((total_steps, chains), dtype=np.float64) for value in values})
    moved = torch.zeros(chains, device=walkers.device, dtype=torch.long)
    for start in range(0, total_steps, batch_steps):
        size = min(batch_steps, total_steps - start)
        previous = walkers
        logs, energies, walkers, states = Obtain_Sampling_batch_gpu(
            model, **options, initial_states=walkers, n_steps=size * retain_every, retain_every=retain_every,
        )
        if not ((states.sum(dim=1) == length).all() & (states >= 0).all() & (states <= nmax).all()).item():
            raise RuntimeError("sampled configurations left the requested fixed-number sector or cutoff")
        positions = states.reshape(size, chains, length)
        before = torch.cat((previous.unsqueeze(0), positions[:-1]), dim=0)
        moved += (positions != before).any(dim=-1).sum(dim=0)
        values = {"energy_per_site": energies.double() / length}
        for distance in config["distances"]:
            values[f"g1_r{distance}"] = local_g1(model, states, logs, distance, nmax, chunk)
        if config["number_blocks"]:
            values.update(local_number_variance(states, config["number_blocks"]))
        for name, local in values.items():
            local = local.reshape(size, chains).cpu().numpy()
            if not np.isfinite(local).all():
                raise RuntimeError(f"non-finite local estimator for {name} after retained step {start}")
            history[name][start:start + size] = local
    return history, (moved.double() / total_steps).cpu().numpy()


def ess_for_sem(variance, sem, count):
    if variance == 0 or sem == 0:
        return None
    return min(float(count), variance / sem**2)


def summarize_series(series, block_sizes, confidence, family_size):
    """Estimate conditional errors from within-chain blocks and chain means."""
    steps, chains = series.shape
    count = series.size
    mean, variance = float(series.mean()), float(series.var(ddof=1))
    chain_means = series.mean(axis=0)
    chain_sem = float(chain_means.std(ddof=1) / math.sqrt(chains))
    diagnostics = []
    for size in block_sizes:
        blocks_per_chain = steps // size
        blocks = series.reshape(blocks_per_chain, size, chains).mean(axis=1)
        sem = float(np.sqrt(blocks.var(axis=0, ddof=1).mean() / (blocks_per_chain * chains)))
        diagnostics.append({"block_size": size, "blocks_per_chain": blocks_per_chain, "sem": sem,
                            "ess": ess_for_sem(variance, sem, count)})
    sem = max(chain_sem, *(row["sem"] for row in diagnostics))
    ess = ess_for_sem(variance, sem, count)
    interval = None
    if ess is not None:
        critical = student_t.isf((1 - confidence) / (2 * family_size), df=chains - 1)
        half_width = float(critical * sem)
        interval = [mean - half_width, mean + half_width]
    split = steps // 2
    return {"mean": mean, "sample_variance": variance, "sem": sem, "ess": ess,
            "tau_int_retained_steps": count / (2 * ess) if ess is not None else None,
            "interval": interval, "confidence": confidence, "interval_family_size": family_size,
            "between_chain_sem": chain_sem, "chain_means": chain_means.tolist(),
            "first_half_chain_means": series[:split].mean(axis=0).tolist(),
            "second_half_chain_means": series[split:].mean(axis=0).tolist(), "blocking": diagnostics}


def evaluate(config):
    started = time.perf_counter()
    device = configure_runtime(config)
    model, checkpoint = load_frozen_model(config, device)
    torch.manual_seed(config["seed"])
    coupling = PHASES[config["phase"]]
    series, move_rates = sample_observables(model, config, coupling)
    diagnostics = {}
    retain_every = config.get("retain_every", 1)
    physics_family = 1 + len(config["distances"])
    for name, data in series.items():
        family_size = 1 if name.startswith("number_variance_") else physics_family
        diagnostics[name] = summarize_series(data, config["block_sizes"], config["confidence"], family_size)
    sampling = {
        "burn_in_proposals": config["chains"] * config["burn_in"],
        "measurement_proposals": config["chains"] * config["steps"] * retain_every,
        "retained_configurations": config["chains"] * config["steps"],
        "move_fraction_by_chain": move_rates.tolist(),
    }
    if retain_every > 1:
        sampling.update(retain_every=retain_every, block_unit="retained steps",
                        block_sizes_proposal_steps=[size * retain_every for size in config["block_sizes"]],
                        move_fraction_basis="State changed between snapshots, including the post-burn-in start.")
        for observable in diagnostics.values():
            for block in observable["blocking"]:
                block["block_size_proposal_steps"] = block["block_size"] * retain_every
    return {
        "checkpoint": checkpoint, "particles": config["length"], "boundary": "periodic", "t": 1.0,
        "u_phys_over_t": coupling, "legacy_u_code": coupling / 2, "series_axes": ["time", "chain"],
        "observables": diagnostics,
        "sampling": sampling,
        "uncertainty": {
            "method": "Maximum SEM across within-chain blocking scales and between-chain means. ESS is capped at the retained count.",
            "intervals": "Conditional Student-t intervals with chains minus one degrees of freedom. Bonferroni across energy and G1. Number variance intervals are pointwise.",
            "assumptions": "Independent stationary chains and sufficiently long time blocks. Check the blocked curves and split chain means for sensitivity. A constant trace has unresolved ESS and interval.",
        },
        "runtime": {"device": str(device), "torch": str(torch.__version__), "job_id": os.environ["SLURM_JOB_ID"],
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "elapsed_seconds": time.perf_counter() - started},
    }, series
