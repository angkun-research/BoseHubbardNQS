"""Numerical implementation of the small-ring ED/VMC bridge."""

from copy import deepcopy
import math
import os
import random
import time

import numpy as np
import scipy
import torch

from ed_fit import FixedSector, build_model, configure_runtime, fidelity_terms, log_amplitudes, one_hot
from ed_protocol import PHASES, model_cost, sector_dimension
# ed_fit exposes the repository's existing model and sampler directory.
from vmc_utils_gpu import (
    Obtain_Sampling_batch_gpu, generate_initial_state, local_energy_on_the_fly_gpu, native_vmc_update,
)


class FullSectorEvaluation:
    def __init__(self, length, nmax, coupling, device, chunk_size):
        self.sector = FixedSector(length, nmax, coupling)
        ground, residual = self.sector.ground_state()
        if len(ground) != sector_dimension(length, nmax):
            raise RuntimeError("enumerated sector does not match the independent counting formula")
        self.reference = dict(self.sector.observables(ground), dimension=len(ground), residual=residual)
        self.logref = torch.from_numpy(np.log(ground)).to(device)
        basis = torch.from_numpy(self.sector.basis).to(device=device, dtype=torch.long)
        self.chunks = basis.split(chunk_size)
        self.channels = nmax + 1

    def wavefunction(self, model):
        logits = log_amplitudes(model, self.chunks, self.channels)
        log_fidelity, _, psi = fidelity_terms(logits, self.logref)
        return log_fidelity.item(), psi.cpu().numpy()

    def score(self, model):
        log_fidelity, psi = self.wavefunction(model)
        if not math.isfinite(log_fidelity) or log_fidelity > 1e-12:
            raise RuntimeError(f"invalid full-sector log fidelity ({log_fidelity})")
        infidelity = -math.expm1(min(0.0, log_fidelity))
        values = self.sector.observables(psi)
        return {
            "fidelity": math.exp(min(0.0, log_fidelity)), "infidelity": infidelity, **values,
            "energy_per_site_error": values["energy_per_site"] - self.reference["energy_per_site"],
            "g1_error": (np.asarray(values["g1"]) - self.reference["g1"]).tolist(),
            "block_variance_error": (np.asarray(values["block_variance"]) - self.reference["block_variance"]).tolist(),
        }

    def exact_energy_update(self, model, optimizer):
        _, psi = self.wavefunction(model)
        h_psi = self.sector.hamiltonian @ psi
        energy = float(psi @ h_psi)
        if not math.isfinite(energy):
            raise RuntimeError("non-finite exact variational energy")
        # For psi = exp(logpsi) / ||exp(logpsi)||, dE/dlogpsi = 2 psi (Hpsi - E psi).
        gradient = torch.from_numpy(2 * psi * (h_psi - energy * psi)).to(self.logref.device)
        optimizer.zero_grad(set_to_none=True)
        for states, derivative in zip(self.chunks, gradient.split([len(chunk) for chunk in self.chunks])):
            output = model(one_hot(states, self.channels))
            output.backward(derivative.to(output.dtype))
        optimizer.step()
        return energy

    def verify_local_energy(self, model, config):
        """Check the sampled Hamiltonian convention against the sparse ED matrix."""
        _, psi = self.wavefunction(model)
        indices = np.linspace(0, len(psi) - 1, 12, dtype=int)
        states = self.sector.basis[indices]
        local = local_energy_on_the_fly_gpu(states, model, **config).cpu().numpy()
        expected = (self.sector.hamiltonian @ psi)[indices] / psi[indices]
        np.testing.assert_allclose(local, expected, rtol=2e-4, atol=2e-4)
        return float(np.max(np.abs(local - expected)))


def save_checkpoint(path, model, settings, arm, step, score):
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"schema_version": 1, "config": settings, "state_dict": weights,
                "arm": arm, "step": step, "exact_score": score}, path)


def sampled_arm(model, evaluator, config, settings, output, sampler_options):
    seed = config["sampler_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    length, nmax = settings["length"], settings["nmax"]
    walkers = [generate_initial_state(length, length, nmax) for _ in range(config["walkers"])]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    history, checkpoints = [], []
    for step in range(1, config["steps"] + 1):
        if config["burn_in"]:
            _, _, walkers, _ = Obtain_Sampling_batch_gpu(
                model, **sampler_options, initial_states=walkers, n_steps=config["burn_in"], burnin=True,
            )
        logs, energies, walkers, samples = Obtain_Sampling_batch_gpu(
            model, **sampler_options, initial_states=walkers, n_steps=config["sample_steps"],
        )
        if not torch.isfinite(energies).all().item():
            raise RuntimeError(f"non-finite sampled local energy at step {step}")
        if not ((samples.sum(dim=1) == length).all() & (samples >= 0).all() & (samples <= nmax).all()).item():
            raise RuntimeError("the sampler left the fixed particle sector or occupation cutoff")
        surrogate = native_vmc_update(model, samples, energies, length, nmax, optimizer, logpsis=logs)
        if not math.isfinite(surrogate):
            raise RuntimeError(f"non-finite sampled update at step {step}")
        history.append({"update": step, "pre_update_sampled_energy_mean": energies.double().mean().item(),
                        "pre_update_local_energy_std": energies.double().std(unbiased=False).item(),
                        "proposals": config["walkers"] * (config["sample_steps"] + config["burn_in"]),
                        "retained_samples": len(energies)})
        if step % config["score_every"] == 0 or step == config["steps"]:
            checkpoints.append(score_checkpoint(model, evaluator, settings, output, "vmc", step))
    return {"updates": history, "checkpoints": checkpoints}


def score_checkpoint(model, evaluator, settings, output, arm, step):
    score = evaluator.score(model)
    name = f"{arm}-{step:06d}.pt"
    save_checkpoint(output / name, model, settings, arm, step, score)
    print(f"arm={arm} step={step} infidelity={score['infidelity']:.8g} energy={score['energy']:.8g}", flush=True)
    return {"step": step, "checkpoint": name, "exact_score": score}


def exact_arm(model, evaluator, config, settings, output):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    history, checkpoints = [], []
    for step in range(1, config["steps"] + 1):
        energy = evaluator.exact_energy_update(model, optimizer)
        history.append({"update": step, "pre_update_exact_energy": energy})
        if step % config["score_every"] == 0 or step == config["steps"]:
            checkpoints.append(score_checkpoint(model, evaluator, settings, output, "exact_energy", step))
    return {"updates": history, "checkpoints": checkpoints}


def run_bridge(config, settings, checkpoint, output):
    started = time.perf_counter()
    length, nmax, width, arch = (settings[key] for key in ("length", "nmax", "width", "arch"))
    coupling = PHASES[settings["phase"]]
    device = configure_runtime(dict(config, seed=settings["seed"]))
    model = build_model(arch, length, nmax, width).to(device).eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != model_cost(arch, length, nmax, width)[0]:
        raise RuntimeError("model parameter count differs from the ED protocol")
    evaluator = FullSectorEvaluation(length, nmax, coupling, device, config["chunk_size"])
    result = {
        "model_config": {key: settings[key] for key in ("length", "nmax", "width", "arch", "phase", "seed")},
        "reference": evaluator.reference, "u_phys_over_t": coupling, "legacy_u_code": coupling / 2,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
                    "device": str(device), "job_id": os.environ["SLURM_JOB_ID"],
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        result["frozen_checkpoint"] = {"arm": checkpoint.get("arm", "ed_fidelity"),
                                       "step": checkpoint.get("step", checkpoint.get("best", {}).get("step")),
                                       "exact_score": evaluator.score(model)}
    else:
        exact_model = deepcopy(model) if config["exact_energy"] else None
        sampler_options = {"L": length, "Nmax": nmax, "t": 1.0, "U": coupling / 2, "mu": 0.0, "pbc": True,
                           "evaluation_batch_size": config["chunk_size"]}
        result["local_energy_max_abs_error"] = evaluator.verify_local_energy(model, sampler_options)
        result["initial"] = score_checkpoint(model, evaluator, settings, output, "initial", 0)
        vmc_started = time.perf_counter()
        result["vmc"] = sampled_arm(model, evaluator, config, settings, output, sampler_options)
        result["vmc"]["elapsed_seconds"] = time.perf_counter() - vmc_started
        if exact_model is not None:
            exact_started = time.perf_counter()
            result["exact_energy"] = exact_arm(exact_model, evaluator, config, settings, output)
            result["exact_energy"]["elapsed_seconds"] = time.perf_counter() - exact_started
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
