"""Sparse fixed-sector references and full-sector exact-fidelity optimization."""

import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import torch
from torch import nn
from torch.nn import functional as F

from ed_protocol import ATTENTION, CHECKPOINT_FRACTIONS, GATE, PHASES, TIGHT_TARGET


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "CriticalBehavior" / "src"))
sys.path.insert(0, str(REPO / "pythonTest"))
import ed as reference_ed
import NeuralNetworkClass as legacy_models


class FixedSector:
    def __init__(self, length, nmax, coupling):
        self.length = length
        self.nmax = nmax
        self.basis = np.asarray(list(reference_ed.occupations(length, length, nmax)), dtype=np.int16)
        self.place = (nmax + 1) ** np.arange(length - 1, -1, -1, dtype=np.int64)
        self.keys = self.basis @ self.place
        moves = [self.hop(site, (site + 1) % length) for site in range(length)]
        rows, cols, strength = map(np.concatenate, zip(*moves))
        size = len(self.basis)
        forward = coo_matrix((-strength, (rows, cols)), shape=(size, size)).tocsr()
        self.hamiltonian = forward + forward.T
        self.hamiltonian.setdiag(coupling * np.sum(self.basis * (self.basis - 1), axis=1) / 2)

    def hop(self, dest, source):
        cols = np.flatnonzero((self.basis[:, dest] < self.nmax) & (self.basis[:, source] > 0))
        rows = np.searchsorted(self.keys, self.keys[cols] + self.place[dest] - self.place[source])
        strength = np.sqrt((self.basis[cols, dest] + 1) * self.basis[cols, source], dtype=np.float64)
        return rows, cols, strength

    def ground_state(self):
        energies, vectors = eigsh(self.hamiltonian, k=1, which="SA", v0=np.ones(len(self.basis)), tol=1e-12)
        psi = vectors[:, 0]
        if psi[np.argmax(np.abs(psi))] < 0:
            psi = -psi
        if np.any(psi <= 0):
            raise RuntimeError("sparse ED did not resolve a strictly positive ground-state vector")
        psi /= np.linalg.norm(psi)
        energy = float(energies[0])
        residual = float(np.linalg.norm(self.hamiltonian @ psi - energy * psi))
        if residual > 1e-9:
            raise RuntimeError(f"sparse ED residual exceeds 1e-9 t ({residual:.3g})")
        return psi, residual

    def observables(self, psi):
        probability = psi**2
        g1 = []
        block_variance = []
        for distance in range(1, self.length // 2 + 1):
            terms = []
            for site in range(self.length):
                rows, cols, strength = self.hop(site, (site + distance) % self.length)
                terms.append(np.sum(strength * psi[rows] * psi[cols]))
            g1.append(float(np.mean(terms)))
        for site in range(self.length):
            counts = np.cumsum(np.roll(self.basis, -site, axis=1)[:, :self.length // 2], axis=1)
            centered = counts - probability @ counts
            block_variance.append(probability @ centered**2)
        energy = float(psi @ (self.hamiltonian @ psi))
        return {"energy": energy, "energy_per_site": energy / self.length, "g1": g1,
                "block_variance": np.mean(block_variance, axis=0).tolist()}


class InvariantConv1D(nn.Module):
    """Circular stride-one convolutions and a translation-invariant readout."""

    def __init__(self, channels, width):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, width, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1, padding_mode="circular"), nn.SiLU(),
        )
        self.readout = nn.Linear(width, 1)

    def forward(self, configurations):
        return self.readout(self.features(configurations.transpose(1, 2)).mean(dim=-1)).squeeze(-1)


def build_model(arch, length, nmax, width):
    if arch == "InvariantConv1D":
        return InvariantConv1D(nmax + 1, width)
    constructor = getattr(legacy_models, arch)
    model = constructor(length, nmax + 1, width) if arch != "JastrowNet" else constructor(length, nmax + 1)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    return model


def one_hot(occupations, channels):
    return F.one_hot(occupations, num_classes=channels).float()


def log_amplitudes(model, chunks, channels):
    with torch.no_grad():
        return torch.cat([model(one_hot(chunk, channels)).double() for chunk in chunks])


def fidelity_terms(logpsi, logref):
    """Evaluate log F and the exact derivative of 1-F with respect to all logits."""
    shifted = logpsi - logpsi.max()
    log_norm_sq = torch.logsumexp(2 * shifted, dim=0)
    log_overlap = torch.logsumexp(shifted + logref, dim=0)
    log_fidelity = 2 * log_overlap - log_norm_sq
    probability = (2 * shifted - log_norm_sq).exp()
    overlap_weight = (shifted + logref - log_overlap).exp()
    gradient = 2 * log_fidelity.exp() * (probability - overlap_weight)
    return log_fidelity, gradient, (shifted - 0.5 * log_norm_sq).exp()


def check_fidelity(logpsi, logref):
    log_fidelity, gradient, psi = fidelity_terms(logpsi, logref)
    shifted_log_fidelity, _, _ = fidelity_terms(logpsi + 500, logref)
    torch.testing.assert_close(log_fidelity, shifted_log_fidelity, atol=1e-12, rtol=0)
    torch.testing.assert_close(psi.square().sum().item(), 1.0, atol=1e-12, rtol=0)
    exact, _, _ = fidelity_terms(logref, logref)
    torch.testing.assert_close(exact.item(), 0.0, atol=1e-12, rtol=0)

    direct_logs = logpsi.clone().requires_grad_()
    root_probability = torch.softmax(2 * direct_logs, dim=0).sqrt()
    direct_fidelity = (root_probability @ logref.exp()).square()
    direct_gradient, = torch.autograd.grad(1 - direct_fidelity, direct_logs)
    torch.testing.assert_close(log_fidelity.exp(), direct_fidelity, atol=1e-12, rtol=0)
    torch.testing.assert_close(gradient, direct_gradient, atol=1e-12, rtol=0)


def check_chunk_gradients(model, chunks, channels, logref):
    direct_logs = torch.cat([model(one_hot(chunk, channels)).double() for chunk in chunks])
    direct_fidelity = (torch.softmax(2 * direct_logs, dim=0).sqrt() @ logref.exp()).square()
    params = [param for param in model.parameters() if param.requires_grad]
    direct_gradients = torch.autograd.grad(1 - direct_fidelity, params)
    for param, direct in zip(params, direct_gradients):
        torch.testing.assert_close(param.grad, direct, atol=2e-7, rtol=1e-4)


def translation_error(model, basis, channels):
    indices = torch.linspace(0, len(basis) - 1, min(32, len(basis)), device=basis.device).long()
    sample = basis[indices]
    with torch.no_grad():
        reference = model(one_hot(sample, channels))
        return max(
            (model(one_hot(sample.roll(shift, dims=1), channels)) - reference).abs().max().item()
            for shift in range(1, sample.shape[1])
        )


def configure_runtime(config):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    # Use the same attention path in the no-grad and gradient passes.
    torch.backends.mha.set_fastpath_enabled(False)
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable on this Slurm node")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device(config["device"])


def fit(config, estimate):
    started = time.perf_counter()
    device = configure_runtime(config)
    length, nmax = config["length"], config["nmax"]
    arch, width = config["arch"], config["width"]
    channels = nmax + 1
    coupling = PHASES[config["phase"]]
    model = build_model(arch, length, nmax, width).to(device).eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != estimate["parameters"]:
        raise RuntimeError(f"counted {parameters} trainable parameters, expected {estimate['parameters']}")

    sector = FixedSector(length, nmax, coupling)
    if len(sector.basis) != estimate["dimension"]:
        raise RuntimeError("enumerated sector does not match the independent counting formula")
    ground, residual = sector.ground_state()
    reference = dict(sector.observables(ground), dimension=len(ground), residual=residual)
    if length == 6:
        legacy = next(reference_ed.sector(length, length, nmax, [coupling]))
        for key in ("energy", "g1", "block_variance"):
            np.testing.assert_allclose(reference[key], legacy[key], atol=1e-9, rtol=0)
        reference["legacy_observables_checked"] = True
    reference_seconds = time.perf_counter() - started

    basis = torch.from_numpy(sector.basis).to(device=device, dtype=torch.long)
    chunks = basis.split(config["chunk_size"])
    logref = torch.from_numpy(np.log(ground)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    step_flops = estimate["estimated_flops_per_step"]
    steps_allowed = min(config["max_steps"], config["max_flops"] // step_flops)
    checkpoints = {0, *[int(np.ceil(fraction * steps_allowed)) for fraction in CHECKPOINT_FRACTIONS]}
    history, best_state, best_logits = [], None, None
    best, tight_step, gate_step = None, None, None
    stop = "flop_budget" if steps_allowed < config["max_steps"] else "step_budget"
    initial_translation = translation_error(model, basis, channels)
    fit_started = time.perf_counter()
    sanity_seconds = 0.0
    for step in range(steps_allowed + 1):
        logits = log_amplitudes(model, chunks, channels)
        log_fidelity, gradient, _ = fidelity_terms(logits, logref)
        if step in checkpoints:
            if step == 0 and length == 6:
                check_fidelity(logits, logref)
            raw = -torch.expm1(log_fidelity).item()
            if not np.isfinite(raw) or raw < -1e-12:
                raise RuntimeError(f"invalid infidelity at step {step} ({raw})")
            score = {"step": step, "infidelity": max(0.0, raw), "training_flops_estimate": step * step_flops}
            history.append(score)
            print(f"step={step} infidelity={score['infidelity']:.8g}", flush=True)
            if best is None or score["infidelity"] < best["infidelity"]:
                best = score
                best_logits = logits
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if score["infidelity"] <= GATE and gate_step is None:
                gate_step = step
            if score["infidelity"] <= TIGHT_TARGET:
                tight_step = step
                stop = "tight_target"
                break
        if step == steps_allowed:
            break
        optimizer.zero_grad(set_to_none=True)
        for states, derivative in zip(chunks, gradient.split(config["chunk_size"])):
            output = model(one_hot(states, channels))
            output.backward(derivative.to(output.dtype))
        if step == 0 and length == 6:
            check_started = time.perf_counter()
            check_chunk_gradients(model, chunks, channels, logref)
            sanity_seconds = time.perf_counter() - check_started
            reference["chunk_gradient_checked"] = True
        optimizer.step()

    training_seconds = time.perf_counter() - fit_started
    model.load_state_dict(best_state)
    best_translation = translation_error(model, basis, channels)
    if arch == "InvariantConv1D" and max(initial_translation, best_translation) > 1e-5:
        raise RuntimeError("the invariant model has an unexpected cyclic translation error")
    _, _, psi = fidelity_terms(best_logits, logref)
    unused = channels**2 if arch == "JastrowNet" else 0
    if arch == "JastrowNet":
        occupations_pair_present = np.add.outer(np.arange(channels), np.arange(channels)) <= length
        unused += (length // 2) * np.count_nonzero(~occupations_pair_present)
    result = {
        "u_phys_over_t": coupling, "legacy_u_code": coupling / 2, "reference": reference,
        "best_observables": sector.observables(psi.cpu().numpy()),
        "checkpoints": history, "best": best, "final": history[-1],
        "first_default_crossing_step": gate_step, "first_tight_crossing_step": tight_step,
        "termination": stop,
        "translation_max_abs_logdiff": {"initial": initial_translation, "best": best_translation},
        "parameters": {"allocated_trainable": parameters, "structurally_used": parameters - int(unused),
                       "unused": int(unused), "global_log_amplitude_shift_is_gauge": True},
        "timing": {"setup_and_reference_seconds": reference_seconds,
                   "fit_seconds_including_sanity": training_seconds, "gradient_sanity_seconds": sanity_seconds},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
                    "cuda": torch.version.cuda, "device": str(device), "job_id": os.environ["SLURM_JOB_ID"],
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "cost": dict(estimate, extra_terminal_score_flops_estimate=estimate["estimated_flops_per_score"]),
    }
    return result, best_state
