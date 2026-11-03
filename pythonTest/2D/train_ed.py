#!/usr/bin/env python3
"""Adam fitting to ED: primary log-MSE or optional experimental infidelity.

CLI default: 2x2. Both CLI and make_ed_dataset use Ly=Lx when Ly is omitted.
All registered models are supported within their geometry constraints.
Full-batch autograd normalizes the entire sector,
never separate minibatches. ED allocation guards do not bound model activations.
Raw ED vectors, not floored targets, define physical overlaps and energies.
Positive t and a nonzero, single-sign real ED vector are required.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

from exact_diagonalization import (
    DEFAULT_MAX_DIMENSION, DEFAULT_MAX_SPARSE_BYTES, _physical_scalar,
    basis_to_one_hot, exact_ground_state,
)
from lattice import _integer, lattice_shape, resolve_boundaries, add_boundary_arguments, resolve_cli_boundaries
from models import ARCH_BUILDERS, ROOT, build_model, load_model, parameter_count, save_model


LOSS_DEFINITION = "mean((logpsi-0.5*logsumexp(2*logpsi)-normalize(log(abs(raw_ED).clamp_min(eps))))**2) over the entire fixed-N sector"
OVERLAP_THRESHOLD = 0.9999
ENERGY_ERROR_PER_SITE_FACTOR = 0.001
LARGER_SYSTEM_OVERLAP_THRESHOLD = 0.99
LOSS_NAMES = ("log_mse", "infidelity")
INFIDELITY_DEFINITION = "1-abs(dot(exp(logpsi-0.5*logsumexp(2*logpsi)), raw_ED/norm(raw_ED)))**2 over the entire fixed-N sector"


def _positive(value, name):
    value = _physical_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def normalized_log_mse(pred_log, target_log):
    """Uniform MSE in normalized log amplitudes over the supplied full sector."""
    if pred_log.ndim != 1 or pred_log.shape != target_log.shape or not pred_log.numel():
        raise ValueError("prediction and target must be matching nonempty vectors")
    if not torch.isfinite(pred_log).all() or not torch.isfinite(target_log).all():
        raise ValueError("prediction and target log amplitudes must be finite")
    pred = pred_log - 0.5 * torch.logsumexp(2 * pred_log, dim=0)
    target = target_log - 0.5 * torch.logsumexp(2 * target_log, dim=0)
    return (pred - target).square().mean()


def normalized_infidelity(pred_log, raw_ed_state):
    """Infidelity to the raw ED vector; no target flooring or minibatch norms."""
    raw = torch.as_tensor(raw_ed_state, device=pred_log.device)
    if (pred_log.ndim != 1 or not pred_log.numel() or raw.shape != pred_log.shape
            or raw.is_complex() or not torch.isfinite(raw).all() or not torch.isfinite(pred_log).all()):
        raise ValueError("prediction and raw ED must be matching finite real vectors")
    raw = raw.detach().to(dtype=pred_log.dtype)
    norm = torch.linalg.vector_norm(raw)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("raw ED must have finite nonzero norm")
    psi = (pred_log - 0.5 * torch.logsumexp(2 * pred_log, dim=0)).exp()
    return 1 - torch.dot(psi, raw / norm).abs().square()


def _training_loss(logs, dataset, loss):
    if loss == "log_mse":
        return normalized_log_mse(logs, dataset["target_log"])
    if loss == "infidelity":
        return normalized_infidelity(logs, dataset["raw_ed_tensor"])
    raise ValueError(f"loss must be one of {LOSS_NAMES}")


def make_ed_dataset(
    Lx, Ly=None, Ntot=None, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True,
    eps=1e-12, dtype=torch.float64, max_dimension=DEFAULT_MAX_DIMENSION,
    max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES,
    *, pbc_x=None, pbc_y=None,
):
    """Return one-hot x, target_log, raw_ed_state, sparse H, basis and metadata.

    The return value is a dictionary. Flooring affects only target_log. Reject
    exact zeros or mixed signs rather than hiding them behind abs/clamp. A
    globally negative ED vector is an equivalent physical global phase.
    """
    Lx, Ly = lattice_shape(Lx, Ly)
    pbc_x, pbc_y = resolve_boundaries(pbc, pbc_x, pbc_y)
    t = _positive(t, "t")
    U, mu = _physical_scalar(U, "U"), _physical_scalar(mu, "mu")
    eps = _positive(eps, "eps")
    if eps >= 1:
        raise ValueError("eps must be smaller than one")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    Ntot = Lx * Ly if Ntot is None else Ntot
    energy, raw, basis, H = exact_ground_state(
        Lx, Ly, Ntot, Nmax, t, U, mu, pbc, max_dimension, max_sparse_bytes, pbc_x=pbc_x, pbc_y=pbc_y
    )
    raw = np.asarray(raw)
    if np.iscomplexobj(raw) or raw.ndim != 1 or len(raw) != len(basis) or not np.isfinite(raw).all():
        raise ValueError("ED target must be a finite real vector matching the sector")
    if np.any(raw == 0):
        raise ValueError("ED target contains exact zeros; log-target fitting rejects zero targets")
    if np.any(raw > 0) and np.any(raw < 0):
        raise ValueError("ED target has mixed signs; a positive-amplitude model cannot fit it")
    raw = np.array(raw, dtype=np.float64, copy=True)
    raw /= np.linalg.norm(raw)
    magnitudes = np.abs(raw)
    target = torch.from_numpy(np.log(np.maximum(magnitudes, eps))).to(dtype=dtype)
    x = torch.from_numpy(basis_to_one_hot(basis, Lx * Ly, Nmax)).to(dtype=dtype)
    applied = H @ raw
    raw_energy = float(raw @ applied)
    floored = np.maximum(magnitudes, eps)
    floored /= np.linalg.norm(floored)
    metadata = {
        "Lx": Lx, "Ly": Ly, "Ntot": int(Ntot), "Nmax": int(Nmax), "t": t, "U": U, "mu": mu,
        "pbc": pbc_x if pbc_x == pbc_y else None, "pbc_x": pbc_x, "pbc_y": pbc_y,
        "dimension": len(basis), "eps": eps, "clamped_target_count": int(np.count_nonzero(magnitudes < eps)),
        "zero_target_count": 0, "target_sign": "single_sign_up_to_global_phase", "loss_definition": LOSS_DEFINITION,
        "raw_ed_reference": {"solver_energy": float(energy), "energy": raw_energy,
                             "energy_variance": float(np.linalg.norm(applied - raw_energy * raw)**2),
                             "norm": float(np.linalg.norm(raw)), "amplitude_overlap_abs_unsquared": 1.0,
                             "minimum_abs_amplitude": float(magnitudes.min())},
        "floored_target_overlap_with_raw_ed_abs_unsquared": float(abs(raw @ floored)),
    }
    return {"x": x, "target_log": target, "raw_ed_state": raw, "H": H,
            "basis": basis, "energy": float(energy), "metadata": metadata}


def _device_dataset(model, dataset):
    parameter = next(model.parameters())
    x = torch.as_tensor(dataset["x"], device=parameter.device, dtype=parameter.dtype)
    y = torch.as_tensor(dataset["target_log"], device=parameter.device, dtype=parameter.dtype).detach()
    dimension = dataset["H"].shape[0]
    if (len(x) != dimension or len(dataset["basis"]) != dimension or y.shape != (dimension,)
            or x.shape != (dimension, model.n_sites, model.Nmax + 1)):
        raise ValueError("x and targets must cover the entire Hamiltonian sector")
    raw = torch.as_tensor(dataset["raw_ed_state"], device=parameter.device, dtype=parameter.dtype).detach()
    return {**dataset, "x": x, "target_log": y, "raw_ed_tensor": raw}


def compare_predictions(model, dataset, loss="log_mse"):
    """Report both objectives, labeled gradients and physical raw-ED metrics.

    Legacy gradient_norm remains the log-MSE gradient norm for either objective.
    Working larger-reference assessment applies to n_sites>=9. This threshold
    is a reporting convention, not a physics criterion.
    """
    if loss not in LOSS_NAMES:
        raise ValueError(f"loss must be one of {LOSS_NAMES}")
    dataset = _device_dataset(model, dataset)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("model must have trainable parameters")
    model.eval()
    with torch.enable_grad():
        logs = model(dataset["x"]).reshape(-1)
        log_mse = normalized_log_mse(logs, dataset["target_log"])
        objective = log_mse if loss == "log_mse" else _training_loss(logs, dataset, loss)
        log_gradients = torch.autograd.grad(log_mse, parameters, retain_graph=loss != "log_mse")
        selected_gradients = log_gradients if loss == "log_mse" else torch.autograd.grad(objective, parameters)
    log_gradient_norm = math.sqrt(sum(g.detach().double().square().sum().item() for g in log_gradients))
    selected_gradient_norm = math.sqrt(sum(g.detach().double().square().sum().item() for g in selected_gradients))
    logs = logs.detach().double().cpu()
    psi = (logs - 0.5 * torch.logsumexp(2 * logs, dim=0)).exp().numpy()
    applied = dataset["H"] @ psi
    energy = float(psi @ applied)
    overlap = float(abs(np.vdot(dataset["raw_ed_state"], psi)))
    error_per_site = (energy - dataset["energy"]) / model.n_sites
    threshold = ENERGY_ERROR_PER_SITE_FACTOR * dataset["metadata"]["t"]
    larger_scope = model.n_sites >= 9
    result = {"normalized_log_mse": log_mse.detach().item(), "gradient_norm": log_gradient_norm,
              "log_mse_gradient_norm": log_gradient_norm, "selected_objective_gradient_norm": selected_gradient_norm,
              "loss_name": loss, "training_loss": objective.detach().item(), "infidelity": 1 - overlap**2,
              "amplitude_overlap_abs_unsquared": overlap, "amplitude_overlap_squared": overlap**2,
              "energy": energy, "energy_per_site": energy / model.n_sites,
              "energy_error_to_ed": energy - dataset["energy"], "energy_error_per_site": error_per_site,
              "energy_variance": float(np.linalg.norm(applied - energy * psi)**2),
              "normalized_sector_norm": float(np.linalg.norm(psi)),
              "fit_gate_passed": bool(overlap >= OVERLAP_THRESHOLD and error_per_site <= threshold),
              "larger_system_fidelity_applied": larger_scope,
              "larger_system_fidelity_converged": bool(overlap >= LARGER_SYSTEM_OVERLAP_THRESHOLD) if larger_scope else None}
    if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite supervised evaluation")
    return result


def train_model_on_ed(model, dataset, lr=1e-3, epochs=2000, device="cpu", eval_every=100, loss="log_mse"):
    """Return (final model, JSON-ready report, best-by-loss CPU state_dict).

    Adam only, full batch. Select the lowest loss among initialization and every
    post-update state, not only reporting epochs. Fixed epoch budget; no early
    stopping, including when the physical fit gate passes or loss plateaus.
    """
    lr = _positive(lr, "lr")
    if loss not in LOSS_NAMES:
        raise ValueError(f"loss must be one of {LOSS_NAMES}")
    epochs, eval_every = _integer(epochs, "epochs", 1), _integer(eval_every, "eval_every", 1)
    model = model.to(device)
    dataset = _device_dataset(model, dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    initial = compare_predictions(model, dataset, loss)
    history = [{"epoch": 0, **initial, "relative_loss_decrease_since_previous_snapshot": None,
                "plateau_since_previous_snapshot": None}]
    best_epoch, best_loss = 0, initial["training_loss"]
    log_mse_at_best = initial["normalized_log_mse"]
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        objective = _training_loss(model(dataset["x"]).reshape(-1), dataset, loss)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("non-finite supervised gradient")
        optimizer.step()
        with torch.no_grad():
            post_logs = model(dataset["x"]).reshape(-1)
            current_loss = _training_loss(post_logs, dataset, loss).item()
        if current_loss < best_loss:
            best_epoch, best_loss = epoch, current_loss
            with torch.no_grad():
                log_mse_at_best = normalized_log_mse(post_logs, dataset["target_log"]).item()
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch % eval_every == 0 or epoch == epochs:
            metrics = compare_predictions(model, dataset, loss)
            previous = history[-1]["training_loss"]
            relative_decrease = (previous - metrics["training_loss"]) / max(abs(previous), 1e-12)
            history.append({"epoch": epoch, **metrics,
                            "relative_loss_decrease_since_previous_snapshot": relative_decrease,
                            "plateau_since_previous_snapshot": abs(relative_decrease) <= 1e-6})
            print(f"epoch={epoch} loss={loss} training_loss={metrics['training_loss']:.8g} log_mse={metrics['normalized_log_mse']:.8g} "
                  f"overlap={metrics['amplitude_overlap_abs_unsquared']:.9f} "
                  f"energy_error/site={metrics['energy_error_per_site']:.8g} variance={metrics['energy_variance']:.8g} "
                  f"selected_gradient_norm={metrics['selected_objective_gradient_norm']:.8g} "
                  f"log_mse_gradient_norm={metrics['log_mse_gradient_norm']:.8g} strict_gate={metrics['fit_gate_passed']} "
                  f"larger_fidelity={metrics['larger_system_fidelity_converged']}")
    report = {"history": history, "final_epoch": epochs,
              "final_metrics": {key: value for key, value in history[-1].items() if key != "epoch"},
              "gradient_definition": "gradient_norm and log_mse_gradient_norm are log-MSE gradient norms; selected_objective_gradient_norm is the gradient norm of loss_name; all Euclidean at the reported weights",
              "loss_name": loss, "training_loss_definition": LOSS_DEFINITION if loss == "log_mse" else INFIDELITY_DEFINITION,
              "best_epoch": best_epoch, "best_training_loss": best_loss,
              "best_normalized_log_mse": best_loss if loss == "log_mse" else None,
              "normalized_log_mse_at_selected_checkpoint": log_mse_at_best,
              "selection": ("strict minimum normalized full-sector log-amplitude MSE over initialization and every post-update state; ties keep earliest"
                            if loss == "log_mse" else "strict minimum full-sector raw-ED infidelity over initialization and every post-update state; ties keep earliest; not selected by log-MSE"),
              "stop": "fixed epoch budget; no early stop on gate or plateau",
              "plateau_definition": "absolute relative selected-objective loss change <=1e-6 between reporting snapshots; descriptive only",
              "larger_system_fidelity": {"scope": "working assessment for n_sites>=9, including rectangles; null outside scope; not a physics criterion",
                                         "metric": "amplitude_overlap_abs_unsquared", "minimum": LARGER_SYSTEM_OVERLAP_THRESHOLD,
                                         "energy_and_variance": "reported separately as diagnostics; not part of this fidelity criterion"},
              "fit_gate": {"amplitude_overlap_abs_unsquared_min": OVERLAP_THRESHOLD,
                           "energy_error_per_site_max": ENERGY_ERROR_PER_SITE_FACTOR * dataset["metadata"]["t"],
                           "combination": "both conditions must pass against raw ED; failure makes no architecture-capacity claim"}}
    return model, report, best_state


def _output_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or path == ROOT or path.exists():
        raise ValueError("output must be a new file under pythonTest/2D/; existing results cannot be overwritten")
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError("output parent must be a directory")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=list(ARCH_BUILDERS), default="FCNet")
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=None, help="default: Ly=Lx when omitted")
    parser.add_argument("--Ntot", type=int)
    parser.add_argument("--Nmax", type=int, default=4)
    add_boundary_arguments(parser)
    parser.add_argument("--width", type=int, default=16, help="FC/Conv/RBM/Transformer width; unused for Jastrow")
    parser.add_argument("--hidden-layers", "--depth", dest="hidden_layers", type=int, default=2, help="FC hidden layer count; unused for other architectures")
    parser.add_argument("--pair-sharing", choices=["displacement", "reflection_orbit"], default="displacement", help="Jastrow table sharing; reflection_orbit is cylinder-only")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=LOSS_NAMES, default="log_mse", help="primary log_mse or experimental raw-ED infidelity; Adam for either")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", help="cpu, cuda[:index], or auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--disable-tf32", action="store_true", help="on CUDA disable matmul/cuDNN TF32 before forwards; default preserves backend flags")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-12)
    for name, default in (("t", 1.0), ("U", 16.74), ("mu", 0.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--max-ed-dimension", type=int, default=DEFAULT_MAX_DIMENSION)
    parser.add_argument("--max-ed-bytes", type=int, default=DEFAULT_MAX_SPARSE_BYTES)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output", type=Path, default=Path("outputs/ed-supervised") / tag / "best.pt")
    args = parser.parse_args(argv)
    try:
        args.Lx, args.Ly = lattice_shape(args.Lx, args.Ly)
        resolve_cli_boundaries(args)
        if args.arch == "GroupEquivariantConv2DNet" and (args.Lx != args.Ly or not args.pbc_x or not args.pbc_y):
            raise ValueError("GroupEquivariantConv2DNet requires a square periodic in both axes")
        if args.arch == "JastrowNet" and args.pair_sharing == "reflection_orbit" and (args.pbc_x or not args.pbc_y):
            raise ValueError("reflection_orbit requires --no-pbc-x --pbc-y")
        args.Ntot = args.Lx * args.Ly if args.Ntot is None else args.Ntot
        if args.Nmax < 0 or not 0 <= args.Ntot <= args.Lx * args.Ly * args.Nmax:
            raise ValueError("invalid occupation cutoff or fixed-number sector")
        args.t, args.lr, args.eps = _positive(args.t, "t"), _positive(args.lr, "lr"), _positive(args.eps, "eps")
        args.U, args.mu = _physical_scalar(args.U, "U"), _physical_scalar(args.mu, "mu")
        if args.eps >= 1 or not 0 <= args.seed < 2**63:
            raise ValueError("eps must be <1 and seed must be in [0, 2**63)")
        for name in ("epochs", "width", "hidden_layers", "eval_every", "max_ed_dimension", "max_ed_bytes"):
            _integer(getattr(args, name), name, 1)
        if args.arch == "Conv2DTransformerNet" and args.width % 4:
            raise ValueError("Conv2DTransformerNet width must be divisible by four heads")
        args.output = _output_path(args.output)
        args.sidecar = _output_path(str(args.output) + ".json")
        if args.device != "auto":
            device = torch.device(args.device)
            if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
                raise ValueError("device must be cpu, cuda[:index], or auto")
            if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
                raise ValueError("requested CUDA device is unavailable")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    recorded_at = datetime.now(timezone.utc).isoformat()
    paths = [ROOT / name for name in ("lattice.py", "exact_diagonalization.py", "models.py", "train_ed.py")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    precision = {"device": str(device), "dtype": args.dtype,
                 "disable_tf32_requested": args.disable_tf32,
                 "disable_tf32_applied": device.type == "cuda" and args.disable_tf32,
                 "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    dtype = getattr(torch, args.dtype)
    dataset = make_ed_dataset(args.Lx, args.Ly, args.Ntot, args.Nmax, args.t, args.U, args.mu, args.pbc,
                              args.eps, dtype, args.max_ed_dimension, args.max_ed_bytes,
                              pbc_x=args.pbc_x, pbc_y=args.pbc_y)
    settings = {} if args.arch == "JastrowNet" else {"n_hidden" if args.arch == "RBMNet" else "hidden_dim": args.width}
    if args.arch == "FCNet":
        settings["hidden_layers"] = args.hidden_layers
    if args.arch == "JastrowNet":
        settings["pair_sharing"] = args.pair_sharing
    model = build_model(args.arch, args.Lx, args.Ly, args.Nmax, args.pbc,
                        pbc_x=args.pbc_x, pbc_y=args.pbc_y, **settings).to(device=device, dtype=dtype)
    model, report, best_state = train_model_on_ed(model, dataset, args.lr, args.epochs, device, args.eval_every, args.loss)
    report.update({"status": "completed", "dataset": dataset["metadata"],
                   "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   "resolved_device": str(device), "width_applied": None if args.arch == "JastrowNet" else args.width,
                   "precision": precision,
                   "parameter_count": parameter_count(model), "source_sha256_at_start": hashes,
                   "source_hashes_recorded_at": recorded_at,
                   "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                   "checkpoint_kind": "best_by_loss_inference_only_no_optimizer_or_exact_resume",
                   "reference_validation_scope": "analytic spectra, independent ladder operators and tiny dense eigensolves; no DMRG comparison",
                   "interpretation": "Supervised fitting is the first gate. Failure alone does not identify architecture capacity or authorize a new VMC campaign."})
    # Preserve final metrics above, then stage the selected best state and verify
    # physical metrics through the inference loader before publishing either file.
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".ed-fit-stage-") as directory:
        checkpoint, sidecar = Path(directory) / "best.pt", Path(directory) / "report.json"
        save_model(checkpoint, model, args.Ntot, args.t, args.U, args.mu)
        saved_model, metadata = load_model(checkpoint, device=device)
        report["best_checkpoint_metrics"] = compare_predictions(saved_model, dataset, args.loss)
        report["checkpoint_metadata"] = metadata
        report["wall_seconds_including_ed_fit_and_staging"] = time.perf_counter() - started
        report["staging"] = "validated before two individually atomic replacements; not a cross-file atomic transaction"
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.output.exists() or args.sidecar.exists():
            raise ValueError("output appeared during training; refusing to overwrite existing results")
        checkpoint.replace(args.output)
        sidecar.replace(args.sidecar)
    print(f"best_epoch={report['best_epoch']} checkpoint={args.output} report={args.sidecar}")
    return report


if __name__ == "__main__":
    main()
