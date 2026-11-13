#!/usr/bin/env python3
"""Train a 2D Bose-Hubbard log-amplitude NQS; ED is explicitly opt-in.

Example: python pythonTest/2D/train.py --Lx 2 --epochs 2 --device cpu
Relative output paths start under pythonTest/2D/. Checkpoints are inference-only;
the adjacent <checkpoint>.json stores settings, history and a fresh post-update
measurement. Variance describes local energies, not uncertainty of the mean.
Outputs are staged and validated before replacement. The two file replacements
are not a cross-file atomic transaction: interruption or an I/O failure between
them can leave a mixed pair. Measurement/serialization failures preserve both.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time

import torch

if __package__:
    from .lattice import lattice_shape, add_boundary_arguments, resolve_cli_boundaries
else:
    from lattice import lattice_shape, add_boundary_arguments, resolve_cli_boundaries
if __package__:
    from .models import ARCH_BUILDERS, ROOT, build_model, load_model, parameter_count, save_model
else:
    from models import ARCH_BUILDERS, ROOT, build_model, load_model, parameter_count, save_model
if __package__:
    from .vmc import (
        DEFAULT_MAX_SR_BYTES, DEFAULT_MAX_SR_PARAMETERS, MAX_JACOBIAN_BATCH_SIZE, MAX_VMAP_JACOBIAN_BATCH_SIZE, SR_CONVENTION, MINSR_CONVENTION,
        check_sr_budget, generate_initial_states, local_energy, native_vmc_update,
        physical_parameters, sample, sr_update,
    )
else:
    from vmc import (
        DEFAULT_MAX_SR_BYTES, DEFAULT_MAX_SR_PARAMETERS, MAX_JACOBIAN_BATCH_SIZE, MAX_VMAP_JACOBIAN_BATCH_SIZE, SR_CONVENTION, MINSR_CONVENTION,
        check_sr_budget, generate_initial_states, local_energy, native_vmc_update,
        physical_parameters, sample, sr_update,
    )


def _output_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or path == ROOT or path.is_dir():
        raise ValueError("output must be a file under pythonTest/2D/")
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError("output parent must be a directory")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=list(ARCH_BUILDERS), default="FCNet")
    parser.add_argument("--Lx", type=int, default=4)
    parser.add_argument("--Ly", type=int)
    parser.add_argument("--Ntot", type=int)
    parser.add_argument("--Nmax", type=int, default=4)
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=32)
    parser.add_argument("--hidden-layers", "--depth", dest="hidden_layers", type=int, default=2, help="FC hidden layer count; other architectures unchanged")
    parser.add_argument("--pair-sharing", choices=["displacement", "reflection_orbit"], default="displacement", help="Jastrow tables; reflection_orbit requires open-x/periodic-y cylinder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--walkers", "--n_walkers", dest="n_walkers", type=int, default=64)
    parser.add_argument("--steps", "--n_steps", dest="n_steps", type=int, default=32)
    parser.add_argument("--burnin", "--burnin_steps", dest="burnin_steps", type=int, default=32)
    parser.add_argument("--initial-burnin", type=int, default=0, help="one-time proposals per fresh walker before any measurement/update; additional to per-measurement --burnin")
    parser.add_argument("--thin", type=int, default=1, help="proposals per retained step; --steps remains retained count, burnin remains proposals")
    parser.add_argument("--seed", type=int, default=123)
    for name, default in (("t", 1.0), ("U", 16.74), ("mu", 0.0), ("lr", 0.001), ("tau", 0.001)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--lr-end", type=float, help="optional positive geometric schedule endpoint")
    parser.add_argument("--tau-end", type=float, help="ridge-only positive geometric schedule endpoint; invalid with MinSR")
    parser.add_argument("--warm-start", type=Path, help="inference weights for explicit SR+SGD; fresh optimizer and walkers, not exact resume")
    add_boundary_arguments(parser)
    parser.add_argument("--update-method", "--update_method", choices=["native", "sr"], default="native")
    parser.add_argument("--optimizer", choices=["adam", "sgd"])
    parser.add_argument("--device", default="auto", help="cpu, cuda[:index], or auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--disable-tf32", action="store_true", help="on CUDA, disable matmul and cuDNN TF32 before any forward; default leaves backend flags unchanged")
    parser.add_argument("--evaluation-batch-size", "--evaluation_batch_size", type=int, default=256)
    parser.add_argument("--max-sr-parameters", type=int, default=DEFAULT_MAX_SR_PARAMETERS)
    parser.add_argument("--max-sr-bytes", type=int, default=DEFAULT_MAX_SR_BYTES)
    parser.add_argument("--sr-solver", choices=["auto", "parameter", "sample"], default="auto")
    parser.add_argument("--sr-regularizer", choices=["ridge", "minsr"], default="ridge", help="ridge is unchanged default; MinSR uses soft spectral pseudoinverse, with tau unused")
    parser.add_argument("--minsr-rpinv", type=float, default=1e-12, help="MinSR relative spectral cutoff, power6 soft filter")
    parser.add_argument("--minsr-apinv", type=float, default=0.0, help="MinSR absolute spectral cutoff")
    parser.add_argument("--jacobian-method", choices=["vjp", "vmap"], default="vjp", help="default VJP reference; functional vmap per-example gradients are opt-in")
    parser.add_argument("--jacobian-batch-size", type=int, default=1, help=f"chunk size: VJP cap{MAX_JACOBIAN_BATCH_SIZE}, vmap cap{MAX_VMAP_JACOBIAN_BATCH_SIZE}; default1")
    parser.add_argument("--ed-reference", action="store_true")
    parser.add_argument("--max-ed-dimension", type=int, default=20_000)
    parser.add_argument("--max-ed-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--output", type=Path, default=Path("outputs/vmc_nqs.pt"))
    args = parser.parse_args(argv)
    try:
        args.Lx, args.Ly = lattice_shape(args.Lx, args.Ly)
        resolve_cli_boundaries(args)
        if args.arch == "JastrowNet" and args.pair_sharing == "reflection_orbit" and (args.pbc_x or not args.pbc_y):
            raise ValueError("reflection_orbit requires --no-pbc-x --pbc-y")
        args.Ntot = args.Lx * args.Ly if args.Ntot is None else args.Ntot
        if args.Nmax < 0 or not 0 <= args.Ntot <= args.Lx * args.Ly * args.Nmax:
            raise ValueError("invalid cutoff or fixed particle number")
        for name in ("hidden_dim", "hidden_layers", "epochs", "n_walkers", "n_steps", "thin", "evaluation_batch_size",
                     "max_sr_parameters", "max_sr_bytes", "max_ed_dimension", "max_ed_bytes", "jacobian_batch_size"):
            if getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if args.arch == "Conv2DTransformerNet" and args.hidden_dim % 4:
            raise ValueError("Conv2DTransformerNet hidden_dim must be divisible by four heads")
        if args.burnin_steps < 0 or args.initial_burnin < 0 or not 0 <= args.seed < 2**63:
            raise ValueError("burnin and initial-burnin must be nonnegative and seed must be in [0, 2**63)")
        args.t, args.U, args.mu = physical_parameters(args.t, args.U, args.mu)
        if any(not math.isfinite(value) or value <= 0 for value in (args.lr, args.tau)):
            raise ValueError("lr and tau must be finite and positive")
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in (args.lr_end, args.tau_end)):
            raise ValueError("lr-end and tau-end must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in (args.minsr_rpinv, args.minsr_apinv)):
            raise ValueError("MinSR cutoffs must be finite and nonnegative")
        if args.sr_regularizer == "minsr":
            if args.update_method != "sr":
                raise ValueError("MinSR requires --update-method sr with native SGD")
            if args.tau_end is not None:
                raise ValueError("tau-end is ridge-only; MinSR has no tau schedule")
            args.tau = None
        cap = MAX_JACOBIAN_BATCH_SIZE if args.jacobian_method == "vjp" else MAX_VMAP_JACOBIAN_BATCH_SIZE
        if args.jacobian_batch_size > cap:
            raise ValueError(f"jacobian-batch-size must be <= {cap} for {args.jacobian_method}")
        args.output = _output_path(args.output)
        args.sidecar = _output_path(str(args.output) + ".json")
        if args.sidecar == args.output:
            raise ValueError("checkpoint and sidecar paths must differ")
        if args.device != "auto":
            device = torch.device(args.device)
            if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
                raise ValueError("device must be cpu, cuda[:index], or auto")
            if device.type == "cuda" and (not torch.cuda.is_available()
                                          or (device.index or 0) >= torch.cuda.device_count()):
                raise ValueError("requested CUDA device is unavailable")
        args.optimizer = args.optimizer or ("adam" if args.update_method == "native" else "sgd")
        if args.update_method == "sr" and args.optimizer != "sgd":
            raise ValueError("SR requires torch.optim.SGD; use --update-method sr --optimizer sgd (Adam is unsupported)")
        if args.warm_start is not None:
            args.warm_start = (ROOT / args.warm_start).resolve() if not args.warm_start.is_absolute() else args.warm_start.resolve()
            if args.update_method != "sr" or args.optimizer != "sgd":
                raise ValueError("warm-start requires explicit --update-method sr --optimizer sgd; it restores inference weights only")
            if not args.warm_start.is_file():
                raise ValueError("warm-start checkpoint does not exist")
            if args.warm_start == args.output or args.output.exists() or args.sidecar.exists():
                raise ValueError("warm-start output must be new; never overwrite the source checkpoint or old results")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return args


def _schedule_value(start, end, epoch, epochs):
    """Geometric endpoints; omission is the exact previous constant value."""
    if end is None or start == end or epochs == 1 or epoch == 1:
        return start
    if epoch == epochs:
        return end
    return math.exp(math.log(start) + (math.log(end) - math.log(start)) * (epoch - 1) / (epochs - 1))


def _load_warm_start(args, requested_model):
    path = args.warm_start
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, metadata = load_model(path, device="cpu")
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("warm-start checkpoint changed during load")
    expected = {"arch": args.arch, "Lx": args.Lx, "Ly": args.Ly, "Nmax": args.Nmax, "Ntot": args.Ntot,
                "pbc_x": args.pbc_x, "pbc_y": args.pbc_y, "t": args.t, "U": args.U, "mu": args.mu,
                "architecture_settings": requested_model.architecture_settings}
    mismatches = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatches:
        raise ValueError(f"warm-start checkpoint mismatch: {', '.join(mismatches)}")
    return loaded, {"checkpoint": str(path), "checkpoint_sha256": checksum,
                    "source_dtype": metadata["dtype"], "training_dtype": args.dtype,
                    "restored": "inference weights only", "optimizer": "fresh torch.optim.SGD, no restored state",
                    "walkers": "fresh fixed-number initialization and burn-in; no restored walkers or RNG state",
                    "exact_resume": False}


def _measurement(model, walkers, args):
    samples, walkers, diagnostics = sample(
        model, walkers, args.n_steps, args.burnin_steps, args.evaluation_batch_size,
        proposal_steps_per_sample=args.thin,
    )
    energies = local_energy(model, samples, args.t, args.U, args.mu, args.evaluation_batch_size)
    statistics = energies.double()
    mean = statistics.mean().item()
    variance = statistics.var(unbiased=False).item()
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise RuntimeError("non-finite measurement summary")
    summary = {
        "energy": mean,
        "energy_per_site": mean / model.n_sites,
        "local_energy_variance": variance,
        "n_samples": len(samples), "sampling": diagnostics,
    }
    return samples, walkers, energies, summary


def main(argv=None):
    args = parse_args(argv)
    source_recorded_at = datetime.now(timezone.utc).isoformat()
    sources = ["train.py", "vmc.py", "models.py", "lattice.py", "exact_diagonalization.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    precision = {"device": str(device), "dtype": args.dtype, "disable_tf32_requested": args.disable_tf32,
                 "disable_tf32_applied": device.type == "cuda" and args.disable_tf32,
                 "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    settings = {} if args.arch == "JastrowNet" else {
        "n_hidden" if args.arch == "RBMNet" else "hidden_dim": args.hidden_dim
    }
    if args.arch == "FCNet":
        settings["hidden_layers"] = args.hidden_layers
    if args.arch == "JastrowNet":
        settings["pair_sharing"] = args.pair_sharing
    model = build_model(args.arch, args.Lx, args.Ly, args.Nmax, args.pbc,
                        pbc_x=args.pbc_x, pbc_y=args.pbc_y, **settings).to(
        device=device, dtype=getattr(torch, args.dtype)
    ).eval()
    warm_start = None
    if args.warm_start is not None:
        model, warm_start = _load_warm_start(args, model)
        model = model.to(device=device, dtype=getattr(torch, args.dtype)).eval()
    sr_budget = None
    if args.update_method == "sr":
        sr_budget = check_sr_budget(model, args.n_steps * args.n_walkers, args.max_sr_parameters, args.max_sr_bytes,
                                    solver=args.sr_solver, jacobian_batch_size=args.jacobian_batch_size, jacobian_method=args.jacobian_method,
                                    regularizer=args.sr_regularizer)
        convention = SR_CONVENTION if args.sr_regularizer == "ridge" else MINSR_CONVENTION
        print(f"SR convention: {convention}; solver={sr_budget['solver']} "
              f"P={sr_budget['parameters']} M={sr_budget['samples']} solve_size={sr_budget['solve_dimension']} "
              f"workspace estimate={sr_budget['estimated_bytes']} bytes")
        print(f"SR Jacobian method={args.jacobian_method} chunk={args.jacobian_batch_size} (vjp1=per-sample reference)")
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=args.lr)
    ed_energy = None
    if args.ed_reference:
        if __package__:
            from .exact_diagonalization import exact_ground_state
        else:
            from exact_diagonalization import exact_ground_state

        ed_energy, _, _, _ = exact_ground_state(
            args.Lx, args.Ly, args.Ntot, args.Nmax, args.t, args.U, args.mu, args.pbc,
            max_dimension=args.max_ed_dimension, max_sparse_bytes=args.max_ed_bytes,
            pbc_x=args.pbc_x, pbc_y=args.pbc_y,
        )
        print(f"ED reference energy: {ed_energy:.10f}")
    print(f"device={device} arch={args.arch} parameters={parameter_count(model)} "
          f"update={args.update_method} optimizer={args.optimizer} lr={args.lr}")
    walkers = generate_initial_states(model.n_sites, args.Ntot, args.Nmax, args.n_walkers, device)
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    initial_equilibration = {
        "executed": args.initial_burnin > 0,
        "proposal_steps_per_walker": args.initial_burnin,
        "total_walker_proposals": args.initial_burnin * args.n_walkers,
        "retained_samples": 0,
        "timing": "once on fresh walkers before any measurement or optimizer step",
        "units": "one bond/direction proposal per walker, not a lattice sweep; independent of thin",
    }
    if args.initial_burnin:
        _, walkers, _ = sample(model, walkers, 0, args.initial_burnin, args.evaluation_batch_size)
        print(f"initial burn-in completed: {args.initial_burnin} proposals per walker "
              f"({initial_equilibration['total_walker_proposals']} total); no updates or retained samples")
    for epoch in range(1, args.epochs + 1):
        lr = _schedule_value(args.lr, args.lr_end, epoch, args.epochs)
        tau = _schedule_value(args.tau, args.tau_end, epoch, args.epochs) if args.sr_regularizer == "ridge" else None
        for group in optimizer.param_groups:
            group["lr"] = lr
        samples, walkers, energies, measurement = _measurement(model, walkers, args)
        if args.update_method == "native":
            native_vmc_update(model, samples, energies, optimizer, args.evaluation_batch_size)
        else:
            _, measurement["sr"] = sr_update(
                model, samples, energies, optimizer, tau, args.max_sr_parameters, args.max_sr_bytes,
                solver=args.sr_solver, return_diagnostics=True, jacobian_batch_size=args.jacobian_batch_size, jacobian_method=args.jacobian_method,
                regularizer=args.sr_regularizer, rpinv=args.minsr_rpinv, apinv=args.minsr_apinv,
            )
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError("non-finite updated parameters; checkpoint not saved")
        history.append({"epoch": epoch, "measurement_timing": "before_update", "lr": lr, "tau": tau, **measurement})
        tau_text = f"{tau:.6g}" if tau is not None else "unused"
        print(f"epoch={epoch} energy={measurement['energy']:.8f} "
              f"variance={measurement['local_energy_variance']:.6g} "
              f"lr={lr:.6g} tau={tau_text} "
              f"accepted={measurement['sampling']['accepted_rate']:.3f} moved={measurement['sampling']['moved_rate']:.3f}")
    # Stage on the checkpoint filesystem, reload those exact weights, measure,
    # and serialize finite JSON before touching either destination file.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".vmc-stage-") as directory:
        checkpoint = Path(directory) / "model.pt"
        sidecar = Path(directory) / "report.json"
        save_model(checkpoint, model, args.Ntot, args.t, args.U, args.mu)
        final_model, metadata = load_model(checkpoint, device=device)
        _, _, _, final = _measurement(final_model, walkers, args)
        final["measurement_timing"] = "after_final_update_from_saved_checkpoint"
        peak_bytes = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_bytes = torch.cuda.max_memory_allocated(device)
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        settings["resolved_device"] = str(device)
        report = {
            "settings": settings, "checkpoint_metadata": metadata, "history": history,
            "source_hashes_recorded_at": source_recorded_at, "source_sha256_at_start": source_hashes, "command": command,
            "warm_start": warm_start,
            "precision": precision,
            "initial_equilibration": initial_equilibration,
            "schedules": {"kind": "geometric endpoints; omitted end is constant; single epoch uses start",
                          "lr_start": args.lr, "lr_end": args.lr_end, "tau_start": args.tau, "tau_end": args.tau_end,
                          "tau_usage": "ridge shift" if args.sr_regularizer == "ridge" else "unused; no tau schedule or ridge"},
            "final_measurement": final, "ed_energy": ed_energy,
            "sr_convention": (SR_CONVENTION if args.sr_regularizer == "ridge" else MINSR_CONVENTION) if args.update_method == "sr" else None,
            "sr_regularizer": args.sr_regularizer if args.update_method == "sr" else None,
            "minsr_cutoffs": {"rpinv": args.minsr_rpinv, "apinv": args.minsr_apinv, "soft_power": 6} if args.sr_regularizer == "minsr" else None,
            "sr_budget": sr_budget, "training_and_final_measurement_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": peak_bytes,
            "cuda_peak_allocated_scope": (
                "Training-device allocator peak, reset after walker initialization through final measurement; "
                "includes resident baseline, updates and staged model reload; excludes setup/ED startup; "
                "allocated bytes, not reserved bytes."
            ) if device.type == "cuda" else None,
            "checkpoint_kind": "inference_only_no_exact_training_resume",
            "uncertainty": "No mean-energy error bar; autocorrelation and effective sample size are not estimated.",
        }
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        args.sidecar.parent.mkdir(parents=True, exist_ok=True)
        # Individually atomic replacements, not a cross-file atomic transaction.
        checkpoint.replace(args.output)
        sidecar.replace(args.sidecar)
    print(f"final energy={final['energy']:.8f}; checkpoint={args.output}; history={args.sidecar}")
    return report


if __name__ == "__main__":
    main()
