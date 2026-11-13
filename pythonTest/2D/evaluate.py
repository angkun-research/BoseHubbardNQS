#!/usr/bin/env python3
"""Frozen-checkpoint measurements; no training or convergence/capacity inference.

Default: 64 walkers x 256 retained steps (~16k), 512 burn-in proposals per
walker, time blocks of 32, selected central correlation row and its reverse,
all diagonals and both directions of Hamiltonian bonds. This is not a claim
that burn-in or block length is sufficient. Check mixing and block stability.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

if __package__:
    from .lattice import bonds_2d
else:
    from lattice import bonds_2d
if __package__:
    from .models import ROOT, load_model
else:
    from models import ROOT, load_model
if __package__:
    from .observables import DEFAULT_MAX_OBSERVABLE_BYTES, measure_observables, sample_frozen
else:
    from observables import DEFAULT_MAX_OBSERVABLE_BYTES, measure_observables, sample_frozen
if __package__:
    from .vmc import generate_initial_states
else:
    from vmc import generate_initial_states


def _new_output(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or path == ROOT or path.exists():
        raise ValueError("evaluation output must be a new file under pythonTest/2D/")
    return path


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--burnin", type=int, default=512, help="proposals per walker, not lattice sweeps")
    parser.add_argument("--thin", type=int, default=1, help="proposals between retained configurations; --steps is retained time count")
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", "--batch-size", type=int, default=256)
    parser.add_argument("--max-observable-bytes", type=int, default=DEFAULT_MAX_OBSERVABLE_BYTES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["checkpoint", "float32", "float64"], default="checkpoint",
                        help="in-memory evaluation dtype; checkpoint preserves saved dtype (default)")
    parser.add_argument("--reference-sites", nargs="+", type=int, help="physical central row(s), with reverse entries/diagonals/bonds")
    parser.add_argument("--pair", nargs=2, type=int, action="append", help="explicit physical (i,j); with only --pair, evaluate exactly the supplied pairs")
    parser.add_argument("--full-rho", action="store_true", help="explicit full matrix, intended for tiny checks")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output", type=Path, default=Path("outputs/frozen-evaluations") / tag / "report.json")
    args = parser.parse_args(argv)
    try:
        if any(getattr(args, name) <= 0 for name in ("walkers", "steps", "thin", "block_length", "evaluation_batch_size", "max_observable_bytes")):
            raise ValueError("walkers, retained steps, thin, block length, batch size and byte budget must be positive")
        if args.burnin < 0 or not 0 <= args.seed < 2**63:
            raise ValueError("burnin must be nonnegative and seed must be in [0, 2**63)")
        if args.full_rho and (args.pair or args.reference_sites):
            raise ValueError("--full-rho cannot be combined with selected pairs or reference sites")
        args.output = _new_output(args.output)
        raw_path = _new_output(str(args.output) + ".samples.npz")
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise ValueError("checkpoint file does not exist")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
            raise ValueError("device must be cpu, cuda[:index], or auto")
        if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
            raise ValueError("requested CUDA device is unavailable")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.manual_seed(args.seed)
    sources = ["evaluate.py", "observables.py", "vmc.py", "models.py", "lattice.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, metadata = load_model(checkpoint, device=device)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during loading; refusing ambiguous weight provenance")
    if args.dtype != "checkpoint":
        model = model.to(dtype=getattr(torch, args.dtype))
    evaluation_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    pairs = None
    references = []
    policy = "explicit_full_matrix"
    if not args.full_rho:
        selected = list(map(tuple, args.pair or []))
        if not args.pair or args.reference_sites:
            references = args.reference_sites or [(model.Ly // 2) * model.Lx + model.Lx // 2]
            if any(not 0 <= site < model.n_sites for site in references):
                parser.error("reference site outside the lattice")
            selected.extend((i, i) for i in range(model.n_sites))
            for i in references:
                selected.extend((i, j) for j in range(model.n_sites))
                selected.extend((j, i) for j in range(model.n_sites))
            for i, j in bonds_2d(model.Lx, model.Ly, pbc_x=model.pbc_x, pbc_y=model.pbc_y):
                selected.extend(((i, j), (j, i)))
            policy = "reference_rows_and_reverses_plus_diagonals_and_directed_hamiltonian_bonds"
        else:
            policy = "exact_explicit_pair_list"
        pairs = list(dict.fromkeys(selected))
        if any(min(i, j) < 0 or max(i, j) >= model.n_sites for i, j in pairs):
            parser.error("pair site outside the lattice")
    # Check output storage before starting a potentially longer burn-in.
    if __package__:
        from .observables import check_observable_budget
    else:
        from observables import check_observable_budget
    budget = check_observable_budget(args.steps * args.walkers, model.n_sites,
                                     model.n_sites**2 if pairs is None else len(pairs), args.max_observable_bytes)
    initial = generate_initial_states(model.n_sites, metadata["Ntot"], model.Nmax, args.walkers, device)
    started = time.perf_counter()
    chains, walkers, sampling = sample_frozen(model, initial, args.steps, args.burnin, args.evaluation_batch_size,
                                             proposal_steps_per_sample=args.thin)
    measured = measure_observables(model, chains, metadata["t"], metadata["U"], metadata["mu"], args.evaluation_batch_size,
                                   pairs=pairs, block_length=args.block_length, max_observable_bytes=args.max_observable_bytes)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    statistics = {name: {key: value for key, value in stats.items() if key != "block_means"}
                  for name, stats in measured["statistics"].items()}
    report = {
        "kind": "frozen_model_observables", "status": "completed", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256, "checkpoint_metadata": metadata,
        "evaluation_dtype": evaluation_dtype,
        "source_sha256_at_start": hashes, "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "resolved_device": str(device), "axes": ["time", "walker", "physical_site"], "flattening": "i=y*Lx+x",
        "pair_policy": policy, "reference_sites": references, "one_body_pairs": measured["one_body_pairs"],
        "pair_axes": ["time", "walker", "selected_physical_pair"], "sampling": sampling,
        "time_units": {"retained_steps_per_walker": args.steps, "proposals_per_retained_step": args.thin,
                       "burnin_proposals_per_walker": args.burnin, "block_length_retained_steps": args.block_length,
                       "block_length_proposals_per_walker": args.block_length * args.thin},
        "mixing_warning": "no accepted occupation changes; no mixing or convergence inference" if not sampling["moved"] else "observed moves do not establish mixing; compare block lengths and independent frozen chains",
        "density_all_samples": measured["density"], "onsite_factorial_moment_all_samples": measured["onsite_factorial_moment"],
        "one_body_pair_means_all_samples": measured["one_body_pair_means"], "one_body_matrix_all_samples": measured["one_body_matrix"],
        "statistics": statistics, "raw_checks_before_symmetrization": measured["diagnostics"], "budget": budget,
        "raw_samples_path": str(raw_path.relative_to(ROOT)), "wall_seconds_sampling_and_measurement": time.perf_counter() - started,
        "uncertainty": "Errors are conditional block-mean estimates, not confidence intervals or ESS. Complete-time-block means are separate from all-sample means; discarded tails are explicit. Increase block length and check stability/mixing. No convergence or capacity claim.",
    }
    report = _json_value(report)
    raw = {"occupations": chains.cpu().numpy(), "physical_pairs": np.asarray(measured["one_body_pairs"], dtype=np.int64).reshape(-1, 2),
           "density": measured["density_samples"].cpu().numpy(), "onsite_factorial_moment": measured["onsite_factorial_moment_samples"].cpu().numpy(),
           "one_body_estimators": measured["one_body_pair_samples"].cpu().numpy(), "local_energy": measured["local_energy_samples"].cpu().numpy(),
           "final_walkers": walkers.cpu().numpy(),
           "retained_postburn_proposal_index": np.arange(1, args.steps + 1, dtype=np.int64) * args.thin}
    if measured["reconstructed_energy_samples"] is not None:
        raw["reconstructed_energy"] = measured["reconstructed_energy_samples"].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".frozen-stage-") as directory:
        staged_json, staged_raw = Path(directory) / "report.json", Path(directory) / "samples.npz"
        staged_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(staged_raw, **raw)
        if args.output.exists() or raw_path.exists():
            raise ValueError("output appeared during evaluation; refusing to overwrite")
        staged_raw.replace(raw_path)
        staged_json.replace(args.output)
    print(f"frozen samples={len(chains)*len(walkers)} pairs={len(measured['one_body_pairs'])} "
          f"moved_rate={sampling['moved_rate']:.4f} complete_blocks/chain={args.steps//args.block_length} "
          f"raw_Hermiticity={measured['diagnostics']['raw_mean_rho_hermiticity_max_abs']}; report={args.output}")
    return report


if __name__ == "__main__":
    main()
