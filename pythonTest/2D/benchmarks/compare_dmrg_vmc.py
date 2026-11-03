#!/usr/bin/env python3
"""Compare one finite-chi DMRG phase with frozen NQS reports, without ranking.

Paths are relative to pythonTest/2D/ unless absolute. Example from repo root:
  .venv/bin/python -B pythonTest/2D/benchmarks/compare_dmrg_vmc.py \
    --reference outputs/dmrg-4x4-cylinder-chi128-spot256 --phase chi256 \
    --reports outputs/2026-10-31-cylinder-pilots/frozen/FCNet-4x4-seed321.json \
    --output benchmarks/results/dmrg-vmc-comparison.json

The last DMRG phase is the default. Means and errors use the frozen report's
statistics.mean (complete blocks, or all samples if no complete block exists).
Conditional SEM is the elementwise maximum of available block/between-chain
SEMs; null means neither exists. This is not a confidence interval, an ESS
estimate, or a finite-chi DMRG error bar. No jobs or model evaluations are run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

from validate_dmrg_reference import ROOT, csv_rows, read_phase, resolve

sys.path.insert(0, str(ROOT))
from models import BOND_CONVENTION, HAMILTONIAN_CONVENTION


PARAMETERS = ("Lx", "Ly", "Ntot", "Nmax", "t", "U", "mu", "pbc_x", "pbc_y")


def statistic(stats, shape):
    mean = np.asarray(stats["mean"], dtype=float)
    if mean.shape != shape or not np.isfinite(mean).all():
        raise ValueError("invalid statistic mean shape or nonfinite value")
    errors = []
    for key in ("block_standard_error", "between_chain_standard_error"):
        if stats.get(key) is not None:
            error = np.asarray(stats[key], dtype=float)
            if error.shape != shape or not np.isfinite(error).all() or (error < 0).any():
                raise ValueError(f"invalid {key}")
            errors.append(error)
    return mean, np.maximum.reduce(errors) if errors else None


def compare_report(report, settings, density, rho, reference_energy):
    if report.get("status") != "completed" or report.get("kind") != "frozen_model_observables":
        raise ValueError("expected a completed frozen_model_observables report")
    metadata = report["checkpoint_metadata"]
    for key in PARAMETERS:
        if key not in metadata or metadata[key] != settings[key]:
            raise ValueError(f"geometry/parameter mismatch: {key}")
    for key, expected in (("hamiltonian_convention", HAMILTONIAN_CONVENTION),
                          ("bond_convention", BOND_CONVENTION)):
        if metadata.get(key) != expected:
            raise ValueError(f"convention mismatch: {key}")
    for flattening in (report.get("flattening"), metadata.get("flattening")):
        if flattening != "i=y*Lx+x":
            raise ValueError("physical index mismatch: expected i=y*Lx+x")
    n = settings["Lx"] * settings["Ly"]
    pairs = report["one_body_pairs"]
    if any(len(pair) != 2 or any(type(i) is not int or not 0 <= i < n for i in pair) for pair in pairs):
        raise ValueError("invalid physical pair indices")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("duplicate physical pairs")
    stats = report["statistics"]
    energy, energy_sem = statistic(stats["energy"], ())
    measured_density, density_sem = statistic(stats["density"], (n,))
    measured_rho, rho_sem = statistic(stats["one_body"], (len(pairs),))
    energy_error = float((energy - reference_energy) / n)
    result = {
        "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
        "checkpoint_metadata": metadata, "sampling": report["sampling"],
        "energy_per_site": float(energy / n), "reference_energy_per_site": reference_energy / n,
        "energy_error_per_site": energy_error, "absolute_energy_error_per_site": abs(energy_error),
        "conditional_energy_sem_per_site": None if energy_sem is None else float(energy_sem / n),
        "all_samples_energy_per_site": float(stats["energy"]["all_samples_mean"]) / n,
        "density": [], "rho": [],
        "statistics_scope": {name: {key: value for key, value in stats[name].items()
                                   if key not in ("mean", "all_samples_mean", "block_standard_error", "between_chain_standard_error")}
                             for name in ("energy", "density", "one_body")},
    }
    for i in range(n):
        result["density"].append({"i": i, "x": i % settings["Lx"], "y": i // settings["Lx"],
            "nqs": float(measured_density[i]), "reference": float(density[i]),
            "difference": float(measured_density[i] - density[i]),
            "conditional_sem": None if density_sem is None else float(density_sem[i])})
    for k, (i, j) in enumerate(pairs):
        delta = measured_rho[k] - rho[i, j]
        result["rho"].append({"i": i, "j": j, "x_i": i % settings["Lx"], "y_i": i // settings["Lx"],
            "x_j": j % settings["Lx"], "y_j": j // settings["Lx"], "nqs_real": float(measured_rho[k]),
            "reference_real": float(rho[i, j].real), "reference_imag": float(rho[i, j].imag),
            "difference_real": float(delta.real), "difference_imag": float(delta.imag),
            "absolute_difference": float(abs(delta)), "conditional_sem_real": None if rho_sem is None else float(rho_sem[k])})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--phase", help="DMRG phase label; default last recorded phase")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    directory, output = resolve(args.reference), resolve(args.output)
    try:
        if not directory.is_relative_to(ROOT) or not directory.is_dir():
            raise ValueError("reference must be an existing directory under 2D/")
        if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
            raise ValueError("output must be a new file under benchmarks/results/")
        metadata = tomllib.loads((directory / "metadata.toml").read_text())
        if metadata["status"] != "completed":
            raise ValueError("DMRG reference is not completed")
        if metadata["conventions"]["flattening"] != "i=y*Lx+x; physical zero-based":
            raise ValueError("DMRG physical index mismatch")
        phases = metadata["phases"]
        phase = phases[-1] if args.phase is None else next((p for p in phases if p["label"] == args.phase), None)
        if phase is None:
            raise ValueError("unknown DMRG phase")
        settings = metadata["settings"]
        bonds = [(int(row["i"]), int(row["j"])) for row in csv_rows(directory / "bonds.csv")]
        density, _, rho, _ = read_phase(directory, phase, settings, bonds)
        paths = [resolve(path) for path in args.reports]
        if any(not path.is_relative_to(ROOT) for path in paths):
            raise ValueError("frozen reports must be under 2D/")
        files = [directory / "metadata.toml", directory / "bonds.csv"]
        files += [directory / f"{phase['label']}-{suffix}.csv" for suffix in ("density", "onsite_factorial", "rho")]
        comparisons = []
        for path in paths:
            row = compare_report(json.loads(path.read_text()), settings, density, rho, phase["energy"])
            row["report"] = str(path.relative_to(ROOT))
            comparisons.append(row)
        result = {
            "kind": "finite_chi_dmrg_frozen_nqs_comparison", "status": "completed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "reference": str(directory.relative_to(ROOT)), "reference_phase": phase,
            "matched_parameters": {key: settings[key] for key in PARAMETERS}, "physical_index": "i=y*Lx+x",
            "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files + paths},
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("compare_dmrg_vmc.py", "validate_dmrg_reference.py")},
            "mean_policy": "statistics.mean: complete time blocks, all samples only when zero complete blocks; all-sample energy separately labeled",
            "conditional_error_policy": "elementwise max of available block SEM and between-chain SEM; null if neither; no DMRG error included",
            "interpretation": "Signed differences from a finite-chi reference, not exact ground-state errors. Conditional sampling errors require stationary, independent chains and decorrelated blocks. No fidelity, convergence, capacity or architecture ranking claim.",
            "comparisons": comparisons,
        }
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(serialized)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        parser.error(str(error))
    print(f"Compared {len(comparisons)} frozen reports against {phase['label']}; output={output}")
    return result


if __name__ == "__main__":
    main()
