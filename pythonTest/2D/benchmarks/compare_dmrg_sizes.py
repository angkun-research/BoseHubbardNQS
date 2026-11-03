#!/usr/bin/env python3
"""Read-only central-strip comparison of two even-Lx, open-x/periodic-y cylinders.

Paths relative to pythonTest/2D/. No sampling or DMRG jobs are launched.
Example:
  .venv/bin/python -B pythonTest/2D/benchmarks/compare_dmrg_sizes.py \
    --references outputs/dmrg-4x4-cylinder-chi128-spot256 outputs/dmrg-6x4-cylinder-chi128 \
    --output benchmarks/results/dmrg-4x4-6x4-central-strip.json

Optional --small-nqs and --large-nqs read frozen JSON plus its saved NPZ for
the respective reference. --block-length overrides the saved time-block length.
Composite observables are formed per time/walker sample before blocking.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import zipfile

import numpy as np

from validate_dmrg_reference import ROOT, csv_rows, read_phase, resolve
from compare_dmrg_vmc import PARAMETERS, HAMILTONIAN_CONVENTION, BOND_CONVENTION

sys.path.insert(0, str(ROOT))
import torch
from observables import block_mean_statistics


def summarize(directory, metadata, label):
    settings = metadata["settings"]
    Lx, Ly = settings["Lx"], settings["Ly"]
    phase = next((p for p in metadata["phases"] if p["label"] == label), None)
    if phase is None:
        raise ValueError(f"missing phase {label} in {directory}")
    bonds = [(int(row["i"]), int(row["j"])) for row in csv_rows(directory / "bonds.csv")]
    density, factorial, rho, _ = read_phase(directory, phase, settings, bonds)
    columns = [Lx // 2 - 1, Lx // 2]
    sites = [y * Lx + x for y in range(Ly) for x in columns]
    local = settings["U"] / 2 * factorial - settings["mu"] * density
    direction_bonds = {"x": [], "y": []}
    for i, j in bonds:
        half_bond_energy = -settings["t"] / 2 * (rho[i, j] + rho[j, i]).real
        local[i] += half_bond_energy
        local[j] += half_bond_energy
        if i in sites and j in sites:
            direction_bonds["x" if i // Lx == j // Lx else "y"].append((i, j))
    np.testing.assert_allclose(local.sum(), phase["energy"], atol=1e-8, rtol=1e-8)
    correlations = {}
    for direction, pairs in direction_bonds.items():
        correlations[direction] = {"unique_physical_bonds": pairs, "bond_count": len(pairs),
            "G": float(np.mean([(rho[i, j] + rho[j, i]).real / 2 for i, j in pairs]))}
    used = [directory / "metadata.toml", directory / "bonds.csv"]
    used += [directory / f"{label}-{suffix}.csv" for suffix in ("density", "onsite_factorial", "rho")]
    return {
        "reference": str(directory.relative_to(ROOT)), "settings": settings, "phase": phase,
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in used},
        "whole_lattice_energy_per_site": phase["energy"] / (Lx * Ly),
        "open_end_site_fraction": 2 / Lx,
        "central_x_columns": columns, "central_physical_sites": sites,
        "central_sites": [{"i": i, "x": i % Lx, "y": i // Lx, "strip_x": i % Lx - columns[0],
                           "density": float(density[i]), "onsite_factorial_moment": float(factorial[i]),
                           "local_energy_density": float(local[i])} for i in sites],
        "central_mean_density": float(density[sites].mean()),
        "central_mean_local_energy_density": float(local[sites].mean()),
        "central_direction_resolved_NN_G": correlations,
        "local_energy_reconstruction_difference": float(local.sum() - phase["energy"]),
    }


def differences(first, second):
    result = {key: second[key] - first[key] for key in
              ("whole_lattice_energy_per_site", "central_mean_density", "central_mean_local_energy_density")}
    result["central_direction_resolved_NN_G"] = {
        direction: second["central_direction_resolved_NN_G"][direction]["G"] - first["central_direction_resolved_NN_G"][direction]["G"]
        for direction in ("x", "y")}
    result["matched_local_energy_density"] = [
        {"strip_x": a["strip_x"], "y": a["y"], "first_physical_i": a["i"], "second_physical_i": b["i"],
         "difference": b["local_energy_density"] - a["local_energy_density"]}
        for a, b in zip(first["central_sites"], second["central_sites"])]
    return result


def central_samples(density, factorial, rho, pairs, settings, bonds, sites):
    """Composite values keep all same-configuration covariance before blocking."""
    lookup = {tuple(pair): k for k, pair in enumerate(pairs)}
    required = {(a, b) for i, j in bonds for a, b in ((i, j), (j, i))}
    missing = sorted(required - lookup.keys())
    if missing:
        raise ValueError(f"missing directed Hamiltonian pairs: {missing}")
    local = settings["U"] / 2 * factorial - settings["mu"] * density
    directions = {"x": [], "y": []}
    for i, j in bonds:
        symmetric = (rho[..., lookup[i, j]] + rho[..., lookup[j, i]]) / 2
        local[..., i] -= settings["t"] * symmetric
        local[..., j] -= settings["t"] * symmetric
        if i in sites and j in sites:
            directions["x" if i // settings["Lx"] == j // settings["Lx"] else "y"].append(symmetric)
    values = {"central_mean_local_energy_density": local[..., sites].mean(-1),
              "central_mean_density": density[..., sites].mean(-1),
              "whole_lattice_energy_per_site": local.mean(-1)}
    for direction in ("x", "y"):
        if not directions[direction]:
            raise ValueError(f"no central {direction} bonds")
        values[f"central_NN_G{direction}"] = np.mean(directions[direction], axis=0)
    return values


def frozen_summary(path, reference, bonds, block_length=None):
    path = resolve(path)
    if not path.is_relative_to(ROOT):
        raise ValueError("frozen report must be under 2D/")
    content = path.read_bytes()
    report = json.loads(content)
    if report.get("status") != "completed" or report.get("kind") != "frozen_model_observables":
        raise ValueError("expected completed frozen_model_observables report")
    settings, metadata = reference["settings"], report["checkpoint_metadata"]
    for key in PARAMETERS:
        if metadata.get(key) != settings[key]:
            raise ValueError(f"NQS geometry/parameter mismatch: {key}")
    for key, expected in (("hamiltonian_convention", HAMILTONIAN_CONVENTION), ("bond_convention", BOND_CONVENTION)):
        if metadata.get(key) != expected:
            raise ValueError(f"NQS convention mismatch: {key}")
    if report.get("flattening") != "i=y*Lx+x" or metadata.get("flattening") != "i=y*Lx+x":
        raise ValueError("NQS physical index mismatch")
    if report.get("axes") != ["time", "walker", "physical_site"] or report.get("pair_axes") != ["time", "walker", "selected_physical_pair"]:
        raise ValueError("NQS report must preserve time/walker/physical axes")
    raw_path = resolve(report["raw_samples_path"])
    if not raw_path.is_relative_to(ROOT):
        raise ValueError("raw samples must be under 2D/")
    # Guard expanded archive storage before loading arrays; no model graphs exist.
    with zipfile.ZipFile(raw_path) as archive:
        if sum(member.file_size for member in archive.infolist()) > 512 * 1024**2:
            raise ValueError("raw NPZ exceeds supported 512 MiB expanded archive budget")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    with np.load(raw_path, allow_pickle=False) as raw:
        occupations, pairs = raw["occupations"], raw["physical_pairs"]
        density, factorial, rho = raw["density"], raw["onsite_factorial_moment"], raw["one_body_estimators"]
        energy = raw["local_energy"]
    n = settings["Lx"] * settings["Ly"]
    if occupations.ndim != 3 or occupations.shape[-1] != n or min(occupations.shape) <= 0:
        raise ValueError("raw occupations must have nonempty time/walker/physical-site axes")
    if not np.issubdtype(occupations.dtype, np.integer) or (occupations < 0).any() or (occupations > settings["Nmax"]).any() or not np.all(occupations.sum(-1) == settings["Ntot"]):
        raise ValueError("raw occupations violate fixed-number/cutoff constraints")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer) or (pairs < 0).any() or (pairs >= n).any():
        raise ValueError("invalid raw physical pairs")
    if pairs.tolist() != report["one_body_pairs"] or len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError("raw pairs disagree with report or contain duplicates")
    time_walkers = occupations.shape[:2]
    for name, array, shape in (("density", density, occupations.shape), ("factorial", factorial, occupations.shape),
                                ("rho", rho, (*time_walkers, len(pairs))), ("energy", energy, time_walkers)):
        if array.shape != shape or not np.isrealobj(array) or not np.isfinite(array).all():
            raise ValueError(f"invalid raw {name} shape or values")
    np.testing.assert_array_equal(density, occupations)
    np.testing.assert_array_equal(factorial, occupations * (occupations - 1))
    saved_stats = report["statistics"]["energy"]
    if time_walkers != (saved_stats["time_steps"], saved_stats["walkers"]):
        raise ValueError("raw time/walker axes disagree with report")
    np.testing.assert_allclose(energy.mean(), saved_stats["all_samples_mean"], atol=1e-10, rtol=1e-10)
    values = central_samples(density, factorial, rho, pairs, settings, bonds, reference["central_physical_sites"])
    np.testing.assert_allclose(values["whole_lattice_energy_per_site"] * n, energy, atol=1e-9, rtol=1e-9)
    block_length = saved_stats["block_length"] if block_length is None else block_length
    targets = {key: reference[key] for key in ("central_mean_local_energy_density", "central_mean_density", "whole_lattice_energy_per_site")}
    targets.update({f"central_NN_G{d}": reference["central_direction_resolved_NN_G"][d]["G"] for d in ("x", "y")})
    statistics = {}
    for key, array in values.items():
        stats = block_mean_statistics(torch.from_numpy(array), block_length)
        stats = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in stats.items() if k != "block_means"}
        errors = [stats[k] for k in ("block_standard_error", "between_chain_standard_error") if stats[k] is not None]
        stats.update(reference=targets[key], difference_to_reference=stats["mean"] - targets[key],
                     conditional_sem=max(errors) if errors else None)
        statistics[key] = stats
    return {"report": str(path.relative_to(ROOT)), "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"], "checkpoint_metadata": metadata,
        "raw_samples": str(raw_path.relative_to(ROOT)), "sampling": report["sampling"],
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(content).hexdigest(), str(raw_path.relative_to(ROOT)): raw_hash},
        "time_walker_shape": time_walkers, "block_length_retained_steps": block_length,
        "central_physical_sites": reference["central_physical_sites"], "statistics": statistics,
        "conditional_error_policy": "max of available block/between-chain SEMs of the COMPOSITE per-sample values; null if neither; no DMRG systematic error included",
        "covariance_policy": "form local energies and bond averages within each raw time/walker sample before averaging over the strip and before time blocking"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, nargs=2, required=True, metavar=("SMALL", "LARGE"))
    parser.add_argument("--phase", default="chi128")
    parser.add_argument("--spot-phase", default="chi256", help="optional phase in smaller reference, kept separate")
    parser.add_argument("--small-nqs", type=Path, help="frozen report matching first geometry; reads existing NPZ only")
    parser.add_argument("--large-nqs", type=Path, help="frozen report matching second geometry; reads existing NPZ only")
    parser.add_argument("--block-length", type=int, help="retained time steps; default each frozen report's saved block length")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths, output = [resolve(p) for p in args.references], resolve(args.output)
        if args.block_length is not None and args.block_length <= 0:
            raise ValueError("block length must be positive")
        if any(not p.is_relative_to(ROOT) or not p.is_dir() for p in paths):
            raise ValueError("references must be directories under 2D/")
        if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
            raise ValueError("output must be new under benchmarks/results/")
        metadata = [tomllib.loads((p / "metadata.toml").read_text()) for p in paths]
        for m in metadata:
            s = m["settings"]
            if m["status"] != "completed" or s["pbc_x"] or not s["pbc_y"] or s["Lx"] < 4 or s["Lx"] % 2:
                raise ValueError("requires completed even-Lx>=4 open-x/periodic-y cylinders")
            if m["conventions"]["flattening"] != "i=y*Lx+x; physical zero-based":
                raise ValueError("physical index mismatch")
        a, b = [m["settings"] for m in metadata]
        for key in ("Ly", "Nmax", "t", "U", "mu", "pbc_x", "pbc_y"):
            if a[key] != b[key]:
                raise ValueError(f"parameter mismatch: {key}")
        if a["Lx"] >= b["Lx"] or a["Ntot"] * b["Lx"] != b["Ntot"] * a["Lx"]:
            raise ValueError("requires increasing Lx and identical particle filling")
        summaries = [summarize(p, m, args.phase) for p, m in zip(paths, metadata)]
        if summaries[0]["phase"]["requested_chi"] != summaries[1]["phase"]["requested_chi"]:
            raise ValueError("reference phases must have equal requested chi")
        result = {"status": "completed", "kind": "finite_chi_central_two_column_comparison",
            "recorded_at": datetime.now(timezone.utc).isoformat(), "references": summaries,
            "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            "larger_minus_smaller_equal_chi": differences(*summaries),
            "definitions": {"physical_index": "i=y*Lx+x", "central": "two central x columns, all y",
                "local_energy_density": "U/2*<n_i(n_i-1)>-mu*<n_i>-t/2*sum_neighbors(rho_ij+rho_ji)",
                "G": "real((rho_ij+rho_ji)/2), average unique NN bonds with BOTH endpoints in central columns, separate x/y",
                "NQS_future_statistics": "Compute local e_i and each direction-resolved bond average for every raw time/walker configuration BEFORE time blocking. Use conditional errors of those composite samples to retain covariance; do not add component SEMs in quadrature."},
            "interpretation": "Finite-size differences at fixed chi; whole-energy/site has changing open-end fraction. Small-system chi spot sensitivity is separate and does not certify larger-system chi accuracy. No thermodynamic or critical convergence claim.",
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("compare_dmrg_sizes.py", "validate_dmrg_reference.py", "compare_dmrg_vmc.py")}}
        result["source_sha256"].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       for name in ("observables.py", "vmc.py", "models.py", "lattice.py")})
        result["frozen_nqs"] = {}
        for name, path, directory, reference in zip(("smaller", "larger"), (args.small_nqs, args.large_nqs), paths, summaries):
            if path is not None:
                bonds = [(int(row["i"]), int(row["j"])) for row in csv_rows(directory / "bonds.csv")]
                result["frozen_nqs"][name] = frozen_summary(path, reference, bonds, args.block_length)
        if any(p["label"] == args.spot_phase for p in metadata[0]["phases"]):
            spot = summarize(paths[0], metadata[0], args.spot_phase)
            result["smaller_system_spot"] = spot
            result["smaller_spot_minus_primary_sensitivity"] = differences(summaries[0], spot)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(serialized)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output), "larger_minus_smaller_equal_chi": result["larger_minus_smaller_equal_chi"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
