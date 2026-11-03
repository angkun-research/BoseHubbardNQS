#!/usr/bin/env python3
"""Assess archived frozen NQS reports against one matched finite-chi DMRG phase.

Paths are relative to pythonTest/2D/ unless absolute. No model loading, sampling
or training. Example from repo root:
  .venv/bin/python -B pythonTest/2D/benchmarks/assess_refinement.py \
    --reference outputs/dmrg-4x4-cylinder-chi128-spot256 --phase chi128 \
    --reports outputs/2026-10-31-cylinder-pilots/frozen/Conv2DNet-4x4-thin16-seed321.json \
    --output benchmarks/results/refinement-4x4.json

Point and precision requirements are separate. A missed point target cannot
pass because its conditional error is large. No ESS, CI, mixing, ground-state
or convergence claim follows from this numerical assessment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch

from compare_dmrg_sizes import ROOT, central_samples, frozen_summary, summarize
from observables import block_mean_statistics
from validate_dmrg_reference import csv_rows, read_phase, resolve


def assess_point(mean, reference, sem, tolerance, precision_limit):
    """Numerical target policy, not a statistical confidence test."""
    values = [mean, reference, tolerance, precision_limit] + ([] if sem is None else [sem])
    if not np.isfinite(values).all() or min(tolerance, precision_limit) < 0 or (sem is not None and sem < 0):
        raise ValueError("assessment requires finite values and nonnegative tolerances/SEM")
    error = abs(mean - reference)
    point_ok = error <= tolerance
    precision_ok = sem is not None and sem <= precision_limit
    within_two_sem = None if sem is None else bool(error <= 2 * sem)
    if point_ok and precision_ok:
        status = "meets_point_and_precision_targets"
    elif sem is None:
        status = "inconclusive_missing_precision"
    elif not point_ok:
        status = "inconclusive_point_target_missed" if within_two_sem else "point_target_missed"
    else:
        status = "inconclusive_precision_target_missed"
    return {"difference": mean - reference, "absolute_error": error,
            "absolute_target_tolerance": tolerance, "conditional_sem": sem,
            "conditional_sem_limit": precision_limit, "point_target_met": bool(point_ok),
            "precision_target_met": bool(precision_ok), "within_two_conditional_sem_of_reference": within_two_sem,
            "meets_point_and_precision_targets": bool(point_ok and precision_ok), "status": status}


def statistics(values, block_length):
    stats = block_mean_statistics(torch.as_tensor(values), block_length)
    stats = {key: value.cpu().tolist() if isinstance(value, torch.Tensor) else value
             for key, value in stats.items() if key != "block_means"}
    errors = [stats[k] for k in ("block_standard_error", "between_chain_standard_error") if stats[k] is not None]
    stats["conditional_sem"] = np.maximum.reduce(errors).tolist() if errors else None
    return stats


def curve_samples(density, factorial, rho, pairs, settings, anchor):
    """Average y within each configuration; preserve directed anchor-row action."""
    Lx, Ly = settings["Lx"], settings["Ly"]
    if not 0 <= anchor < Lx * Ly or anchor % Lx not in (Lx // 2 - 1, Lx // 2):
        raise ValueError("anchor must be in one of the central two x columns")
    lookup = {tuple(pair): k for k, pair in enumerate(pairs)}
    line_pairs = [(anchor, (anchor // Lx) * Lx + x) for x in range(Lx)]
    missing = [pair for pair in line_pairs if pair not in lookup]
    if missing:
        raise ValueError(f"missing directed central-anchor line pairs: {missing}")
    np.testing.assert_allclose(rho[..., lookup[anchor, anchor]], density[..., anchor], atol=1e-12, rtol=0)
    return {"density_per_x": density.reshape(*density.shape[:2], Ly, Lx).mean(2),
            "factorial_per_x": factorial.reshape(*factorial.shape[:2], Ly, Lx).mean(2),
            "anchor_line": rho[..., [lookup[pair] for pair in line_pairs]], "line_pairs": line_pairs}


def split_half_drift(values):
    """Paired walker differences preserve covariance between the two halves."""
    half = len(values) // 2
    if not half:
        return {"status": "insufficient_time_steps"}
    first, last = values[:half], values[-half:]
    paired = last.mean(0) - first.mean(0)
    sem = float(paired.std(ddof=1) / len(paired)**.5) if len(paired) >= 2 else None
    return {"first_half_mean": float(first.mean()), "last_half_mean": float(last.mean()),
            "last_minus_first": float(paired.mean()), "paired_between_chain_conditional_sem": sem,
            "time_steps_per_half": half, "discarded_middle_steps": len(values) - 2 * half,
            "interpretation": "drift diagnostic only; paired walker SEM assumes independent chains; no stationarity or mixing proof"}


def assess_report(path, reference, bonds, density_ref, factorial_ref, rho_ref, block_length, anchor):
    frozen = frozen_summary(path, reference, bonds, block_length)
    settings = reference["settings"]
    with np.load(resolve(frozen["raw_samples"]), allow_pickle=False) as raw:
        density, factorial, rho, pairs = (raw[key] for key in
            ("density", "onsite_factorial_moment", "one_body_estimators", "physical_pairs"))
    # frozen_summary already validated the archive/resource/physical contracts.
    values = central_samples(density, factorial, rho, pairs, settings, bonds, reference["central_physical_sites"])
    curves = curve_samples(density, factorial, rho, pairs, settings, anchor)
    checks = {}
    for key, kind in (("whole_lattice_energy_per_site", "energy"), ("central_mean_density", "density"),
                      ("central_NN_Gx", "G"), ("central_NN_Gy", "G")):
        stats = frozen["statistics"][key]
        target = stats["reference"]
        tolerance = .01 if kind == "density" else (.01 if kind == "energy" else .05) * abs(target)
        precision = .005 if kind == "density" else (.0025 if kind == "energy" else .025) * abs(target)
        checks[key] = assess_point(stats["mean"], target, stats["conditional_sem"], tolerance, precision)
    Lx, Ly = settings["Lx"], settings["Ly"]
    density_stats = statistics(curves["density_per_x"], block_length)
    factorial_stats = statistics(curves["factorial_per_x"], block_length)
    line_stats = statistics(curves["anchor_line"], block_length)
    profiles = []
    for x in range(Lx):
        row = {"x": x, "physical_sites": [y * Lx + x for y in range(Ly)]}
        for name, stats, target in (("density", density_stats, density_ref.reshape(Ly, Lx).mean(0)[x]),
                                    ("onsite_factorial_moment", factorial_stats, factorial_ref.reshape(Ly, Lx).mean(0)[x])):
            row[name] = {key: None if stats[key] is None else stats[key][x] for key in
                         ("mean", "all_samples_mean", "block_standard_error", "between_chain_standard_error", "conditional_sem")}
            row[name]["reference"] = float(target)
            row[name]["difference"] = row[name]["mean"] - float(target)
        row["density"]["target_check"] = assess_point(row["density"]["mean"], row["density"]["reference"],
                                                      row["density"]["conditional_sem"], .01, .005)
        profiles.append(row)
    line = []
    for k, (i, j) in enumerate(curves["line_pairs"]):
        row = {"physical_pair": [i, j], "x_i": i % Lx, "y_i": i // Lx, "x_j": j % Lx, "y_j": j // Lx,
               "dx": j % Lx - i % Lx, "reference_real": float(rho_ref[i, j].real),
               "reference_imag": float(rho_ref[i, j].imag)}
        row.update({key: None if line_stats[key] is None else line_stats[key][k] for key in
                    ("mean", "all_samples_mean", "block_standard_error", "between_chain_standard_error", "conditional_sem")})
        row["difference_real"] = row["mean"] - row["reference_real"]
        line.append(row)
    frozen.update({"summary_target_checks": checks,
        "all_four_summary_point_and_precision_targets_met": all(c["meets_point_and_precision_targets"] for c in checks.values()),
        "profiles_per_x": profiles, "anchor_site": anchor, "anchor_same_y_line": line,
        "curve_averaging": "density and factorial: mean over all y at each x PER raw time/walker before blocking; line: directed rho[anchor,j], same y, unsymmetrized",
        "curve_block_scope": {key: density_stats[key] for key in ("block_length", "complete_blocks_per_chain", "used_time_steps", "discarded_tail_steps", "status")},
        "block_length_stability": {str(length): {key: statistics(array, length) for key, array in values.items()}
                                   for length in (32, 64, 128)},
        "split_half_energy_per_site_drift": split_half_drift(values["whole_lattice_energy_per_site"])})
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--phase", default="chi128")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--block-length", type=int, default=64, help="primary retained-time block, not selected from stability scan")
    parser.add_argument("--anchor-site", type=int, help="default x=Lx//2,y=Ly//2; must be in central strip and present in NPZ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        directory, output = resolve(args.reference), resolve(args.output)
        if not directory.is_relative_to(ROOT) or not directory.is_dir():
            raise ValueError("reference must be an existing directory under 2D/")
        if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
            raise ValueError("output must be new under benchmarks/results/")
        if args.block_length <= 0:
            raise ValueError("block length must be positive")
        metadata = tomllib.loads((directory / "metadata.toml").read_text())
        settings = metadata["settings"]
        if metadata["status"] != "completed" or settings["pbc_x"] or not settings["pbc_y"] or settings["Lx"] < 4 or settings["Lx"] % 2:
            raise ValueError("requires completed even-Lx>=4 open-x/periodic-y cylinder reference")
        if metadata["conventions"]["flattening"] != "i=y*Lx+x; physical zero-based":
            raise ValueError("DMRG physical index mismatch")
        reference = summarize(directory, metadata, args.phase)
        phase = reference["phase"]
        bonds = [(int(row["i"]), int(row["j"])) for row in csv_rows(directory / "bonds.csv")]
        density_ref, factorial_ref, rho_ref, _ = read_phase(directory, phase, settings, bonds)
        anchor = args.anchor_site if args.anchor_site is not None else (settings["Ly"] // 2) * settings["Lx"] + settings["Lx"] // 2
        rows = [assess_report(path, reference, bonds, density_ref, factorial_ref, rho_ref, args.block_length, anchor)
                for path in args.reports]
        sources = ["benchmarks/assess_refinement.py", "benchmarks/compare_dmrg_sizes.py", "benchmarks/compare_dmrg_vmc.py",
                   "benchmarks/validate_dmrg_reference.py", "observables.py", "models.py", "vmc.py", "lattice.py"]
        record = {"status": "completed", "kind": "frozen_refinement_point_and_precision_assessment",
            "recorded_at": datetime.now(timezone.utc).isoformat(), "reference": reference,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
            "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            "primary_block_length": args.block_length, "reports": rows,
            "targets": {"relative_energy_error": .01, "relative_G_error": .05, "absolute_density_error": .01,
                        "energy_SEM_fraction_abs_reference": .0025, "G_SEM_fraction_abs_reference": .025, "absolute_density_SEM": .005},
            "target_scope": "energy=whole-lattice E/site; density=central-eight-site mean at Ly4; Gx/Gy=directional averages of unique bonds with both endpoints in central two columns. Central local energy, per-x profiles and anchor line are separate diagnostics; aggregate targets do not guarantee every site/bond.",
            "policy": "Each point target AND its precision limit must both hold. Point failure within2conditionalSEM of reference is inconclusive, never a pass; missing SEM cannot pass. No automatic ground-state/fidelity/convergence/architecture acceptance or ranking.",
            "limitations": "Conditional SEM=max(block,between-chain); no DMRG systematic error included. Stability scan is descriptive, not used to choose the smallest SEM. Split-half drift uses paired per-walker differences. No ESS/CI or proof of mixing/stationarity/thermodynamic convergence."}
        serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(serialized)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        parser.error(str(error))
    print(f"Assessed {len(rows)} archived reports against {phase['label']}; output={output}")
    return record


if __name__ == "__main__":
    main()
