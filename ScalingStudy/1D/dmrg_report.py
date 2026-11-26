"""Summarize matched, periodic DMRG bond-dimension sweeps on a compute node."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import tarfile
import tomllib


ENERGY_TOLERANCE = 1e-3
CORRELATION_FLOOR = 5e-4


def measurements(directory, archive):
    paths = sorted(directory.glob("correlations-periodic-*.toml"))
    paths.extend(sorted(directory.glob("dmrg-refine-*/correlations-periodic-*.toml")))
    seen = {path.name for path in paths}
    for path in paths:
        yield str(path), tomllib.loads(path.read_text())
    if archive:
        with tarfile.open(archive, "r:gz") as bundle:
            for entry in bundle:
                name = Path(entry.name).name
                if entry.isfile() and name.startswith("correlations-periodic-") and name not in seen:
                    with bundle.extractfile(entry) as handle:
                        yield f"{archive}:{name}", tomllib.loads(handle.read().decode())


def summarize_cap(runs):
    preferred = max(runs, key=lambda row: (row["sweep_converged"], row["sweeps"], -row["energy_density"]))
    row = dict(preferred, scoring_ready=preferred["sweep_converged"], fixed_cap_energy_margin=0.0,
               fixed_cap_g1_margins=[0.0] * len(preferred["g1"]))
    if preferred["sweep_converged"]:
        row["convergence_evidence"] = "solver"
        return row
    # A longer, same-cap repeat can establish stability at the observable
    # tolerances even when the solver's absolute energy criterion keeps cycling.
    earlier = [candidate for candidate in runs if candidate["sweeps"] < preferred["sweeps"]]
    if not earlier:
        return row
    check = max(earlier, key=lambda candidate: candidate["sweeps"])
    energy_shift = abs(preferred["energy_density"] - check["energy_density"])
    shifts = [abs(a - b) for a, b in zip(preferred["g1"], check["g1"])]
    stable = energy_shift <= ENERGY_TOLERANCE / 10 and all(
        shift <= (0.05 * abs(g1) + CORRELATION_FLOOR) / 10 for shift, g1 in zip(shifts, preferred["g1"])
    )
    return row | {"scoring_ready": stable, "convergence_evidence": "repeat_at_same_cap",
                  "fixed_cap_check_path": check["path"], "fixed_cap_energy_margin": energy_shift,
                  "fixed_cap_g1_margins": shifts}


def group_report(length, coupling, rows):
    runs_by_cap = defaultdict(list)
    for row in rows:
        runs_by_cap[row["cap"]].append(row)
    rows = [summarize_cap(runs_by_cap[cap]) for cap in sorted(runs_by_cap)]
    comparable = [row for row in rows if row["scoring_ready"]]
    result = {"runs": rows}
    if len(comparable) < 2:
        return result | {"status": "need_two_converged_caps"}
    previous, reference = comparable[-2:]
    older = [row for row in comparable if row["cap"] < reference["cap"]]
    if not older:
        return result | {"status": "need_distinct_reference_caps"}

    def value(row, distance):
        return row["g1"][distance - 1]

    distances = {length // 4, length // 2}
    if coupling >= 8:
        resolved = [
            r for r in range(1, length // 2 + 1)
            if value(reference, r) - abs(value(reference, r) - value(previous, r))
            - previous["fixed_cap_g1_margins"][r - 1] - reference["fixed_cap_g1_margins"][r - 1] > 1e-2
        ]
        distances.add(max(resolved, default=1))
    tolerances = {r: 0.05 * abs(value(reference, r)) + CORRELATION_FLOOR for r in sorted(distances)}
    margins = {r: abs(value(previous, r) - value(reference, r)) +
               previous["fixed_cap_g1_margins"][r - 1] + reference["fixed_cap_g1_margins"][r - 1]
               for r in tolerances}
    energy_margin = (abs(previous["energy_density"] - reference["energy_density"]) +
                     previous["fixed_cap_energy_margin"] + reference["fixed_cap_energy_margin"])
    result |= {"reference": reference["cap"], "reference_check": previous["cap"],
               "energy_margin": energy_margin, "g1_margins": margins, "g1_tolerances": tolerances}
    if energy_margin > ENERGY_TOLERANCE / 10 or any(
        margins[r] > tolerance / 10 for r, tolerance in tolerances.items()
    ):
        return result | {"status": "reference_margin_exceeds_goal"}

    for row in older:
        energy_error = abs(row["energy_density"] - reference["energy_density"])
        energy_uncertainty = energy_margin + row["fixed_cap_energy_margin"]
        pass_energy = energy_error + energy_uncertainty <= ENERGY_TOLERANCE
        miss_energy = energy_error - energy_uncertainty > ENERGY_TOLERANCE
        passes_g1, misses_g1 = True, False
        for r, tolerance in tolerances.items():
            error = abs(value(row, r) - value(reference, r))
            uncertainty = 1.05 * margins[r] + row["fixed_cap_g1_margins"][r - 1]
            passes_g1 &= error + uncertainty <= tolerance
            misses_g1 |= error - uncertainty > tolerance
        if pass_energy and passes_g1:
            row["passes"] = True
        elif miss_energy or misses_g1:
            row["passes"] = False
        else:
            row["passes"] = None
    passing = [row for row in older if row["passes"] is True]
    if not passing:
        return result | {"status": "no_certified_crossing_below_reference"}
    smallest = min(passing, key=lambda row: (row["chi"], row["cap"]))
    smaller_runs = [row for row in rows if row["cap"] < smallest["cap"]]
    return result | {"status": "crossing", "chi_to_target": smallest["chi"], "passing_cap": smallest["cap"],
                     "smaller_unconverged_caps": [row["cap"] for row in smaller_runs if row.get("passes") is None],
                     "largest_failing_cap_below_crossing": max(
                         (row["cap"] for row in smaller_runs if row.get("passes") is False), default=None
                     )}


def cutoff_comparison(report, length):
    baseline = report[f"L{length}-U2-n6"]["runs"]
    expanded = report[f"L{length}-U2-n8"]["runs"]
    original = {run["cap"]: run for run in baseline if run["scoring_ready"]}
    checked = {run["cap"]: run for run in expanded if run["scoring_ready"]}
    common_caps = original.keys() & checked.keys()
    if not common_caps:
        return {"status": "need_matching_converged_cap"}
    cap = max(common_caps)
    lower, upper = original[cap], checked[cap]
    energy_shift = abs(lower["energy_density"] - upper["energy_density"])
    shifts = {r: abs(lower["g1"][r - 1] - upper["g1"][r - 1]) for r in (length // 4, length // 2)}
    return {"status": "matched_cap_comparison", "cap": cap,
            "achieved_chi": [lower["chi"], upper["chi"]], "energy_density_shift": energy_shift,
            "g1_shifts": shifts,
            "within_reference_margin": energy_shift <= ENERGY_TOLERANCE / 10 and all(
                shift <= (0.05 * abs(upper["g1"][r - 1]) + CORRELATION_FLOOR) / 10
                for r, shift in shifts.items()
            )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path, run in measurements(args.directory, args.archive):
        length, coupling = run["L"], run["U_phys"]
        if coupling not in (2.0, 3.275, 8.0) or run["nmax"] not in (6, 8):
            continue
        data = run["sectors"]["0"]
        if abs(data["particle_count"] - length) > 1e-8 or len(data["g1"]) != length // 2:
            raise ValueError(f"invalid particle count or correlation profile in {path}")
        row = {"path": path, "cap": run["chi_requested"], "chi": data["chi"],
               "energy_density": data["energy"] / length, "g1": data["g1"],
               "block_variance": data["block_variance"],
               "sweep_converged": data["converged_sweeps"],
               "seed": run.get("seed", 123), "sweeps": data["sweeps"]}
        groups[length, coupling, run["nmax"]].append(row)
    report = {}
    for (length, coupling, cutoff), rows in sorted(groups.items()):
        key = f"L{length}-U{coupling:g}-n{cutoff}"
        report[key] = group_report(length, coupling, rows)
    for length, coupling, cutoff in groups:
        if coupling == 2.0 and cutoff == 8 and (length, coupling, 6) in groups:
            report[f"L{length}-U2-n8"]["cutoff_comparison"] = cutoff_comparison(report, length)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
