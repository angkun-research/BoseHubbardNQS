#!/usr/bin/env python3
"""Read-only CSV/TOML checks; write a new JSON validation/sensitivity record.

From the repository root:
  .venv/bin/python -B pythonTest/2D/benchmarks/validate_dmrg_reference.py \
    --reference outputs/dmrg-3x3-cylinder \
    --ed-reference benchmarks/results/3x3-cylinder-independent-ed-observables.json \
    --output benchmarks/results/dmrg-3x3-validation.json

Omit --ed-reference for a finite-chi sensitivity report over the phases in one
run directory. This helper never launches Julia or any optimization. All paths
are relative to pythonTest/2D/ unless absolute; outputs stay under results/.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    return (ROOT / value).resolve()


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_phase(directory, phase, settings, bonds):
    n = settings["Lx"] * settings["Ly"]
    label = phase["label"]
    density = np.empty(n)
    factorial = np.empty(n)
    rho = np.empty((n, n), dtype=np.complex128)
    for suffix, key, array in (("density", "density", density), ("onsite_factorial", "n_nminus1", factorial)):
        rows = csv_rows(directory / f"{label}-{suffix}.csv")
        assert len(rows) == n and {int(row["i"]) for row in rows} == set(range(n))
        for row in rows:
            i = int(row["i"])
            assert (int(row["x"]), int(row["y"])) == (i % settings["Lx"], i // settings["Lx"])
            array[i] = float(row[key])
    rows = csv_rows(directory / f"{label}-rho.csv")
    assert len(rows) == n * n and {(int(row["i"]), int(row["j"])) for row in rows} == set(np.ndindex(n, n))
    for row in rows:
        i, j = int(row["i"]), int(row["j"])
        assert [int(row[k]) for k in ("x_i", "y_i", "x_j", "y_j")] == [i % settings["Lx"], i // settings["Lx"], j % settings["Lx"], j // settings["Lx"]]
        rho[i, j] = complex(float(row["real"]), float(row["imag"]))
    assert np.isfinite(density).all() and np.isfinite(factorial).all() and np.isfinite(rho).all()
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8, rtol=0)
    np.testing.assert_allclose(rho.diagonal(), density, atol=1e-8, rtol=0)
    np.testing.assert_allclose(density.sum(), settings["Ntot"], atol=1e-8, rtol=0)
    assert np.linalg.eigvalsh(rho).min() >= -1e-8
    energy = settings["U"] / 2 * factorial.sum() - settings["mu"] * density.sum()
    energy -= settings["t"] * sum((rho[i, j] + rho[j, i]).real for i, j in bonds)
    np.testing.assert_allclose(energy, phase["energy"], atol=1e-8, rtol=1e-8)
    return density, factorial, rho, float(energy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ed-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-atol", type=float, default=1e-6)
    parser.add_argument("--observable-atol", type=float, default=5e-5)
    args = parser.parse_args(argv)
    directory, output = resolve(args.reference), resolve(args.output)
    if not directory.is_relative_to(ROOT) or not directory.is_dir():
        parser.error("reference must be an existing directory under 2D/")
    if not output.is_relative_to(ROOT / "benchmarks/results") or output.exists():
        parser.error("output must be a new file under benchmarks/results/")
    if not all(np.isfinite(v) and v > 0 for v in (args.energy_atol, args.observable_atol)):
        parser.error("tolerances must be finite and positive")
    metadata = tomllib.loads((directory / "metadata.toml").read_text())
    assert metadata["status"] == "completed"
    settings = metadata["settings"]
    n = settings["Lx"] * settings["Ly"]
    sites = csv_rows(directory / "sites.csv")
    assert len(sites) == n and {int(row["i"]) for row in sites} == set(range(n))
    expected_mapping = [i % settings["Lx"] * settings["Ly"] + (i // settings["Lx"] if (i % settings["Lx"]) % 2 == 0 else settings["Ly"] - 1 - i // settings["Lx"]) + 1 for i in range(n)]
    for row in sites:
        i = int(row["i"])
        assert int(row["mps_site_one_based"]) == expected_mapping[i]
        assert (int(row["x"]), int(row["y"])) == (i % settings["Lx"], i // settings["Lx"])
    assert sorted(expected_mapping) == list(range(1, n + 1))
    bonds = [(int(row["i"]), int(row["j"])) for row in csv_rows(directory / "bonds.csv")]
    assert len(bonds) == len(set(bonds)) == metadata["bond_count"] and all(i < j for i, j in bonds)
    expected_edges = []
    for i, j in combinations(range(n), 2):
        dx, dy = abs(i % settings["Lx"] - j % settings["Lx"]), abs(i // settings["Lx"] - j // settings["Lx"])
        x_step = min(dx, settings["Lx"] - dx) if settings["pbc_x"] else dx
        y_step = min(dy, settings["Ly"] - dy) if settings["pbc_y"] else dy
        if (dy == 0 and x_step == 1) or (dx == 0 and y_step == 1):
            expected_edges.append((i, j))
    assert bonds == expected_edges, "bonds disagree with independent physical-coordinate adjacency"
    report = {"status": "passed", "reference": str(directory.relative_to(ROOT)),
              "recorded_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.suffix in (".toml", ".csv")},
              "tolerances": {"total_energy": args.energy_atol, "observable_max_abs": args.observable_atol},
              "phases": [], "interpretation": "Finite-chi reference. A spot difference measures sensitivity, not a rigorous error bar or a convergence proof."}
    gold = None
    if args.ed_reference:
        path = resolve(args.ed_reference)
        gold = json.loads(path.read_text())
        for key in ("Lx", "Ly", "Ntot", "Nmax", "pbc_x", "pbc_y", "t", "U", "mu"):
            assert settings[key] == gold[key], f"ED geometry/parameter mismatch: {key}"
        assert bonds == [tuple(pair) for pair in gold["bonds"]]
        report["independent_ED_reference"] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "energy": gold["energy"]}
    data = []
    for phase in metadata["phases"]:
        density, factorial, rho, rebuilt = read_phase(directory, phase, settings, bonds)
        data.append((density, factorial, rho))
        record = {"label": phase["label"], "requested_chi": phase["requested_chi"], "energy": phase["energy"],
                  "energy_per_site": phase["energy_per_site"], "energy_from_observables": rebuilt,
                  "variance_status": phase["energy_variance_status"], "variance": phase.get("energy_variance"),
                  "particle_number_error": float(abs(density.sum() - settings["Ntot"]))}
        if gold is not None:
            differences = {"energy_abs": abs(phase["energy"] - gold["energy"]),
                           "density_max_abs": float(np.max(abs(density - gold["density"]))),
                           "onsite_factorial_max_abs": float(np.max(abs(factorial - gold["onsite_factorial_moment"]))),
                           "rho_full_max_abs": float(np.max(abs(rho - gold["one_body_matrix"])))}
            record["ED_differences"] = differences
            if differences["energy_abs"] > args.energy_atol or max(v for k, v in differences.items() if k != "energy_abs") > args.observable_atol:
                report["status"] = "failed_ED_tolerance"
        report["phases"].append(record)
    if len(data) == 2:
        density0, factorial0, rho0 = data[0]
        density1, factorial1, rho1 = data[1]
        bulk = [i for i in range(n)
                if (settings["pbc_x"] or 0 < i % settings["Lx"] < settings["Lx"] - 1)
                and (settings["pbc_y"] or 0 < i // settings["Lx"] < settings["Ly"] - 1)]
        report["spot_sensitivity"] = {"energy_per_site_difference": metadata["phases"][1]["energy_per_site"] - metadata["phases"][0]["energy_per_site"],
            "density_max_abs": float(np.max(abs(density1 - density0))), "onsite_factorial_max_abs": float(np.max(abs(factorial1 - factorial0))),
            "rho_full_max_abs": float(np.max(abs(rho1 - rho0))), "rho_bulk_max_abs": float(np.max(abs((rho1 - rho0)[np.ix_(bulk, bulk)]))) if bulk else None,
            "bulk_physical_sites": bulk, "bulk_definition": "exclude both end layers along each open axis; retain periodic axes; same physical pairs at both chi values; null bulk difference if no interior remains"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "phases": report["phases"], "spot_sensitivity": report.get("spot_sensitivity"), "output": str(output)}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
