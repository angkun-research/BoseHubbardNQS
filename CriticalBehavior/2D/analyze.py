"""Summarize periodic square-torus DMRG and finite-size estimates."""

import csv
from itertools import combinations, groupby
import json
from math import isclose, log, pi, sin, sqrt
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly

BENCHMARK = 16.7424


def read_points(source):
    runs = []
    separate_sectors = {}
    for path in sorted(source.glob("L*.toml")):
        run = tomllib.loads(path.read_text())
        key = tuple(run[field] for field in
                    ("L", "U", "nmax", "chi_requested", "seed", "sweeps_requested"))
        if run["mode"] in ("hole", "particle"):
            if run["status"] == "complete":
                offset = "-1" if run["mode"] == "hole" else "1"
                separate_sectors.setdefault(key, {})[offset] = (run["sectors"][offset], path.name)
        elif "0" in run["sectors"]:
            runs.append((path, run, key))

    points = []
    for path, run, key in runs:
        if run["status"] != "complete" and run["mode"] != "gap":
            continue
        sectors = run["sectors"]
        sector_files = {offset: path.name for offset in sectors}
        if run["mode"] in ("gap", "neutral"):
            for offset, (result, filename) in separate_sectors.get(key, {}).items():
                if offset not in sectors:
                    sectors[offset] = result
                    sector_files[offset] = filename
        neutral = sectors["0"]
        point = {key: run[key] for key in
                 ("L", "U", "nmax", "chi_requested", "seed", "sweeps_requested")}
        point.update({key: neutral[key] for key in
                      ("energy", "n0", "nkx", "nky", "ratio", "density_range")})
        point.update({key: neutral.get(key) for key in ("G_mid_x", "G_mid_y", "G_mid_diag")})
        point["n0_minus_nk"] = point["n0"] - (point["nkx"] + point["nky"]) / 2
        ratio = point["ratio"]
        point["xi_over_L"] = (sqrt(ratio / (1 - ratio)) / (2 * point["L"] * sin(pi / point["L"]))
                              if 0 < ratio < 1 else None)
        point["relative_axis_difference"] = abs(point["nkx"] - point["nky"]) / point["n0"]
        point["gap"] = point["mu_mid"] = None
        if {"1", "-1"} <= sectors.keys():
            particle, hole = sectors["1"]["energy"], sectors["-1"]["energy"]
            point["gap"] = particle + hole - 2 * point["energy"]
            point["mu_mid"] = (particle - hole) / 2
        point["scaled_gap"] = point["gap"] * point["L"] if point["gap"] is not None else None
        point["achieved_chi"] = max(sector["chi"] for sector in sectors.values())
        point["final_sweep_energy_change"] = max(
            abs(sector["sweep_energies"][-1] - sector["sweep_energies"][-2])
            for sector in sectors.values())
        point["final_truncation"] = max(sector["sweep_truncations"][-1] for sector in sectors.values())
        point["file"] = path.name
        point["sector_files"] = json.dumps(sector_files, sort_keys=True)
        points.append(point)
    return points


def scans(points, observable="ratio"):
    """Match chi and the complete coupling grid to preserve slope comparisons."""
    output = []
    key = lambda point: (point["nmax"], point["chi_requested"], point["seed"],
                         point["sweeps_requested"], point["L"])
    for settings, rows in groupby(sorted(points, key=key), key=key):
        by_coupling = {}
        for row in rows:
            if row[observable] is None:
                continue
            if row["U"] not in by_coupling or row["gap"] is not None:
                by_coupling[row["U"]] = row
        rows = sorted(by_coupling.values(), key=lambda row: row["U"])
        if len(rows) >= 3:
            output.append((settings, rows))
    return output


def crossings(points, observable):
    output = []
    for (a, first), (b, second) in combinations(scans(points, observable), 2):
        grid = [row["U"] for row in first]
        if a[:-1] != b[:-1] or grid != [row["U"] for row in second]:
            continue
        fit_a, fit_b = (PchipInterpolator(grid, [row[observable] for row in rows],
                                        extrapolate=False) for rows in (first, second))
        difference = PPoly(fit_a.c - fit_b.c, fit_a.x, extrapolate=False)
        diff_nodes = difference(grid)
        def nu_from_slopes(slope_a, slope_b):
            if slope_a * slope_b <= 0:
                return None
            slope_ratio = slope_b / slope_a
            return log(b[-1] / a[-1]) / log(slope_ratio) if slope_ratio > 1 else None

        linear = []
        for i, (x1, x2, d1, d2) in enumerate(zip(grid[:-1], grid[1:], diff_nodes[:-1], diff_nodes[1:])):
            if d1 * d2 < 0:
                slopes = [(rows[i+1][observable] - rows[i][observable]) / (x2 - x1)
                          for rows in (first, second)]
                linear.append({"U": float(x1 - d1 * (x2 - x1) / (d2 - d1)),
                               "nu_effective": nu_from_slopes(*slopes)})
        roots = sorted(set(float(x) for x in difference.roots() if np.isfinite(x)))

        def nu_at(coupling):
            slope_a = float(fit_a.derivative()(coupling))
            slope_b = float(fit_b.derivative()(coupling))
            return nu_from_slopes(slope_a, slope_b)

        output.append({"sizes": [a[-1], b[-1]], "nmax": a[0], "chi": a[1],
                       "seed": a[2], "sweeps": a[3],
                       "crossings": [{"U": x, "nu_effective": nu_at(x)} for x in roots],
                       "linear_crossings": linear,
                       "nu_effective_at_published_Uc": nu_at(BENCHMARK)})
    return output


def benchmark_points(points):
    return [point for point in points if isclose(point["U"], BENCHMARK, rel_tol=0, abs_tol=1e-9)]


def size_estimates(points):
    key = lambda point: (point["nmax"], point["chi_requested"], point["seed"])
    critical = benchmark_points(points)
    output = []
    for settings, group in groupby(sorted(critical, key=key), key=key):
        by_size = {}
        for point in group:
            existing = by_size.get(point["L"])
            if (existing is None or (point["gap"] is not None, point["sweeps_requested"])
                    > (existing["gap"] is not None, existing["sweeps_requested"])):
                by_size[point["L"]] = point
        rows = sorted(by_size.values(), key=lambda point: point["L"])
        for a, b in combinations(rows, 2):
            size_log = log(b["L"] / a["L"])
            row = dict(zip(("nmax", "chi", "seed"), settings))
            row["sizes"] = [a["L"], b["L"]]
            row["sweeps"] = [a["sweeps_requested"], b["sweeps_requested"]]
            for observable, label in (("n0", "eta_raw"), ("n0_minus_nk", "eta_subtracted")):
                row[label] = 1 - log(b[observable] / a[observable]) / size_log
            row["z_effective"] = (-log(b["gap"] / a["gap"]) / size_log
                                  if a["gap"] and b["gap"] and min(a["gap"], b["gap"]) > 0 else None)
            output.append(row)
    return output


def compare_ed(source, points):
    path = source / "ed.json"
    if not path.exists():
        return []
    ed = json.loads(path.read_text())
    baseline = ed["sectors"]["0"]
    ed_gap = ed["sectors"]["1"]["energy"] + ed["sectors"]["-1"]["energy"] - 2 * baseline["energy"]
    result = []
    for point in points:
        if (point["L"], point["U"], point["nmax"]) != (ed["L"], ed["U"], ed["nmax"]):
            continue
        sector_files = json.loads(point["sector_files"])
        sectors = {offset: tomllib.loads((source / filename).read_text())["sectors"][offset]
                   for offset, filename in sector_files.items()}
        result.append({"file": point["file"],
            "sector_energy_errors": {offset: sector["energy"] - ed["sectors"][offset]["energy"]
                                     for offset, sector in sectors.items()},
            "ratio_error": point["ratio"] - baseline["ratio"],
            "n0_error": point["n0"] - baseline["n0"],
            "gap_error": point["gap"] - ed_gap if point["gap"] is not None else None})
    return result


def convergence_at_benchmark(points):
    output = []
    observables = ("energy", "n0", "n0_minus_nk", "ratio", "gap")
    for a, b in combinations(benchmark_points(points), 2):
        if a["L"] != b["L"]:
            continue
        changed = [key for key in ("chi_requested", "nmax", "seed") if a[key] != b[key]]
        if len(changed) != 1:
            continue
        output.append({"L": a["L"], "parameter": changed[0],
                       "files": [a["file"], b["file"]],
                       "second_minus_first": {key: b[key] - a[key] for key in observables
                                              if a[key] is not None and b[key] is not None}})
    return output


def plot(points, target):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.3), layout="constrained")
    axes = axes.ravel()
    for settings, rows in scans(points):
        nmax, chi, seed, _, L = settings
        if nmax == 4 and seed == 123:
            axes[0].plot([row["U"] for row in rows], [row["ratio"] for row in rows],
                         "o-", label=f"$L={L},\\ \\chi={chi}$", markersize=3)
    axes[0].axvline(BENCHMARK, color="0.5", linestyle=":", label="QMC")
    axes[0].set(xlabel="$U/t$", ylabel="$1-\\overline{n(k_{min})}/n(0)$")
    axes[0].legend(fontsize=7)
    for settings, rows in scans(points, "scaled_gap"):
        nmax, chi, seed, _, L = settings
        if nmax == 4 and seed == 123:
            axes[1].plot([row["U"] for row in rows], [row["scaled_gap"] for row in rows],
                         "o-", label=f"$L={L},\\ \\chi={chi}$", markersize=3)
    if axes[1].lines:
        axes[1].legend(fontsize=7)
    axes[1].axvline(BENCHMARK, color="0.5", linestyle=":")
    axes[1].set(xlabel="$U/t$", ylabel="$L\\,\\Delta_c/t$")

    critical = [row for row in benchmark_points(points) if row["nmax"] == 4 and row["seed"] == 123]
    for chi in sorted({row["chi_requested"] for row in critical}):
        rows = sorted((row for row in critical if row["chi_requested"] == chi), key=lambda row: row["L"])
        if rows:
            length = np.array([row["L"] for row in rows])
            axes[2].plot(length, [row["n0_minus_nk"] / row["L"]**(1 - 0.03816)
                                 for row in rows], "o-", label=f"$\\chi={chi}$")
        gap_rows = [row for row in rows if row["gap"] is not None]
        if gap_rows:
            axes[3].plot([row["L"] for row in gap_rows], [row["scaled_gap"] for row in gap_rows],
                         "o-", label=f"$\\chi={chi}$")
    axes[2].set(xlabel="$L$", ylabel="$(n(0)-\\overline{n(k_{min})})/L^{1-\\eta_{XY}}$")
    axes[3].set(xlabel="$L$", ylabel="$L\\,\\Delta_c/t$")
    for ax in axes[2:]:
        ax.set_xticks(sorted({row["L"] for row in points}))
        if ax.lines:
            ax.legend(fontsize=8)
    fig.savefig(target)
    plt.close(fig)


def main(source, target):
    points = read_points(source)
    if not points:
        raise SystemExit(f"No completed DMRG points in {source}")
    summary = {"published_Uc": BENCHMARK, "point_count": len(points),
               "crossings": {key: crossings(points, key)
                             for key in ("ratio", "xi_over_L", "scaled_gap")},
               "pairwise_exponents_at_published_Uc": size_estimates(points),
               "convergence_at_published_Uc": convergence_at_benchmark(points),
               "ed_comparison": compare_ed(source, points)}
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with target.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(sorted(points, key=lambda p: (p["L"], p["U"], p["nmax"], p["chi_requested"])))
    plot(points, target.with_suffix(".pdf"))


if __name__ == "__main__":
    main(*(Path(argument) for argument in sys.argv[1:]))
