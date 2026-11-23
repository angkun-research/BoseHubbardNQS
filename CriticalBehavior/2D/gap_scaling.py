"""Analyze square-torus charge-gap crossings in the convention of Fig. 2."""

from itertools import combinations
import json
from math import log
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import BENCHMARK, benchmark_points, convergence_at_benchmark, read_points

NU_XY = 0.671718
PRIMARY_GRID = (16.60, BENCHMARK, 16.90, 17.10)


def square_curves(points, grid):
    """Use a single bond dimension and sweep schedule across U at each L."""
    curves = {}
    for L in sorted({point["L"] for point in points}):
        by_setting = {}
        for point in points:
            if point["L"] != L or point["gap"] is None or point["nmax"] != 4 or point["seed"] != 123:
                continue
            if point["U"] in grid:
                setting = point["chi_requested"], point["sweeps_requested"]
                by_setting.setdefault(setting, {})[point["U"]] = point
        complete = [setting for setting, rows in by_setting.items() if set(rows) == set(grid)]
        if complete:
            curves[L] = [by_setting[max(complete)][U] for U in grid]
    return curves


def fit_gap_curves(curves, grid):
    g_ref = 1 / BENCHMARK
    x = 1 / np.asarray(grid) - g_ref
    results = {}
    for L, rows in curves.items():
        y = [row["scaled_gap"] for row in rows]
        slope, intercept = np.polyfit(x, y, 1)
        results[L] = dict(intercept=float(intercept), slope=float(slope),
                          chi=rows[0]["chi_requested"],
                          max_residual=float(np.max(np.abs(y - (slope*x + intercept)))))
    pairs = []
    for a, b in combinations(results, 2):
        first, second = results[a], results[b]
        slope_diff = first["slope"] - second["slope"]
        cross = (second["intercept"] - first["intercept"]) / slope_diff if slope_diff else None
        slope_ratio = second["slope"] / first["slope"] if first["slope"] else 0
        pairs.append(dict(sizes=[a, b],
                          crossing_U=(1 / (cross + g_ref) if cross is not None
                                      and min(x) <= cross <= max(x) else None),
                          nu_effective=(log(b/a) / log(slope_ratio) if slope_ratio > 1 else None)))
    return results, pairs


def fixed_exponent_fit(curves, grid):
    """Leading-order fit to L*gap = A + B*(g_c - g)*L**(1/nu_XY)."""
    if len(curves) < 2:
        return None
    g_ref = 1 / BENCHMARK
    design, measured = [], []
    for L, rows in curves.items():
        weight = L**(1 / NU_XY)
        for U, point in zip(grid, rows):
            design.append([1, weight * (g_ref - 1/U), weight])
            measured.append(point["scaled_gap"])
    coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    _, slope, shift = coefficients
    U = 1 / (g_ref + shift/slope) if slope else None
    predicted = np.asarray(design) @ coefficients
    return dict(U=U if U is not None and min(grid) <= U <= max(grid) else None,
                max_residual=float(np.max(np.abs(predicted - measured))))


def critical_checks(points):
    benchmark = [row for row in benchmark_points(points) if row["nmax"] == 4 and row["seed"] == 123]
    best_gap, best_neutral = {}, {}
    for point in sorted(benchmark, key=lambda p: (p["chi_requested"], p["sweeps_requested"])):
        best_neutral[point["L"]] = point
        if point["gap"] is not None:
            best_gap[point["L"]] = point
    gaps, correlations = [], []
    for a, b in combinations(sorted(best_gap), 2):
        first, second = best_gap[a], best_gap[b]
        gaps.append(dict(sizes=[a, b], chi=[first["chi_requested"], second["chi_requested"]],
                         z_effective=log(first["gap"] / second["gap"]) / log(b/a)))
    for a, b in combinations(sorted(L for L in best_neutral if L % 2 == 0), 2):
        first, second = best_neutral[a], best_neutral[b]
        row = dict(sizes=[a, b], chi=[first["chi_requested"], second["chi_requested"]])
        for observable in ("G_mid_x", "G_mid_y", "G_mid_diag"):
            a_val, b_val = first[observable], second[observable]
            if a_val and b_val:
                row[observable] = log(a_val / b_val) / log(b/a) - 1
        correlations.append(row)
    return gaps, correlations


def plot(points, curves, fits, grid, target):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for L, rows in curves.items():
        g = 1 / np.asarray(grid)
        line, = axes[0].plot(g, [row["scaled_gap"] for row in rows], "o", label=f"L = {L}")
        fit = fits[L]
        axes[0].plot(g, fit["intercept"] + fit["slope"] * (g - 1/BENCHMARK),
                     color=line.get_color())
        by_u = {row["U"]: row for row in points if row["L"] == L and row["gap"] is not None
                and row["chi_requested"] == fit["chi"] and row["nmax"] == 4 and row["seed"] == 123}
        on_mott_side = sorted(U for U in by_u if U >= BENCHMARK)
        axes[1].plot([(1/BENCHMARK - 1/U) * L**(1/NU_XY) for U in on_mott_side],
                     [by_u[U]["scaled_gap"] for U in on_mott_side], "o-", color=line.get_color())
    axes[0].axvline(1/BENCHMARK, linestyle=":", color="0.5", label="QMC")
    axes[0].set(xlabel="$t/U$", ylabel="$L\\Delta_c/t$")
    axes[0].legend()
    axes[1].set(xlabel="$(t/U_c-t/U)L^{1/\\nu_{XY}}$", ylabel="$L\\Delta_c/t$")
    fig.savefig(target)
    plt.close(fig)


def main(source, target, cap=None):
    points = read_points(source)
    if cap is not None:
        points = [point for point in points if point["chi_requested"] <= cap]
    grid = PRIMARY_GRID
    curves = square_curves(points, grid)
    if len(curves) < 2:
        grid = (16.0, BENCHMARK, 17.5)
        curves = square_curves(points, grid)
    if len(curves) < 2:
        raise SystemExit("At least two complete square-torus charge-gap curves are required")
    fits, crossings = fit_gap_curves(curves, grid)
    gap_exponents, midpoint_exponents = critical_checks(points)
    control_curves = square_curves(
        [point for point in points if point["L"] != 6 or point["chi_requested"] <= 512], grid)
    _, control_crossings = fit_gap_curves(control_curves, grid)
    report = dict(coupling_grid=grid, fixed_nu=NU_XY, fixed_z=1, fits=fits, crossings=crossings,
                  fixed_exponent_fit=fixed_exponent_fit(curves, grid),
                  L6_through_512_crossings=control_crossings,
                  narrower_window_fit=fixed_exponent_fit(
                      {L: rows[:-1] for L, rows in curves.items()}, grid[:-1]),
                  gap_exponents_at_published_Uc=gap_exponents,
                  eta_effective_at_published_Uc=midpoint_exponents,
                  convergence_at_published_Uc=convergence_at_benchmark(points),
                  benchmark=benchmark_points(points))
    target.with_suffix(".json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    plot(points, curves, fits, grid, target.with_suffix(".pdf"))


if __name__ == "__main__":
    source, target = map(Path, sys.argv[1:3])
    main(source, target, int(sys.argv[3]) if len(sys.argv) > 3 else None)
