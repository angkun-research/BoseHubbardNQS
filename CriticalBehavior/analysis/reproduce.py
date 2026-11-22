"""Extract K from published PBC DMRG block variances and locate the RG separatrix."""

import argparse
import csv
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares


def extract_k(variance, length, fraction):
    half = length // 2
    width = max(2, round(length * fraction))
    block = np.arange(max(1, half - width), half + 1)
    x = np.log(np.sin(np.pi * block / length)) / np.pi**2
    return float(np.polyfit(x, np.asarray(variance)[block - 1], 1)[0])


def rg_invariant(k):
    return np.log(k / 2) + 2 / k


def zeta(first, last, k_first, k_last):
    """Integrate the RG equation between two measured points on the same flow."""
    if not 0 < k_last < k_first:
        return np.nan
    pole = rg_invariant(np.clip(2., k_last, k_first))
    target = 2 * np.log(last / first)

    def residual(value):
        integral, _ = quad(lambda k: 1 / (k*k * (rg_invariant(k) - value)),
                           k_last, k_first, epsabs=1e-10)
        return integral - target

    # Approach the pole only as far as necessary to bracket the physical root.
    for distance in (1e-2, 1e-4, 1e-6, 1e-8):
        upper = pole - distance
        if residual(upper) > 0:
            return brentq(residual, pole - 4, upper, xtol=1e-13)
    return np.nan


def published_data(archive):
    with ZipFile(archive) as source:
        for name in source.namelist():
            match = re.search(r"/data/pbc/DMRG/sigma2_L(\d+)_N(\d+)_nmax06_", name)
            if match is None:
                continue
            length, number = map(int, match.groups())
            if length != number:
                raise ValueError(f"Unexpected filling in {name}")
            raw = np.loadtxt(io.BytesIO(source.read(name)), ndmin=2)
            for row in raw:
                if len(row) != 1 + length // 2:
                    raise ValueError(f"Unexpected data width in {name}")
                yield round(float(row[0]), 6), length, row[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fractions = [1/16, 1/12, 1/8, 3/16, 1/4]
    data = {(u, length): vals for u, length, vals in published_data(args.archive)}
    couplings = sorted({u for u, _ in data if 3.24 <= u <= 3.35})
    curves = {}
    roots = []
    point_rows = []
    for fraction in fractions:
        k = {key: extract_k(values, key[1], fraction) for key, values in data.items()}
        for l1, l2 in [(48, 64), (64, 96), (64, 128), (96, 128)]:
            points = [(u, zeta(l1, l2, k[u, l1], k[u, l2])) for u in couplings
                      if (u, l1) in k and (u, l2) in k]
            points = [(u, value) for u, value in points if np.isfinite(value)]
            point_rows.extend((fraction, l1, l2, u, value, k[u, l1], k[u, l2])
                              for u, value in points)
            curves[fraction, l1, l2] = points
            for lo, hi in [(3.270, 3.300), (3.270, 3.290), (3.270, 3.346)]:
                x, y = np.asarray([(u, value) for u, value in points if lo <= u <= hi]).T
                slope, intercept = np.polyfit(x - 3.275, y - 1, 1)
                roots.append({"window_fraction": fraction, "sizes": [l1, l2],
                              "coupling_range": [lo, hi], "U_phys_c": float(3.275 - intercept / slope),
                              "slope": float(slope), "bracketed": bool(min(y) <= 1 <= max(y))})
    reference = next(r for r in roots if r["sizes"] == [96, 128] and r["window_fraction"] == 1/16
                     and r["coupling_range"] == [3.270, 3.300])
    values = [r["U_phys_c"] for r in roots if r["sizes"][0] >= 64 and r["bracketed"]]
    # Continue the measured Mott-side flows through a fixed interval around K=2.
    # This probes the essential singularity within the RG equation itself.
    rg_scales = []
    uc = reference["U_phys_c"]
    for u, invariant in curves[1/16, 96, 128]:
        if u <= uc or invariant >= 1:
            continue
        rg_time = quad(lambda k: 0.5 / (k*k * (rg_invariant(k) - invariant)),
                       1.75, 2.25, epsabs=1e-10)[0]
        rg_scales.append({"U_phys": u, "zeta": invariant, "log_scale_ratio": rg_time})
    scale_fits = []
    for center in (min(values), uc, max(values)):
        delta = np.array([row["U_phys"] - center for row in rg_scales])
        target = np.array([row["log_scale_ratio"] for row in rg_scales])
        fit = least_squares(lambda p: p[0] + p[1] / delta**p[2] - target,
                            [-5., 5., .5], bounds=([-100, 0, .05], [100, 200, 2.]))
        scale_fits.append({"U_phys_c_assumed": center, "sigma": float(fit.x[2]),
                           "offset": float(fit.x[0]), "amplitude": float(fit.x[1]),
                           "rms_log_scale": float(np.sqrt(np.mean(fit.fun**2)))})
    # Fit the limiting K freely as a check on the leading critical-flow amplitude.
    u_near = min(couplings, key=lambda u: abs(u - reference["U_phys_c"]))
    flow = {}
    for minimum in [16, 32, 48]:
        x, y = np.array(sorted((l, extract_k(v, l, 1/16)) for (u, l), v in data.items()
                               if u == u_near and l >= minimum)).T
        fit = least_squares(lambda p: p[0] + 1 / (np.log(x) + p[1]) - y, [2., 0.])
        flow[str(minimum)] = {"U_phys": u_near, "K_limit": float(fit.x[0]),
                              "rms": float(np.sqrt(np.mean(fit.fun**2)))}
    summary = {"primary": reference, "U_code_c": reference["U_phys_c"] / 2,
               "large_pair_bracketed_U_phys_range": [min(values), max(values)],
               "critical_K_fits": flow, "Mott_RG_scale_from_continued_flow": rg_scales, "Mott_RG_scale_exponent_fits": scale_fits,
               "all_critical_coupling_fits": roots}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "all_critical_coupling_fits"}, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["block_window_fraction", "L_min", "L_max", "U_phys", "zeta", "K_min", "K_max"])
        writer.writerows(point_rows)
    fig, (ax, scale_ax) = plt.subplots(1, 2, figsize=(9.1, 4.0), layout="constrained")
    for l1, l2 in [(48, 64), (64, 96), (96, 128)]:
        uv, zv = np.asarray(curves[1/16, l1, l2]).T
        ax.plot(uv, zv, "o-", label=f"$L={l1}, {l2}$", markersize=4)
    ax.axhline(1, lw=1, color="0.45", ls="--")
    ax.axvline(reference["U_phys_c"], lw=1, color="0.45", ls=":")
    ax.set(xlabel=r"$U_{\rm phys}/t$", ylabel=r"RG invariant $\zeta$")
    ax.legend(frameon=False)
    delta = np.array([row["U_phys"] - uc for row in rg_scales])
    rg_time = [row["log_scale_ratio"] for row in rg_scales]
    scale_ax.loglog(delta, rg_time, "o", label="Continued RG flow")
    fit = scale_fits[1]
    fine = np.geomspace(delta.min(), delta.max(), 100)
    scale_ax.loglog(fine, fit["offset"] + fit["amplitude"] / fine**fit["sigma"],
                    "--", label=rf"Fit: $\sigma={fit['sigma']:.2f}$")
    scale_ax.set(xlabel=r"$(U_{\rm phys}-U_c)/t$", ylabel="RG time across $K=1.75$ to $2.25$")
    scale_ax.legend(frameon=False)
    fig.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
