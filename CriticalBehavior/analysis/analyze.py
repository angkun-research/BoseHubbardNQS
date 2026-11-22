"""Analyze independent ITensor DMRG and ED results for the 1D BKT study."""

import argparse
import json
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from reproduce import extract_k, published_data, zeta


def exponent(x, values):
    slope, intercept = np.polyfit(np.log(x), np.log(values), 1)
    return float(-slope), float(np.exp(intercept))


def charge_gap(data):
    sectors = data["sectors"]
    return sectors["1"]["energy"] + sectors["-1"]["energy"] - 2 * sectors["0"]["energy"]


def load_results(directory):
    itensor = []
    for path in sorted(directory.glob("*.toml")):
        with path.open("rb") as stream:
            result = tomllib.load(stream)
        if "ITensorMPS" in result:
            itensor.append(result)
    ed = [row for path in sorted(directory.glob("ed-*.json")) for row in json.loads(path.read_text())]
    return itensor, ed


def highest_chi(data, boundary, mode, nmax=6):
    best = {}
    for row in sorted(data, key=lambda d: d["chi_requested"]):
        if row["boundary"] == boundary and row["mode"] == mode and row["nmax"] == nmax:
            best[row["L"], row["U_phys"]] = row
    return best


def correlation_fits(rings, critical, flow_shift):
    results = []
    for fraction in (1/4, 1/3, 1/2):
        for minimum in (8, 12, 16, 24, 32):
            series = sorted((length, data) for (length, u), data in rings.items()
                            if u == critical and length >= minimum)
            if len(series) < 3:
                continue
            lengths = np.asarray([length for length, _ in series])
            # Linear interpolation in log G versus log chord removes integer rounding of r/L.
            values = []
            for length, data in series:
                dist = np.arange(1, length // 2 + 1)
                g1 = np.asarray(data["sectors"]["0"]["g1"])
                x = np.log(np.sin(np.pi * dist / length))
                values.append(np.exp(np.interp(np.log(np.sin(np.pi * fraction)), x, np.log(g1))))
            chord = lengths / np.pi * np.sin(np.pi * fraction)
            eta_plain, _ = exponent(chord, values)
            log_flow = np.log(lengths) + flow_shift
            eta_log, _ = exponent(chord, np.asarray(values) / log_flow**0.125)
            log_power, _ = exponent(log_flow, np.asarray(values) * chord**0.25)
            results.append({"fraction": fraction, "L": lengths.tolist(),
                            "eta_raw": eta_plain, "eta_log_corrected": eta_log,
                            "log_power_if_eta_quarter": -log_power})
    return results


def independent_rg(rings):
    pairs = [(12, 16), (16, 24), (24, 32), (32, 48), (48, 64)]
    couplings = sorted({coupling for _, coupling in rings})
    results = []
    for first, last in pairs:
        for fraction in (1/16, 1/8, 1/4):
            points = []
            for u in couplings:
                if not 3.20 <= u <= 3.3578 or (first, u) not in rings or (last, u) not in rings:
                    continue
                k1, k2 = [extract_k(rings[size, u]["sectors"]["0"]["block_variance"], size, fraction)
                          for size in (first, last)]
                invariant = zeta(first, last, k1, k2)
                if np.isfinite(invariant):
                    points.append([u, invariant, k1, k2])
            brackets = []
            for left, right in zip(points[:-1], points[1:]):
                if left[1] > 1 >= right[1]:
                    brackets.append(float(left[0] + (right[0] - left[0]) * (1 - left[1]) / (right[1] - left[1])))
            results.append({"L": [first, last], "block_fraction": fraction,
                            "U_phys_c_interpolated": brackets, "points": points})
    return results


def benchmark_errors(itensor, ed):
    reference = {(d["L"], d["U_phys"], d["nmax"]): d["sectors"]["0"] for d in ed}
    comparisons = []
    for row in itensor:
        key = row["L"], row["U_phys"], row["nmax"]
        if key not in reference or row["boundary"] != "periodic" or row["mode"] != "correlations":
            continue
        actual, exact = row["sectors"]["0"], reference[key]
        comparisons.append({"L": key[0], "U_phys": key[1], "nmax": key[2],
                            "abs_energy_error": abs(actual["energy"] - exact["energy"]),
                            "g1_max_abs_error": float(np.max(np.abs(np.subtract(actual["g1"], exact["g1"])))),
                            "block_variance_max_abs_error": float(np.max(np.abs(np.subtract(
                                actual["block_variance"], exact["block_variance"])))),
                            "ED_residual": exact["residual"]})
    return comparisons


def published_errors(rings, archive):
    if not archive.is_file():
        return []
    comparisons = []
    for u, length, variance in published_data(archive):
        if (length, u) not in rings:
            continue
        simulated = rings[length, u]["sectors"]["0"]["block_variance"]
        comparisons.append({"L": length, "U_phys": u,
                            "max_abs_variance_error": float(np.max(np.abs(np.subtract(variance, simulated)))),
                            "K_error": extract_k(simulated, length, 1/16) - extract_k(variance, length, 1/16)})
    return sorted(comparisons, key=lambda row: (row["L"], row["U_phys"]))


def sensitivity(data, critical):
    rows = []
    for boundary in ("periodic", "open"):
        filtered = [d for d in data if d["mode"] == "correlations" and d["boundary"] == boundary
                    and d["U_phys"] == critical]
        for length in sorted({row["L"] for row in filtered}):
            for d in sorted((r for r in filtered if r["L"] == length), key=lambda r: (r["nmax"], r["chi_requested"])):
                s = d["sectors"]["0"]
                row = {"L": length, "boundary": boundary, "nmax": d["nmax"], "chi": d["chi_requested"],
                       "energy": s["energy"], "g1_half": s["g1"][-1], "last_delta_E": s["last_delta_E"],
                       "last_truncation": s["last_truncation"], "density_cov_sum": s["density_cov_sum"],
                       "converged_sweeps": s["converged_sweeps"]}
                if boundary == "periodic":
                    row["K"] = extract_k(s["block_variance"], length, 1/16)
                rows.append(row)
    return rows


def gap_analysis(gaps, critical, uc):
    groups = {u: sorted((l, charge_gap(row)) for (l, coupling), row in gaps.items() if coupling == u)
              for _, u in gaps}
    z = []
    for start in (32, 64, 128):
        rows = [(l, gap) for l, gap in groups.get(critical, []) if l >= start]
        if len(rows) >= 2:
            l, gap = np.array(rows).T
            value, _ = exponent(l, gap)
            z.append({"L": l.astype(int).tolist(), "z_effective": value})
    plateau = []
    for u, rows in sorted(groups.items()):
        if u <= critical or len(rows) < 3:
            continue
        lengths, values = np.asarray(rows).T
        drift = abs(values[-2] / values[-1] - 1)
        if drift > .10:
            continue
        # OBC: test both leading empirical finite-size forms, and retain the spread.
        estimates = [float(np.polyfit(lengths**(-p), values, 1)[1]) for p in (1, 2)]
        plateau.append({"U_phys": u, "gap_max_L": float(values[-1]), "relative_final_drift": float(drift),
                        "gap_inf_linear_invL": estimates[0], "gap_inf_linear_invL2": estimates[1]})
    essential = []
    for field in ("gap_max_L", "gap_inf_linear_invL", "gap_inf_linear_invL2"):
        points = [row for row in plateau if row[field] > 0]
        if len(points) < 4:
            continue
        x = np.array([row["U_phys"] - uc for row in points])
        y = np.log([row[field] for row in points])
        for subset in (slice(None), slice(None, -1)):
            if len(x[subset]) < 4:
                continue
            delta = x[subset]
            target = y[subset]
            residual = lambda p: p[0] - p[1] / delta**p[2] - target
            fit = least_squares(residual, [1., 2., .5], bounds=([-30, 0, .01], [100, 100, 5]))
            essential.append({"estimator": field, "U_range": [float(delta[0] + uc), float(delta[-1] + uc)],
                              "sigma_effective": float(fit.x[2]), "rms_log_gap": float(np.sqrt(np.mean(fit.fun**2)))})
    return {"gap_vs_L": groups, "z_fits": z, "large_L_mott_gaps": plateau, "essential_gap_fits": essential}


def plot(rings, gaps, critical, flow_shift, output):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 4), layout="constrained")
    values = sorted((l, d["sectors"]["0"]["g1"][-1]) for (l, u), d in rings.items() if u == critical)
    if values:
        lengths, g1 = np.asarray(values).T
        corrected = g1 / (np.log(lengths) + flow_shift)**.125
        left.loglog(lengths, corrected, "o", label="ITensor DMRG")
        left.loglog(lengths, corrected[-1] * (lengths / lengths[-1])**(-.25), "--", label=r"$L^{-1/4}$")
        left.legend(frameon=False)
    left.set(xlabel="$L$ (periodic)", ylabel=r"$G_1(L/2)/[\log L+C]^{1/8}$")
    for u, rows in sorted(gaps["gap_vs_L"].items()):
        l, delta = np.asarray(rows).T
        right.loglog(l, delta, "o-", label=f"{u:g}")
    right.set(xlabel="$L$ (open)", ylabel=r"Charge gap $\Delta_c/t$")
    if gaps["gap_vs_L"]:
        right.legend(title=r"$U_{\rm phys}/t$", frameon=False, ncols=2, fontsize=8)
    fig.savefig(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--published-archive", type=Path, default=Path(__file__).resolve().parents[1] / "raw" / "published-source.zip")
    args = parser.parse_args()
    itensor, ed = load_results(args.directory)
    rings = highest_chi(itensor, "periodic", "correlations")
    gaps = highest_chi(itensor, "open", "gap")
    critical = 3.275
    uc = json.loads((args.directory / "reproduction.json").read_text())["primary"]["U_phys_c"]
    critical_rings = sorted((l, extract_k(d["sectors"]["0"]["block_variance"], l, 1/16))
                            for (l, u), d in rings.items() if u == critical and l >= 16)
    if len(critical_rings) >= 2:
        lengths, k = np.asarray(critical_rings).T
        fit = least_squares(lambda p: 2 + 1 / (np.log(lengths) + p[0]) - k, [0.], bounds=([-1.5], [20.]))
        flow_shift = float(fit.x[0])
    else:
        flow_shift = 0.
    gap_results = gap_analysis(gaps, critical, uc)
    ed_gaps = sorted((d["L"], charge_gap(d)) for d in ed if d["U_phys"] == critical and d["nmax"] == 6)
    ed_z = []
    if len(ed_gaps) >= 2:
        ed_lengths, ed_delta = np.array(ed_gaps).T
        ed_z = [{"L": ed_lengths[i:].astype(int).tolist(),
                 "z_effective": exponent(ed_lengths[i:], ed_delta[i:])[0]}
                for i in range(len(ed_gaps) - 1)]
    result = {"num_ITensor": len(itensor), "num_ED": len(ed), "U_phys_for_exponents": critical,
              "ED_comparison": benchmark_errors(itensor, ed),
              "published_comparison": published_errors(rings, args.published_archive),
              "critical_sensitivity": sensitivity(itensor, critical),
              "RG_pairs": independent_rg(rings),
              "RG_pairs_nmax4": independent_rg(highest_chi(itensor, "periodic", "correlations", nmax=4)),
              "critical_flow_log_shift": flow_shift, "critical_K_L": critical_rings,
              "eta_fits": correlation_fits(rings, critical, flow_shift), "ED_critical_gaps": ed_gaps, "ED_z_fits": ed_z, **gap_results}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(rings, gap_results, critical, flow_shift, args.output)
    print(f"Analyzed {len(itensor)} ITensor runs and {len(ed)} ED size/coupling/cutoff points")
    for key in ("RG_pairs", "RG_pairs_nmax4"):
        print(key, [(row["L"], row["block_fraction"], row["U_phys_c_interpolated"])
                    for row in result[key] if row["U_phys_c_interpolated"]])
    for key in ("critical_K_L", "ED_z_fits", "z_fits", "large_L_mott_gaps", "essential_gap_fits"):
        print(key, result[key])
    print("cutoff sensitivity at critical coupling", [
        (row["L"], row["nmax"], row["chi"], row.get("K"), row["g1_half"])
        for row in result["critical_sensitivity"] if row["boundary"] == "periodic" and row["L"] >= 24
    ])
    print("eta fits, Lmin >= 16", [row for row in result["eta_fits"] if row["L"][0] >= 16])


if __name__ == "__main__":
    main()
