#!/usr/bin/env python3
"""Assemble presentation data from independently assessed frozen VMC results.

The selection JSON has a `runs` list: model, Lx, assessment, updates (optional).
Every assessment must contain predeclared confirmation seed 654. Tables never
select the most favorable evaluation seed. A partial deck can be written for
review, but the renderer requires status=complete for publication.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["FC", "Jastrow", "Conv2D", "RBM", "Transformer"]
COLORS = dict(zip(MODELS, ["#3575a2", "#bb7937", "#3d9367", "#a25283", "#7562aa"]))
ARCH = dict(zip(MODELS, ["FCNet", "JastrowNet", "Conv2DNet", "RBMNet", "TransformerNet"]))


def read(path):
    return json.loads((ROOT / path).read_text())


def confirmation(assessment, progress=False):
    matches = [r for r in assessment["reports"] if read(r["report"])["settings"]["seed"] == 654]
    if not matches and progress:
        matches = [r for r in assessment["reports"] if read(r["report"])["settings"]["seed"] == 321]
    if len(matches) != 1:
        raise ValueError("need exactly one independent seed-654 confirmation per selection")
    return matches[0]


def metric(report, name):
    return report["statistics"][name]


def series(label, x, values, errors=None, **kwargs):
    result = dict(label=label, x=x, y=values, **kwargs)
    if errors is not None:
        result["yerr"] = errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", default="slides/deck.json")
    parser.add_argument("--progress", action="store_true", help="explicitly labeled audited snapshot; permits unfinished comparisons")
    args = parser.parse_args()
    selection = read(args.selection)
    deck = read("slides/base_slides.json")
    deck["sources"] += [args.selection, "slides/assemble_results.py", "slides/base_slides.json"]
    deck["sources"] += selection.get("sources", [])
    deck["sources"] += [p.name for p in sorted(ROOT.glob("*.pdf"))
                        if re.fullmatch(r"BoseNQS\d{2}-\d{2}-\d{4}\.pdf", p.name)]
    chosen = {}
    for run in selection["runs"]:
        key = (run["model"], run["Lx"])
        if key in chosen:
            raise ValueError(f"duplicate selected model/size: {key}")
        assessment = read(run["assessment"])
        baseline = run["model"] == "Jastrow"
        report = confirmation(assessment, args.progress or baseline)
        seeds = [read(r["report"])["settings"]["seed"] for r in assessment["reports"]]
        if not args.progress and not baseline and (set(seeds) != {321, 654} or len(seeds) != 2):
            raise ValueError("final comparisons require exactly the screening and confirmation evaluations")
        report = dict(report)
        report["evaluation_seed"] = read(report["report"])["settings"]["seed"]
        report["both_seed_summary_targets_met"] = set(seeds) == {321,654} and all(
            r["all_four_summary_point_and_precision_targets_met"] for r in assessment["reports"])
        metadata = report["checkpoint_metadata"]
        if (metadata["Lx"] != run["Lx"] or metadata["Ly"] != 4
                or metadata["arch"] != ARCH[run["model"]]
                or metadata["pbc_x"] or not metadata["pbc_y"]):
            raise ValueError("selection model/geometry mismatch")
        if run["model"] == "Jastrow" and metadata["architecture_settings"].get("pair_sharing", "displacement") != "displacement":
            raise ValueError("main comparison requires basic Jastrow; boundary-orbit experiments remain archived")
        chosen[key] = (run, report)
        deck["sources"] += [run["assessment"], report["report"]]
        training_record = Path(str(report["checkpoint"]) + ".json")
        if not training_record.is_absolute():
            training_record = ROOT / training_record
        if training_record.is_file():
            deck["sources"].append(str(training_record.relative_to(ROOT)))
    slides = deck["slides"]
    slides.insert(8, dict(kind="table", title="Capacity and loss checks",
        columns=["Candidate", "Objective / optimizer", "Absolute overlap", "Overlap gate"],
        rows=[["FC64, 2 layers", "Adam fit → width lift", "0.997904", "pass"],
              ["RBM64", "Adam fit → widening", "0.992508", "pass"],
              ["Transformer64", "log-MSE / Adam", "0.990323", "pass"]],
        fontsize=31,
        footer="3×3 cylinder. FC/RBM enlarged from qualified states, with independent raw-ED checks. Basic Jastrow is a physical baseline; no further adaptation or common neural accuracy requirement."))
    slides.insert(4, dict(kind="table", title="Check the references", fontsize=32,
        columns=["Check", "Observed difference / conclusion"],
        rows=[["3×3: DMRG versus ED", "Energy: 4.27×10⁻¹⁰ t; full ρ: 3.08×10⁻⁹"],
              ["4×4: χ128 → χ256 spot", "Energy/site: 1.64×10⁻⁶ t"],
              ["4×4 → 6×4, fixed χ128", "Central energy: 1.6%; NN G: 3–4%"],
              ["Occupation cutoff", "Matched Nmax=4; larger-size cutoff convergence untested"]],
        widths=[.42,.58],
        footer="Tiny ED also has analytic-limit and independent ladder-operator checks. Finite-χ and finite-size effects are separate from Monte Carlo SEM; modest length changes do not establish bulk convergence."))
    deck["sources"] += ["benchmarks/results/dmrg-nqs-4x4-6x4-central-strip.json",
                         "benchmarks/results/ed-cutoff-wavefunction-check.json",
                         "benchmarks/results/2026-10-31-exact-jastrow-sr/report.json",
                         "benchmarks/results/2026-10-31-exact-jastrow-orbit-sr/report.json",
                         "benchmarks/results/2026-10-31-all-model-refinement/checker-width-qualification.json",
                         "benchmarks/results/2026-10-31-all-model-refinement/checker-jastrow-orbit-qualification-review.json"]
    small = read("slides/small_system_minsr.json")
    slides.append(small["slide"])
    observables = read("slides/observable_definitions.json")
    slides.extend(observables["slides"])
    deck["sources"] += ["slides/small_system_minsr.json", small["source"],
                         "slides/observable_definitions.json", *observables["sources"]]
    slides += [dict(kind="table", title="VMC accuracy targets",
        columns=["Observable", "Point difference", "Conditional SEM limit"],
        rows=[["Energy / site", "≤1% of |DMRG|", "≤0.25% of |DMRG|"],
              ["Central NN Gx, Gy", "≤5% of |DMRG|", "≤2.5% of |DMRG|"],
              ["Central mean density", "≤0.01", "≤0.005"]],
        footer="Central strip = two middle x columns. Form composite observables before blocking. Point and precision criteria must both pass; a large error bar cannot earn a pass.")]
    refinement_rows = []
    for Lx in [4, 6]:
        for stage, filename in [("B: +400 updates", f"Conv2DNet-{Lx}x4-stageB-assessment.json"),
                                ("C: +500 updates", f"Conv-stageC-{Lx}x4-independent-assessment.json")]:
            path = f"benchmarks/results/2026-10-31-vmc-refinement/{filename}"
            assessment = read(path)
            report = confirmation(assessment) if stage.startswith("C") else assessment["reports"][0]
            stat = metric(report, "whole_lattice_energy_per_site")
            refinement_rows.append([f"{Lx}×4", stage,
                f'{stat["mean"]:.6f} ± {stat["conditional_sem"]:.6f}',
                f'{100*stat["difference_to_reference"]/abs(stat["reference"]):.2f}%'])
            deck["sources"] += [path, report["report"]]
    slides.append(dict(kind="table", title="Tighten Conv2D VMC", fontsize=32,
        columns=["Cylinder", "Continuation", "Energy/site ± SEM", "DMRG gap"],
        rows=refinement_rows, widths=[.14,.29,.38,.19],
        footer="Same Conv32 architecture. Stage B: M=1024; C: M=2048 with decreasing learning rate and SR shift. Fresh walkers equilibrated for 4096 proposals. B uses seed321; C independent confirmation seed654."))
    for Lx in [4, 6]:
        rows = []
        failures = []
        for model in MODELS:
            if (model, Lx) not in chosen:
                if args.progress or model == "Jastrow":
                    rows.append([model, "—", "not assessed", "—", "baseline" if model == "Jastrow" else "pending"])
                continue
            run, report = chosen[model, Lx]
            stat = metric(report, "whole_lattice_energy_per_site")
            gap = 100 * stat["difference_to_reference"] / abs(stat["reference"])
            passes = report["both_seed_summary_targets_met"]
            baseline = model == "Jastrow"
            label = f'{model}\nseed {report["evaluation_seed"]}' if args.progress or report["evaluation_seed"] != 654 else model
            rows.append([label, str(report["checkpoint_metadata"]["parameter_count"]),
                f'{stat["mean"]:.6f} ± {stat["conditional_sem"]:.6f}',
                f"{gap:.2f}%", "baseline" if baseline else ("pass" if passes else "below target")])
            if not passes and not baseline:
                failures.append(len(rows))
        slides.append(dict(kind="table", title=f"Compare: {Lx}×4 cylinder", fontsize=31,
            columns=["Model", "Parameters", "Energy/site ± SEM", "Gap", "Neural gate"], rows=rows,
            widths=[.18,.15,.33,.12,.22], highlight_rows=failures,
            footer=("Progress snapshot: confirmation seed 654 when available, screening seed 321 otherwise. A neural pass requires both seeds and all four targets. Jastrow is a fixed baseline, exempt from that gate." if args.progress else
                    "Energy table: independent seed 654, CPU float64. Neural gate requires both seeds to pass energy, density and both NN directions. Basic Jastrow is a physical baseline; not an all-distance correlation gate.")))
    for Lx in [4, 6]:
        available = [(model, chosen[model, Lx][1]) for model in MODELS if (model, Lx) in chosen]
        if not available:
            continue
        panels = []
        for name, title in [("central_NN_Gx", "$G_x$"), ("central_NN_Gy", "$G_y$")]:
            data = [metric(r, name) for _, r in available]
            panels.append(dict(title=title, xlabel="Model", ylabel="Central NN correlation",
                xticks=list(range(1,len(data)+1)), xticklabels=[model for model,_ in available],
                series=[series("VMC ± SEM", list(range(1,len(data)+1)), [s["mean"] for s in data],
                    [s["conditional_sem"] for s in data], marker="o", linestyle="none", color="#3575a2")],
                reference_lines=[dict(y=data[0]["reference"], label="DMRG χ=128")]))
        slides.append(dict(kind="plots", title=f"One-body correlations: {Lx}×4", panels=panels,
            footer="Both endpoints lie inside the central strip; directed bond estimates are averaged per configuration. Errors are max(block, between-chain SEM), conditional on stationarity and independent walkers."))
        for observable, label in [("density", r"$\langle n\rangle_y$"),
                                  ("onsite_factorial_moment", r"$\langle n(n-1)\rangle_y$")]:
            groups = [available[:3], available[3:]] if len(available) > 3 else [available]
            panels = []
            for group in groups:
                if not group:
                    continue
                curves = []
                reference = group[0][1]["profiles_per_x"]
                curves.append(series("DMRG", [p["x"] for p in reference],
                    [p[observable]["reference"] for p in reference], color="black", linestyle="--"))
                for model, report in group:
                    profiles = report["profiles_per_x"]
                    curves.append(series(model, [p["x"] for p in profiles],
                        [p[observable]["mean"] for p in profiles],
                        [p[observable]["conditional_sem"] for p in profiles], color=COLORS[model], marker="o"))
                panels.append(dict(title=f"{Lx}×4 cylinder", xlabel="Open-axis position x", ylabel=label,
                    series=curves, legend_size=23))
            slides.append(dict(kind="plots", title=f"{'Density' if observable=='density' else 'Occupation fluctuations'}: {Lx}×4",
                panels=panels, footer="Average over periodic y within each raw sample before blocking; error bars retain covariance. DMRG and VMC use the same local occupation cutoff Nmax=4."))
    anchor_panels = []
    for Lx in [4, 6]:
        curves = []
        available = [(model, chosen[model,Lx][1]) for model in MODELS if (model,Lx) in chosen]
        for index, (model, report) in enumerate(available):
            points = [p for p in report["anchor_same_y_line"] if p["dx"] != 0]
            if index == 0:
                curves.append(series("DMRG", [p["dx"] for p in points],
                    [p["reference_real"] for p in points], color="black", linestyle="--"))
            curves.append(series(model, [p["dx"] for p in points], [p["mean"] for p in points],
                [p["conditional_sem"] for p in points], color=COLORS[model], marker="o"))
        anchor_panels.append(dict(title=f"{Lx}×4: central anchor, same y", xlabel="Signed separation dx",
            ylabel=r"$\langle b_i^\dagger b_j\rangle$", series=curves, legend_size=22))
    slides.append(dict(kind="plots", title="Correlations beyond NN", panels=anchor_panels,
        footer="Raw directed anchor-line estimates with conditional SEM; diagonal excluded. Individual pairs are noisier than central NN averages. The 5% summary target does not apply to this whole curve."))
    budget_rows = []
    for (model, Lx), (run, report) in sorted(chosen.items(), key=lambda item:(item[0][1], MODELS.index(item[0][0]))):
        architecture = report["checkpoint_metadata"]["architecture_settings"]
        configuration = ", ".join(f"{k}={v}" for k,v in architecture.items()) or "pair tables"
        budget_rows.append([f"{model}, {Lx}×4", configuration, str(run.get("updates", "see run manifest"))])
    slides.append(dict(kind="table", title="Configurations and budgets", fontsize=27,
        columns=["Model and cylinder", "Architecture settings", "Update lineage"], rows=budget_rows,
        widths=[.30,.42,.28],
        footer="Architecture-specific optimization budgets; this is an accuracy comparison, not an equal-compute ranking. Warm starts restore inference weights only; optimizers and walkers are fresh."))
    slides.append(dict(kind="text", title="Takeaways and next steps", fontsize=41, step=.14,
        lines=selection.get("takeaways", ["Review selected results and sampling diagnostics before publication."]),
        footer="Finite cylinders at U/t=16.74 are a building benchmark. No thermodynamic limit, critical exponents, larger-system fidelity or universal architecture ranking is inferred."))
    appendix = read("slides/ridge_sr_appendix.json")
    slides.extend(appendix["slides"])
    slides.append(observables["estimator_appendix"])
    deck["sources"] += ["slides/ridge_sr_appendix.json", "benchmarks/results/minsr-paper-comparison.json",
        "benchmarks/results/2026-10-31-all-model-refinement/checker-sr-1d-2d-same-batch.json",
        "benchmarks/results/2026-10-31-all-model-refinement/checker-sr-legacy-centering-gauge-audit.json"]
    comparison_path = "benchmarks/results/2026-10-31-all-model-refinement/checker-rbm-matched-minsr300-comparison.json"
    if (ROOT / comparison_path).is_file():
        comparison = read(comparison_path)
        settings = comparison["matched_settings"]
        differences = comparison["intentional_settings_differences"]
        runtime = comparison["observed_runtime"]
        slides.append(dict(kind="table", title="Matched optimizer comparison", fontsize=31,
            columns=["Setting", "Ridge SR", "MinSR soft filter"],
            rows=[["Model and geometry", "RBM64, 4×4 cylinder", "same"],
                  ["Starting weights", "same checkpoint", "same checkpoint"],
                  ["Updates / samples per update", f'{settings["epochs"]} / {settings["n_walkers"]*settings["n_steps"]}', "same"],
                  ["Learning rate", f'{settings["lr"]:g} → {settings["lr_end"]:g}', "same"],
                  ["Regularization", f'τ = {differences["tau"][0]:g}', f'r = {differences["minsr_rpinv"][1]:g}, a = 0'],
                  ["Observed shared-GPU time", f'{runtime["ridge"]["subprocess_wall_seconds"]:.0f} s', f'{runtime["minsr"]["subprocess_wall_seconds"]:.0f} s']],
            widths=[.40,.30,.30],
            footer="One optimizer seed; two evaluation seeds probe measurement variation, not independent training runs. Float64 Gram/solve/eigensolve; native SGD. Shared-GPU times are not isolated throughput benchmarks."))
        rows = []
        for seed in [321,654]:
            for method in ["ridge","minsr"]:
                record = next(q for q in comparison["arms"][method] if q["evaluation_seed"] == seed)
                energy = record["statistics"]["whole_lattice_energy_per_site"]
                rows.append(["Ridge SR" if method == "ridge" else "MinSR", str(seed),
                    f'{energy["mean"]:.6f} ± {energy["conditional_sem"]:.6f}', f'{record["energy_gap_percent"]:.2f}%'])
        slides.append(dict(kind="table", title="First comparison: no clear winner", fontsize=32,
            columns=["Optimizer", "Evaluation seed", "Energy/site ± SEM", "DMRG gap"],
            rows=rows, widths=[.22,.22,.38,.18],
            footer="Both arms miss the energy target; central density and both NN directions pass. Paired conditional differences are unresolved (1.30 / 0.15 SEM). This short, single-training-seed test establishes no general advantage."))
        deck["sources"].append(comparison_path)
    expected = {(model, Lx) for model in MODELS if model != "Jastrow" for Lx in [4,6]}
    deck["status"] = "complete" if expected.issubset(chosen) and selection.get("audited") else "partial_template"
    if args.progress and selection.get("audited"):
        deck["status"] = "audited_progress_snapshot"
        deck["title"] = "Two-dimensional NQS: benchmark progress"
        deck["slides"][0]["title"] = "Bose–Hubbard NQS\n2D benchmark progress"
        deck["slides"][0]["subtitle"] = "Checked results and unresolved accuracy targets"
    deck["sources"] = list(dict.fromkeys(deck["sources"]))
    (ROOT / args.output).write_text(json.dumps(deck, indent=2) + "\n")
    print(f'{len(slides)} slides; status={deck["status"]}')


if __name__ == "__main__":
    main()
