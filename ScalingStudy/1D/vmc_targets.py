"""Compare frozen VMC evaluation intervals with a matched DMRG reference."""

import argparse
import json
from pathlib import Path

from dmrg_report import CORRELATION_FLOOR, ENERGY_TOLERANCE


def classify(observable, reference, margin, relative, absolute):
    interval = observable.get("interval") if observable else None
    if interval is None:
        return {"status": "unresolved_interval", "reference": reference, "reference_margin": margin}
    a, b = interval[0] - reference - margin, interval[1] - reference + margin
    lower_tolerance = relative * max(0.0, abs(reference) - margin) + absolute
    upper_tolerance = relative * (abs(reference) + margin) + absolute
    if -lower_tolerance <= a <= b <= lower_tolerance:
        status = "pass"
    elif a > upper_tolerance or b < -upper_tolerance:
        status = "miss"
    else:
        status = "unresolved"
    return {"status": status, "reference": reference, "reference_margin": margin,
            "conservative_tolerance": lower_tolerance, "signed_error_interval": [a, b],
            "ess": observable.get("ess")}


def compare(evaluations, dmrg):
    first = evaluations[0]
    length, cutoff = first["config"]["length"], first["config"]["nmax"]
    coupling = first["u_phys_over_t"]
    signatures = {(run["config"]["length"], run["config"]["nmax"], run["u_phys_over_t"],
                   run["config"]["checkpoint"]) for run in evaluations}
    seeds = [run["config"]["seed"] for run in evaluations]
    if len(evaluations) < 2 or len(signatures) != 1 or len(set(seeds)) != len(seeds):
        raise ValueError("comparison requires at least two distinct seeds on the same frozen checkpoint")
    key = f"L{length}-U{coupling:g}-n{cutoff}"
    reference = dmrg.get(key)
    if reference is None or reference["status"] not in ("crossing", "no_certified_crossing_below_reference"):
        return {"status": "reference_unresolved", "key": key,
                "dmrg_status": reference["status"] if reference else "missing"}
    row = next(run for run in reference["runs"] if run["cap"] == reference["reference"])
    results = {}
    for evaluation in evaluations:
        observables = evaluation["observables"]
        scores = {"energy_per_site": classify(observables.get("energy_per_site"), row["energy_density"],
                                             reference["energy_margin"], 0, ENERGY_TOLERANCE)}
        for key_r, margin in reference["g1_margins"].items():
            distance = int(key_r)
            name = f"g1_r{distance}"
            scores[name] = classify(observables.get(name), row["g1"][distance - 1], margin, 0.05, CORRELATION_FLOOR)
        results[str(evaluation["config"]["seed"])] = scores
    passes = [all(score["status"] == "pass" for score in scores.values()) for scores in results.values()]
    misses = [any(score["status"] == "miss" for score in scores.values()) for scores in results.values()]
    status = "pass" if all(passes) else "miss" if all(misses) else "unresolved"
    return {"status": status, "key": key, "reference_cap": row["cap"], "independent_evaluations": results}


def read_result(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmrg", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.pilot_root.is_dir():
        report = {"status": "pilot_not_run"}
    else:
        reference = read_result(args.dmrg)
        report = {}
        for phase in ("sf", "bkt", "mott"):
            evaluations = sorted(args.pilot_root.glob(f"{phase}-eval-*/result.json"))
            report[phase] = (compare([read_result(path) for path in evaluations], reference)
                             if len(evaluations) >= 2 else {"status": "evaluation_incomplete"})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
