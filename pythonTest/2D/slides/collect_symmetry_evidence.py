"""Freeze compact square-symmetry results for portable slide reconstruction.

This is read-only with respect to all benchmark records and checkpoints.
It copies metrics and 338 orbit predictions, not Monte Carlo arrays or weights.
Run after the selected trainer and independent result review have completed.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "benchmarks/results/2026-11-02-square-symmetry"


def collect(trainer_path, review_path):
    sources = {}

    def read(path):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("evidence must be local to pythonTest/2D")
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    trainer = read(trainer_path)
    review = read(review_path)
    if trainer["status"] != "completed":
        raise ValueError("require completed retained trainer record")
    if not review.get("approved", False):
        raise ValueError("require independent result review with approved=true")
    if review.get("trainer_report_sha256") != hashlib.sha256(trainer_path.read_bytes()).hexdigest():
        raise ValueError("independent approval does not match the selected trainer bytes")
    if review.get("checkpoint_sha256") != trainer.get("checkpoint_sha256"):
        raise ValueError("independent approval does not match the selected checkpoint")
    if trainer["full_dimension"] != 19855 or trainer["orbit_dimension"] != 338:
        raise ValueError("unexpected study sector")
    audit = read(STUDY / "reproducible-ed-audit.json")
    independent = read(STUDY / "checker-independent-symmetry-audit.json")
    reference = read(ROOT / "benchmarks/results/2026-11-01-periodic-focus/checker-3x3-torus-independent-ed.json")
    checks = {name: read(STUDY / name) for name in
              ("checker-gcnn-equivariance-v2.json", "checker-worker-orbit-loss.json", "checker-readiness.json")}
    baselines = []
    paths = [(name, ROOT / f"outputs/2026-10-30-ed-3x3-w32-e10000/{name}-seed123.pt.json")
             for name in ("FCNet", "JastrowNet", "Conv2DNet", "RBMNet")]
    paths += [("TransformerNet", ROOT / "outputs/2026-10-30-ed-3x3-Transformer32-f32-e10000/TransformerNet-seed123.pt.json"),
              ("Conv2DTransformerNet", ROOT / "outputs/2026-11-01-backup-extension/Hybrid32-3x3-ed-logmse10000-seed123.pt.json")]
    for name, path in paths:
        data = read(path)
        baselines.append({"architecture": name, "parameter_count": data["parameter_count"],
                          "settings": data["settings"], "metrics": data["best_checkpoint_metrics"]})
    # Copy all trainer diagnostics, so plotted curves and scalar conclusions can
    # be independently traced without a local checkpoint or a GPU.
    return {"status": "independently_reviewed_progress_snapshot", "ed_audit": audit,
            "ed_reference": reference, "independent_symmetry_audit": independent,
            "training": trainer, "result_review": review, "checks": checks,
            "historical_baselines": baselines, "archival_sources_sha256": sources,
            "scope": "One 3x3 N9 cutoff4 torus GCNN supervised fit; Adam/log-MSE; no VMC or architecture ranking claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.trainer, args.review)
    output = ROOT / "slides/symmetry-evidence.json"
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
