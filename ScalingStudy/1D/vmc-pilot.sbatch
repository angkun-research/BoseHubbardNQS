#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
python=${2:?absolute Python executable required}
report_l8=${3:?L=8 screen report required}
report_l10=${4:?L=10 screen report required}
threads=${SLURM_CPUS_PER_TASK:-1}
root="$PWD/vmc-l16-pilot-$SLURM_JOB_ID"
export PYTHONNOUSERSITE=1

status=$("$python" -B -c '
import json
import sys

for length, path in zip((8, 10), sys.argv[1:]):
    with open(path) as handle:
        report = json.load(handle)
    if report["missing_results"]:
        raise ValueError(f"incomplete L={length} confirmation")
    if report[f"l{length}_screen"]["FCNet"][f"l{length}_all_three_pass"] is not True:
        print("ineligible")
        break
else:
    print("eligible")
' "$report_l8" "$report_l10")
if [[ $status != eligible ]]; then
    echo "FCNet did not pass the all-three gate at both ring sizes. No VMC trial run."
    exit 0
fi

for phase in sf bkt mott; do
    trial="$root/$phase"
    "$python" -B "$repo/ScalingStudy/1D/vmc_train.py" --screen-l8 "$report_l8" \
        --screen-l10 "$report_l10" --arch FCNet --length 16 --phase "$phase" \
        --width 49 --seed 101 --steps 100 --threads "$threads" --output "$trial"
    checkpoint=$("$python" -B -c '
import json
import sys
with open(sys.argv[1]) as handle:
    print(json.load(handle)["checkpoints"][-1]["path"])
' "$trial/result.json")
    for seed in 411 733; do
        "$python" -B "$repo/ScalingStudy/1D/vmc_evaluate.py" \
            --checkpoint "$trial/$checkpoint" --length 16 --phase "$phase" --distances 1 2 3 4 8 \
            --number-blocks 4 8 --chains 32 --burn-in 1024 --steps 1024 --seed "$seed" \
            --block-sizes 32 64 128 --threads "$threads" --output "$root/$phase-eval-$seed"
    done
done
