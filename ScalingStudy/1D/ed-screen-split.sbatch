#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
python=${2:?absolute Python executable required}
model=${3:?screen model required}
stage=${4:?tuning, confirmation, or report required}
length=${5:-8}
runner="$repo/ScalingStudy/1D/ed_fidelity.py"
root="$PWD/ed-l$length-screen/$model-split"
threads=${SLURM_CPUS_PER_TASK:-1}
export PYTHONNOUSERSITE=1
mkdir -p -- "$root"
phase_filter=

case "$stage" in
    tuning)
        number=${SLURM_ARRAY_TASK_ID:?array index required}
        values=(2000 8000 32000)
        manifest="$root/tuning-$number.json"
        screen_options=(--length "$length")
        if [[ $length == 10 ]]; then
            previous=${6:?L=8 confirmation manifest is required for L=10}
            screen_options+=(--from-confirmation "$previous")
        fi
        "$python" -B "$runner" plan-screen --arch "$model" --threads "$threads" "${screen_options[@]}" \
            --output-root "$root/tuning" > "$manifest"
        column=band
        ;;
    confirmation)
        number=${SLURM_ARRAY_TASK_ID:?array index required}
        values=(101 202 303)
        manifest="$root/confirmation-$number.json"
        "$python" -B "$runner" select-screen --manifest "$root/tuning-0.json" \
            --output-root "$root/confirmation" > "$manifest"
        column=seed
        ;;
    report)
        "$python" -B "$runner" report-screen --manifest "$root/confirmation-0.json" > "$root/report.json"
        cat "$root/report.json"
        exit
        ;;
    *)
        echo "Unknown screen stage: $stage" >&2
        exit 2
        ;;
esac

if [[ ${SLURM_ARRAY_TASK_COUNT:-0} == 9 ]]; then
    phases=(sf bkt mott)
    phase_filter=${phases[$((number % 3))]}
    number=$((number / 3))
fi

indices=$("$python" -B -c '
import json
import sys

with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
key, value, phase = sys.argv[2], int(sys.argv[3]), sys.argv[4]
for index, trial in enumerate(manifest["trials"]):
    if trial[key] == value and (not phase or trial["phase"] == phase):
        print(index)
' "$manifest" "$column" "${values[$number]}" "$phase_filter")
for index in $indices; do
    echo "Running manifest row $index from $manifest"
    "$python" -B "$runner" task --manifest "$manifest" --index "$index"
done
