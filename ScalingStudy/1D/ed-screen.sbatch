#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
python=${2:?absolute Python executable required}
model=${3:?screen model required}
runner="$repo/ScalingStudy/1D/ed_fidelity.py"
root="$PWD/ed-l8-screen/$model-$SLURM_JOB_ID"
threads=${SLURM_CPUS_PER_TASK:-1}
export PYTHONNOUSERSITE=1
mkdir -p -- "$root"

run_manifest() {
    local manifest=$1
    local task_count index
    task_count=$("$python" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "$manifest")
    for ((index=0; index<task_count; index++)); do
        echo "Running manifest row $index of $task_count from $manifest"
        "$python" -B "$runner" task --manifest "$manifest" --index "$index"
    done
}

"$python" -B "$runner" plan-screen --arch "$model" --threads "$threads" \
    --output-root "$root/tuning" > "$root/tuning.json"
run_manifest "$root/tuning.json"
"$python" -B "$runner" select-screen --manifest "$root/tuning.json" \
    --output-root "$root/confirmation" > "$root/confirmation.json"
run_manifest "$root/confirmation.json"
"$python" -B "$runner" report-screen --manifest "$root/confirmation.json" > "$root/report.json"
cat "$root/report.json"
