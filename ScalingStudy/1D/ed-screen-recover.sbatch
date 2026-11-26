#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
python=${2:?absolute Python executable required}
model=${3:?screen model required}
stage=${4:?tuning or confirmation required}
index=${SLURM_ARRAY_TASK_ID:?array index required}
runner="$repo/ScalingStudy/1D/ed_fidelity.py"
root="$PWD/ed-l10-screen/$model-split"
export PYTHONNOUSERSITE=1

case "$stage" in
    tuning)
        manifest="$root/tuning-0.json"
        ;;
    confirmation)
        manifest="$root/confirmation-$index.json"
        partial="$manifest.tmp-$SLURM_JOB_ID"
        "$python" -B "$runner" select-screen --manifest "$root/tuning-0.json" \
            --output-root "$root/confirmation" > "$partial"
        mv -T -- "$partial" "$manifest"
        ;;
    *)
        echo "Unknown screen stage: $stage" >&2
        exit 2
        ;;
esac

output=$("$python" -B -c '
from pathlib import Path
import json
import sys

manifest, index, root, stage = sys.argv[1:]
trial = json.loads(Path(manifest).read_text())["trials"][int(index)]
output = Path(trial["output"]).resolve()
if not output.is_relative_to(Path(root).resolve() / stage):
    raise ValueError(f"Unexpected trial output: {output}")
print(output)
' "$manifest" "$index" "$root" "$stage")

if [[ -f $output/result.json ]]; then
    echo "Preserving completed trial: $output"
    exit 0
fi
if [[ -d $output ]]; then
    interrupted="$output.interrupted-$SLURM_JOB_ID"
    if [[ -e $interrupted ]]; then
        echo "Archive already exists: $interrupted" >&2
        exit 3
    fi
    mv -T -- "$output" "$interrupted"
    echo "Preserved interrupted trial in $interrupted"
fi
exec "$python" -B "$runner" task --manifest "$manifest" --index "$index"
