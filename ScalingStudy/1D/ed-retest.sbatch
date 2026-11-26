#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
action=${2:?plan, task, retry, or merge required}
# Default to the archived environment. A replacement runtime must still meet
# ed_retest.py's exact-replay compatibility checks.
python=${BHNQS_ED_RETEST_PYTHON:-"$PWD/torch-cpu-6652940/bin/python"}
runner="$repo/ScalingStudy/1D/ed_retest.py"
root="$PWD/ed-l10-screen"
manifest="$PWD/ed-l10-mott-retest.json"
originals=()
for model in InvariantConv1D TransformerNet ViTNet; do
    originals+=(--original "$root/$model-split/confirmation-0.json")
done
export PYTHONNOUSERSITE=1
case "$action" in
    plan)
        "$python" -B "$runner" plan "${originals[@]}" --output-root "$PWD/ed-l10-mott-retest" > "$manifest.tmp"
        mv -T -- "$manifest.tmp" "$manifest"
        ;;
    task|retry)
        index=${3:?trial index required}
        if [[ $action == retry ]]; then
            output=$("$python" -B -c '
import json
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
trial = json.loads(manifest.read_text())["trials"][int(sys.argv[2])]
output = Path(trial["config"]["output"]).resolve()
if not output.is_relative_to((manifest.parent / "ed-l10-mott-retest").resolve()):
    raise ValueError(f"Unexpected retest path: {output}")
print(output)
' "$manifest" "$index")
            if [[ -f $output/result.json ]]; then
                echo "Preserving completed retest: $output"
                exit 0
            fi
            if [[ -d $output ]]; then
                archive="$output.interrupted-$SLURM_JOB_ID"
                if [[ -e $archive ]]; then
                    echo "Retest archive already exists: $archive" >&2
                    exit 3
                fi
                mv -T -- "$output" "$archive"
                echo "Preserved interrupted retest in $archive"
            fi
        fi
        exec "$python" -B "$runner" task --manifest "$manifest" --index "$index"
        ;;
    merge)
        report="$PWD/ed-l10-mott-retest-report.json"
        "$python" -B "$runner" merge "${originals[@]}" --retest "$manifest" > "$report.tmp"
        mv -T -- "$report.tmp" "$report"
        ;;
    *)
        echo "Unknown action: $action" >&2
        exit 2
        ;;
esac
