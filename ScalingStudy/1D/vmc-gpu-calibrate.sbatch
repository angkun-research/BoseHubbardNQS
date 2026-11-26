#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
report_l8=${2:-"$PWD/ed-l8-screen/FCNet-6653047/report.json"}
report_l10=${3:-"$PWD/ed-l10-screen/FCNet-split/report.json"}
for report in "$report_l8" "$report_l10"; do
    if [[ ! -r $report ]]; then
        echo "Provide readable FCNet L=8 and L=10 report paths: $report" >&2
        exit 2
    fi
done
export PYTHONNOUSERSITE=1
nvidia-smi -L
python=${4:-}
if [[ -z $python ]]; then
    source /etc/profile.d/modules.sh
    module load anaconda3/2024.06
    env_dir="$PWD/torch-cu124-$SLURM_JOB_ID"
    python -m venv --system-site-packages "$env_dir"
    python="$env_dir/bin/python"
    "$python" -m pip --disable-pip-version-check install --no-cache-dir --retries 2 --timeout 30 \
        --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0'
fi
"$python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'

for walkers in 64 256 1024; do
    "$python" -B "$repo/ScalingStudy/1D/vmc_train.py" --arch FCNet --length 16 --phase bkt \
        --width 49 --seed 101 --steps 100 --walkers "$walkers" --burn-in 32 --sample-steps 8 \
        --chunk-size 256 --threads "${SLURM_CPUS_PER_TASK:-1}" --device cuda \
        --screen-l8 "$report_l8" --screen-l10 "$report_l10" \
        --output "$PWD/vmc-gpu-calibrate-$SLURM_JOB_ID/walkers-$walkers"
done
