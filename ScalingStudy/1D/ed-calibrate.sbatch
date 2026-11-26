#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
smoke_job=${2:?completed CPU smoke job ID required}
python="$PWD/torch-cpu-$smoke_job/bin/python"
threads=${SLURM_CPUS_PER_TASK:-1}
export PYTHONNOUSERSITE=1

for model in FCNet RBMNet Conv1DNet Conv1DPoolNet TransformerNet ConvTransformer ViTNet InvariantConv1D; do
    "$python" -B "$repo/ScalingStudy/1D/ed_fidelity.py" trial --length 8 \
        --phase bkt --arch "$model" --band 8000 --seed 101 --max-steps 12 \
        --max-flops 1000000000000 --threads "$threads" \
        --output "$PWD/ed-calibration-$SLURM_JOB_ID/$model"
done
