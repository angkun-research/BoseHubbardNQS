#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
source /etc/profile.d/modules.sh
module load anaconda3/2024.06
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The Anaconda module supplies a compatible SciPy and NumPy pair.
env_dir="$PWD/torch-cpu-$SLURM_JOB_ID"
python -m venv --system-site-packages "$env_dir"
python="$env_dir/bin/python"
"$python" -m pip --disable-pip-version-check install --no-cache-dir --retries 2 --timeout 20 \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
"$python" -c 'import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__)'

for spec in FCNet:sf RBMNet:bkt Conv1DNet:mott Conv1DPoolNet:sf TransformerNet:bkt ConvTransformer:mott ViTNet:sf InvariantConv1D:bkt; do
    model=${spec%:*}
    phase=${spec#*:}
    "$python" -B "$repo/ScalingStudy/1D/ed_fidelity.py" trial --length 6 \
        --phase "$phase" --arch "$model" --width 8 --seed 101 --max-steps 1 \
        --max-flops 1000000000 --output "$PWD/ed-smoke-$SLURM_JOB_ID/$model"
done
