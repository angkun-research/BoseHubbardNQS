#!/bin/bash -l
set -euo pipefail
module load anaconda3/2024.06
export PYTHONNOUSERSITE=1 MPLCONFIGDIR="$PWD/matplotlib"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
study_dir=$1
program=$2
shift 2
exec python -B -u "$study_dir/$program" "$@"
