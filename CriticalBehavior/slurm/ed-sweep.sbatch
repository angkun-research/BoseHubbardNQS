#!/bin/bash -l
set -euo pipefail
study_dir=$1
lengths=(4 6 8 10 12 8 8 10 10)
cutoffs=(6 6 6 6 6 4 8 4 8)
length=${lengths[SLURM_ARRAY_TASK_ID]}
nmax=${cutoffs[SLURM_ARRAY_TASK_ID]}
exec "$study_dir/slurm/run.sbatch" "$study_dir" src/ed.py "$length" "$nmax" \
    "$PWD/ed-L${length}-n${nmax}.json" 3.20 3.24 3.275 3.31 3.3578 4.0
