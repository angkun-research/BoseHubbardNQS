#!/bin/bash -l
set -euo pipefail
study_dir=$1
manifest=$2
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$study_dir/slurm/sweeps/$manifest")
read -r length coupling nmax boundary chi mode <<< "$row"
output="$PWD/${mode}-${boundary}-L${length}-U${coupling}-n${nmax}-chi${chi}.toml"
exec "$study_dir/slurm/run-itensor.sbatch" "$study_dir" "$length" "$coupling" "$nmax" "$boundary" "$chi" "$mode" "$output"
