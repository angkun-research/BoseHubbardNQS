#!/bin/bash -l
set -euo pipefail
module load julia/1.11.3
export JULIA_DEPOT_PATH="$PWD/julia-generic:$PWD/julia-depot:$HOME/.julia:"
export JULIA_CPU_TARGET=generic OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
study_dir=$1
shift
exec julia --startup-file=no --threads=1 --compiled-modules=existing --pkgimages=existing \
    --project="$study_dir" "$study_dir/src/dmrg.jl" "$@"
