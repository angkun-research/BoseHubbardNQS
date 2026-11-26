#!/usr/bin/env bash
set -euo pipefail

repo=${1:?absolute repository path required}
manifest=${2:?absolute manifest path required}
depot_root=${3:?absolute path to the ITensor depot parent required}
case_line=$(sed -n "${SLURM_ARRAY_TASK_ID:?array job required}p" "$manifest")
read -r lattice coupling cutoff bond_cap extra <<< "$case_line"
if [[ ! $lattice =~ ^[0-9]+$ || ! $coupling =~ ^[0-9]+([.][0-9]+)?$ ||
      ! $cutoff =~ ^[0-9]+$ || ! $bond_cap =~ ^[0-9]+$ || -n ${extra:-} ]]; then
    echo "Invalid or missing manifest row: $case_line" >&2
    exit 2
fi

output="correlations-periodic-L${lattice}-U${coupling}-n${cutoff}-chi${bond_cap}.toml"
if [[ -n ${BHNQS_DMRG_REFINE:-} ]]; then
    refine_dir="dmrg-refine-$SLURM_ARRAY_JOB_ID"
    mkdir -p -- "$refine_dir"
    output="$refine_dir/$output"
fi
if [[ -s $output ]]; then
    echo "Output already present: $output" >&2
    exit 3
fi

source /etc/profile.d/modules.sh
module load julia/1.11.3
export JULIA_DEPOT_PATH="$depot_root/julia-generic:$depot_root/julia-depot:$HOME/.julia:"
export JULIA_CPU_TARGET=generic OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --startup-file=no --project="$repo/CriticalBehavior" --threads=1 \
    --compiled-modules=existing --pkgimages=existing \
    "$repo/CriticalBehavior/src/dmrg.jl" "$lattice" "$coupling" "$cutoff" \
    periodic "$bond_cap" correlations "$output"
