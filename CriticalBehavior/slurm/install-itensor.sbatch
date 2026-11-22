#!/bin/bash -l
set -euo pipefail
module load julia/1.11.3
export JULIA_DEPOT_PATH="$PWD/julia-generic:$PWD/julia-depot:$HOME/.julia:"
export JULIA_CPU_TARGET=generic JULIA_NUM_PRECOMPILE_TASKS=2 JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec julia --startup-file=no --project="$1" -e '
    using Pkg
    if !isfile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
        Pkg.Registry.add("General")
        Pkg.Registry.update()
    end
    Pkg.instantiate(update_registry=false)
    Pkg.precompile()
    using ITensors, ITensorMPS
    println("ITensors ", pkgversion(ITensors), " ITensorMPS ", pkgversion(ITensorMPS))'
