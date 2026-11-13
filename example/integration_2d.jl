# Run from the repository root: ./julia-local example/integration_2d.jl
using BoseHubbardNQS

# Tiny generated fixture, removed afterwards. CLI output guards require 2D/ paths.
root = normpath(joinpath(@__DIR__, "..", "pythonTest", "2D"))
mktempdir(root; prefix=".integration-example-") do directory
    path = joinpath(directory, "tiny.pt")
    fit = train_ed_2d(; Lx=3, Ly=1, Ntot=2, Nmax=2, pbc=false,
                       epochs=1, width=4, eval_every=1, device="cpu", output=path)
    model = load_nn_2d(path) # CPU default; device="auto" or "cuda:0" opts in
    configurations = [0 1; 1 0; 1 1] # sites x walkers; x fastest
    println(model.metadata)
    println(nn_value(model, configurations[:,1]))
    println(nn_values(model, configurations; chunk_size=1024))
    measured = evaluate_2d(; checkpoint=path, walkers=2, steps=2, burnin=0,
                            block_length=1, device="cpu", output=joinpath(directory,"observables.json"))
    println(measured["paths"])
end
