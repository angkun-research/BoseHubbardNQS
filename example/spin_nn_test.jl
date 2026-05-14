using BoseHubbardNQS

# Example usage
model = load_nn("data/spin_net.pt")

sigma = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
val = nn_value(model, sigma)

println("spin config = ", sigma)
println("NN scalar output = ", val)

# Sketch of how you would use this in a QMC loop:
function ansatz_amplitude(model, sigma::Vector{Int})
    return nn_value(model, sigma)
end

for _ in 1:3
    trial_sigma = rand(0:1, 10)
    println("trial = ", trial_sigma, "  ansatz = ", ansatz_amplitude(model, trial_sigma))
end