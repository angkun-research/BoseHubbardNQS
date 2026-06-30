using BoseHubbardNQS

# Example usage
MODEL_PATH = "data/bose_hubbard_nqs.pt"

L = 8
Nmax = 8
hidden_dim = 128
model = load_nn(MODEL_PATH;
                 arch="FCNet", n_sites=L, nmax=Nmax, hidden_dim=hidden_dim,
                 map_location="cpu")

# Example boson occupation configuration (length = n_sites)
sigma = [1 for _ in 1:L] #[1, 1, 0, 2, 1, 1, 0, 2]
val = nn_value(model, sigma)

println("occupation config = ", sigma)
println("NN scalar output = ", val)


for _ in 1:3
    trial_sigma = rand(0:2, length(sigma))
    println("trial = ", trial_sigma, "  ansatz log amp = ", ansatz_log_amplitude(model, trial_sigma))
end



# Normalization of the ansatz amplitudes on the one sector basis
using PythonCall

# Stable logsumexp for a Julia vector
function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

Ntot = 8

ed = pyimport("pythonTest.BoseHubbardED")
basis_py = ed.generate_basis(L, Ntot, Nmax)
basis = [pyconvert(Vector{Int}, s) for s in basis_py]

# Evaluate raw log amplitudes on full sector basis
logpsi_raw = [nn_value(model, s) for s in basis]

# Same normalization constant as in TrainEDstates.py: logZ = logsumexp(2*logpsi)
logZ = logsumexp(2.0 .* logpsi_raw)

# Evaluate any test config with normalized quantities
sigma = [1 for _ in 1:L]
raw = nn_value(model, sigma)
logamp_norm = raw - 0.5 * logZ
amp_norm = exp(logamp_norm)
prob = exp(2.0 * logamp_norm)

println("sigma = ", sigma)
println("raw log amp = ", raw)
println("normalized log amp = ", logamp_norm)
println("normalized amp = ", amp_norm)
println("probability = ", prob)