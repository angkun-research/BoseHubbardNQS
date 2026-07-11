using BoseHubbardNQS
using Random
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const MODEL_PATH = joinpath(REPO_ROOT, "data", "bose_hubbard_nqs.pt")

const L = 8
const Nmax = 8
const HIDDEN_DIM = 128
const N_STEPS = 100_000_000
const REPORT_EVERY = 1_000_000

model = load_nn(
    MODEL_PATH;
    arch = "FCNet",
    n_sites = L,
    nmax = Nmax,
    hidden_dim = HIDDEN_DIM,
    map_location = "cpu",
)

function rss_mb()
    pid = getpid()
    out = read(`ps -o rss= -p $pid`, String)
    return parse(Int, strip(out)) / 1024.0
end

function random_config(L, Nmax)
    return rand(0:Nmax, L)
end

println("Starting stress test")
println("L = $L, Nmax = $Nmax, steps = $N_STEPS")
println("Initial RSS = $(round(rss_mb(), digits=2)) MB")

amplitudes = Vector{Float64}(undef, N_STEPS)

t0 = time()
for step in 1:N_STEPS
    sigma = random_config(L, Nmax)
    amplitudes[step] = ansatz_log_amplitude(model, sigma)

    if step % REPORT_EVERY == 0
        #GC.gc()
        elapsed = time() - t0
        println(
            "step = $step",
            "  last_amp = $(amplitudes[step])",
            "  RSS = $(round(rss_mb(), digits=2)) MB",
            "  elapsed = $(round(elapsed, digits=2)) s",
            "  avg_step = $(round(elapsed / step * 1e3, digits=3)) ms"
        )
    end
end

println("Done")
println("Final RSS = $(round(rss_mb(), digits=2)) MB")
println("Mean amplitude = $(mean(amplitudes))")