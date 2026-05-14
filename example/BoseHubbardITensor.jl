using ITensors, ITensorMPS

L = 8
Nmax = 8
t = 1.0
U = 2.0
mu = 0.0

sites = siteinds("Boson", L; dim=Nmax+1, conserve_qns=true)

ampo = OpSum()
for j in 1:L-1
    ampo += -t, "Adag", j, "A", j+1
    ampo += -t, "Adag", j+1, "A", j
end
# add periodic boundary condition
ampo += -t, "Adag", L, "A", 1
ampo += -t, "Adag", 1, "A", L
for j in 1:L
    ampo += (U), "N * N", j   # gives U * n(n-1)
    ampo += -(U), "N", j      # gives U * n(n-1)
    ampo += -mu, "N", j
end

H = MPO(ampo, sites)

Ntot = L  # choose total boson number sector
state0 = [j <= Ntot ? "1" : "0" for j in 1:L]  # labels for Boson sites
psi0 = random_mps(sites, state0; linkdims=20)   # or randomMPS(sites, state0; ...)
energy, psi = dmrg(H, psi0; nsweeps=20, maxdim=[50,100,200,400], cutoff=[1e-10])
println(energy)