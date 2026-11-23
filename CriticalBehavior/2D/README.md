# Unit-filled square-lattice Bose–Hubbard transition

This study uses $L\times L$ square tori, $L\ge 3$, at zero temperature and fixed particle number $N=L^2$. Both physical axes wrap. With hopping $t=1$, the Hamiltonian is
$$
H=-t\sum_{\langle ij\rangle}(b_i^\dagger b_j+b_j^\dagger b_i)
  +\frac U2\sum_i n_i(n_i-1).
$$
The interaction $U$ and inclusive cutoff `nmax` follow [the established 2D ED model](../../pythonTest/2D/exact_diagonalization.py). The $L=2$ bond convention in that model gives only two neighbors per site, so the finite-size series begins at $L=3$.

The [numerical results](RESULTS.md) compare the DMRG estimates through $\chi=1024$ with the published benchmarks and include the pilot and exact-diagonalization checks. The [charge-gap scaling method](GAP-SCALING.md) describes the fits. See the [sweep manifests](sweeps/) and [archived measurements and figures](results/) for the inputs and outputs.

## Published targets

The zero-temperature transition belongs to the three-dimensional XY universality class. A separate BKT transition occurs at finite temperature on the superfluid side. [Bose–Hubbard worm QMC](https://arxiv.org/abs/0710.2703)

| Quantity | Target | Source |
| --- | ---: | --- |
| $U_c/t$ | $16.7424(1)$ | [Chen et al., PRL 110, 170403 (2013)](https://doi.org/10.1103/PhysRevLett.110.170403) |
| $\nu$, correlation-length exponent | $0.671718(23)$ | [Hasenbusch, PRB 112, 184512 (2025)](https://doi.org/10.1103/1d4g-483z) |
| $\eta$, anomalous dimension | $0.03816(2)$ | [Hasenbusch, PRB 112, 184512 (2025)](https://doi.org/10.1103/1d4g-483z) |
| $z$, dynamical exponent | $1$ | [Emergent Lorentz invariance at fixed filling](https://doi.org/10.1103/PhysRevLett.110.170403) |
| $\omega$, leading correction-to-scaling exponent | $0.789(4)$ | [Hasenbusch, PRB 100, 224517 (2019)](https://doi.org/10.1103/PhysRevB.100.224517) |

The early Bose–Hubbard worm QMC result $t/U_c=0.05974(3)$ corresponds to $U_c/t\simeq16.739(8)$. Söyler et al. reported $16.7424(5)$ for the clean limit in 2011. The static exponents follow the three-dimensional XY universality class. For example, hyperscaling gives the order-parameter exponent $\beta=\nu(1+\eta)/2\simeq0.34868$. [Capogrosso-Sansone et al. (2008)](https://arxiv.org/abs/0710.2703) · [Söyler et al. (2011)](https://arxiv.org/abs/1107.5437) · [Periodic Bose–Hubbard QMC and XY universality](https://pmc.ncbi.nlm.nih.gov/articles/PMC5133547/)

## Finite-size observables

- Compute equal-time $n(\mathbf{k})=L^{-2}\sum_{ij} e^{i\mathbf{k}\cdot(\mathbf{r}_j-\mathbf{r}_i)}\langle b_i^\dagger b_j\rangle$, using both first nonzero momenta $(2\pi/L,0)$ and $(0,2\pi/L)$ and denoting their mean by $\bar n_k$. Their difference and the range of site densities diagnose symmetry breaking by the finite-bond-dimension MPS.
- Find pairwise crossings of $R=1-\bar n_k/n(0)$ using only the common sampled coupling interval. Near the transition $R=F((U-U_c)L^{1/\nu})$ to leading order, so $\partial_U R\propto L^{1/\nu}$ and a pairwise slope ratio yields an effective $\nu$. The analysis also checks crossings of the lattice second-moment ratio $\xi_{\rm 2nd}/L=\sqrt{R/(1-R)}/[2L\sin(\pi/L)]$. Drift across size pairs and sensitivity to interpolator and bond dimension set the accuracy of this method.
- At the published coupling $n(0)\sim L^{2-z-\eta}$. The code reports `eta_raw` and `eta_subtracted` assuming $z=1$, with the latter using $n(0)-\bar n_k$ to reduce a momentum-independent short-distance contribution. At this same coupling $\Delta_c=E(N+1)+E(N-1)-2E(N)\sim L^{-z}$. Report these as finite-size tests at the published coupling. The energies of all three sectors must be stable for a gap estimate.
- The charged-sector scan also checks crossings of $L\Delta_c$ with $z=1$ fixed. This gives a complementary coupling estimate and slope test with finite-size corrections of its own.
- Check the $3\times3$ torus against the existing ED Hamiltonian in all three particle sectors. Compare multiple bond dimensions and onsite cutoffs at fixed $L,U$. Small sweep-to-sweep energy changes and small discarded weights are diagnostics. Explicit changes in the measured observables with bond dimension, cutoff and starting state determine which points are resolved. Use the analysis PDF and individual estimates to check scale drift before interpreting any exponent as asymptotic.

For the stopping check, use a 1% relative change between the two largest bond dimensions as the target for the gap and one-body observables at the benchmark coupling. A crossing must also be stable against changing the observable and dropping the smallest size before it can support a thermodynamic estimate. Keep the largest completed torus in the report with its measured sensitivity if the test fails.

## Reproduce the analysis

The [measurement archive](results/measurements.tar.gz) contains the solver TOMLs and ED reference, including the partial pilot records needed to recover the split $L=6$ sectors. An archive replay reproduced all four JSON reports, the 65-row pilot CSV, and the three retained figures up to PDF creation timestamps.

Run the following from `CriticalBehavior/2D/` with Python 3.11 or later and NumPy, SciPy, and Matplotlib installed.

```bash
work_dir=$(mktemp -d)
tar -xzf results/measurements.tar.gz -C "$work_dir"

python3 analyze.py "$work_dir/pilot" "$work_dir/summary"
python3 gap_scaling.py "$work_dir/pilot" "$work_dir/gap-scaling-baseline"
python3 gap_scaling.py "$work_dir/staged" "$work_dir/gap-scaling-through-512" 512
python3 gap_scaling.py "$work_dir/staged" "$work_dir/gap-scaling-final"
```

## Rerun the solver

Use Julia 1.11 and the [pinned environment](../Project.toml) and [manifest](../Manifest.toml). The [sweep manifests](sweeps/) specify the model and solver settings. Each staged row solves one particle-number sector. To repeat the three-sector ED check from `CriticalBehavior/2D/`, run the following.

```bash
julia --project=.. -e 'using Pkg; Pkg.instantiate()'
work_dir=$(mktemp -d)
while IFS=$'\t' read -r name L U nmax offset seed stages; do
  julia --project=.. --threads=1 staged_dmrg.jl "$L" "$U" "$nmax" \
    "$offset" "$seed" "$stages" "$work_dir/$name"
done < sweeps/staged-smoke.tsv
python3 validate_ed.py "$work_dir/ed.json"
python3 analyze.py "$work_dir" "$work_dir/smoke"
```

The ED comparison appears under `ed_comparison` in `smoke.json`. Each solver process uses one Julia thread. On systems with a job scheduler, the independent manifest rows can run concurrently with distinct output prefixes.
