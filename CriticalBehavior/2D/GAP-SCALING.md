# Charge-gap scaling method

The [numerical results](RESULTS.md) compare the pilot and staged estimates with published benchmarks.

Capogrosso-Sansone et al. measured particle and hole gaps from the imaginary-time single-particle Green function using worm QMC. Their Fig. 2 plots the total scaled gap $L\Delta_c/t$ against $t/U$ for $L=10,20,40,80$, using straight-line crossings to locate $(t/U)_c=0.05974(3)$. The critical scaling uses $z=1$ and $\nu=0.6715$ as universality inputs. The later Bose–Hubbard QMC benchmark is $U_c/t=16.7424(1)$. [Capogrosso-Sansone et al., 2008](https://arxiv.org/abs/0710.2703) · [Chen et al., 2013](https://doi.org/10.1103/PhysRevLett.110.170403)

At unit filling $N=L^2$, the neutral ground-state energy $E_0(N)$ is the reference for the particle-addition threshold $\mu_+=E_0(N+1)-E_0(N)$ and the particle-removal threshold $\mu_-=E_0(N)-E_0(N-1)$. The charge gap is the width between these thresholds, $\Delta_c=\mu_+-\mu_-=E_0(N+1)+E_0(N-1)-2E_0(N)$. This second difference cancels any chemical-potential term proportional to particle number. The primary analysis uses fully periodic square tori.

## Measurements and fits

- Scan $U/t=16.60,16.7424,16.90,17.10$ on $L=4,5,6$. The point at $17.80$ extends the Mott-side plot for $L=4,5$ and for the $L=6$ curve through $\chi=512$. Keep a single bond-dimension and sweep setting across couplings within each size when fitting slopes. Compare pairwise straight-line crossings with a common fit at $z=1,\nu=0.671718$ and vary the interaction window. The latter reproduces the scaling assumption of the paper.
- Compare the $5\times5$ gap and correlations at $\chi=384,512$, and compare $6\times6$ at $\chi=384,512,768,1024$. At the published coupling, also compare $N_{\max}=4,5$ for $6\times6$ at $\chi=512$. Sensitivity to the last sweeps and to $\chi$ is assessed alongside the size dependence.
- Extract effective $z$ from $\Delta_c$ at the published coupling. With $z=1$, estimate effective $\nu$ from the slopes of the scaled-gap curves. For $\eta$, check axial and diagonal equal-time correlators at midpoint separation on the even tori. A pair of even sizes gives an effective exponent and provides a check on the [pilot momentum-occupation analysis](RESULTS.md#exact-diagonalization-check-and-pilot-results).
- The primary fits use square tori because the finite-size scaling function depends on aspect ratio.

## Solver and analysis

The [staged solver](staged_dmrg.jl) runs one particle-number sector per process using the same Hamiltonian as the pilot. It replaces its restart checkpoint after every sweep and saves a TOML measurement at each completed stage from $\chi=384$ onward or at the final stage in the manifest. Restart a stopped run with the same manifest row and output prefix. The initial state uses the original snake ordering with a central particle or hole seed.

The manifests [staged-square-scan.tsv](sweeps/staged-square-scan.tsv), [staged-L6-benchmark.tsv](sweeps/staged-L6-benchmark.tsv), and [staged-L6-scan.tsv](sweeps/staged-L6-scan.tsv) control the runs. Rows specify the output prefix, $L$, $U/t$, $N_{\max}$, particle-number offset, seed, and comma-separated $\chi$:sweep-count stages. The [analysis](gap_scaling.py) joins the three sectors with matching run settings and generates the crossing and scaling figure and machine-readable results. It reports the effective slopes separately from the fixed-exponent fit.

<a id="completed-follow-up"></a>

## Results

See [critical coupling and exponents](RESULTS.md#critical-coupling-and-exponents). The [replay instructions](README.md#reproduce-the-analysis) regenerate the figures and fit data.
