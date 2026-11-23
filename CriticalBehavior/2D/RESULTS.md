# 2D Bose–Hubbard numerical results

The model is at zero temperature with one boson per site on a square lattice periodic in both directions. The onsite term is $Un_i(n_i-1)/2$. The superfluid–Mott transition is in the 3D XY universality class.

## Critical coupling and exponents

The largest calculation uses a $6\times6$ torus and bond dimension $\chi=1024$. The numerical column reports finite-size estimates.

| Quantity | Our numerical result | Published benchmark |
| --- | --- | --- |
| Critical interaction $U_c/t$ | **$16.833$** from the $L=5,6$ charge-gap crossing with $z=1$ fixed | $16.7424(1)$ |
| Critical interaction $U_c/t$ with both exponents fixed | $16.858$ from a common fit over $L=4,5,6$ with $z=1$ and $\nu=0.671718$ fixed | $16.7424(1)$ |
| Correlation-length exponent $\nu$ | $\nu_{\rm eff}=0.721$ from the $L=5,6$ scaled-gap slopes with $z=1$ fixed | $0.671718(23)$ |
| Dynamical exponent $z$ | $z_{\rm eff}=1.052$ from the $L=5,6$ gap ratio at the published $U_c$ | $1$ |
| Anomalous dimension $\eta$ | $\eta_{\rm eff}=0.028$ along $x$, $0.007$ along $y$, and $-0.078$ along the diagonal, from $L=4,6$ midpoint correlations at the published $U_c$ with $z=1$ fixed | $0.03816(2)$ |

The critical interaction comes from Bose–Hubbard QMC. The values of $\nu$ and $\eta$ come from finite-size scaling of an improved 3D XY clock model, and $z=1$ follows from the emergent Lorentz invariance at the unit-filled transition. [Chen et al.](https://doi.org/10.1103/PhysRevLett.110.170403) · [Hasenbusch](https://doi.org/10.1103/1d4g-483z)

The largest-pair crossing is 0.54% above the published critical coupling. The same pair gives $z_{\rm eff}$ and $\nu_{\rm eff}$ about 5% and 7% above their benchmarks. The two axial correlators on $6\times6$ differ by about 0.9% even at $\chi=1024$. The spread across midpoint correlators leaves the universal $\eta$ unresolved. [Final scaling figure](results/gap-scaling-final.pdf) · [Fit details](results/gap-scaling-final.json)

### Charge-gap crossings

Following the published QMC procedure, we fit straight lines to $L\Delta_c/t$ versus $t/U$ near the transition. In the final scan, the fits use maximum bond dimensions $\chi=384,512,1024$ for $L=4,5,6$, respectively. [Capogrosso-Sansone et al., Fig. 2](https://arxiv.org/pdf/0710.2703#page=2) · [Analysis method](GAP-SCALING.md)

| Pair of torus sizes | Crossing $U^*/t$ | Slope-based $\nu_{\rm eff}$ |
| --- | ---: | ---: |
| $L=4,5$ | 16.893 | 0.766 |
| $L=4,6$ | 16.861 | 0.745 |
| $L=5,6$ | 16.833 | 0.721 |

Dropping the largest interaction from the common fit gives $U^*/t=16.856$. The coarse-grid baseline gave $16.841$ for the $L=4,5$ crossing and $16.889$ for a common fit over $L=3,4,5$. The narrower baseline window yielded no bracketed common-fit crossing. The drift across size pairs and the pilot observable dependence limit the precision of a thermodynamic coupling estimate. [Baseline figure](results/gap-scaling-baseline.pdf) · [Baseline analysis](results/gap-scaling-baseline.json)

## Bond dimension and onsite cutoff

The table shows the staged charge gaps at the published $U_c/t=16.7424$ with inclusive onsite cutoff $N_{\max}=4$.

| Torus | $\Delta_c/t$ at $\chi=384$ | $\chi=512$ | $\chi=768$ | $\chi=1024$ |
| --- | ---: | ---: | ---: | ---: |
| $5\times5$ | 2.63198 | 2.63319 | — | — |
| $6\times6$ | 2.15131 | 2.15961 | 2.16941 | 2.17368 |

On $5\times5$, the gap changed by 0.046% from $\chi=384$ to 512. On $6\times6$, it changed by 0.65% from $\chi=512$ to 1024, with 0.20% on the final step. Increasing the largest bond dimension moved the $L=5,6$ crossing from $16.899$ to $16.833$ and moved $z_{\rm eff}$ from $1.087$ to $1.052$. [Analysis through $\chi=512$](results/gap-scaling-through-512.json) · [Final analysis](results/gap-scaling-final.json)

At $\chi=256$, increasing $N_{\max}$ from 4 to 5 and changing the starting seed each altered the $5\times5$ gap by less than $2\times10^{-4}t$. On $6\times6$ at $\chi=512$, raising $N_{\max}$ from 4 to 5 changed the gap by 0.044% and $n(0)$ by 0.024%. These checks measure sensitivity at the published coupling and the stated finite bond dimensions.

## Exact-diagonalization check and pilot results

On the $3\times3$ torus at $U/t=16.7424$ and $N_{\max}=4$, DMRG at $\chi=128$ agrees with sparse ED within $1.9\times10^{-7}t$ in the neutral energy, $2.5\times10^{-6}t$ in the charge gap, and $1.0\times10^{-8}$ in the one-body momentum ratio. The ED neutral sector has 19,855 states on an 18-bond torus. [ED reference](results/ed.json)

The pilot used monotone cubic interpolation for $L=4,5$ at $N_{\max}=4$. Across observables, its highest-$\chi$ crossings range from $U^*/t=15.884$ to $16.931$. The momentum-ratio crossing at $16.734$ is particularly close to the benchmark, and its drift with $\chi$ and the spread across observables show the size of the numerical systematics. At the highest scanned $\chi$ for each observable, linear interpolation gives $16.734$ for $R$, $15.909$ for $\xi_{\rm 2nd}/L$, and $16.911$ for $L\Delta_c$. [Pilot figure](results/summary.pdf) · [Pilot fit details](results/summary.json)

| Pilot observable | $\chi$ | Crossing $U^*/t$ | $\nu_{\rm eff}$ |
| --- | ---: | ---: | ---: |
| Momentum ratio $R$ | 128 | 16.932 | 0.714 |
| Momentum ratio $R$ | 256 | 16.802 | 0.757 |
| Momentum ratio $R$ | 384 | 16.734 | 0.755 |
| Second-moment ratio $\xi_{\rm 2nd}/L$ | 128 | 16.152 | 0.791 |
| Second-moment ratio $\xi_{\rm 2nd}/L$ | 256 | 15.979 | 0.817 |
| Second-moment ratio $\xi_{\rm 2nd}/L$ | 384 | 15.884 | 0.825 |
| Scaled charge gap $L\Delta_c$ | 256 | 16.931 | 0.740 |

The $6\times6$ pilot gap changed from $2.07838t$ at $\chi=128$ to $2.14064t$ at $\chi=256$, a 3.00% increase. The $L=5,6$ gap ratio at $\chi=256$ gave $z_{\rm eff}=1.119$. On $6\times6$, $n(0)$ changed by 1.09% from $\chi=256$ to 384. These changes motivated the staged bond-dimension study above.

At the published coupling, the pilot $L=4,5$ estimate at $\chi=384$ gave $\eta_{\rm eff}=-0.085$ from $n(0)$ and $-0.084$ after subtracting the average occupation at the first nonzero momenta. For $L=5,6$, the estimate from $n(0)$ shifted from $-0.140$ to $-0.088$ as $\chi$ increased from 256 to 384. That shift alone exceeds the predicted $\eta$. The $6\times6$ pilot also gave a midpoint chemical potential of $\mu_{\rm mid}/t=6.22487$ at $\chi=256$, compared with the published $\mu_c/t=6.21(2)$. [Bose–Hubbard QMC](https://doi.org/10.1103/PhysRevLett.110.170403) · [Pilot measurements](results/summary.csv)

## Reproduce the results

The [measurement archive](results/measurements.tar.gz) contains the original sector energies, correlation measurements, and ED reference data. The [study README](README.md#reproduce-the-analysis) gives the replay commands for the pilot and the final analysis. The pilot CSV records the source of each charge-gap sector.
