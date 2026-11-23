# Periodic square-lattice pilot results

The [charge-gap follow-up](GAP-SCALING.md#completed-follow-up) extends this pilot through $\chi=1024$.

The study treats the zero-temperature superfluid–Mott transition at exactly one boson per site on fully periodic square tori. The published targets are $U_c/t=16.7424(1)$, $\nu=0.671718(23)$, $\eta=0.03816(2)$, and $z=1$. The critical point has emergent Lorentz invariance and belongs to the three-dimensional XY universality class. [Bose–Hubbard QMC](https://doi.org/10.1103/PhysRevLett.110.170403) · [3D XY scaling](https://doi.org/10.1103/1d4g-483z)

## Exact-diagonalization check

At $L=3$, $U/t=16.7424$, and `nmax=4`, the established periodic ED model has 18 bonds and neutral-sector dimension 19,855. The $\chi=128$ DMRG absolute errors relative to ED are $1.9\times10^{-7}t$ in the neutral energy, $2.5\times10^{-6}t$ in the charge gap, and $1.0\times10^{-8}$ in the one-body momentum ratio. The [ED reference data](results/ed.json) record all three sector energies and both first nonzero momenta.

## Finite-size crossings and correlation-length exponent

These crossings use the $L=4,5$ pair, `nmax=4` and monotone cubic interpolation inside the sampled interval. The effective $\nu$ comes from the ratio of local slopes. The scaled-gap analysis fixes $z=1$.

| Observable | Requested $\chi$ | Crossing $U^*/t$ | $\nu_{\rm eff}$ |
| --- | ---: | ---: | ---: |
| Momentum ratio $R$ | 128 | 16.932 | 0.714 |
| Momentum ratio $R$ | 256 | 16.802 | 0.757 |
| Momentum ratio $R$ | 384 | 16.734 | 0.755 |
| Second-moment ratio $\xi_{\rm 2nd}/L$ | 128 | 16.152 | 0.791 |
| Second-moment ratio $\xi_{\rm 2nd}/L$ | 256 | 15.979 | 0.817 |
| Second-moment ratio $\xi_{\rm 2nd}/L$ | 384 | 15.884 | 0.825 |
| Scaled charge gap $L\Delta_c$ | 256 | 16.931 | 0.740 |

The $\chi=384$ momentum-ratio crossing lies within $0.01t$ of the published critical coupling. Raising $\chi$ from 256 to 384 moved that crossing by $0.068t$, and the three observables span $1.05t$ at their highest scanned $\chi$. The smaller $L=3,4$ pair at $\chi=256$ puts the momentum-ratio crossing at $17.243t$ and the scaled-gap crossing at $17.068t$. This drift limits the coupling result to a finite-size comparison. Linear interpolation gives $U^*/t=16.734,\ 15.909,\ 16.911$ for the three highest-$\chi$ entries, respectively. The corresponding $\nu_{\rm eff}$ values are $0.778,\ 0.829,\ 0.713$, compared with the published $0.672$.

## Scaling at the published coupling

These values use $U/t=16.7424$, onsite cutoff `nmax=4`, and seed 123. The momentum occupation $n(0)$ is defined with a factor $L^{-2}$ in the Fourier sum.

| $L$ | Requested $\chi$ | $n(0)$ | $R$ | $\Delta_c/t$ |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 256 | 4.01288 | 0.80805 | 4.55178 |
| 4 | 384 | 5.54750 | 0.81377 | 3.33554 |
| 5 | 256 | 7.07786 | 0.81468 | 2.62534 |
| 5 | 384 | 7.06684 | 0.81365 | 2.63157 |
| 6 | 128 | 8.78012 | 0.82237 | 2.07838 |
| 6 | 256 | 8.71239 | 0.81803 | 2.14064 |
| 6 | 384 | 8.61738 | 0.81485 | — |

At $\chi=256$, the charge-gap exponents for size pairs $(3,4)$, $(4,5)$ and $(5,6)$ are $z_{\rm eff}=1.081,\ 1.072,\ 1.119$. The same-parity pairs give $1.077$ for $(3,5)$ and $1.093$ for $(4,6)$. The $\chi=384$ estimate from $(4,5)$ is approximately $1.06$. The gaps show approximately inverse-linear scaling over these sizes, with visible deviations from $z=1$.

At $\chi=384$, the $(4,5)$ pair gives $\eta_{\rm eff}=-0.085$ from $n(0)$, or $-0.084$ using $n(0)-\bar n_k$, with $z=1$ fixed. The $(5,6)$ raw estimate moves from $-0.140$ at $\chi=256$ to $-0.088$ at $\chi=384$. This shift is larger than the published anomalous dimension. The data resolve growth close to $n(0)\propto L$ and leave the universal value of $\eta$ unresolved. The $6\times6$ midpoint chemical potential at $\chi=256$ is $\mu_{\rm mid}/t=6.22487$ compared with the published $\mu_c/t=6.21(2)$. [Bose–Hubbard QMC](https://doi.org/10.1103/PhysRevLett.110.170403)

## Bond-dimension limit

At $L=5$, raising $\chi$ from 256 to 384 changes $n(0)$ by 0.16% and the gap by 0.24%. Raising the onsite cutoff from 4 to 5 and changing the seed at $\chi=256$ each change the gap by less than $2\times10^{-4}t$. At $L=6$, $n(0)$ changes by 1.09% from $\chi=256$ to 384 and the charge gap changes by 3.00% from $\chi=128$ to 256. The $6\times6$ torus exceeds the 1% convergence target and sets the stopping point. Its largest available neutral result uses $\chi=384$, and its largest verified three-sector gap uses $\chi=256$.

## Reproducibility

The solver was checked against `pythonTest/2D`. The [figure](results/summary.pdf), [65 measurement rows](results/summary.csv), and [crossings, exponents, and convergence comparisons](results/summary.json) document the exact-diagonalization and DMRG runs. The CSV records the TOML source of each sector. The [replay archive](results/measurements.tar.gz) contains the sector measurements and ED reference.
