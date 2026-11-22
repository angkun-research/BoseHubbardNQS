# 1D Bose–Hubbard numerical results

The chain is at unit filling and zero temperature. Here $t=1$, $U_{\rm phys}$ multiplies $n_i(n_i-1)/2$, and $U_{\rm code}=U_{\rm phys}/2$ in the original 1D Python implementation.

## Critical coupling reproduced from published DMRG data

Fitting the BKT RG invariant $\zeta$ across periodic sizes $L=96,128$ gives a crossing of $\zeta=1$ at $U_{\rm phys,c}/t=3.27602$, or **$U_{\rm code,c}/t=1.63801$**. Varying the fit interval in block length and interaction and using size pairs drawn from $L=64,96,128$ gives $U_{\rm phys,c}/t=3.27399$ to $3.27617$. This is a sensitivity range across analysis choices. The published estimate is $3.275(2)$, or $U_{\rm code,c}/t=1.6375(10)$. The published data used `Nmax=6`. [Thamm et al.](https://doi.org/10.1103/4xdz-5vq7) · [Reproduction figure](figures/reproduction.pdf) · [Fit details](results/reproduction.json)

A fit of $K(L)=K_\infty+1/(\ln L+C)$ at $U_{\rm phys}/t=3.275$ gives $K_\infty=1.9972$ from $L\ge48$. The critical prediction is $K_c=2$. For the essential singularity, continuing the measured Mott-side RG flows across $1.75\le K\le2.25$ gives a logarithmic scale ratio proportional to $(U-U_c)^{-\sigma}$ plus a constant. The fitted effective power is $\sigma=0.52$ at the primary crossing, and varies from $0.50$ to $0.75$ across the tested crossing values. This is a test of the BKT continuation of the measured flows. Close to $K=2$, the RG denominator has the form $(1-\zeta)+(K-2)^2/8$, yielding the asymptotic $\sigma=1/2$ when $1-\zeta$ is linear in $U-U_c$. An ordinary correlation-length power-law exponent $\nu$ does not describe this transition. [Thamm et al.](https://doi.org/10.1103/4xdz-5vq7)

## Independent Julia ITensor and ED calculations

The Julia ITensor ground state on an eight-site ring at $U_{\rm phys}/t=3.275$ agrees with sparse ED to $1.1\times10^{-10}t$ in ground-state energy, $9.3\times10^{-10}$ in the largest pointwise one-body correlation difference, and $7.3\times10^{-11}$ in block number variance. The ED eigenpair residual was $4.1\times10^{-13}t$.

Across the three block-fit windows, the independent ITensor $L=24,32$ crossing estimates give $U_{\rm phys,c}/t=3.263$–$3.275$. The $L=32,48$ pair gives $3.260$–$3.268$ with `Nmax=6` and $3.260$–$3.267$ with `Nmax=4`. The $L=48,64$ pair has one bracketed estimate, $3.262$. At the benchmark coupling, the $L=64$ ITensor estimate of $K$ is about $0.0073$ below the published value for the same size. This finite-bond-dimension sensitivity limits the precision of the independent crossings. Use the published-data result above for the thermodynamic critical coupling. The original `U_code=1.6789` is on the Mott side.

At $L=32$ and matched bond cap $\chi=384$, changing `Nmax` from 4 to 6 changes $K$ by $8.2\times10^{-4}$ and $G_1(L/2)$ by $1.5\times10^{-4}$. The `Nmax=5,6,8` values at that size agree within $3.4\times10^{-6}$ in $K$ and $4.0\times10^{-7}$ in $G_1(L/2)$. At $L=48$, `Nmax=6` and 8 agree to about $8\times10^{-5}$ in $K$ at the same bond cap. These checks support `Nmax=5` for routine near-critical 1D runs. The largest rings and the published critical-coupling fit used `Nmax=6`.

| Observable | Numerical result | BKT prediction |
|---|---|---|
| Limiting Luttinger parameter | $K_\infty=1.9972$ from published rings, $L\ge48$ | $K_c=2$ |
| One-body correlation exponent | $\eta=0.250$–$0.252$ for Julia ITensor rings, $L=16$–64, after the leading logarithmic correction | $\eta=1/4$ |
| Dynamical exponent | $z_{\rm eff}=1.0011$ from open ITensor chains, $L=32,64,128,256$ | $z=1$ |
| Essential exponent in $\ln\xi \sim (U-U_c)^{-\sigma}$ | $\sigma_{\rm eff}=0.52$ from continuing the published RG flow at the primary crossing | $\sigma=1/2$ |

For comparison, a plain power-law fit of $G_1$ on the same ITensor rings gives $\eta=0.231$–$0.233$. Fixing $\eta=1/4$ and fitting the power of the logarithm gives $0.112$–$0.126$, close to $1/8$. The available ring sizes constrain these fits with either the power-law or logarithmic exponent fixed. Periodic ED through $L=12$ gives $z_{\rm eff}=0.983$. The Mott-side charge gaps at $L\le256$ are too size-dependent near the transition to fit the essential exponent directly. [Independent plot](figures/independent.pdf) · [Independent fit details](results/independent.json)

## Slurm verification

The final analysis job `6635752` completed with exit code `0:0`. It analyzed 81 Julia ITensor outputs and 54 ED size/coupling/cutoff points. The analysis job was submitted before the cutoff-matched array `6635856`, and that array was added to its dependency list while it was pending. Slurm accounting on November 20, 2026 shows the last cutoff-matched task ended at 18:56:27 and the final analysis started at 20:48:09.

| Work | Slurm job ID |
|---|---|
| Published-data reproduction | `6635852` |
| Sparse ED sweep | `6635630` |
| Eight-site ITensor comparison | `6635623` |
| 32-site warm-start check | `6635837` |
| Periodic ring sweep, `rings.tsv` | `6635637` |
| Open-chain sweep, `chains.tsv` | `6635638` |
| Cutoff sweep, `cutoffs.tsv` | `6635701` |
| Matched-bond-dimension cutoff sweep, `cutoff-matched.tsv` | `6635856` |
| Final independent analysis | `6635752` |

Open-chain task `6635638_3` reached its wall-time limit at $L=256$ and $\chi=192$. The corresponding $\chi=288$ task `6635638_26` completed and supplies the $L=256$ critical gap used here. The job logs and original outputs are under `$SCRATCHDIR/CriticalBehavior`. The local `raw/` directory includes the ITensor/ED comparison and the completed sweep outputs.
