# Literature follow-up on the zero-temperature square torus

Capogrosso-Sansone et al. measured particle and hole gaps from the imaginary-time single-particle Green function using worm QMC. Their Fig. 2 plots the total scaled gap $L\Delta_c/t$ against $t/U$ for $L=10,20,40,80$, using straight-line crossings to locate $(t/U)_c=0.05974(3)$. The critical scaling uses $z=1$ and $\nu=0.6715$ as universality inputs. The later Bose–Hubbard QMC benchmark is $U_c/t=16.7424(1)$. [Capogrosso-Sansone et al., 2008](https://arxiv.org/abs/0710.2703) · [Chen et al., 2013](https://doi.org/10.1103/PhysRevLett.110.170403)

At unit filling $N=L^2$, the neutral ground-state energy $E_0(N)$ is the reference for the particle-addition threshold $\mu_+=E_0(N+1)-E_0(N)$ and the particle-removal threshold $\mu_-=E_0(N)-E_0(N-1)$. The charge gap is the width between these thresholds, $\Delta_c=\mu_+-\mu_-=E_0(N+1)+E_0(N-1)-2E_0(N)$. This second difference cancels any chemical-potential term proportional to particle number. The primary analysis uses fully periodic square tori.

## Measurements

- Scan $U/t=16.60,16.7424,16.90,17.10$ on $L=4,5,6$. The point at $17.80$ extends the Mott-side plot for $L=4,5$ and for the $L=6$ curve through $\chi=512$. Keep a single bond-dimension and sweep setting across couplings within each size when fitting slopes. Compare pairwise straight-line crossings with a common fit at $z=1,\nu=0.671718$ and vary the interaction window. The latter reproduces the scaling assumption of the paper.
- Compare the $5\times5$ gap and correlations at $\chi=384,512$, and compare $6\times6$ at $\chi=384,512,768,1024$. At the published coupling, also compare $N_{\max}=4,5$ for $6\times6$ at $\chi=512$. Sensitivity to the last sweeps and to $\chi$ is assessed alongside the size dependence.
- Extract effective $z$ from $\Delta_c$ at the published coupling. With $z=1$, estimate effective $\nu$ from the slopes of the scaled-gap curves. For $\eta$, check axial and diagonal equal-time correlators at midpoint separation on the even tori. A pair of even sizes gives an effective exponent and provides a check on the momentum-occupation analysis in the [first study](RESULTS.md).
- The primary fits use square tori because the finite-size scaling function depends on aspect ratio.

## Solver and analysis

The [staged solver](staged_dmrg.jl) runs one particle-number sector per process using the same model as the first study. It replaces its restart checkpoint after every sweep and saves a TOML measurement at each completed stage from $\chi=384$ onward or at the final stage in the manifest. Restart a stopped run with the same manifest row and output prefix. The initial state uses the original snake ordering with a central particle or hole seed.

The manifests [staged-square-scan.tsv](sweeps/staged-square-scan.tsv), [staged-L6-benchmark.tsv](sweeps/staged-L6-benchmark.tsv), and [staged-L6-scan.tsv](sweeps/staged-L6-scan.tsv) control the runs. Rows specify the output prefix, $L$, $U/t$, $N_{\max}$, particle-number offset, seed, and comma-separated $\chi$:sweep-count stages. The [analysis](gap_scaling.py) joins the three sectors with matching run settings and generates the crossing and scaling figure and machine-readable results. It reports the effective slopes separately from the fixed-exponent fit.

## Existing-data baseline

Applying the Fig. 2 straight-line analysis to the existing $\chi=256$ coarse grid $U/t=16.0,16.7424,17.5$ gives a $4\times4$–$5\times5$ crossing at $U^*/t=16.841$ and a slope-based $\nu_{\rm eff}=0.768$. A common linear fit across $L=3,4,5$ with $z=1$ and fixed $\nu$ gives $U^*/t=16.889$. The narrower window does not yield a bracketed common-fit crossing. This baseline motivates the denser sampling around the tip. [Baseline figure](results/gap-scaling-baseline.pdf) · [Baseline analysis](results/gap-scaling-baseline.json)

## Completed follow-up

All 54 production array tasks and both analysis jobs completed with exit code 0 on November 22–23, 2026. The $\chi=1024$ jobs took 3 hours 31 minutes to 3 hours 43 minutes. The final analysis completed at 04:32 EST on November 23. [Final figure](results/gap-scaling-final.pdf) · [Final analysis](results/gap-scaling-final.json) · [Analysis through $\chi=512$](results/gap-scaling-through-512.json)

At the published $U_c/t=16.7424$, the $5\times5$ charge gap changed by 0.046% from $\chi=384$ to 512. The $6\times6$ gap rose from $2.15961t$ at $\chi=512$ to $2.16941t$ at 768 and $2.17368t$ at 1024, a 0.65% change from 512 to 1024 and a 0.20% change on the final step. At $\chi=512$ the $6\times6$ cutoff check from $N_{\max}=4$ to 5 changed the gap by 0.044% and $n(0)$ by 0.024%. These cutoff comparisons are at the tested finite bond dimension.

The $5\times5$–$6\times6$ gap crossing moved from $U^*/t=16.899$ through $\chi=512$ to $16.833$ with the $6\times6$ at $\chi=1024$. The final three-size fit with $z=1$ and $\nu=0.671718$ gives $U^*/t=16.858$, or $16.856$ in the narrower window. The $5,6$ slope ratio gives $\nu_{\rm eff}=0.721$. At the published coupling their gap ratio gives $z_{\rm eff}=1.052$, compared with $1.087$ through $\chi=512$. The even-size midpoint correlators give $\eta_{\rm eff}=0.028$ and $0.007$ along the axes and $-0.078$ on the diagonal. The two axial correlators on $6\times6$ still differ by about 0.9% at $\chi=1024$. The spread between correlators leaves the universal $\eta$ unresolved.
