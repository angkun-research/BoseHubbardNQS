# BKT transition of the unit-filled 1D Bose–Hubbard model

For the zero-temperature square lattice with periodic boundaries, see the [2D study](2D/README.md) and its [numerical results through $\chi=1024$](2D/RESULTS.md).

We study the zero-temperature chain at fixed $N=L$ with nearest-neighbor hopping $t=1$. This directory uses `U_phys` for the coefficient in the conventional onsite term $U_{\rm phys}n(n-1)/2$. The original 1D Python and Julia examples use `U_code = U_phys / 2`. The occupation cutoff `nmax` is inclusive.

See [RESULTS.md](RESULTS.md) for the numerical findings and verification.

## Folder layout

| Location | Contents |
|---|---|
| [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml) | Pinned Julia environment |
| [`src/`](src/) | Julia ITensor DMRG and Python sparse ED |
| [`analysis/`](analysis/) | Published-data reproduction, fits, and figure generation |
| [`sweeps/`](sweeps/) | DMRG and ED sweep manifests |
| [`figures/`](figures/) | Published-data and independent-study PDFs |
| [`results/`](results/) | Archived measurements, fit results, and CSV data |
| [`2D/`](2D/) | Periodic square-lattice DMRG, scaling analysis, and replay archive |

## Methods

- `analysis/reproduce.py` reads the published periodic DMRG block variances of Thamm et al. at `nmax=6`. It fits $K(L)$ from the slope of variance versus $\pi^{-2}\log[\sin(\pi\ell/L)]$ near half the ring, solves the RG flow equation for its invariant $\zeta$ between two sizes, and interpolates to $\zeta=1$. Varying the size pair, subsystem window and coupling window measures analysis sensitivity. The primary fit uses sizes 96 and 128 and a local linear interpolation over `U_phys = 3.270–3.300`. The paper uses a Bayesian interpolation for its final quoted uncertainty. [Paper](https://doi.org/10.1103/4xdz-5vq7) · [Data and analysis](https://github.com/DelMaestroGroup/papers-code-BKTFlowBoseHubbard/tree/009358b5c115ff2ec7bc6e89245ae7ddd2db7e78)
- `src/dmrg.jl` uses Julia ITensors and ITensorMPS with exact particle-number conservation. The periodic ring gives the RG and one-body correlation measurements. The open chain gives $\Delta_c(L)=E_0(N+1)+E_0(N-1)-2E_0(N)$. The manifests in [`sweeps/`](sweeps/) include `rings.tsv`, `chains.tsv`, `cutoffs.tsv`, and `cutoff-matched.tsv`. They specify length, `U_phys`, occupation cutoff, boundary, maximum bond dimension and measurement. Independent runs vary cutoff and bond dimension at selected couplings. [ITensor DMRG](https://docs.itensor.org/ITensorMPS/dev/DMRG.html) · [Bose–Hubbard charge-gap scaling](https://arxiv.org/abs/1212.2219)
- `src/ed.py` is a sparse periodic-ring reference. It computes energies, one-body correlations and block variances in the number basis. `analysis/analyze.py` compares ED with ITensor and fits the extracted observables. The correlation exponent at the transition is fitted with the leading BKT log correction removed, using $G_1(r)\propto r^{-\eta}[\log r+C]^{1/8}$. The correction follows by integrating the running correlation exponent $1/[2K(L)]$ along the critical flow. Gap fits at fixed coupling estimate the dynamical exponent $z$.

## Archived results

- [Published-data figure](figures/reproduction.pdf), [fit inputs and estimates](results/reproduction.json), and [RG data points](results/reproduction.csv) document the critical coupling and essential scaling analysis.
- [Independent figure](figures/independent.pdf) and [fit results](results/independent.json) record the Julia ITensor and ED measurements.
- The [measurement archive](results/measurements.tar.gz) contains 81 Julia ITensor TOML files and nine ED JSON files. Sector keys `0`, `-1`, and `1` are particle-number offsets from unit filling. Entries in `g1` and `block_variance` start at distance or block length one. Each ITensor output records the requested and achieved bond dimensions and sweep diagnostics.

## Reproduce the analysis

Run these commands from `CriticalBehavior/` with Python 3.11 or later and NumPy, SciPy, and Matplotlib installed. The published-data analysis uses the pinned dataset from Thamm et al. The independent analysis uses the archived DMRG and ED measurements.

```bash
work_dir=$(mktemp -d)
tar -xzf results/measurements.tar.gz -C "$work_dir"
curl -fsSL https://codeload.github.com/DelMaestroGroup/papers-code-BKTFlowBoseHubbard/zip/009358b5c115ff2ec7bc6e89245ae7ddd2db7e78 -o "$work_dir/published-source.zip"
printf '%s  %s\n' a77532354d152c544565a513ec1e8ddd49ab8840d1a0f95cd4c1ce89c1a16e6f \
  "$work_dir/published-source.zip" | sha256sum -c -
python3 analysis/reproduce.py "$work_dir/published-source.zip" "$work_dir/reproduction"
python3 analysis/analyze.py "$work_dir" "$work_dir/independent" \
  --published-archive "$work_dir/published-source.zip"
```

## Rerun the solver

Use Julia 1.11 with the pinned environment. The DMRG manifests list $L$, $U_{\rm phys}/t$, inclusive `nmax`, boundary, maximum bond dimension, and observable mode. The [ED manifest](sweeps/ed.tsv) lists the chain length, occupation cutoff, and coupling grid.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
work_dir=$(mktemp -d)
julia --project=. --threads=1 src/dmrg.jl 8 3.275 6 periodic 128 correlations \
  "$work_dir/itensor-smoke.toml"
python3 src/ed.py 8 6 "$work_dir/ed-L8-n6.json" 3.275
```
