# BKT transition of the unit-filled 1D Bose–Hubbard model

We study the zero-temperature chain at fixed $N=L$ with nearest-neighbor hopping $t=1$. This directory uses `U_phys` for the coefficient in the conventional onsite term $U_{\rm phys}n(n-1)/2$. The original 1D Python and Julia examples use `U_code = U_phys / 2`. The occupation cutoff `nmax` is inclusive.

See [RESULTS.md](RESULTS.md) for the numerical findings and verified Slurm status.

## Folder layout

| Location | Contents |
|---|---|
| [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml) | Pinned Julia environment |
| [`src/`](src/) | Julia ITensor DMRG and Python sparse ED |
| [`analysis/`](analysis/) | Published-data reproduction, fits, and figure generation |
| [`slurm/`](slurm/) | ISAAC-NG job scripts and [sweep manifests](slurm/sweeps/) |
| [`figures/`](figures/) | Published-data and independent-study PDFs |
| [`results/`](results/) | Processed JSON and CSV used to document the findings |
| `raw/` | Local solver outputs and the downloaded literature archive, ignored by Git |

## Methods

- `analysis/reproduce.py` reads the published periodic DMRG block variances of Thamm et al. at `nmax=6`. It fits $K(L)$ from the slope of variance versus $\pi^{-2}\log[\sin(\pi\ell/L)]$ near half the ring, solves the RG flow equation for its invariant $\zeta$ between two sizes, and interpolates to $\zeta=1$. Varying the size pair, subsystem window and coupling window measures analysis sensitivity. The primary fit uses sizes 96 and 128 and a local linear interpolation over `U_phys = 3.270–3.300`. The paper uses a Bayesian interpolation for its final quoted uncertainty. [Paper](https://doi.org/10.1103/4xdz-5vq7) · [Data and analysis](https://github.com/DelMaestroGroup/papers-code-BKTFlowBoseHubbard/tree/009358b5c115ff2ec7bc6e89245ae7ddd2db7e78)
- `src/dmrg.jl` uses Julia ITensors and ITensorMPS with exact particle-number conservation. The periodic ring gives the RG and one-body correlation measurements. The open chain gives $\Delta_c(L)=E_0(N+1)+E_0(N-1)-2E_0(N)$. The manifests in [`slurm/sweeps/`](slurm/sweeps/) include `rings.tsv`, `chains.tsv`, `cutoffs.tsv`, and `cutoff-matched.tsv`. They specify length, `U_phys`, occupation cutoff, boundary, maximum bond dimension and measurement. Independent runs vary cutoff and bond dimension at selected couplings. [ITensor DMRG](https://docs.itensor.org/ITensorMPS/dev/DMRG.html) · [Bose–Hubbard charge-gap scaling](https://arxiv.org/abs/1212.2219)
- `src/ed.py` is a sparse periodic-ring reference. It computes energies, one-body correlations and block variances in the number basis. `analysis/analyze.py` compares ED with ITensor and fits the extracted observables. The correlation exponent at the transition is fitted with the leading BKT log correction removed, using $G_1(r)\propto r^{-\eta}[\log r+C]^{1/8}$. The correction follows by integrating the running correlation exponent $1/[2K(L)]$ along the critical flow. Gap fits at fixed coupling estimate the dynamical exponent $z$.

## Archived results

- [Published-data figure](figures/reproduction.pdf), [fit inputs and estimates](results/reproduction.json), and [RG data points](results/reproduction.csv) document the critical coupling and essential scaling analysis.
- [Independent figure](figures/independent.pdf) and [fit results](results/independent.json) record the Julia ITensor and ED measurements.
- The local `raw/` directory contains 81 Julia ITensor TOML files and nine ED JSON files. Sector keys `0`, `-1`, and `1` are particle-number offsets from unit filling. Entries in `g1` and `block_variance` start at distance or block length one. Each ITensor output records the requested and achieved bond dimensions and sweep diagnostics.

## Running on ISAAC-NG

Run numerical work through Slurm. From `CriticalBehavior/`, create a fresh run directory on Lustre before submitting. The example uses the general campus account `acf-utk0011`. Use the project account if the workload is charged to one.

Download the pinned source archive for the published dataset into `raw/`.

```bash
mkdir -p -- raw
curl -fsSL https://codeload.github.com/DelMaestroGroup/papers-code-BKTFlowBoseHubbard/zip/009358b5c115ff2ec7bc6e89245ae7ddd2db7e78 -o raw/published-source.zip
printf '%s  %s\n' a77532354d152c544565a513ec1e8ddd49ab8840d1a0f95cd4c1ce89c1a16e6f raw/published-source.zip | sha256sum -c -
```

The installation wrapper populates the private Julia depot inside the current Lustre run directory and uses `julia/1.11.3`. It precompiles packages for a generic CPU target so the same cache works on heterogeneous compute nodes. Python ED and analysis use `anaconda3/2024.06`.

```bash
study_dir=$PWD
mkdir -p -- "${SCRATCHDIR:?}/CriticalBehavior"
work_dir=$(mktemp -d "$SCRATCHDIR/CriticalBehavior/run.XXXXXX")
```

### Reanalyze the archived runs

The archived raw outputs can regenerate the results without rerunning ED or DMRG. These files are local to the existing workspace. A fresh clone can recover them from the original study's Lustre directory using the commands below, or regenerate them by rerunning the simulations. With the pinned archive and a fresh Lustre run directory, stage the files together. The analysis reads `reproduction.json` from the same directory as the raw outputs.

If the local raw outputs are missing and the original ISAAC-NG run is still available, recover them first.

```bash
mkdir -p -- "$study_dir/raw"
original_run="$SCRATCHDIR/CriticalBehavior"
cp -- "$original_run"/correlations-*.toml "$original_run"/gap-*.toml \
  "$original_run"/ed-*.json "$original_run"/itensor-smoke.toml \
  "$original_run"/warm-validation.toml "$study_dir/raw/"
```

```bash
cp -- "$study_dir"/raw/*.toml "$study_dir"/raw/ed-*.json "$study_dir/results/reproduction.json" "$work_dir/"
sbatch --parsable --account=acf-utk0011 --partition=short --qos=short --time=00:10:00 \
  --cpus-per-task=1 --mem=3G --chdir="$work_dir" --output="$work_dir/analysis-%j.log" \
  "$study_dir/slurm/run.sbatch" "$study_dir" analysis/analyze.py "$work_dir" "$work_dir/independent"
```

The analysis writes `independent.json` and `independent.pdf` to the run directory. The published-data command in the next section regenerates `reproduction.json`, `reproduction.csv`, and `reproduction.pdf` independently.

### Rerun the simulations

Start with an empty run directory and submit the setup, published-data analysis, and ED jobs.

```bash
sbatch --parsable --account=acf-utk0011 --partition=short --qos=short --time=00:20:00 \
  --cpus-per-task=2 --mem=6G --chdir="$work_dir" --output="$work_dir/setup-%j.log" \
  "$study_dir/slurm/install-itensor.sbatch" "$study_dir"

sbatch --parsable --account=acf-utk0011 --partition=short --qos=short --time=00:10:00 \
  --cpus-per-task=1 --mem=3G --chdir="$work_dir" --output="$work_dir/reproduce-%j.log" \
  "$study_dir/slurm/run.sbatch" "$study_dir" analysis/reproduce.py "$study_dir/raw/published-source.zip" "$work_dir/reproduction"

sbatch --parsable --account=acf-utk0011 --partition=short --qos=short --time=00:20:00 \
  --cpus-per-task=1 --mem=6G --array=0-8%6 --chdir="$work_dir" --output="$work_dir/ed-%A_%a.log" \
  "$study_dir/slurm/ed-sweep.sbatch" "$study_dir"
```

Once Julia setup has passed, submit the DMRG manifests. Allow three hours for the open-chain array. The longest completed chain task in the study took about two hours and six minutes. The archived run has 81 ITensor outputs because the $L=256$, $\chi=192$ gap job timed out. A rerun that completes this task will produce an additional output. The analysis selects the $\chi=288$ result for that gap when both are available.

```bash
for kind in rings chains cutoffs cutoff-matched; do
  count=$(wc -l < "$study_dir/slurm/sweeps/$kind.tsv")
  walltime=01:35:00
  if [[ $kind == chains ]]; then
    walltime=03:00:00
  fi
  sbatch --parsable --account=acf-utk0011 --partition=campus --qos=campus --time="$walltime" \
    --cpus-per-task=1 --mem=5G --array="0-$((count - 1))%12" \
    --chdir="$work_dir" --output="$work_dir/$kind-%A_%a.log" \
    "$study_dir/slurm/sweep.sbatch" "$study_dir" "$kind.tsv"
done
```

The archive also includes an eight-site ITensor comparison to ED and a warm-start check of the 32-site open-chain charge gap. The following jobs rerun those checks after Julia setup.

```bash
sbatch --parsable --account=acf-utk0011 --partition=short --qos=short --time=00:15:00 \
  --cpus-per-task=1 --mem=5G --chdir="$work_dir" --output="$work_dir/itensor-ed-%j.log" \
  "$study_dir/slurm/run-itensor.sbatch" "$study_dir" 8 3.275 6 periodic 128 correlations "$work_dir/itensor-smoke.toml"

sbatch --parsable --account=acf-utk0011 --partition=short --qos=short --time=00:20:00 \
  --cpus-per-task=1 --mem=5G --chdir="$work_dir" --output="$work_dir/warm-check-%j.log" \
  "$study_dir/slurm/run-itensor.sbatch" "$study_dir" 32 3.275 6 open 192 gap "$work_dir/warm-validation.toml" --warm-gap
```

Record the job IDs. Monitor each job with `squeue -j "$job_id"` and check `sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed` and the logs before submitting the analysis command. For a fresh simulation, use its raw outputs and its generated `reproduction.json`.
