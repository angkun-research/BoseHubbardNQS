# BoseHubbardNQS
Neural network quantum states for bose hubbard model

## Usage

Everything here should be run under the project root directory.

### 1. Create the local Python environment

This project expects a local virtual environment in `.venv/`.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

The Julia wrapper uses this same Python environment through `PythonCall`. The helper script [julia-local] sets:

- `JULIA_CONDAPKG_BACKEND=Null`
- `JULIA_PYTHONCALL_EXE=.venv/bin/python`

so Julia will use the local Python interpreter instead of creating a separate one.

If you have not instantiated the Julia environment yet:

```bash
./julia-local -e 'using Pkg; Pkg.instantiate()'
```

### 2. Train a neural network with `TrainEDstates.py`

The training script is [pythonTest/TrainEDstates.py]. It:

1. builds an exact-diagonalization dataset for the Bose-Hubbard model,
2. trains one of the neural-network architectures,
3. evaluates overlap with the ED ground state,
4. saves the trained model to `data/bose_hubbard_nqs.pt`. Note that we need to create the folder `data/` first.

Run it with:

```bash
.venv/bin/python TrainEDstates.py
```

Before running, edit the following parameters in the `if __name__ == "__main__":` block of [pythonTest/TrainEDstates.py]:

- `L`: number of lattice sites
- `Nmax`: maximum boson occupation per site
- `Ntot`: total boson number
- `t`: hopping amplitude
- `U`: onsite interaction
- `mu`: chemical potential
- `pbc`: periodic boundary conditions (`True` or `False`)
- `target`: training target, typically `"logabs"`
- `hidden_dim`: hidden layer size
- `arch_name`: one of `"FCNet"`, `"JastrowNet"`, `"Conv1DNet"`, `"RBMNet"`

Example configuration already present in the script:

```python
L = 8
Nmax = 8
Ntot = 8
t = 1.0
U = 2.0
mu = 0.0
pbc = True
target = "logabs"
hidden_dim = 128
arch_name = "FCNet"
```

After training, the model is saved as:

```python
MODEL_PATH = Path("data/bose_hubbard_nqs.pt")
save_model(MODEL_PATH, model, arch_name, L, Nmax, hidden_dim)
```

### 3. Load the trained network from Julia

The Julia module exports `load_nn`, `nn_value`, and `ansatz_log_amplitude` from [src/BoseHubbardNQS.jl].

A complete example is given in [example/Bose_Hubbard_NQS_test.jl]. Run it with:

```bash
./julia-local Bose_Hubbard_NQS_test.jl
```

Core usage:

```julia
using BoseHubbardNQS

MODEL_PATH = "data/bose_hubbard_nqs.pt"

L = 8
Nmax = 8
hidden_dim = 128

model = load_nn(
    MODEL_PATH;
    arch="FCNet",
    n_sites=L,
    nmax=Nmax,
    hidden_dim=hidden_dim,
    map_location="cpu",
)

sigma = [1 for _ in 1:L]
val = nn_value(model, sigma)

println("occupation config = ", sigma)
println("NN scalar output = ", val)
println("ansatz log amplitude = ", ansatz_log_amplitude(model, sigma))
```

The example file also shows how to:

- evaluate the model on random occupation configurations,
- generate the fixed-particle-number basis using `pythonTest.BoseHubbardED`,
- compute the sector normalization constant
  `logZ = logsumexp(2 .* logpsi_raw)`,
- convert the raw network output into normalized amplitudes and probabilities.

Important: the Julia load arguments must match the architecture and dimensions used when training the model, for example `arch="FCNet"`, `n_sites=8`, `nmax=8`, and `hidden_dim=128`.



## Install it as a julia module for other projects

To use it beyond the project root:
Download the repo to the root (/home/BoseHubbardNQS/). Open Julia from home root and install the module to julia global environment
```julia
]
dev ~/BoseHubbardNQS
```
Launch julia from root
```bash
./BoseHubbardNQS/julia-local BoseHubbardNQS/example/Bose_Hubbard_NQS_test.jl
```