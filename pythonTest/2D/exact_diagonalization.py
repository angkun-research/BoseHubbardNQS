"""Self-contained fixed-number 2D Bose-Hubbard exact reference.

H = -t sum_<ij> (b_i^dag b_j + b_j^dag b_i)
    + (U/2) sum_i n_i(n_i-1) - mu sum_i n_i.

U is twice the coefficient called U in the existing 1D code. Occupations satisfy
0 <= n_i <= Nmax, including Nmax. No runtime dependency on the 1D implementation.
Physical parameters are finite real scalars, with t >= 0 and U >= 0.
"""

from numbers import Real

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from lattice import _integer, bonds_2d, lattice_shape


DEFAULT_MAX_DIMENSION = 20_000
DEFAULT_MAX_SPARSE_BYTES = 512 * 1024**2
MAX_COUNTING_WORK = 1_000_000


def _sector(n_sites, Ntot, Nmax):
    n_sites = _integer(n_sites, "n_sites", 1)
    Nmax = _integer(Nmax, "Nmax")
    Ntot = _integer(Ntot, "Ntot")
    if Ntot > n_sites * Nmax:
        raise ValueError("Ntot exceeds n_sites * Nmax")
    return n_sites, Ntot, Nmax


def count_basis_states(n_sites, Ntot, Nmax=4):
    """Exact bounded-sector count, using integer DP before enumerating states.

    Particle-hole symmetry reduces work near full filling. No basis is allocated.
    At most MAX_COUNTING_WORK DP cells may be processed; larger requests raise
    ValueError before allocating the particle-count-sized workspace.
    """
    n_sites, Ntot, Nmax = _sector(n_sites, Ntot, Nmax)
    target = min(Ntot, n_sites * Nmax - Ntot)
    if target == 0 or n_sites == 1:
        return 1
    if n_sites * (target + 1) > MAX_COUNTING_WORK:
        raise ValueError(f"sector counting work exceeds {MAX_COUNTING_WORK} DP cells")
    counts = [1] + [0] * target
    for _ in range(n_sites):
        updated = []
        window = 0
        for number in range(target + 1):
            window += counts[number]
            if number > Nmax:
                window -= counts[number - Nmax - 1]
            updated.append(window)
        counts = updated
    return counts[target]


def _check_dimension(dimension, max_dimension):
    limit = _integer(max_dimension, "max_dimension", 1)
    if dimension > limit:
        raise ValueError(f"sector dimension {dimension} exceeds max_dimension={limit}")


def generate_basis(n_sites, Ntot, Nmax=4, max_dimension=DEFAULT_MAX_DIMENSION):
    """Enumerate bounded fixed-number occupation tuples in lexicographic order."""
    n_sites, Ntot, Nmax = _sector(n_sites, Ntot, Nmax)
    dimension = count_basis_states(n_sites, Ntot, Nmax)
    _check_dimension(dimension, max_dimension)
    if Ntot == 0 or Ntot == n_sites * Nmax:
        return [(0 if Ntot == 0 else Nmax,) * n_sites]
    basis = []
    state = [0] * n_sites

    def choices(site, remaining):
        lower = max(0, remaining - (n_sites - site - 1) * Nmax)
        return iter(range(lower, min(Nmax, remaining) + 1))

    # Explicit depth-first traversal preserves lexicographic order without a
    # Python recursion-depth limit or copying partial configurations per branch.
    stack = [(Ntot, choices(0, Ntot))]
    while stack:
        remaining, occupations = stack[-1]
        site = len(stack) - 1
        occupation = next(occupations, None)
        if occupation is None:
            stack.pop()
            continue
        state[site] = occupation
        if site == n_sites - 1:
            basis.append(tuple(state))
        else:
            rest = remaining - occupation
            stack.append((rest, choices(site + 1, rest)))
    return basis


def basis_to_one_hot(basis, n_sites, nmax):
    """Return float32 one-hot features shaped (dimension, n_sites, nmax+1)."""
    n_sites = _integer(n_sites, "n_sites", 1)
    nmax = _integer(nmax, "nmax")
    occupations = np.asarray(basis)
    if len(basis) == 0:
        return np.empty((0, n_sites, nmax + 1), dtype=np.float32)
    if (occupations.ndim != 2 or occupations.shape[1] != n_sites
            or not np.issubdtype(occupations.dtype, np.integer)
            or np.any(occupations < 0) or np.any(occupations > nmax)):
        raise ValueError("basis must contain integer occupations in [0, nmax]")
    features = np.zeros((len(basis), n_sites, nmax + 1), dtype=np.float32)
    features[np.arange(len(basis))[:, None], np.arange(n_sites), occupations] = 1
    return features


def estimate_sparse_bytes(dimension, n_sites, n_bonds):
    """Conservative planning estimate, not a hard process-memory bound.

    Allow one diagonal and two directed hops per bond per column, even when
    occupation constraints forbid them. Budget 64 bytes per possible entry for
    COO arrays, CSR conversion and temporaries; also budget Python basis/index
    storage and 32 float64 eigensolver vectors. No dense matrix is included.
    """
    dimension = _integer(dimension, "dimension", 1)
    n_sites = _integer(n_sites, "n_sites", 1)
    n_bonds = _integer(n_bonds, "n_bonds")
    entries = dimension * (1 + 2 * n_bonds)
    return 64 * entries + dimension * (256 + 40 * n_sites + 32 * 8)


def _physical_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and real")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be finite and real") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if name in ("t", "U") and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def build_bose_hubbard_hamiltonian(
    Lx, Ly=None, Ntot=None, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True,
    max_dimension=DEFAULT_MAX_DIMENSION,
    max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES,
    *, pbc_x=None, pbc_y=None,
):
    """Return (CSR Hamiltonian, basis); guard allocations before enumeration.

    Defaults are a square periodic lattice at unit filling. Both dimension and
    estimated sparse-memory limits must pass; increase either explicitly if needed.
    Counting also has a fixed MAX_COUNTING_WORK budget before workspace allocation.
    """
    Lx, Ly = lattice_shape(Lx, Ly)
    n_sites = Lx * Ly
    Ntot = n_sites if Ntot is None else Ntot
    n_sites, Ntot, Nmax = _sector(n_sites, Ntot, Nmax)
    t, U, mu = (_physical_scalar(value, name)
                for name, value in (("t", t), ("U", U), ("mu", mu)))
    dimension = count_basis_states(n_sites, Ntot, Nmax)
    _check_dimension(dimension, max_dimension)
    memory_limit = _integer(max_sparse_bytes, "max_sparse_bytes", 1)
    # Reject even the minimum storage before constructing a potentially huge
    # lattice, including a vacuum sector with many sites but only one state.
    if estimate_sparse_bytes(dimension, n_sites, 0) > memory_limit:
        raise ValueError(f"estimated sparse memory exceeds max_sparse_bytes={memory_limit}")
    bonds = bonds_2d(Lx, Ly, pbc, pbc_x=pbc_x, pbc_y=pbc_y)
    estimate = estimate_sparse_bytes(dimension, n_sites, len(bonds))
    if estimate > memory_limit:
        raise ValueError(f"estimated sparse memory {estimate} bytes exceeds "
                         f"max_sparse_bytes={memory_limit}")
    basis = generate_basis(n_sites, Ntot, Nmax, max_dimension)
    state_to_index = {state: idx for idx, state in enumerate(basis)}
    capacity = dimension * (1 + 2 * len(bonds))
    rows = np.empty(capacity, dtype=np.int64)
    columns = np.empty(capacity, dtype=np.int64)
    values = np.empty(capacity, dtype=np.float64)
    entry = 0
    for col, state in enumerate(basis):
        rows[entry], columns[entry] = col, col
        values[entry] = sum(0.5 * U * n * (n - 1) - mu * n for n in state)
        entry += 1
        if t == 0:
            continue
        for i, j in bonds:
            for src, dst in ((i, j), (j, i)):
                if state[src] > 0 and state[dst] < Nmax:
                    connected = list(state)
                    connected[src] -= 1
                    connected[dst] += 1
                    rows[entry] = state_to_index[tuple(connected)]
                    columns[entry] = col
                    values[entry] = -t * np.sqrt(state[src] * (state[dst] + 1))
                    entry += 1
    H = coo_matrix((values[:entry], (rows[:entry], columns[:entry])),
                   shape=(dimension, dimension)).tocsr()
    H.eliminate_zeros()
    return H, basis


def exact_ground_state(
    Lx, Ly=None, Ntot=None, Nmax=4, t=1.0, U=16.74, mu=0.0, pbc=True,
    max_dimension=DEFAULT_MAX_DIMENSION,
    max_sparse_bytes=DEFAULT_MAX_SPARSE_BYTES,
    *, pbc_x=None, pbc_y=None,
):
    """Return (energy, normalized real state, basis, CSR Hamiltonian)."""
    H, basis = build_bose_hubbard_hamiltonian(
        Lx, Ly, Ntot, Nmax, t, U, mu, pbc, max_dimension, max_sparse_bytes, pbc_x=pbc_x, pbc_y=pbc_y
    )
    diagonal = H.diagonal()
    if H.nnz == np.count_nonzero(diagonal):
        # Includes zero matrices and degenerate atomic limits, for which ARPACK
        # can break down. Choose one normalized ground vector deterministically.
        index = int(np.argmin(diagonal))
        state = np.zeros(len(basis))
        state[index] = 1
        return float(diagonal[index]), state, basis, H
    # A seeded start makes reference solves reproducible without restricting
    # the initial vector to a particular lattice symmetry sector.
    initial = np.random.default_rng(0).normal(size=len(basis))
    energies, states = eigsh(H, k=1, which="SA", v0=initial)
    state = states[:, 0]
    if state[np.argmax(np.abs(state))] < 0:
        state = -state
    return float(energies[0]), state, basis, H
