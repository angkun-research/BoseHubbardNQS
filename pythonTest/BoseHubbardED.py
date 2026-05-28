import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
import numpy as np

def generate_basis(L, Ntot, Nmax):
    """
    Generate all bosonic occupation-number basis states
    (n_0, ..., n_{L-1}) with:
        sum_i n_i = Ntot
        0 <= n_i <= Nmax
    """
    basis = []
    state = [0] * L

    def rec(site, remaining):
        if site == L - 1:
            if remaining <= Nmax:
                state[site] = remaining
                basis.append(tuple(state))
            return

        max_occ = min(Nmax, remaining)
        for n in range(max_occ + 1):
            state[site] = n
            rec(site + 1, remaining - n)

    rec(0, Ntot)
    return basis

def basis_to_one_hot(basis, n_sites, nmax):
    """
    Convert a list of basis states like
        [(n1,...,nL), ...]
    into one-hot encoded tensors of shape
        (num_basis, n_sites, nmax + 1)

    local occupation n_i is encoded as a one-hot vector of length nmax+1.
    """
    channels = nmax + 1
    num_basis = len(basis)

    x = np.zeros((num_basis, n_sites, channels), dtype=np.float32)

    for idx, state in enumerate(basis):
        for site, occ in enumerate(state):
            x[idx, site, occ] = 1.0

    return x

def build_bose_hubbard_hamiltonian(L, Ntot, Nmax, t, U, mu=0.0, pbc=True):
    """
    Build the full many-body Bose-Hubbard Hamiltonian in the fixed-N sector.

    Hamiltonian convention matches the Julia example:
        H = -t sum_<ij> (b_i^dag b_j + h.c.)
            + sum_i [U n_i (n_i - 1) - mu n_i]
    """
    basis = generate_basis(L=L, Ntot=Ntot, Nmax=Nmax)
    dim = len(basis)
    state_to_index = {state: idx for idx, state in enumerate(basis)}

    H = np.zeros((dim, dim), dtype=np.float64)

    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc:
        bonds.append((L - 1, 0))

    for col, state in enumerate(basis):
        # Diagonal onsite term
        diag = 0.0
        for n in state:
            diag += U * n * (n - 1) - mu * n
        H[col, col] = diag

        # Hopping terms
        for i, j in bonds:
            n_i = state[i]
            n_j = state[j]

            # b_i^dagger b_j : move one boson j -> i
            if n_j > 0 and n_i < Nmax:
                new_state = list(state)
                new_state[i] += 1
                new_state[j] -= 1
                row = state_to_index[tuple(new_state)]
                H[row, col] += -t * np.sqrt((n_i + 1) * n_j)

            # b_j^dagger b_i : move one boson i -> j
            if n_i > 0 and n_j < Nmax:
                new_state = list(state)
                new_state[i] -= 1
                new_state[j] += 1
                row = state_to_index[tuple(new_state)]
                H[row, col] += -t * np.sqrt((n_j + 1) * n_i)

    return H, basis


def exact_ground_state(L, Ntot, Nmax, t, U, mu=0.0, pbc=True):
    H, basis = build_bose_hubbard_hamiltonian(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )
    # evals, evecs = np.linalg.eigh(H)
    evals, evecs = eigsh(csr_matrix(H), k=1, which="SA")
    ground_energy = evals[0]
    ground_state = evecs[:, 0]
    return ground_energy, ground_state, basis, H


def print_largest_components(state_vector, basis, n_show=10):
    weights = np.abs(state_vector) ** 2
    order = np.argsort(weights)[::-1]

    print(f"Top {n_show} basis components:")
    for k in order[:n_show]:
        print(
            f"basis={basis[k]}   amp={state_vector[k]: .8f}   prob={weights[k]:.8f}"
        )

def density_profile(ground_state, basis):
    """
    Compute <n_i> for each site i from a ground-state vector in the occupation basis.

    Returns:
        dens: array of shape (L,)
    """
    dim = len(basis)
    if dim == 0:
        return np.array([])

    L = len(basis[0])
    probs = np.abs(ground_state) ** 2
    dens = np.zeros(L, dtype=np.float64)

    for amp2, state in zip(probs, basis):
        for i, n_i in enumerate(state):
            dens[i] += amp2 * n_i

    return dens


def one_body_density_matrix(ground_state, basis, Nmax):
    """
    Compute the one-body density matrix rho[i, j] = <b_i^dagger b_j>.

    Args:
        ground_state: eigenvector in the occupation basis
        basis: list of occupation tuples
        Nmax: local occupancy cutoff used to build the basis

    Returns:
        rho: complex array of shape (L, L)
    """
    dim = len(basis)
    if dim == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    L = len(basis[0])
    basis_index = {state: idx for idx, state in enumerate(basis)}
    rho = np.zeros((L, L), dtype=np.complex128)

    for ket_idx, ket_state in enumerate(basis):
        ket_amp = ground_state[ket_idx]

        for j in range(L):
            n_j = ket_state[j]
            if n_j == 0:
                continue

            for i in range(L):
                if i == j:
                    rho[i, j] += np.conjugate(ket_amp) * ket_amp * n_j
                    continue

                n_i = ket_state[i]
                if n_i >= Nmax:
                    continue

                new_state = list(ket_state)
                new_state[i] += 1
                new_state[j] -= 1
                bra_idx = basis_index[tuple(new_state)]

                matrix_element = np.sqrt((n_i + 1) * n_j)
                rho[i, j] += np.conjugate(ground_state[bra_idx]) * ket_amp * matrix_element

    return rho

if __name__ == "__main__":
    # Match the Julia example as closely as possible
    L = 8
    Nmax = 8
    Ntot = 8
    t = 1.0
    U = 2.0
    mu = 0.0
    pbc = True

    energy, psi0, basis, H = exact_ground_state(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )
    # -7.459790610498828 

    dens = density_profile(psi0, basis)
    rho = one_body_density_matrix(psi0, basis, Nmax)

    print("Hilbert-space dimension:", len(basis))
    print("Hamiltonian shape:", H.shape)
    print("Ground-state energy:", energy)
    print("Ground-state norm:", np.linalg.norm(psi0))
    print("Density profile:", dens)
    print("One-body density matrix:\n", rho)

    #print_largest_components(psi0, basis, n_show=10)

    rho_plot = np.real_if_close(rho)   # or np.abs(rho) if you want magnitudes

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rho_plot, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$\langle b_i^\dagger b_j \rangle$")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("One-body density matrix")
    plt.tight_layout()
    #plt.show()
    fig.savefig("fig_rho.pdf", dpi=150, bbox_inches="tight")