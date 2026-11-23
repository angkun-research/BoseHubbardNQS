"""Check three fixed-number sectors of the periodic 3x3 model against ED."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pythonTest" / "2D"))
from exact_diagonalization import exact_ground_state
from lattice import bonds_2d


def one_body_vector(state, basis, nmax):
    indices = {occupations: idx for idx, occupations in enumerate(basis)}
    # The ED state is translation-invariant, so one reference origin suffices.
    rho = np.zeros(len(basis[0]))
    for idx, occupations in enumerate(basis):
        rho[0] += occupations[0] * state[idx] ** 2
        if occupations[0] == nmax:
            continue
        for j, occupation in enumerate(occupations[1:], start=1):
            if occupation == 0:
                continue
            new = list(occupations)
            new[0] += 1
            new[j] -= 1
            rho[j] += (state[idx] * state[indices[tuple(new)]]
                       * np.sqrt(new[0] * occupation))
    return rho


def main(path):
    length = 3
    nmax = 4
    coupling = 16.7424
    sectors = {}
    for offset in (0, -1, 1):
        energy, state, basis, _ = exact_ground_state(length, Ntot=length**2 + offset,
            Nmax=nmax, U=coupling, pbc=True, max_dimension=60_000)
        result = {"energy": energy, "basis_dimension": len(basis)}
        if offset == 0:
            rho = one_body_vector(state, basis, nmax).reshape(length, length)
            momentum = np.fft.fft2(rho).real
            n0 = momentum[0, 0]
            nkx, nky = momentum[0, 1], momentum[1, 0]
            result.update(n0=n0, nkx=nkx, nky=nky, ratio=1 - (nkx+nky)/(2*n0))
        sectors[str(offset)] = result
    path.write_text(json.dumps({"L": length, "nmax": nmax, "U": coupling,
        "bonds": len(bonds_2d(length)), "sectors": sectors}, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
