"""Sparse periodic-ring ED for checking the DMRG Hamiltonian and observables."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def occupations(length, particles, nmax):
    if length == 1:
        if 0 <= particles <= nmax:
            yield (particles,)
        return
    for n in range(max(0, particles - (length - 1) * nmax), min(nmax, particles) + 1):
        for tail in occupations(length - 1, particles - n, nmax):
            yield (n,) + tail


def sector(length, particles, nmax, couplings):
    basis = np.asarray(list(occupations(length, particles, nmax)), dtype=np.int16)
    place = (nmax + 1) ** np.arange(length - 1, -1, -1, dtype=np.int64)
    keys = basis @ place

    def hop(dest, source):
        cols = np.flatnonzero((basis[:, dest] < nmax) & (basis[:, source] > 0))
        rows = np.searchsorted(keys, keys[cols] + place[dest] - place[source])
        amplitude = np.sqrt((basis[cols, dest] + 1) * basis[cols, source], dtype=np.float64)
        return rows, cols, amplitude

    moves = [hop(i, (i + 1) % length) for i in range(length)]
    row, col, amplitude = map(np.concatenate, zip(*moves))
    dimension = len(basis)
    forward = coo_matrix((-amplitude, (row, col)), shape=(dimension, dimension)).tocsr()
    h = forward + forward.T
    doublons = np.sum(basis * (basis - 1), axis=1) / 2
    v0 = np.ones(dimension)
    for coupling in couplings:
        h.setdiag(coupling * doublons)
        e, wave = eigsh(h, k=1, which="SA", v0=v0, tol=1e-12)
        psi = v0 = wave[:, 0]
        result = {"energy": float(e[0]), "residual": float(np.linalg.norm(h @ psi - e[0] * psi)),
                  "dimension": dimension}
        if length == particles:
            probability = psi**2
            block_counts = np.cumsum(basis[:, :length // 2], axis=1)
            centered = block_counts - probability @ block_counts
            result["block_variance"] = (probability @ centered**2).tolist()
            result["g1"] = []
            for r in range(1, length // 2 + 1):
                rows, cols, amplitude = hop(0, r)
                result["g1"].append(float(np.sum(psi[rows] * psi[cols] * amplitude)))
        yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int, choices=range(4, 13))
    parser.add_argument("nmax", type=int, choices=range(2, 9))
    parser.add_argument("output", type=Path)
    parser.add_argument("couplings", type=float, nargs="+")
    args = parser.parse_args()
    length = args.length
    sectors = [list(sector(length, length + d, args.nmax, args.couplings)) for d in [0, -1, 1]]
    data = [{"L": length, "U_phys": u, "nmax": args.nmax, "boundary": "periodic",
             "sectors": {str(d): results[i] for d, results in zip([0, -1, 1], sectors)}}
            for i, u in enumerate(args.couplings)]
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
