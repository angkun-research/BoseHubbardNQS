"""D4 coordinate conventions for square-torus regular-representation features.

Index r+4*m means R**r S**m, R(x,y)=(-y,x), S(x,y)=(x,-y).
For g=(t,q), pullback action is (g*x)(r)=x(q^-1(r-t)).
Feature action additionally replaces orientation h by q^-1*h.
These helpers contain geometry only, not ED, training or a particle-number mask.
"""
import torch

if __package__:
    from .lattice import _integer
else:
    from lattice import _integer

D4_ELEMENTS = tuple(range(8))


def d4_product(left, right):
    """Index of left*right (right acts first)."""
    r, m = left % 4, left // 4
    s, n = right % 4, right // 4
    return (r + (-1 if m else 1) * s) % 4 + 4 * (m ^ n)


def d4_inverse(element):
    return element if element >= 4 else (-element) % 4


def d4_coordinate(element, x, y):
    """Apply R**r S**m to integer coordinates about the origin."""
    if element >= 4:
        y = -y
    for _ in range(element % 4):
        x, y = -y, x
    return x, y


def kernel_pullbacks():
    """Long tensor [h,delta_index] locating h^-1*delta in a3x3 stencil."""
    rows = []
    for h in D4_ELEMENTS:
        row = []
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                sx, sy = d4_coordinate(d4_inverse(h), x, y)
                row.append((sy + 1) * 3 + sx + 1)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def site_pullback(L, element, tx=0, ty=0, device="cpu"):
    """Return p with transformed[...,r]=original[...,p[r]], i=y*L+x."""
    L = _integer(L, "L", 1)
    element = _integer(element, "D4 element")
    if element >= 8:
        raise ValueError("D4 element must lie in0..7")
    indices = []
    for y in range(L):
        for x in range(L):
            sx, sy = d4_coordinate(d4_inverse(element), x - tx, y - ty)
            indices.append((sy % L) * L + sx % L)
    return torch.tensor(indices, dtype=torch.long, device=device)


def space_group_pullbacks(L):
    """All8*L² pullbacks, ordered (D4element,ty,tx); rows need not be faithful."""
    L = _integer(L, "L", 1)
    return torch.stack([site_pullback(L, q, tx, ty)
                        for q in D4_ELEMENTS for ty in range(L) for tx in range(L)])


def transform_features(features, element, tx=0, ty=0):
    """Pull back (batch,orientation,channel,y,x) regular features."""
    if features.ndim != 5 or features.shape[1] != 8 or features.shape[-1] != features.shape[-2]:
        raise ValueError("regular features must have shape(batch,8,channels,L,L)")
    L = features.shape[-1]
    sites = site_pullback(L, element, tx, ty, features.device)
    orientations = torch.tensor([d4_product(d4_inverse(element), h) for h in D4_ELEMENTS],
                                device=features.device)
    return features[:, orientations].flatten(-2)[..., sites].reshape_as(features)
