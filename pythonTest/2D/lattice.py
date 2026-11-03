"""Rectangular square lattices, flattened as ``i = y * Lx + x``.

Each unordered nearest-neighbor bond occurs once. In particular, periodic
length-two axes have one edge, unlike the parallel bonds in the existing 1D
code. Length-one axes never contribute self-hopping. Legacy pbc sets both axes;
explicit pbc_x/pbc_y overrides take precedence independently.
"""

from numbers import Integral


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def lattice_shape(Lx, Ly=None):
    """Validate dimensions; an omitted Ly gives a square lattice."""
    Lx = _integer(Lx, "Lx", 1)
    Ly = Lx if Ly is None else _integer(Ly, "Ly", 1)
    return Lx, Ly


def resolve_boundaries(pbc=True, pbc_x=None, pbc_y=None):
    """Resolve each optional axis override against the legacy both-axis flag."""
    if not isinstance(pbc, bool):
        raise ValueError("pbc must be a boolean")
    for name, value in (("pbc_x", pbc_x), ("pbc_y", pbc_y)):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean or None")
    return pbc if pbc_x is None else pbc_x, pbc if pbc_y is None else pbc_y


def add_boundary_arguments(parser):
    """Shared CLI flags; overrides win regardless of global-flag ordering."""
    global_flags = parser.add_mutually_exclusive_group()
    global_flags.add_argument("--pbc", dest="pbc", action="store_true", default=True,
                              help="set both axes periodic before explicit axis overrides")
    global_flags.add_argument("--no-pbc", dest="pbc", action="store_false",
                              help="set both axes open before explicit axis overrides")
    for axis in ("x", "y"):
        flags = parser.add_mutually_exclusive_group()
        flags.add_argument(f"--pbc-{axis}", dest=f"pbc_{axis}", action="store_true", default=None)
        flags.add_argument(f"--no-pbc-{axis}", dest=f"pbc_{axis}", action="store_false")


def resolve_cli_boundaries(args):
    """Resolve flags without changing the historical fully periodic default."""
    args.pbc_x, args.pbc_y = resolve_boundaries(args.pbc, args.pbc_x, args.pbc_y)


def bonds_2d(Lx, Ly=None, pbc=True, *, pbc_x=None, pbc_y=None):
    """Return sorted unique site pairs for open or periodic boundaries."""
    Lx, Ly = lattice_shape(Lx, Ly)
    pbc_x, pbc_y = resolve_boundaries(pbc, pbc_x, pbc_y)
    bonds = set()
    for y in range(Ly):
        for x in range(Lx):
            site = y * Lx + x
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if (nx >= Lx and not pbc_x) or (ny >= Ly and not pbc_y):
                    continue
                nx, ny = nx % Lx, ny % Ly
                neighbor = ny * Lx + nx
                if neighbor != site:
                    bonds.add(tuple(sorted((site, neighbor))))
    return sorted(bonds)
