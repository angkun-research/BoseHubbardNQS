"""Small real log-amplitude models for flattened 2D one-hot occupations.

Inputs are (batch, Lx*Ly, Nmax+1); outputs are (batch,). Forward passes are
deterministic in both train and eval mode. No model enforces particle number;
sector restriction belongs to sampling or normalization, not the ansatz.

JastrowNet enforces translations along each periodic
axis, and x/y reflections for any boundary choice. Sharing keeps axes distinct,
even on square grids. The default displacement mode uses symmetric occupation
tables; cylinder reflection orbits retain ordered edge/bulk occupations. Sharing
reduces capacity relative to independent site-pair
tables and excludes higher-body terms in log amplitude. FCNet, RBMNet, the
flattened Conv2DNet head, and position-aware TransformerNet impose no spatial
invariance. GroupEquivariantConv2DNet enforces the square-torus translations,
rotations and reflections using regular-representation features and pooling.
Checkpoint format version 2 records explicit axis boundaries;
architecture_settings records every configurable architecture parameter.
"""

from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from lattice import _integer, lattice_shape, resolve_boundaries
from square_group import D4_ELEMENTS, d4_inverse, d4_product, kernel_pullbacks
from exact_diagonalization import _physical_scalar


ROOT = Path(__file__).resolve().parent
HAMILTONIAN_CONVENTION = "-t*sum_bonds(b_i^dag*b_j+h.c.)+U/2*sum_i(n_i*(n_i-1))-mu*sum_i(n_i)"
BOND_CONVENTION = "unique_unordered_nearest_neighbors_no_self_bonds"


def _setup(model, Lx, Ly, Nmax, pbc, settings, pbc_x=None, pbc_y=None):
    model.Lx, model.Ly = lattice_shape(Lx, Ly)
    model.Nmax = _integer(Nmax, "Nmax")
    model.pbc_x, model.pbc_y = resolve_boundaries(pbc, pbc_x, pbc_y)
    # None explicitly means mixed boundaries, never a shorthand for open axes.
    model.pbc = model.pbc_x if model.pbc_x == model.pbc_y else None
    model.n_sites = model.Lx * model.Ly
    model.channels = model.Nmax + 1
    model.architecture_settings = settings


def _input(model, x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (model.n_sites, model.channels):
        raise ValueError(f"input must have shape (batch, {model.n_sites}, {model.channels})")
    return x.to(dtype=next(model.parameters()).dtype)


class FCNet(nn.Module):
    """Tanh MLP with positive hidden_layers (default two); depth is an alias.

    Default depth preserves the legacy net.0/net.2/net.4 state_dict keys.
    depth overrides the default hidden_layers=2; conflicting nondefault values
    are rejected. Only FCNet changes capacity through this setting.
    """
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=64, *,
                 hidden_layers=2, depth=None, pbc_x=None, pbc_y=None):
        super().__init__()
        hidden_dim = _integer(hidden_dim, "hidden_dim", 1)
        hidden_layers = _integer(hidden_layers, "hidden_layers", 1)
        if depth is not None:
            depth = _integer(depth, "depth", 1)
            if hidden_layers != 2 and hidden_layers != depth:
                raise ValueError("depth and hidden_layers disagree")
            hidden_layers = depth
        self.hidden_layers = hidden_layers
        _setup(self, Lx, Ly, Nmax, pbc, {"hidden_dim": hidden_dim, "hidden_layers": hidden_layers}, pbc_x, pbc_y)
        layers = [nn.Linear(self.n_sites * self.channels, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(_input(self, x).flatten(1)).squeeze(-1)


class JastrowNet(nn.Module):
    """Onsite table plus all unordered pairs, shared by absolute (dx, dy).

    Periodic distances use the minimum along each axis; open distances do not wrap.
    Default displacement tables are averaged with their transpose. Optional
    reflection_orbit is restricted to open-x/periodic-y cylinders: onsite tables
    use x-reflection orbits, pairs use reflected unordered x pairs and minimal
    periodic |dy|. Reflected canonicalization swaps occupation roles. Only same-x
    or self-reflection pair classes are transpose-symmetric; edge/bulk tables
    retain ordered occupation roles. Pair buffers avoid Python forward loops;
    gather storage scales as batch * number_of_site_pairs.
    """
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, *, pbc_x=None, pbc_y=None,
                 pair_sharing="displacement"):
        super().__init__()
        if pair_sharing not in ("displacement", "reflection_orbit"):
            raise ValueError("pair_sharing must be displacement or reflection_orbit")
        _setup(self, Lx, Ly, Nmax, pbc, {"pair_sharing": pair_sharing}, pbc_x, pbc_y)
        self.pair_sharing = pair_sharing
        if pair_sharing == "reflection_orbit":
            if self.pbc_x or not self.pbc_y:
                raise ValueError("reflection_orbit requires open x and periodic y")
            pairs = torch.triu_indices(self.n_sites, self.n_sites, offset=1)
            ordered_pairs, keys = [], []
            for i, j in pairs.T.tolist():
                if i % self.Lx > j % self.Lx:
                    i, j = j, i
                left, right = i % self.Lx, j % self.Lx
                reflected = (self.Lx - 1 - right, self.Lx - 1 - left)
                if reflected < (left, right):
                    left, right = reflected
                    i, j = j, i
                dy = abs(i // self.Lx - j // self.Lx)
                keys.append((left, right, min(dy, self.Ly - dy)))
                ordered_pairs.append((i, j))
            classes = sorted(set(keys))
            indices = {key: index for index, key in enumerate(classes)}
            self.register_buffer("pair_sites", torch.tensor(ordered_pairs, dtype=torch.long).reshape(-1, 2).T.contiguous())
            self.register_buffer("pair_classes", torch.tensor([indices[key] for key in keys], dtype=torch.long))
            self.register_buffer("pair_orbits", torch.tensor(classes, dtype=torch.long).reshape(-1, 3))
            self.register_buffer("symmetric_classes", torch.tensor(
                [left == right or left + right == self.Lx - 1 for left, right, _ in classes], dtype=torch.bool))
            self.register_buffer("onsite_classes", torch.tensor(
                [min(i % self.Lx, self.Lx - 1 - i % self.Lx) for i in range(self.n_sites)], dtype=torch.long))
            self.onsite = nn.Parameter(torch.zeros((self.Lx + 1) // 2, self.channels))
            self.pair = nn.Parameter(torch.zeros(len(classes), self.channels, self.channels))
            return
        pairs = torch.triu_indices(self.n_sites, self.n_sites, offset=1)
        displacements = []
        for i, j in pairs.T.tolist():
            dx, dy = abs(i % self.Lx - j % self.Lx), abs(i // self.Lx - j // self.Lx)
            if self.pbc_x:
                dx = min(dx, self.Lx - dx)
            if self.pbc_y:
                dy = min(dy, self.Ly - dy)
            displacements.append((dx, dy))
        classes = sorted(set(displacements))
        indices = {displacement: index for index, displacement in enumerate(classes)}
        self.register_buffer("pair_sites", pairs)
        self.register_buffer("pair_classes", torch.tensor(
            [indices[d] for d in displacements], dtype=torch.long
        ))
        self.register_buffer("displacements", torch.tensor(classes, dtype=torch.long).reshape(-1, 2))
        self.onsite = nn.Parameter(torch.zeros(self.channels))
        self.pair = nn.Parameter(torch.zeros(len(classes), self.channels, self.channels))

    def forward(self, x):
        occupations = _input(self, x).argmax(dim=-1)
        if self.pair_sharing == "displacement":
            onsite = self.onsite[occupations].sum(dim=1)
            tables = 0.5 * (self.pair + self.pair.transpose(-1, -2))
        else:
            onsite = self.onsite[self.onsite_classes, occupations].sum(dim=1)
            tables = torch.where(self.symmetric_classes[:, None, None],
                                 0.5 * (self.pair + self.pair.transpose(-1, -2)), self.pair)
        left, right = self.pair_sites
        return onsite + tables[
            self.pair_classes, occupations[:, left], occupations[:, right]
        ].sum(dim=1)


def lift_jastrow_reflection_orbit(model):
    """Lift displacement cylinder weights exactly; no optimizer or jitter.

    Old symmetric pair tables are copied by (dx, minimal periodic |dy|).
    Non-self-reflection orbits subsequently allow ordered occupation tables.
    """
    if not isinstance(model, JastrowNet) or model.pair_sharing != "displacement":
        raise ValueError("expected displacement JastrowNet")
    result = JastrowNet(model.Lx, model.Ly, model.Nmax, pbc_x=model.pbc_x, pbc_y=model.pbc_y,
                        pair_sharing="reflection_orbit").to(device=model.onsite.device, dtype=model.onsite.dtype)
    lookup = {tuple(key): k for k, key in enumerate(model.displacements.tolist())}
    with torch.no_grad():
        result.onsite.copy_(model.onsite[None].expand_as(result.onsite))
        tables = 0.5 * (model.pair + model.pair.transpose(-1, -2))
        for k, (left, right, dy) in enumerate(result.pair_orbits.tolist()):
            result.pair[k].copy_(tables[lookup[right - left, dy]])
    return result.train(model.training)


class Conv2DNet(nn.Module):
    """One 3x3 convolution and a dense head; no translation invariance.

    Each axis is padded separately, circular when periodic and zero otherwise,
    including length-one axes. Zero feature padding is not an occupation-zero
    one-hot outside site. Legacy convolution/head state_dict keys are unchanged.
    """
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=64, *, pbc_x=None, pbc_y=None):
        super().__init__()
        hidden_dim = _integer(hidden_dim, "hidden_dim", 1)
        _setup(self, Lx, Ly, Nmax, pbc, {"hidden_dim": hidden_dim}, pbc_x, pbc_y)
        self.conv = nn.Conv2d(self.channels, hidden_dim, kernel_size=3, padding=0)
        self.head = nn.Sequential(
            nn.SiLU(), nn.Flatten(), nn.Linear(hidden_dim * self.n_sites, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = _input(self, x).transpose(1, 2).reshape(-1, self.channels, self.Ly, self.Lx)
        x = F.pad(x, (1, 1, 0, 0), mode="circular" if self.pbc_x else "constant")
        x = F.pad(x, (0, 0, 1, 1), mode="circular" if self.pbc_y else "constant")
        return self.head(self.conv(x)).squeeze(-1)


class RBMNet(nn.Module):
    """Binary-hidden RBM expression interpreted directly as log amplitude."""
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, n_hidden=64, *, pbc_x=None, pbc_y=None):
        super().__init__()
        n_hidden = _integer(n_hidden, "n_hidden", 1)
        _setup(self, Lx, Ly, Nmax, pbc, {"n_hidden": n_hidden}, pbc_x, pbc_y)
        visible = self.n_sites * self.channels
        self.visible_bias = nn.Parameter(torch.zeros(visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.W = nn.Parameter(0.01 * torch.randn(visible, n_hidden))

    def forward(self, x):
        visible = _input(self, x).flatten(1)
        return visible @ self.visible_bias + F.softplus(visible @ self.W + self.hidden_bias).sum(1)


class TransformerNet(nn.Module):
    """One non-autoregressive, one-head attention block; learned x/y positions."""
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=32, feedforward_dim=None, *, pbc_x=None, pbc_y=None):
        super().__init__()
        hidden_dim = _integer(hidden_dim, "hidden_dim", 1)
        feedforward_dim = 2 * hidden_dim if feedforward_dim is None else _integer(
            feedforward_dim, "feedforward_dim", 1
        )
        _setup(self, Lx, Ly, Nmax, pbc,
               {"hidden_dim": hidden_dim, "feedforward_dim": feedforward_dim}, pbc_x, pbc_y)
        self.occupation_embedding = nn.Linear(self.channels, hidden_dim)
        self.x_embedding = nn.Parameter(0.02 * torch.randn(self.Lx, hidden_dim))
        self.y_embedding = nn.Parameter(0.02 * torch.randn(self.Ly, hidden_dim))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=1, dim_feedforward=feedforward_dim,
            dropout=0.0, activation="relu", batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        positions = (self.y_embedding[:, None] + self.x_embedding[None, :]).reshape(self.n_sites, -1)
        tokens = self.occupation_embedding(_input(self, x)) + positions
        return self.head(self.encoder(tokens).mean(dim=1)).squeeze(-1)


class Conv2DTransformerNet(nn.Module):
    """2D generalization of the 1D ConvTransformer, for optional backup studies.

    A boundary-aware 3x3 convolution and SiLU produce physical-order tokens
    i=y*Lx+x. One four-head encoder uses FF width 2h and ReLU, followed by
    the original flattened dense/SiLU/scalar head. There are no positional
    embeddings or mean pooling. The flattened head imposes no translation
    invariance. Unlike the 1D module's framework-default dropout=0.1, dropout
    is fixed to zero for deterministic VMC forwards in train and eval mode.
    """
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=32, *, pbc_x=None, pbc_y=None):
        super().__init__()
        hidden_dim = _integer(hidden_dim, "hidden_dim", 1)
        if hidden_dim % 4:
            raise ValueError("Conv2DTransformerNet hidden_dim must be divisible by four heads")
        _setup(self, Lx, Ly, Nmax, pbc, {"hidden_dim": hidden_dim}, pbc_x, pbc_y)
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, hidden_dim, kernel_size=3, padding=0), nn.SiLU(),
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=2 * hidden_dim,
            activation="relu", dropout=0.0, batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_dim * self.n_sites, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = _input(self, x).transpose(1, 2).reshape(-1, self.channels, self.Ly, self.Lx)
        x = F.pad(x, (1, 1, 0, 0), mode="circular" if self.pbc_x else "constant")
        x = F.pad(x, (0, 0, 1, 1), mode="circular" if self.pbc_y else "constant")
        tokens = self.conv(x).flatten(2).transpose(1, 2)
        return self.fc(self.transformer(tokens)).squeeze(-1)


class GroupEquivariantConv2DNet(nn.Module):
    """Square-PBC D4 regular-representation CNN with invariant scalar readout.

    Features have shape(batch,8,hidden_dim,L,L), h=R**r S**m, index r+4*m.
    Cross-correlation uses K(h^-1 delta) for lifting and
    W(h^-1 delta,h^-1 a) for group convolution, including all8 input a.
    Shared orientation biases and pointwise SiLU preserve equivariance.
    Mean over space/orientation followed by Linear(hidden_dim,1) is invariant.
    No particle-number mask or extra position encoding is introduced.
    """
    def __init__(self, Lx, Ly=None, Nmax=4, pbc=True, hidden_dim=16, *, pbc_x=None, pbc_y=None):
        super().__init__()
        hidden_dim = _integer(hidden_dim, "hidden_dim", 1)
        _setup(self, Lx, Ly, Nmax, pbc, {"hidden_dim": hidden_dim}, pbc_x, pbc_y)
        if self.Lx != self.Ly or not self.pbc_x or not self.pbc_y:
            raise ValueError("GroupEquivariantConv2DNet requires a square periodic in both axes")
        self.register_buffer("kernel_pullback", kernel_pullbacks())
        self.register_buffer("relative_group", torch.tensor([
            [d4_product(d4_inverse(h), a) for a in D4_ELEMENTS] for h in D4_ELEMENTS], dtype=torch.long))
        self.lift_weight = nn.Parameter(torch.empty(hidden_dim, self.channels, 3, 3))
        self.lift_bias = nn.Parameter(torch.empty(hidden_dim))
        self.group_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim, 8, 3, 3))
        self.group_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.kaiming_uniform_(self.lift_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.group_weight, a=math.sqrt(5))
        nn.init.uniform_(self.lift_bias, -1 / math.sqrt(9 * self.channels), 1 / math.sqrt(9 * self.channels))
        nn.init.uniform_(self.group_bias, -1 / math.sqrt(72 * hidden_dim), 1 / math.sqrt(72 * hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)

    def lift_features(self, x):
        """One-hot(batch,sites,channels) -> SiLU lift(batch,8,h,L,L)."""
        x = _input(self, x).transpose(1, 2).reshape(-1, self.channels, self.Ly, self.Lx)
        h = self.lift_weight.shape[0]
        weights = self.lift_weight.flatten(-2)[..., self.kernel_pullback]
        weights = weights.permute(2, 0, 1, 3).reshape(8 * h, self.channels, 3, 3)
        lifted = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), weights, self.lift_bias.repeat(8))
        return F.silu(lifted).reshape(-1, 8, h, self.Ly, self.Lx)

    def group_features(self, features):
        """A true D4 group convolution+SiLU, mixing all relative orientations."""
        h = self.group_weight.shape[0]
        if features.ndim != 5 or tuple(features.shape[1:]) != (8, h, self.Ly, self.Lx):
            raise ValueError("features must have shape(batch,8,hidden_dim,L,L)")
        weights = self.group_weight.flatten(-2)[:, :, self.relative_group]
        indices = self.kernel_pullback[None, None, :, None, :].expand(h, h, 8, 8, 9)
        weights = weights.gather(-1, indices).permute(2, 0, 3, 1, 4).reshape(8 * h, 8 * h, 3, 3)
        result = F.conv2d(F.pad(features.flatten(1, 2), (1, 1, 1, 1), mode="circular"),
                          weights, self.group_bias.repeat(8))
        return F.silu(result).reshape(-1, 8, h, self.Ly, self.Lx)

    def forward_features(self, x):
        return self.group_features(self.lift_features(x))

    def forward(self, x):
        return self.head(self.forward_features(x).mean(dim=(1, 3, 4))).squeeze(-1)


BASELINE_ARCHITECTURES = ("FCNet", "JastrowNet", "Conv2DNet", "RBMNet", "TransformerNet")
GENERAL_ARCHITECTURES = (*BASELINE_ARCHITECTURES, "Conv2DTransformerNet")
ARCH_BUILDERS = {model.__name__: model for model in (FCNet, JastrowNet, Conv2DNet, RBMNet, TransformerNet, Conv2DTransformerNet, GroupEquivariantConv2DNet)}


def build_model(arch, Lx, Ly=None, Nmax=4, pbc=True, *, pbc_x=None, pbc_y=None, **settings):
    if arch not in ARCH_BUILDERS:
        raise ValueError(f"unknown architecture {arch!r}; choose from {list(ARCH_BUILDERS)}")
    return ARCH_BUILDERS[arch](Lx, Ly, Nmax, pbc, pbc_x=pbc_x, pbc_y=pbc_y, **settings)


def parameter_count(model):
    """Raw trainable parameter count, including Jastrow redundancies."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def spatial_symmetries(model):
    """Guaranteed symmetries; additional accidental ones depend on parameters."""
    if isinstance(model, GroupEquivariantConv2DNet):
        return ["x_periodic_translations", "y_periodic_translations", "D4_rotations_reflections"]
    if isinstance(model, JastrowNet):
        return (["x_reflection", "y_reflection"]
                + (["x_periodic_translations"] if model.pbc_x else [])
                + (["y_periodic_translations"] if model.pbc_y else []))
    return []


def save_model(path, model, Ntot=None, t=1.0, U=16.74, mu=0.0):
    """Save an inference checkpoint under 2D/ (relative paths start there).

    Stores no optimizer, random-generator, or walker state; not an exact training
    resume checkpoint. Path resolution rejects symlink escapes before writing.
    """
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("checkpoint output must stay under pythonTest/2D/")
    arch = type(model).__name__
    if arch not in ARCH_BUILDERS or type(model) is not ARCH_BUILDERS[arch]:
        raise ValueError("unsupported model type")
    Ntot = model.n_sites if Ntot is None else _integer(Ntot, "Ntot")
    if Ntot > model.n_sites * model.Nmax:
        raise ValueError("Ntot exceeds n_sites * Nmax")
    t, U, mu = (_physical_scalar(value, name)
                for name, value in (("t", t), ("U", U), ("mu", mu)))
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("inference checkpoints support float32 and float64")
    metadata = {
        "format_version": 2, "arch": arch,
        "Lx": model.Lx, "Ly": model.Ly, "Nmax": model.Nmax, "pbc": model.pbc,
        "pbc_x": model.pbc_x, "pbc_y": model.pbc_y,
        "architecture_settings": dict(model.architecture_settings),
        "dtype": str(dtype).removeprefix("torch."),
        "Ntot": Ntot, "t": t, "U": U, "mu": mu,
        "hamiltonian_convention": HAMILTONIAN_CONVENTION,
        "bond_convention": BOND_CONVENTION,
        "flattening": "i=y*Lx+x", "output": "real_unnormalized_log_amplitude",
        "checkpoint_kind": "inference",
        "parameter_count": parameter_count(model), "spatial_symmetries": spatial_symmetries(model),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, path)
    return path


def load_model(path, device="cpu"):
    """Return (eval-mode model, metadata), preserving saved floating dtype."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    expected = {
        "hamiltonian_convention": HAMILTONIAN_CONVENTION,
        "bond_convention": BOND_CONVENTION, "flattening": "i=y*Lx+x",
        "output": "real_unnormalized_log_amplitude", "checkpoint_kind": "inference",
    }
    if metadata.get("format_version") not in (1, 2) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported checkpoint format or physical convention")
    if metadata["format_version"] == 1:
        px, py = resolve_boundaries(metadata["pbc"])
        if any(key in metadata and metadata[key] != value for key, value in (("pbc_x", px), ("pbc_y", py))):
            raise ValueError("format-1 axes conflict with legacy pbc")
        metadata["pbc_x"], metadata["pbc_y"] = px, py
    else:
        if not all(key in metadata and isinstance(metadata[key], bool) for key in ("pbc_x", "pbc_y")):
            raise ValueError("format-2 checkpoints require explicit boolean axis boundaries")
        px, py = metadata["pbc_x"], metadata["pbc_y"]
        if "pbc" in metadata and metadata["pbc"] is not (px if px == py else None):
            raise ValueError("legacy pbc field conflicts with explicit axes")
    if metadata["arch"] == "FCNet":
        metadata["architecture_settings"].setdefault("hidden_layers", 2)
    if metadata["arch"] == "JastrowNet":
        metadata["architecture_settings"].setdefault("pair_sharing", "displacement")
    if metadata.get("dtype") not in ("float32", "float64"):
        raise ValueError("unsupported checkpoint dtype")
    for name in ("t", "U", "mu"):
        metadata[name] = _physical_scalar(metadata[name], name)
    metadata["Ntot"] = _integer(metadata["Ntot"], "Ntot")
    model = build_model(metadata["arch"], metadata["Lx"], metadata["Ly"],
                        metadata["Nmax"], pbc_x=px, pbc_y=py, **metadata["architecture_settings"])
    if metadata["Ntot"] > model.n_sites * model.Nmax:
        raise ValueError("Ntot exceeds n_sites * Nmax")
    model.to(device=device, dtype=getattr(torch, metadata["dtype"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata
