import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from pythonTest.NeuralNetworkClass import basis_to_one_hot
from pythonTest.NeuralNetworkClass import FCNet, Conv1DNet, JastrowNet, RBMNet

MODEL_PATH = Path("data/bose_hubbard_nqs.pt")

ARCH_BUILDERS = {
    "FCNet": lambda L, C, H: FCNet(n_sites=L, channels=C, hidden_dim=H),
    "JastrowNet": lambda L, C, H: JastrowNet(n_sites=L, channels=C),
    "Conv1DNet": lambda L, C, H: Conv1DNet(n_sites=L, channels=C, hidden_dim=H),
    "RBMNet": lambda L, C, H: RBMNet(n_sites=L, channels=C, n_hidden=H),
}

def build_model(arch, n_sites, nmax, hidden_dim=128):
    channels = nmax + 1
    if arch not in ARCH_BUILDERS:
        raise ValueError(f"unknown architecture: {arch}")
    model = ARCH_BUILDERS[arch](n_sites, channels, hidden_dim)
    model.arch = arch
    model.n_sites = n_sites
    model.nmax = nmax
    model.hidden_dim = hidden_dim
    return model


def load_model(path=MODEL_PATH, arch=None, n_sites=None, nmax=None, hidden_dim=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    # New style: full checkpoint with metadata.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        arch_f = arch or ckpt.get("arch", None)
        n_sites_f = n_sites if n_sites is not None else ckpt.get("n_sites", None)
        nmax_f = nmax if nmax is not None else ckpt.get("nmax", None)
        hidden_f = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 128)

        model = build_model(arch=arch_f, n_sites=int(n_sites_f), nmax=int(nmax_f), hidden_dim=int(hidden_f))
        model.load_state_dict(sd)
        model.eval()
        return model

    raise ValueError("unsupported checkpoint format")

@torch.no_grad()
def predict_scalar(model, occupation_config, nmax=None):
    cfg = [int(v) for v in occupation_config]
    n_sites = len(cfg)

    if nmax is None:
        if not hasattr(model, "nmax"):
            raise ValueError("nmax not provided and not found on model")
        nmax = int(model.nmax)

    if any(v < 0 or v > nmax for v in cfg):
        raise ValueError(f"occupation values must be within [0, {nmax}]")

    x_np = basis_to_one_hot([tuple(cfg)], n_sites=n_sites, nmax=int(nmax))
    x = torch.tensor(x_np, dtype=torch.float32)
    y = model(x)
    return float(y.item())