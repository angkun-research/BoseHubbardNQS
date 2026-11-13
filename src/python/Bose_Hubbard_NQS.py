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


def load_model(path=MODEL_PATH, arch=None, n_sites=None, nmax=None, hidden_dim=None, map_location="cpu", device=None):
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and isinstance(ckpt.get("metadata"), dict):
        raise ValueError("2D checkpoint: use load_nn_2d instead of load_nn")

    # New style: full checkpoint with metadata.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        arch_f = arch or ckpt.get("arch", None)
        n_sites_f = n_sites if n_sites is not None else ckpt.get("n_sites", None)
        nmax_f = nmax if nmax is not None else ckpt.get("nmax", None)
        hidden_f = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 128)

        model = build_model(arch=arch_f, n_sites=int(n_sites_f), nmax=int(nmax_f), hidden_dim=int(hidden_f))
        model.load_state_dict(sd)
        if device is not None:
            model.to(resolve_device(device))
        model.eval()
        return model

    raise ValueError("unsupported checkpoint format")

# Transport keeps the 1D float32 forward contract unchanged.
from bosehubbard_nqs_inference import resolve_device, predict_scalar, predict_values
