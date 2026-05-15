import torch
import torch.nn as nn
import numpy as np

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

class FCNet(nn.Module):
    def __init__(self, n_sites: int, channels: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = n_sites * channels
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, n_sites, Channels), entries are occupations 0,1,...,Nmax
        x = x.view(x.size(0), -1)  # flatten the input
        return self.net(x.float()).squeeze(-1)