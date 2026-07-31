import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

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
    

class JastrowNet(nn.Module):
    def __init__(self, n_sites, channels):
        super().__init__()
        self.n_sites = n_sites
        self.channels = channels
        self.max_dist = n_sites // 2
        self.onsite = nn.Parameter(torch.zeros(channels))
        self.pair = nn.Parameter(torch.zeros(self.max_dist + 1, channels, channels))

    def forward(self, x):
        levels = torch.arange(self.channels, device=x.device, dtype=x.dtype)
        occ = (x * levels).sum(dim=-1).long()

        logpsi = self.onsite[occ].sum(dim=1)

        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                d = min(j - i, self.n_sites - (j - i))
                logpsi = logpsi + self.pair[d, occ[:, i], occ[:, j]]
        #psi = torch.exp(logpsi)
        return logpsi
    
class Conv1DNet(nn.Module):
    def __init__(self, n_sites, channels, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular'),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * n_sites, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, n_sites, Channels)
        x = x.transpose(1, 2)  # (batch, Channels, n_sites)
        x = self.conv(x)  # (batch, 1)
        return x.squeeze(-1)


class RBMNet(nn.Module):
    """
    Simple real-valued RBM wavefunction.
    Input x is expected to be one-hot encoded with shape
        (batch, n_sites, channels)
    """
    def __init__(self, n_sites,  channels, n_hidden):
        super().__init__()
        self.n_sites = n_sites
        self.channels = channels
        self.n_visible = n_sites * channels
        self.n_hidden = n_hidden

        self.visible_bias = nn.Parameter(torch.zeros(self.n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(self.n_hidden))
        self.W = nn.Parameter(0.01 * torch.randn(self.n_visible, self.n_hidden))
    
    def free_energy(self, v):
        """
        v: (batch, n_visible), binary or real-valued visible units
        returns: (batch,)
        """
        linear_term = v @ self.visible_bias
        hidden_term = F.softplus(self.hidden_bias + v @ self.W).sum(dim=1)
        return -(linear_term + hidden_term)

    def forward(self, v):
        """
        Returns an unnormalized log-score.
        Larger value means higher probability up to a constant shift.
        """
        # flatten (batch, n_sites, channels) -> (batch, n_visible)
        v = v.view(v.size(0), -1)
        prob_unnorm = -self.free_energy(v) #torch.exp(-self.free_energy(v))
        return prob_unnorm
    

def save_model(path, model, arch, n_sites, nmax, hidden_dim=128, extra=None):
    ckpt = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "n_sites": int(n_sites),
        "nmax": int(nmax),
        "hidden_dim": int(hidden_dim),
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)
    print(f"saved model to {Path(path).resolve()}")



class Conv1DPoolNet(nn.Module):
    def __init__(self, n_sites, channels, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular'),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular'),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular'),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, n_sites, Channels)
        x = x.transpose(1, 2)  # (batch, Channels, n_sites)
        x = self.conv(x)  # (batch, 1)
        return x.squeeze(-1)

    

class TransformerNet(nn.Module):
    def __init__(self, n_sites, channels, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(channels, hidden_dim)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            activation='relu',
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * n_sites, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)     # [B, n_sites, hidden_dim]
        x = self.transformer(x)    # [B, n_sites, hidden_dim]
        x = self.fc(x)            # [B, 1]
        return x.squeeze(-1)

class ConvTransformer(nn.Module):
    def __init__(self, n_sites, channels, hidden_dim=256, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode='circular',
            ),
            nn.SiLU(),
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            activation='relu',
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * n_sites, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, n_sites, Channels)
        x = x.transpose(1, 2)  # (batch, Channels, n_sites)
        x = self.conv(x)          # [B, hidden_dim, L]
        x = x.transpose(1, 2)   # [B, L, hidden_dim]
        x = self.transformer(x)   # [B, L, hidden_dim]
        x = self.fc(x) 
        return x.squeeze(-1)

class ViTNet(nn.Module):
    """
    Minimal Vision-Transformer-like model for NQS.

    Input:
      x with shape [B, L, C] (one-hot occupations)
      or shape [B, L] (integer occupations 0..Nmax)

    Output:
      scalar per sample:
        - log-amplitude if return_log=True
        - amplitude if return_log=False
    """
    def __init__(
        self,
        n_sites: int,
        channels: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.channels = channels

        self.token_embed = nn.Linear(self.channels, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_sites + 1, hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
            enable_nested_tensor=False)  # disable nested tensor for compatibility

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_log: bool = True):
        B, L, C = x.shape

        tok = self.token_embed(x)                      # [B, L, d_model]
        cls = self.cls_token.expand(B, -1, -1)        # [B, 1, d_model]
        h = torch.cat([cls, tok], dim=1) + self.pos_embed[:, :L+1, :]
        h = self.encoder(h)                            # [B, L+1, d_model]
        log_amp = self.head(h[:, 0]).squeeze(-1)      # use CLS token

        return log_amp