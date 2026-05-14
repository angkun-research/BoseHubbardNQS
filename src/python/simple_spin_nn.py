# simple_spin_nn.py
import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path("data/spin_net.pt")

class SpinNet(nn.Module):
    def __init__(self, n_sites: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sites, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: shape (batch, n_sites)
        return self.net(x).squeeze(-1)


def make_toy_dataset(n_samples: int, n_sites: int):
    # Random 0/1 spin configurations.
    x = torch.randint(0, 2, (n_samples, n_sites), dtype=torch.float32)

    # Toy scalar target:
    # replace this with your exact amplitude / log-amplitude / local observable target.
    magnetization = 2.0 * x.sum(dim=1) - n_sites
    y = (magnetization / n_sites) ** 2

    return x, y


def train_and_save(n_sites: int = 10, hidden_dim: int = 32, epochs: int = 2000, lr: float = 1e-3):
    model = SpinNet(n_sites=n_sites, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_train, y_train = make_toy_dataset(n_samples=4000, n_sites=n_sites)

    for epoch in range(epochs):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"epoch={epoch:4d}  loss={loss.item():.6e}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_sites": n_sites,
            "hidden_dim": hidden_dim,
        },
        MODEL_PATH,
    )
    print(f"saved model to {MODEL_PATH.resolve()}")


def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu")
    model = SpinNet(
        n_sites=checkpoint["n_sites"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_scalar(model, spin_config):
    # spin_config: Python list or 1D array of 0/1 values
    x = torch.tensor(spin_config, dtype=torch.float32).unsqueeze(0)
    y = model(x)
    return float(y.item())


if __name__ == "__main__":
    train_and_save()
    model = load_model(MODEL_PATH)
    test_cfg = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    print("test config:", test_cfg)
    print("network output:", predict_scalar(model, test_cfg))