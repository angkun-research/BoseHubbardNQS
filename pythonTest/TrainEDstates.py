import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from BoseHubbardED import exact_ground_state, basis_to_one_hot
from NeuralNetworkClass import FCNet


def make_ed_dataset(L, Ntot, Nmax, t, U, mu=0.0, pbc=True):
    """
    Build training data from exact diagonalization.

    Returns:
        x: one-hot encoded basis states, shape (dim, L, Nmax+1)
        y: exact ground-state amplitudes, shape (dim,)
        energy: ground-state energy
        basis: list of basis configurations
        psi0: exact ground-state vector
    """
    energy, psi0, basis, H = exact_ground_state(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )

    x = basis_to_one_hot(basis, n_sites=L, nmax=Nmax)

    # ED eigenvectors are already normalized.
    # For this Bose-Hubbard Hamiltonian with real coefficients,
    # the ground state can be taken real.
    y = np.asarray(psi0, dtype=np.float32)

    return x, y, energy, basis, psi0


def train_fc_on_ed(
    L,
    Ntot,
    Nmax,
    t,
    U,
    mu=0.0,
    pbc=True,
    hidden_dim=64,
    lr=1e-3,
    epochs=2000,
    batch_size=128,
    device="cpu",
):
    x, y, energy, basis, psi0 = make_ed_dataset(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FCNet(n_sites=L, channels=Nmax + 1, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        total_loss /= len(dataset)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:4d}  loss={total_loss:.6e}")

    return model, {
        "ground_energy": energy,
        "basis": basis,
        "psi0": psi0,
        "x": x,
        "y": y,
    }


@torch.no_grad()
def compare_predictions(model, x, y, n_show=10, device="cpu"):
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    pred = model(x_tensor).cpu().numpy()

    print("First few exact vs predicted amplitudes:")
    for i in range(min(n_show, len(y))):
        print(f"{i:4d}   exact={y[i]: .8f}   pred={pred[i]: .8f}")

    mse = np.mean((pred - y) ** 2)
    print("full-basis MSE:", mse)

    overlap_num = np.dot(pred, y)
    overlap_den = np.linalg.norm(pred) * np.linalg.norm(y)
    if overlap_den > 0:
        print("normalized overlap:", overlap_num / overlap_den)

if __name__ == "__main__":
    L = 8
    Nmax = 8
    Ntot = 8
    t = 1.0
    U = 2.0
    mu = 0.0
    pbc = True

    model, data = train_fc_on_ed(
        L=L,
        Ntot=Ntot,
        Nmax=Nmax,
        t=t,
        U=U,
        mu=mu,
        pbc=pbc,
        hidden_dim=128,
        lr=1e-3,
        epochs=1000,
        batch_size=128,
        device="cpu",
    )

    compare_predictions(model, data["x"], data["y"], n_show=10)
    # test a single state
    teststate = [1,2,3,4,0,0,0,0]
    test_x = basis_to_one_hot([teststate], n_sites=L, nmax=Nmax)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_pred = model(test_x_tensor).item()
    print(f"Test state {teststate}  predicted amplitude: {test_pred:.8f}")