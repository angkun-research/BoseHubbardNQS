import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from BoseHubbardED import exact_ground_state, basis_to_one_hot
from NeuralNetworkClass import FCNet,JastrowNet, Conv1DNet


def make_ed_dataset(L, Ntot, Nmax, t, U, mu=0.0, pbc=True, target="amplitude", eps=1e-12):
    energy, psi0, basis, H = exact_ground_state(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )
    x = basis_to_one_hot(basis, n_sites=L, nmax=Nmax)

    if target == "amplitude":
        y = np.asarray(psi0, dtype=np.float32)
    elif target == "logabs":
        y = np.log(np.abs(psi0) + eps).astype(np.float32)
    else:
        raise ValueError(f"unknown target mode: {target}")

    return x, y, energy, basis, psi0


def train_model_on_ed(
    model,
    x,
    y,
    target="amplitude",
    lr=1e-3,
    epochs=2000,
    batch_size=128,
    device="cpu",
):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
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

    return model

@torch.no_grad()
def compare_predictions(model, x, y, target="amplitude", device="cpu"):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    pred_raw = model(x_tensor).cpu().numpy()

    if target == "logabs":
        pred = np.exp(pred_raw)
        pred /= np.linalg.norm(pred)
        y_cmp = np.exp(y)
        y_cmp /= np.linalg.norm(y_cmp)
    else:
        pred = pred_raw
        y_cmp = y

    mse = np.mean((pred - y_cmp) ** 2)
    overlap = np.dot(pred, y_cmp) / (np.linalg.norm(pred) * np.linalg.norm(y_cmp))

    print("full-basis MSE:", mse)
    print("normalized overlap:", overlap)

if __name__ == "__main__":
    L = 8
    Nmax = 8
    Ntot = 8
    t = 1.0
    U = 2.0
    mu = 0.0
    pbc = True

    model_kind = "fc" #"jastrow"   # or "fc"
    target = "logabs" if model_kind == "jastrow" else "amplitude"

    x, y, energy, basis, psi0 = make_ed_dataset(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc, target=target
    )

    hidden_dim = 128
    #model = FCNet(n_sites=L, channels=Nmax + 1, hidden_dim=hidden_dim)
    model = Conv1DNet(n_sites=L, channels=Nmax + 1, hidden_dim=hidden_dim)
    #model = JastrowNet(n_sites=L, channels=Nmax + 1)

    model = train_model_on_ed(
        model, x, y, target=target, lr=1e-3, epochs=1000, batch_size=128
    )

    compare_predictions(model, x, y, target=target)
    # test a single state
    teststate = [1,2,3,4,0,0,0,0]
    test_x = basis_to_one_hot([teststate], n_sites=L, nmax=Nmax)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_pred = model(test_x_tensor).item()
    print(f"Test state {teststate}  predicted amplitude: {test_pred:.8f}")