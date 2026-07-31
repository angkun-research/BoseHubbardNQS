import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from BoseHubbardED import exact_ground_state, basis_to_one_hot
from NeuralNetworkClass import Conv1DPoolNet
from utils import pick_best_device

def make_ed_dataset(L, Ntot, Nmax, t, U, mu=0.0, pbc=True, target="amplitude", eps=1e-12):
    energy, psi0, basis, H = exact_ground_state(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc
    )
    print(f"ED ground state energy: {energy:.6e}")
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
    # batch_size=128,
    device="cpu",
):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # dataset = TensorDataset(x_tensor, y_tensor)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        if target == "logabs":
            logpsi = model(x_tensor)
            pred = logpsi - 0.5 * torch.logsumexp(2.0 * logpsi, dim=0)

            y_norm = y_tensor - 0.5 * torch.logsumexp(2.0 * y_tensor, dim=0)
            loss = loss_fn(pred, y_norm)
        else:
            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0  or epoch == epochs - 1:
            print(f"epoch={epoch:4d}  loss={loss.item():.6e}")

    return model

@torch.no_grad()
def compare_predictions(model, x, y, target="amplitude", device="cpu"):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor= torch.tensor(y, dtype=torch.float32, device=device).reshape(-1)
    pred_raw = model(x_tensor).reshape(-1) #.cpu().numpy()

    if target == "logabs":
        logZ = torch.logsumexp(2.0 * pred_raw, dim=0)
        pred_cmp = pred_raw - 0.5 * torch.logsumexp(2.0 * pred_raw, dim=0)
        y_cmp = y_tensor - 0.5 * torch.logsumexp(2.0 * y_tensor, dim=0)
        psi_pred = torch.exp(pred_cmp)
        psi_true = torch.exp(y_cmp)
        pred_norm = torch.exp(0.5 * torch.logsumexp(2.0 * pred_raw, dim=0))
    else:
        pred_cmp = pred_raw
        y_cmp = y_tensor
        psi_pred = pred_cmp
        psi_true = y_cmp
        pred_norm = torch.linalg.vector_norm(pred_raw)
        logZ = torch.zeros(1, device=device)  # not used in this case

    mse = torch.mean((pred_cmp - y_cmp) ** 2)
    # norm_pred = np.linalg.norm(pred)
    # overlap = np.dot(pred, y_cmp) / (norm_pred * np.linalg.norm(y_cmp))
    overlap = torch.dot(psi_pred, psi_true)

    print(f"full-basis MSE: {mse.item():.6e}")
    print(f"normalized overlap: {overlap.item():.6e}")
    return mse.item(), overlap.item(), logZ.item()

if __name__ == "__main__":
    DEVICE = pick_best_device()
    print(f"Using device: {DEVICE}")

    L = 8 #8
    Nmax = 4 #8
    Ntot = L #8
    t = 1.0
    U_model = 3.3578 #4.0 #3.3578
    U = U_model/2 #2.0
    mu = 0.0
    pbc = True

    target = "logabs"  #"amplitude"

    x, y, energy, basis, psi0 = make_ed_dataset(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc, target=target
    )

    hidden_dim = 128

    arch_name = "Conv1DPoolNet"
    print(f"Using architecture: {arch_name}")
    model = Conv1DPoolNet(n_sites=L, channels=Nmax + 1, hidden_dim=hidden_dim)
    model = train_model_on_ed(
        model, x, y, target=target, lr=1e-3, epochs=10000, device=DEVICE
    )
    print(f"Fitting results for L={L}")
    _, _, logZ = compare_predictions(model, x, y, target=target, device=DEVICE)

    print(f"Testing on larger system size L={L+2}")
    x_test, y_test, energy_test, basis_test, psi0_test = make_ed_dataset(
        L=L+2, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc, target=target
    )
    _, _, logZ_test = compare_predictions(model, x_test, y_test, target=target, device=DEVICE)