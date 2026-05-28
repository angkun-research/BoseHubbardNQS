import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from BoseHubbardED import exact_ground_state, basis_to_one_hot
from NeuralNetworkClass import FCNet,JastrowNet, Conv1DNet, RBMNet
from utils import pick_best_device

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

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:4d}  loss={loss.item():.6e}")

    return model

@torch.no_grad()
def compare_predictions(model, x, y, target="amplitude", device="cpu"):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor= torch.tensor(y, dtype=torch.float32, device=device).reshape(-1)
    pred_raw = model(x_tensor).reshape(-1) #.cpu().numpy()

    if target == "logabs":
        # logpred = pred_raw
        # pred_shift = logpred - np.max(logpred)
        # pred = np.exp(pred_shift)
        # pred /= np.linalg.norm(pred)

        # y_shift = y - np.max(y)
        # y_cmp = np.exp(y_shift)
        # y_cmp /= np.linalg.norm(y_cmp)
        
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
    eps = torch.finfo(psi_pred.dtype).eps
    denom = torch.linalg.vector_norm(psi_pred) * torch.linalg.vector_norm(psi_true)
    overlap = torch.dot(psi_pred, psi_true) / torch.clamp(denom, min=eps)

    print(f"full-basis MSE: {mse.item():.6e}")
    print(f"normalized overlap: {overlap.item():.6e}")
    return mse.item(), overlap.item(), logZ.item()

if __name__ == "__main__":
    DEVICE = pick_best_device()
    print(f"Using device: {DEVICE}")

    L = 8
    Nmax = 8
    Ntot = 8
    t = 1.0
    U = 2.0
    mu = 0.0
    pbc = True

    target = "logabs"  #"amplitude"

    x, y, energy, basis, psi0 = make_ed_dataset(
        L=L, Ntot=Ntot, Nmax=Nmax, t=t, U=U, mu=mu, pbc=pbc, target=target
    )

    hidden_dim = 128

    # Choose one: "FCNet", "JastrowNet", "Conv1DNet", "RBMNet"
    arch_name = "RBMNet"

    arch_builders = {
        "FCNet":       lambda: FCNet(n_sites=L, channels=Nmax + 1, hidden_dim=hidden_dim),
        "JastrowNet":  lambda: JastrowNet(n_sites=L, channels=Nmax + 1),
        "Conv1DNet":   lambda: Conv1DNet(n_sites=L, channels=Nmax + 1, hidden_dim=hidden_dim),
        "RBMNet":      lambda: RBMNet(n_sites=L, channels=Nmax + 1, n_hidden=hidden_dim),
    }

    print(f"Using architecture: {arch_name}")
    model = arch_builders[arch_name]()
    model = train_model_on_ed(
        model, x, y, target=target, lr=1e-3, epochs=10000, device=DEVICE
    )

    _, _, logZ = compare_predictions(model, x, y, target=target, device=DEVICE)
    # test a single state
    teststate = [1 for _ in range(L)]#[1,2,3,4,0,0,0,0]
    teststate[0] = 2
    def single_state_eval_logabs(teststate, logZ):
        state_x = basis_to_one_hot([teststate], n_sites=L, nmax=Nmax)
        state_x_tensor = torch.as_tensor(state_x, dtype=torch.float32, device=DEVICE)
        logpsi_raw = model(state_x_tensor).reshape(())

        # 3) Normalized quantities
        logamp_norm = logpsi_raw - 0.5 * logZ
        amp_norm = torch.exp(logamp_norm)
        prob = torch.exp(2.0 * logamp_norm)
        print(f"State: {teststate} amp_norm={amp_norm.item():.6e}  prob={prob.item():.6e}")

    teststate0 = [1 for _ in range(L)]
    teststate1 = teststate0.copy()
    teststate1[0] = 2
    teststate2 = teststate0.copy()
    teststate2[0] = 0
    single_state_eval_logabs(teststate0, logZ)
    single_state_eval_logabs(teststate1, logZ)
    single_state_eval_logabs(teststate2, logZ)