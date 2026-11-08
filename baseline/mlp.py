"""Standard baseline: a small float32 MLP (784-256-10, ReLU) trained with Adam on CPU (PyTorch)."""
import numpy as np
import torch
import torch.nn as nn

from common.bench import Timer, best_of


def train(xtr, ytr, hidden=256, epochs=5, batch=128, lr=1e-3, seed=0):
    """Returns (model, training seconds)."""
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(xtr.reshape(len(xtr), -1).astype(np.float32) / 255.0)
    Ytr = torch.from_numpy(ytr)
    model = nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    with Timer() as t:
        for _ in range(epochs):
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(Xtr), batch):
                b = perm[i:i + batch]
                opt.zero_grad()
                lossf(model(Xtr[b]), Ytr[b]).backward()
                opt.step()
    model.eval()
    return model, t.s


def run(xtr, ytr, xte, yte, hidden=256, epochs=5):
    model, train_s = train(xtr, ytr, hidden, epochs)
    Xte = torch.from_numpy(xte.reshape(len(xte), -1).astype(np.float32) / 255.0)

    def infer():
        with torch.no_grad():
            return model(Xte).argmax(1).numpy()

    pred, t_inf = best_of(infer)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": f"MLP 784-{hidden}-10 fp32 (Adam, {epochs} ep)",
        "acc": float((pred == yte).mean()),
        "train_s": train_s,
        "infer_s": t_inf,
        "mem_bytes": n_params * 4,
    }
