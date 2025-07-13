#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from enn.config import Config
from enn.model import ENNModelWithSparsityControl

def make_xor_data(N=4096, T=30):
    bits = np.random.randint(0, 2, (N, T, 1)).astype(np.float32)
    trg  = np.zeros_like(bits)
    trg[:, 3:] = np.logical_xor(bits[:, :-3], bits[:, 1:-2])
    return bits, trg

def main():
    cfg = Config("synthetic")
    cfg.num_layers   = 2
    cfg.num_neurons  = 30
    cfg.num_states   = 1
    batch_size       = 64

    X_np, Y_np = make_xor_data(N=4096, T=30)
    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    ds, loader = TensorDataset(X, Y), DataLoader(X, batch_size=batch_size, shuffle=True)

    model   = ENNModelWithSparsityControl(cfg)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    print(f"Training ENN on XOR-in-time: epochs={cfg.epochs}, batch={batch_size}")
    for epoch in range(1, cfg.epochs + 1):
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)                 # [B, T, 1]
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {epoch:02d}/{cfg.epochs}  BCE loss = {tot/len(loader):.6f}")

    # quick accuracy check
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.random.randint(0,2,(100,30,1)).astype(np.float32))
        preds = torch.sigmoid(model(xb)).round()
        print("Accuracy:", (preds==xb[...,0:1]).float().mean().item())

if __name__ == "__main__":
    main()
