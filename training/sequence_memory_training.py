#!/usr/bin/env python
# training/sequence_copy_profiler.py
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from enn.config import Config
from enn.model  import ENNModelWithSparsityControl


DEBUG_NO_GATES = True
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------- #
# Synthetic copy-task generator
# -------------------------------------------------------------------- #
def make_seq_copy_data(N=1024, T=15, S=5):
    seq = np.random.randn(N, T, S).astype(np.float32)
    pad = np.zeros_like(seq)
    X   = np.concatenate([seq, pad], axis=1)   # input: seq + zeros
    Y   = np.concatenate([pad, seq], axis=1)   # target: zeros + seq
    return X, Y


# -------------------------------------------------------------------- #
# One training epoch
# -------------------------------------------------------------------- #
def train_epoch(loader, model, opt, loss_fn):
    model.train()
    tot_loss, tot_gnorm = 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out  = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        tot_gnorm += torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        tot_loss  += loss.item()
    return tot_loss / len(loader), tot_gnorm / len(loader)


# -------------------------------------------------------------------- #
def main():
    # -------- config & data -------- #
    cfg = Config("synthetic")
    cfg.num_layers  = 2
    N, T, S         = 1024, 200, 5
    cfg.num_neurons = 2 * T
    cfg.num_states  = S
    cfg.buffer_size = int(0.3 * T)
    cfg.epochs      = 114  # Standardized training period
    batch_size      = cfg.batch_size  # Use config batch size

    X_np, Y_np = make_seq_copy_data(N, T, S)
    X_np = (X_np - X_np.mean()) / (X_np.std() + 1e-6)
    Y_np = (Y_np - Y_np.mean()) / (Y_np.std() + 1e-6)

    ds     = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(Y_np))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # -------- model -------- #
    model = ENNModelWithSparsityControl(cfg).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    # optional dense baseline
    if DEBUG_NO_GATES:
        base = nn.Linear(cfg.num_neurons * cfg.num_states,
                         cfg.num_neurons * cfg.num_states).to(DEVICE)
        base_opt = torch.optim.Adam(base.parameters(), lr=1e-3)

    loss_hist = []
    t0 = time.perf_counter()

    # -------- profiler -------- #
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs"),
        record_shapes=True,
        with_stack=True
    ) as prof:

        for epoch in range(1, cfg.epochs + 1):
            with record_function("epoch"):
                mse, gnorm = train_epoch(loader, model, opt, loss_fn)

            # optional baseline update
            if DEBUG_NO_GATES:
                base_opt.zero_grad()
                flat_x = next(iter(loader))[0].reshape(batch_size, -1).to(DEVICE)
                flat_y = next(iter(loader))[1].reshape(batch_size, -1).to(DEVICE)
                base_loss = loss_fn(base(flat_x), flat_y)
                base_loss.backward(); base_opt.step()

            # logging
            spars = (model.neuron_states != 0).float().mean().item()
            loss_hist.append(mse)
            print(f"Ep {epoch:03d}/{cfg.epochs}  "
                  f"MSE={mse:8.2e}  g={gnorm:6.2e}  active={spars:.3f}")

            prof.step()               # advance profiler schedule

    # -------- quick visual check -------- #
    model.eval()
    with torch.no_grad():
        pred = model(ds[0][0].unsqueeze(0).to(DEVICE))[0].cpu()

    plt.plot(ds[0][1][:, 0], label="target")
    plt.plot(pred[:, 0], '--', label="pred")
    plt.title("Sequence-copy (state-0)")
    plt.legend(); plt.show()

    plt.figure()
    plt.plot(loss_hist); plt.yscale("log"); plt.title("Training loss"); plt.show()

    print(f"Total wall-time: {time.perf_counter() - t0:.1f}s")


# -------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
