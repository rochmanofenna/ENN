#!/usr/bin/env python
# training/sequence_copy_profiler.py

import math, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data   import TensorDataset, DataLoader, random_split
from torch.profiler     import profile, ProfilerActivity
from torch.optim.lr_scheduler import CosineAnnealingLR

from enn.config import Config
from enn.model  import ENNModelWithSparsityControl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_seq_copy_data(N=1024, T=15, S=5):
    seq = np.random.randn(N, T, S).astype(np.float32)
    pad = np.zeros_like(seq)
    return np.concatenate([seq, pad], axis=1), np.concatenate([pad, seq], axis=1)

def sparsity_schedule(epoch, total):
    return 0.05 * min(1.0, epoch / (0.8 * total))

def k_schedule(epoch, total, S):
    return max(1, math.ceil((0.6 - 0.4*(epoch/total)) * S))

def train_epoch(loader, model, opt, loss_fn, max_grad_norm=0.5):
    model.train()
    tot_loss, tot_gnorm = 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE).double(), yb.to(DEVICE).double()
        opt.zero_grad()

        out  = model(xb)
        mse  = loss_fn(out, yb)

# inside train_epoch, after you get `mse = loss_fn(out, yb)`:
        loss = mse + model.mask_l1 + model.thr_l1
        loss.backward()

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        tot_gnorm += gn
        opt.step()
        tot_loss  += mse.item()

    return tot_loss / len(loader), tot_gnorm / len(loader)

def eval_loss(loader, model, loss_fn):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE).double(), yb.to(DEVICE).double()
            tot += loss_fn(model(xb), yb).item()
    return tot / len(loader)

def main():
    cfg = Config()
    cfg.num_layers         = 2
    cfg.num_states         = 5
    cfg.num_neurons        = 2*200
    cfg.buffer_size        = int(0.3*200)
    cfg.epochs             = 114
    cfg.sparsity_threshold = 0.0
    cfg.l1_lambda          = 1e-4

    X, Y = make_seq_copy_data(1024, 200, cfg.num_states)
    X = (X - X.mean()) / (X.std() + 1e-6)
    Y = (Y - Y.mean()) / (Y.std() + 1e-6)

    full_ds   = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    n_val     = int(0.1 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    model   = ENNModelWithSparsityControl(cfg).to(DEVICE).double()
    opt     = torch.optim.Adam(model.parameters(), lr=3e-3, eps=1e-8)
    lr_sched= CosineAnnealingLR(opt, T_max=int(0.6*cfg.epochs), eta_min=3e-4)
    loss_fn = nn.MSELoss()

    best_val, wait, patience = float("inf"), 0, 10
    loss_hist, val_hist = [], []

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU],
                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                 on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs"),
                 record_shapes=False, with_stack=False) as prof:

        for ep in range(1, cfg.epochs+1):
            # —–– Instead of reassigning, update the param’s .data (or drop this entirely)
            with torch.no_grad():
                targ = sparsity_schedule(ep, cfg.epochs)
                model.sparsity_thr.data.copy_(torch.logit(torch.tensor(
                    targ, device=model.sparsity_thr.device)))
    
            # train + L1 penalties
            mse, gnorm = train_epoch(train_loader, model, opt, loss_fn)
            loss_hist.append(mse)
            # step LR
            if mse > 1e-6:
                lr_sched.step()
    
            # validate + early stop …
            val_mse = eval_loss(val_loader, model, loss_fn)
            # … your existing checkpoint/stop logic …
    
            # logging
            lr     = opt.param_groups[0]["lr"]
            active = (model.neuron_states != 0).float().mean().item()
            print(f"Ep {ep:03d}/{cfg.epochs}"
                  f"  train={mse:.2e}  val={val_mse:.2e}"
                  f"  g={gnorm:.2e}  lr={lr:.1e}"
                  f"  active={active:.3f}"
                  f"  thr={torch.sigmoid(model.sparsity_thr).item():.3f}"
                  f"  k={model.low_power_k}")
    
            prof.step()
    

    best_val, best_epoch = float("inf"), 0
    
    if val_mse < best_val:
        best_val, best_epoch = val_mse, ep
        torch.save(model.state_dict(), "best_enn.pt")
        wait = 0
    else:
        wait += 1

    print(f"\nBest val MSE={best_val:.2e} in {wait} epochs")
    print(f"Total time: {time.perf_counter()-t0:.1f}s")

    with torch.no_grad():
        sample_x, sample_y = full_ds[0]
        pred = model(sample_x.unsqueeze(0).to(DEVICE).double())[0].cpu()
    
    # final plots…
    plt.figure(); plt.plot(loss_hist, label="train"); plt.plot(val_hist, label="val")
    plt.yscale("log"); plt.legend(); plt.title("MSE history"); plt.show()
    sample_x, sample_y = full_ds[0]
    with torch.no_grad():
        # invert the sigmoid so that sigmoid(thr.data) ≈ target_schedule
        target = sparsity_schedule(ep, cfg.epochs)
        model.sparsity_thr.data.copy_(torch.logit(torch.tensor(target, device=model.sparsity_thr.device)))
    
    plt.figure(); plt.plot(sample_y[:,0], label="tgt"); plt.plot(pred[:,0],"--",label="pred")
    plt.legend(); plt.show()

if __name__ == "__main__":
    main()
