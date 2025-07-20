# train_synthetic.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from enn.model import ENNModelWithSparsityControl
from enn.config import Config

# 1) Generate synthetic data
#    Let's build N samples of length T, each with S states.
N, T, S = 500, 20, 5
# Create a mixture of sine + noise
t = np.linspace(0, 2*np.pi, T)
sinusoids = np.sin(t)[None, :, None]            # shape [1, T, 1]
X = sinusoids + 0.1 * np.random.randn(N, T, 1)  # shape [N, T, 1]
# Tile to S dimensions (so we have S features per time-step)
X = np.tile(X, (1, 1, S))                       # shape [N, T, S]

# Create a toy target: e.g. sum of last time-step features
y = X[:,-1,:].sum(axis=1)                       # shape [N,]

# 2) Wrap in a DataLoader
tensor_X = torch.tensor(X, dtype=torch.float32)  # [N, T, S]
tensor_y = torch.tensor(y, dtype=torch.float32)  # [N]
dataset = TensorDataset(tensor_X, tensor_y)
loader  = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# 3) Build your model, optimizer, loss
config = Config()  # now takes no args
config.input_dim = S  # Set input dimension to match data features
model  = ENNModelWithSparsityControl(config)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn= torch.nn.MSELoss()

# 4) Training loop
model.train()
for epoch in range(1, 6):             # 5 epochs
    total_loss = 0.0
    for batch_X, batch_y in loader:
        opt.zero_grad()
        out = model(batch_X)          # -> [batch, num_neurons, num_states]
        # simple read‐out: mean over neurons & states
        preds = out.mean(dim=(1,2))   # -> [batch]
        loss  = loss_fn(preds, batch_y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:>2}  loss = {total_loss/len(loader):.4f}")

# 5) Quick inference on new random input
model.eval()
with torch.no_grad():
    test_x = torch.tensor(np.random.randn(10, T, S), dtype=torch.float32)
    out    = model(test_x).mean(dim=(1,2))
    print("Test preds:", out.numpy())
