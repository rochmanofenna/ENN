# enn/model.py
import asyncio, torch
import torch.nn as nn

from enn.memory           import ShortTermBuffer, state_decay, reset_neuron_state, temporal_proximity_scaling
from enn.state_collapse   import StateAutoEncoder, advanced_state_collapse
from enn.sparsity_control import dynamic_sparsity_control, low_power_state_collapse
from enn.scheduler        import PriorityTaskScheduler


class ENNModelWithSparsityControl(nn.Module):
    """Entangled-NN core with memory + adaptive sparsity."""

    def __init__(self, cfg):
        super().__init__()
        # ── hyper-params ──────────────────────────────────────────
        self.num_layers   = cfg.num_layers
        self.num_neurons  = cfg.num_neurons
        self.num_states   = cfg.num_states
        self.decay_rate   = cfg.decay_rate
        self.recency_fact = cfg.recency_factor
        self.buffer_size  = cfg.buffer_size
        self.low_power_k  = cfg.low_power_k
        self.sparsity_thr = cfg.sparsity_threshold
        self.l1_lambda    = getattr(cfg, "l1_lambda", 1e-4)

        # ── persistent state ─────────────────────────────────────
        self.register_buffer("neuron_states",
                             torch.zeros(self.num_neurons, self.num_states))

        # ── learnable parameters ─────────────────────────────────
        self.entanglement = nn.Parameter(torch.randn(self.num_neurons,
                                                     self.num_states))
        self.mixing  = nn.Parameter(torch.eye(self.num_neurons))
        self.readout = nn.Linear(self.num_states, self.num_states, bias=False)

        # ── helpers ──────────────────────────────────────────────
        self.short_buffers = [ShortTermBuffer(self.buffer_size)
                              for _ in range(self.num_neurons)]
        self.autoencoder   = StateAutoEncoder(self.num_states,
                                              cfg.compressed_dim)
        self.scheduler     = PriorityTaskScheduler()

    # ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, N, S]"""
        dev = x.device
        self.neuron_states = self.neuron_states.to(dev)

        # L1 regulariser on mask (returned for loss term)
        self.mask_l1 = self.l1_lambda * torch.sigmoid(self.entanglement).mean()

        for _ in range(self.num_layers):
            # 1) prune + decay
            self.neuron_states = dynamic_sparsity_control(
                self.neuron_states, self.sparsity_thr)
            self.neuron_states = state_decay(
                self.neuron_states, self.decay_rate)

            # 2) entangle + mix
            mask = torch.sigmoid(self.entanglement).unsqueeze(0)  # [1,N,S]
            x    = x * mask
            x    = torch.einsum("bns,nm->bms", x, self.mixing)
            self.neuron_states = x.mean(0)                        # update mem

            # 3) recency-weighted memory (vectorised)
            buf_stack = []
            for i, buf in enumerate(self.short_buffers):
                buf.add_to_buffer(self.neuron_states[i])
                acts = buf.get_recent_activations()
                if acts and all(a.size(-1) == self.num_states for a in acts):
                    buf_stack.append(torch.stack(acts, 0))  # [L,S]
                else:
                    buf_stack.append(self.neuron_states[i].unsqueeze(0))
            buf_stack  = torch.nn.utils.rnn.pad_sequence(buf_stack,
                                                         batch_first=True)
            L          = buf_stack.size(1)
            weights    = self.recency_fact ** torch.arange(
                            L - 1, -1, -1, device=dev).view(1, L, 1)
            self.neuron_states = (buf_stack * weights).sum(1) / weights.sum(1)

            # 4) collapse + low-power
            self.neuron_states = advanced_state_collapse(
                self.neuron_states, self.autoencoder, importance_threshold=0.)
            self.neuron_states = low_power_state_collapse(
                self.neuron_states, top_k=self.low_power_k)

        return self.readout(x)
