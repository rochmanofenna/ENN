# enn/model.py
import asyncio
import torch
import torch.nn as nn

from enn.memory           import ShortTermBuffer, state_decay, reset_neuron_state, temporal_proximity_scaling
from enn.state_collapse   import StateAutoEncoder, advanced_state_collapse
from enn.sparsity_control import low_power_state_collapse, dynamic_sparsity_control
from enn.scheduler        import PriorityTaskScheduler
from enn.attention        import probabilistic_path_activation


class ENNModelWithSparsityControl(nn.Module):
    """
    Minimal, gate-free debug ENN:
      • trainable entanglement mask (N×S)
      • trainable mixing matrix   (N×N) for time copying
      • small read-out            (S→S) to mix state features
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_layers   = cfg.num_layers
        self.num_neurons  = cfg.num_neurons
        self.num_states   = cfg.num_states
        self.decay_rate   = cfg.decay_rate
        self.recency_fact = cfg.recency_factor
        self.buffer_size  = cfg.buffer_size
        self.low_power_k  = cfg.low_power_k
        self.sparsity_threshold = cfg.sparsity_threshold

        # persistent state
        self.register_buffer("neuron_states",
                             torch.zeros(self.num_neurons, self.num_states))

        # trainable parameters
        self.entanglement = nn.Parameter(torch.randn(self.num_neurons,
                                                     self.num_states))
        self.mixing       = nn.Parameter(torch.eye(self.num_neurons))
        self.readout      = nn.Linear(self.num_states, self.num_states,
                                      bias=False)

        # helper modules
        self.short_buffers = [ShortTermBuffer(self.buffer_size)
                              for _ in range(self.num_neurons)]
        self.autoencoder   = StateAutoEncoder(self.num_states,
                                              cfg.compressed_dim)
        self.scheduler     = PriorityTaskScheduler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, N, S]
        device = x.device
        self.neuron_states = self.neuron_states.to(device)

        for _ in range(self.num_layers):
            # decay with pruning
            pruned  = dynamic_sparsity_control(self.neuron_states, self.sparsity_threshold)
            decayed = state_decay(pruned, self.decay_rate)
            self.neuron_states = reset_neuron_state(decayed)
            x = x * torch.sigmoid(self.entanglement).unsqueeze(0)
            # x = probabilistic_path_activation(x, activation_probability=0.1)
            x = torch.einsum("bns,nm->bms", x, self.mixing)
            self.neuron_states = x.mean(dim=0)

            # recency-weighted memory
            new_states = []
            for i, buf in enumerate(self.short_buffers):
                buf.add_to_buffer(self.neuron_states[i])
                rec = buf.get_recent_activations()
                if rec and all(r.size(-1) == self.num_states for r in rec):
                    rec_t = torch.stack(rec, dim=0).to(device)
                    new_states.append(
                        temporal_proximity_scaling(rec_t, self.recency_fact)
                    )
                else:
                    new_states.append(self.neuron_states[i])
            self.neuron_states = torch.stack(new_states, dim=0)

            # 5) (optional) collapse & low-power
            self.neuron_states = advanced_state_collapse(
                self.neuron_states, self.autoencoder, importance_threshold=0.0
            )
            self.neuron_states = low_power_state_collapse(
                self.neuron_states, top_k=self.low_power_k
            )

        # final read-out mixes state features
        return self.readout(x)

    async def async_process_event(self, neuron_state, data_input, prio: int = 1):
        self.scheduler.add_task(self.async_neuron_update(neuron_state,
                                                         data_input), prio)
        await self.scheduler.process_tasks()

    async def async_neuron_update(self, neuron_state, data_input, thr: float = 0.5):
        if data_input.mean().item() > thr:
            neuron_state.copy_(torch.sigmoid(data_input))
            await asyncio.sleep(0)
        return neuron_state

    def reset_memory(self):
        self.neuron_states.zero_()
        for buf in self.short_buffers:
            buf.buffer.clear()

