# enn/model.py
import asyncio, torch
import torch.nn as nn

from enn.memory           import ShortTermBuffer, state_decay, reset_neuron_state, temporal_proximity_scaling
from enn.state_collapse   import StateAutoEncoder, advanced_state_collapse
from enn.sparsity_control import dynamic_sparsity_control, low_power_state_collapse
from enn.scheduler        import PriorityTaskScheduler
from enn.validation       import validate_config, validate_tensor_dimensions, handle_device_mismatch


class ENNModelWithSparsityControl(nn.Module):
    """Entangled-NN core with memory + adaptive sparsity."""

    def __init__(self, cfg):
        super().__init__()
        
        # Validate configuration
        validate_config(cfg)
        
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
        
        # ── input projection for temporal data ───────────────────
        # Projects [batch, time, features] to [batch, num_neurons, num_states]
        self.input_projection = nn.Linear(cfg.input_dim if hasattr(cfg, 'input_dim') else self.num_states, 
                                         self.num_neurons * self.num_states)

        # ── helpers ──────────────────────────────────────────────
        self.short_buffers = [ShortTermBuffer(self.buffer_size)
                              for _ in range(self.num_neurons)]
        self.autoencoder   = StateAutoEncoder(self.num_states,
                                              cfg.compressed_dim)
        self.scheduler     = PriorityTaskScheduler()

    # ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Can be either:
           - [batch, num_neurons, num_states] (direct ENN format)
           - [batch, time_steps, features] (temporal data)
        """
        dev = x.device
        self.neuron_states = self.neuron_states.to(dev)
        
        # Validate and handle input tensor
        if x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D")
            
        # Handle temporal input: [batch, time, features] -> [batch, num_neurons, num_states]
        if x.dim() == 3 and x.size(1) != self.num_neurons:
            # Assume temporal format [batch, time, features]
            batch_size, time_steps, features = x.shape
            if time_steps == 0:
                raise ValueError("Time dimension cannot be zero")
            # Take the last timestep and project to neuron space
            x_last = x[:, -1, :]  # [batch, features]
            x_projected = self.input_projection(x_last)  # [batch, num_neurons * num_states]
            x = x_projected.view(batch_size, self.num_neurons, self.num_states)
        elif x.dim() == 2:
            # Assume [batch, features] format, add neuron dimension
            batch_size, features = x.shape
            x_projected = self.input_projection(x)
            x = x_projected.view(batch_size, self.num_neurons, self.num_states)

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
    
    async def async_process_event(self, neuron_state: torch.Tensor, 
                                 input_data: torch.Tensor, priority: int):
        """
        Asynchronous event processing for individual neuron states.
        
        Args:
            neuron_state: Current state of the neuron [num_states]
            input_data: Input data for the event [num_states] 
            priority: Priority level for the event (higher = more important)
        """
        import asyncio
        
        # Add event to scheduler with priority
        await self.scheduler.add_task(
            task_id=f"neuron_update_{id(neuron_state)}", 
            priority=priority,
            data=input_data
        )
        
        # Process the update asynchronously
        await asyncio.sleep(0.001)  # Simulate async processing
        
        # Update neuron state with decay and input
        decayed_state = state_decay(neuron_state, self.decay_rate)
        updated_state = decayed_state + 0.1 * input_data
        
        # Apply sparsity control
        sparse_state = dynamic_sparsity_control(updated_state, self.sparsity_thr)
        
        # Update the global neuron states (ensure proper dimensions)
        if sparse_state.dim() == 1 and self.neuron_states.dim() == 2:
            self.neuron_states[0] = sparse_state
        else:
            # If dimensions don't match, take the mean over batch/time dimensions
            if sparse_state.dim() > 1:
                sparse_state = sparse_state.mean(0)
            self.neuron_states[0] = sparse_state[:self.num_states]
        
        return sparse_state
    
    def reset_memory(self):
        """Reset all neuron states and memory buffers."""
        self.neuron_states.zero_()
        for buffer in self.short_buffers:
            buffer.buffer.clear()
