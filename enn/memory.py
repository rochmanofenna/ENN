import torch
import torch.nn as nn
import math
from collections import deque

def state_decay(neuron_state, decay_rate=0.1):
    """
    Applies decay to a neuron's state to reduce the impact of older data.

    Parameters:
    - neuron_state: Tensor representing the neuron's current state.
    - decay_rate: Rate at which the state decays over time.

    Returns:
    - Decayed neuron state.
    """
    factor = math.exp(-decay_rate)
    return neuron_state * factor

def reset_neuron_state(neuron_state):
    """
    Resets the neuron's state, effectively 'forgetting' outdated entanglements.

    Parameters:
    - neuron_state: Tensor representing the neuron's current state.

    Returns:
    - Reset neuron state (zeroed out).
    """
    return torch.zeros_like(neuron_state)

class ShortTermBuffer:
    def __init__(self, max_len: int = 30):
        self.max_len = max_len
        self.buffer  = []

    def add_to_buffer(self, state: torch.Tensor):
        if len(self.buffer) == self.max_len:
            self.buffer.pop(0)              # FIFO
        self.buffer.append(state.detach())  # keep it small & grad-free

    def get_recent_activations(self):
        return self.buffer

def temporal_proximity_scaling(history: torch.Tensor,
                               recency_factor: float = 0.9) -> torch.Tensor:
    """
    history: [L, S]  (L ≤ buffer_size; no list-of-tensors anymore)
    returns : [S]
    """
    # newest entry is last → weights [r^(L-1), …, r^0]
    L = history.size(0)
    weights = recency_factor ** torch.arange(L - 1, -1, -1,
                                             dtype=history.dtype,
                                             device=history.device)
    weighted = history * weights.unsqueeze(-1)        # [L,S]
    return weighted.sum(dim=0) / weights.sum()        # [S]


