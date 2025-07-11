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

class ShortTermBuffer(nn.Module):
    def __init__(self, buffer_size: int = 5):
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_to_buffer(self, state_tensor):
        self.buffer.append(state_tensor.detach())

    def get_recent_activations(self):
        return list(self.buffer)

def temporal_proximity_scaling(weights: torch.Tensor, recency_factor: float = 0.9) -> torch.Tensor:
    """
    weights: Tensor of shape [T, num_states] containing the last T activations for one neuron.
    recency_factor: float in (0,1], higher means more emphasis on recent entries.

    Returns a single 1D tensor [num_states] by weighting each row by recency and
    then averaging across time.
    """
    # Number of time steps
    T = weights.shape[0]
    # Indices 0...(T-1); 0 = oldest, T-1 = newest
    time_indices = torch.arange(T, dtype=torch.float32, device=weights.device)
    # Reverse so newest has highest exponent: recency_factor**(T-1-idx)
    scaling = recency_factor ** ((T - 1) - time_indices)
    # Expand to multiply across the num_states dimension
    scaling = scaling.unsqueeze(1)  # shape [T,1]
    # Apply weights and sum across time, then normalize by total scaling
    weighted = weights * scaling        # shape [T, num_states]
    return weighted.sum(dim=0) / scaling.sum()  # shape [num_states]

