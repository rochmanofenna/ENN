import torch

# ----------------------------------------------------------------------
# Simple helpers used by layers.process_entangled_neuron_layer
# ----------------------------------------------------------------------

def compute_data_complexity(x: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar ‘complexity score’ for an input vector.

    Current heuristic: mean absolute value in [0, 1].
    You can swap this out for variance, entropy, etc. later.
    """
    return torch.mean(torch.abs(x))


def dynamic_scaling(complexity_score: torch.Tensor,
                    max_states: int = 5) -> int:
    """
    Map a 0–1 complexity score to an integer 1 … max_states.
    Ensures we never return 0.
    """
    # convert to Python float, clamp to [0, 1]
    c = float(torch.clamp(complexity_score, 0.0, 1.0).item())
    k = int(c * max_states)
    return max(1, min(max_states, k))


def entropy_based_prune(neuron_state, importance_threshold=0.1):
    return torch.where(neuron_state > importance_threshold, neuron_state, torch.zeros_like(neuron_state))
    
def context_collapse(neuron_state, threshold=0.5):
    entropy = -torch.sum(neuron_state * torch.log(neuron_state + 1e-9))
    if entropy < threshold:
        mean_val = torch.mean(neuron_state)
        return mean_val.expand_as(neuron_state)
    return neuron_state
