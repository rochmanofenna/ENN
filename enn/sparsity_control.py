import torch

@torch.no_grad()
def dynamic_sparsity_control(states: torch.Tensor,
                             threshold: float = 0.05) -> torch.Tensor:
    """
    Zero-out values with |x| < threshold, **in-place** to save memory.
    Keeps gradients for the surviving entries.
    """
    if threshold <= 0.0:
        return states
    mask = states.abs() < threshold          # True = prune
    states = states.clone()                  # keep autograd history
    states[mask] = 0.0
    return states
                        # pruned tensor


def event_trigger(input_data, threshold=0.1):
    """
    Detects significant changes in input data to trigger sparse backpropagation.

    Parameters:
    - input_data: Current input tensor.
    - threshold: Threshold for detecting significant changes.

    Returns:
    - Boolean indicating whether sparse backpropagation should be triggered.
    """
    change_magnitude = torch.mean(torch.abs(input_data - input_data.mean()))
    return change_magnitude > threshold

def sparse_backpropagation(model, input_data, criterion, optimizer, sparsity_triggered):
    """
    Executes sparse backpropagation if triggered by input changes.

    Parameters:
    - model: Neural network model.
    - input_data: Input tensor.
    - criterion: Loss function.
    - optimizer: Optimizer for model training.
    - sparsity_triggered: Boolean indicating if backpropagation is triggered.
    """
    if sparsity_triggered:
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)  # Assuming `target` is defined elsewhere
        loss.backward()
        optimizer.step()
        
def low_power_state_collapse(neuron_states, top_k: int = 3):
    """
    Keep only the top-k highest-magnitude activations for each neuron.
    Works for either:
      • a 1-D tensor  -> single neuron       (shape: [num_states])
      • a 2-D tensor  -> batch of neurons    (shape: [num_neurons, num_states])
    All other values are zeroed out.
    """
    if neuron_states.ndim == 1:
        # ----- single-neuron vector -----
        kth_value = torch.topk(neuron_states, k=top_k, largest=True).values[-1]
        return torch.where(neuron_states >= kth_value,
                           neuron_states,
                           torch.zeros_like(neuron_states))

    # ----- batch of neurons (2-D) -----
    # Ensure top_k doesn't exceed the number of states
    actual_k = min(top_k, neuron_states.size(-1))
    if actual_k <= 0:
        return torch.zeros_like(neuron_states)
    
    # top_vals: [num_neurons, actual_k]
    top_vals, _ = torch.topk(neuron_states, k=actual_k, dim=-1, largest=True)
    # threshold per neuron = its k-th largest value  → shape [num_neurons, 1]
    threshold = top_vals[:, -1].unsqueeze(-1)
    return torch.where(neuron_states >= threshold,
                       neuron_states,
                       torch.zeros_like(neuron_states))




