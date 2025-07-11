import torch

def context_aware_initialization(num_neurons, num_states, entanglement_matrix, method="xavier"):
    if method == "xavier":
        scale = torch.sqrt(torch.tensor(2.0) / (num_neurons + num_states))
    elif method == "he":
        scale = torch.sqrt(torch.tensor(2.0) / num_neurons)
    else:
        raise ValueError("Unsupported initialization method")
    # Random initial weights scaled by chosen initialization
    weights = scale * torch.randn(num_neurons, num_states)
    # Create mask for positive entanglement and apply scaling
    ent_matrix_t = torch.tensor(entanglement_matrix)  # ensure it's a tensor if not already
    mask = ent_matrix_t > 0  
    weights[mask] *= ent_matrix_t[mask]
    # (Where entanglement_matrix <= 0, weights stay the same as multiplying by 1.0)
    return weights

