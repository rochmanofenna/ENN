import torch

def compute_data_complexity(data_input):
    complexity_score = torch.var(data_input)
    return torch.clamp(complexity_score, 0, 1)

def dynamic_scaling(data_complexity, max_states=5):
    scaled = (data_complexity * max_states).clamp(min=1.0)
    num_active_states = int(scaled.item())
    return num_active_states


def activate_neuron(neuron_state, data_input):
    complexity_score = compute_data_complexity(data_input)
    active_states = dynamic_scaling(complexity_score)
    neuron_state[:active_states] = torch.sigmoid(data_input[:active_states])
    return neuron_state

