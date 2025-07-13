
import torch
import torch.nn.functional as F

def attention_gate(neuron_activations, threshold=0.1):
    query = neuron_activations
    key = neuron_activations.transpose(0, 1)
    
    # Calculate attention scores and ensure it has the shape [10, 5]
    attention_scores = F.softmax(torch.matmul(query, key) / (neuron_activations.size(-1) ** 0.5), dim=-1)
    attention_scores = attention_scores[:, :neuron_activations.size(1)]  # Adjust size to match neuron_activations
    
    gated_activations = torch.where(attention_scores > threshold, neuron_activations, torch.zeros_like(neuron_activations))
    return gated_activations



def probabilistic_path_activation(neuron_activations, activation_probability=0.2):
    random_mask = (torch.rand_like(neuron_activations) < activation_probability)
    activated_paths = neuron_activations * random_mask.float()
    return activated_paths

