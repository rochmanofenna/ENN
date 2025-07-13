from enn.attention import attention_gate, probabilistic_path_activation
from enn.weight_sharing import dynamic_weight_sharing
from enn.core import activate_neuron
from enn.utils import compute_data_complexity, dynamic_scaling
import torch

# def process_entangled_neuron_layer(data_batch,
#                                   weights,
#                                   num_neurons=10,
#                                   num_states=5,
#                                   attention_threshold=0.5,
#                                   activation_probability=0.2):
#    """
#    data_batch: 3D tensor  [batch_size, num_neurons, num_states]
#    weights:    2D tensor  [num_neurons, num_states]
#    """
#    outputs = []
#    device = data_batch.device

#    for data_input in data_batch:  # data_input: [num_neurons, num_states]
        # --- 1) Per‐neuron state activation ---
#        neuron_activations = torch.zeros(num_neurons, num_states, device=device)
#        for n in range(num_neurons):
            # Feed each neuron's own state‐vector slice into activate_neuron
#            neuron_activations[n] = activate_neuron(
#                neuron_activations[n],
#                data_input[n]
#            )

        # --- 2) Weight sharing across entangled neurons ---
       # neuron_activations = dynamic_weight_sharing(neuron_activations, weights)

        # --- 3) Probabilistic pathway gating ---
       # neuron_activations = probabilistic_path_activation(
           # neuron_activations,
           # activation_probability
       # )

        # --- 4) Attention‐based gating ---
       # neuron_activations = attention_gate(
           # neuron_activations,
           # threshold=attention_threshold
       # )

#        outputs.append(neuron_activations)

    # Stack back into shape [batch_size, num_neurons, num_states]
#    return torch.stack(outputs, dim=0) ***\\\

# --- TEMPORARY for debugging ---
def process_entangled_neuron_layer(x, entanglement, *_):
    # x: [batch, num_neurons, num_states]
    # entanglement: [num_neurons, num_states] (nn.Parameter)
    return x * torch.sigmoid(entanglement).unsqueeze(0)

def mini_batch_processing(data_stream, batch_size=4):
    """
    Processes data in mini-batches for real-time updates.
    
    Parameters:
    - data_stream: Continuous data stream for real-time processing.
    - batch_size: Number of data samples to process per mini-batch.
    
    Returns:
    - Generator yielding mini-batches for processing.
    """
    data_iter = iter(data_stream)
    while True:
        mini_batch = list()
        for _ in range(batch_size):
            try:
                mini_batch.append(next(data_iter))
            except StopIteration:
                break
        if not mini_batch:
            break
        yield torch.stack(mini_batch)
