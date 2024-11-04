import torch
import torch.nn as nn
from enn.memory import ShortTermBuffer, state_decay, reset_neuron_state, temporal_proximity_scaling
from enn.state_collapse import StateAutoEncoder, advanced_state_collapse
from enn.sparsity_control import dynamic_sparsity_control, event_trigger, low_power_state_collapse
from enn.layers import process_neuron_layer

class ENNModelWithSparsityControl(nn.Module):
    def __init__(self, config):
        super(ENNModelWithSparsityControl, self).__init__()
        self.num_layers = config.num_layers
        self.num_neurons = config.num_neurons
        self.num_states = config.num_states
        self.decay_rate = config.decay_rate
        self.recency_factor = config.recency_factor
        self.buffer_size = config.buffer_size
        self.importance_threshold = config.importance_threshold
        self.compressed_dim = config.compressed_dim
        self.sparsity_threshold = config.sparsity_threshold
        self.low_power_k = config.low_power_k
        
        # Initialize neuron states, short-term buffers, and autoencoder
        self.neuron_states = torch.zeros(self.num_neurons, self.num_states)
        self.short_term_buffers = [ShortTermBuffer(buffer_size=self.buffer_size) for _ in range(self.num_neurons)]
        self.autoencoder = StateAutoEncoder(input_dim=self.num_states, compressed_dim=self.compressed_dim)

    def forward(self, x):
        """
        Forward pass with memory decay, adaptive buffering, advanced state collapse, and sparsity control.

        Parameters:
        - x: Input tensor for batch processing.

        Returns:
        - Processed output after passing through layers.
        """
        for _ in range(self.num_layers):
            # Apply dynamic sparsity control to neuron states
            self.neuron_states = dynamic_sparsity_control(self.neuron_states, self.sparsity_threshold)
            
            # Decay the state of each neuron
            self.neuron_states = state_decay(self.neuron_states, self.decay_rate)
            
            # Process the layer and update neuron states
            x = process_neuron_layer(x, self.num_neurons, self.num_states)
            
            # Store recent activations in the short-term buffer
            for i in range(self.num_neurons):
                self.short_term_buffers[i].add_to_buffer(self.neuron_states[i])
                
            # Retrieve recent activations and apply temporal proximity scaling
            for i in range(self.num_neurons):
                recent_activations = torch.tensor(self.short_term_buffers[i].get_recent_activations())
                if recent_activations.numel() > 0:
                    self.neuron_states[i] = temporal_proximity_scaling(recent_activations, self.recency_factor)

            # Apply advanced state collapse to each neuron
            for i in range(self.num_neurons):
                self.neuron_states[i] = advanced_state_collapse(
                    self.neuron_states[i], self.autoencoder, self.importance_threshold
                )

            # Apply low-power state collapse if in low-resource mode
            self.neuron_states = low_power_state_collapse(self.neuron_states, top_k=self.low_power_k)

        return x

    def reset_memory(self):
        """
        Resets the memory of all neurons, clearing short-term buffers.
        """
        self.neuron_states = reset_neuron_state(self.neuron_states)
        for buffer in self.short_term_buffers:
            buffer.buffer.clear()


