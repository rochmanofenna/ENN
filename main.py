import torch
from config import Config
from enn.model import ENNModelWithMemory

# Initialize the model with configuration settings
config = Config()
model = ENNModelWithMemory(config)

# Example batch processing
input_data = torch.rand((8, config.num_neurons))  # Example batch with 8 samples
output = model(input_data)
print("Batch Processing Output with Memory:", output)

# Reset the model's memory after processing
model.reset_memory()
print("Memory reset. Neuron states:", model.neuron_states)
