import torch
import asyncio
from enn.config import Config
from enn.model import ENNModelWithSparsityControl

# Initialize the model with configuration settings
config = Config()
model = ENNModelWithSparsityControl(config)

# Example batch processing
input_data = torch.rand((8, config.num_neurons, config.num_states))  # Adjusted input shape
output = model(input_data)
print("Batch Processing Output with Memory:", output)

# Asynchronous event processing example with PriorityTaskScheduler
async def async_event_processing_example(model, input_data):
    # Demonstrate processing events asynchronously with priority-based updates
    neuron_state = model.neuron_states  # Example neuron state to update asynchronously
    priority = 1  # Priority level for event

    # Schedule asynchronous neuron updates based on input importance
    await model.async_process_event(neuron_state, input_data, priority)

# Run the asynchronous event processing example
asyncio.run(async_event_processing_example(model, input_data[0]))  # Process the first sample asynchronously

# Reset the model's memory after processing
model.reset_memory()
print("Memory reset. Neuron states:", model.neuron_states)
