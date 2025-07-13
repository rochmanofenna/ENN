import torch
import asyncio
import torch
import asyncio
from enn.layers import mini_batch_processing  # Add this import** 
from enn.scheduler import PriorityTaskScheduler

async def async_neuron_update(neuron_state, data_input, priority_threshold=0.5):
    # Calculate importance of the new data
    data_importance = torch.mean(data_input)
    
    if data_importance.item() > priority_threshold:
        # Update the neuron state if data importance exceeds threshold
        neuron_state = torch.sigmoid(data_input)
        await asyncio.sleep(0)  # Yield control for asynchronous processing
    
    return neuron_state

async def handle_event(neuron_state, data_input, scheduler, priority=1):
    # Schedule neuron update task based on priority
    update_task = async_neuron_update(neuron_state, data_input)
    scheduler.add_task(update_task, priority)
    await scheduler.process_tasks()
    
async def process_data_stream(data_stream, neuron_state, scheduler, batch_size=4):
    for mini_batch in mini_batch_processing(data_stream, batch_size):
        for data_input in mini_batch:
            tasks = [handle_event(neuron_state, data_input, scheduler, priority=1) for data_input in mini_batch]
            await asyncio.gather(*tasks)
            


