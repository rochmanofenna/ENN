import heapq
import asyncio

class PriorityTaskScheduler:
    def __init__(self):
        self.task_queue = []
        self.completed_tasks = {}

    def add_task(self, task, priority):
        # Add tasks with priority to the queue (lower values indicate higher priority)
        heapq.heappush(self.task_queue, (priority, task))
    
    async def add_task(self, task_id: str, priority: int, data=None):
        """Add a task to the async queue with metadata."""
        task_data = {
            'id': task_id,
            'data': data,
            'timestamp': asyncio.get_event_loop().time()
        }
        heapq.heappush(self.task_queue, (priority, task_data))
        
    def get_pending_tasks(self):
        """Get number of pending tasks."""
        return len(self.task_queue)

    async def process_tasks(self):
        while self.task_queue:
            priority, task = heapq.heappop(self.task_queue)
            if callable(task):
                await task
            else:
                # Handle task data
                self.completed_tasks[task['id']] = task
