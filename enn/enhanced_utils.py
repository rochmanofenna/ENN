"""
Enhanced utilities for ENN including logging, checkpointing, and monitoring.
"""
import torch
import logging
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ENNLogger:
    """Enhanced logging for ENN training and inference."""
    
    def __init__(self, name: str = "ENN", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def info(self, msg: str, **kwargs):
        if kwargs:
            msg += f" | {kwargs}"
        self.logger.info(msg)
    
    def error(self, msg: str, **kwargs):
        if kwargs:
            msg += f" | {kwargs}"
        self.logger.error(msg)
    
    def warning(self, msg: str, **kwargs):
        if kwargs:
            msg += f" | {kwargs}"
        self.logger.warning(msg)

class ENNCheckpointer:
    """Model checkpointing and state management."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch: int, loss: float, 
                       config: Any, metadata: Optional[Dict] = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config.__dict__,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"enn_checkpoint_epoch_{epoch}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Save latest checkpoint reference
        latest_path = os.path.join(self.checkpoint_dir, "latest.json")
        with open(latest_path, 'w') as f:
            json.dump({'latest_checkpoint': filename, 'epoch': epoch}, f)
    
    def load_checkpoint(self, model, optimizer, checkpoint_path: Optional[str] = None):
        """Load model checkpoint."""
        if checkpoint_path is None:
            # Load latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest.json")
            if not os.path.exists(latest_path):
                raise FileNotFoundError("No checkpoint found")
            
            with open(latest_path, 'r') as f:
                latest_info = json.load(f)
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_info['latest_checkpoint'])
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metadata', {})

class MetricsTracker:
    """Track and analyze training metrics."""
    
    def __init__(self):
        self.metrics = {
            'loss': [],
            'sparsity': [],
            'gradient_norm': [],
            'memory_usage': [],
            'epoch_time': []
        }
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics."""
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[f"{metric}_mean"] = sum(values) / len(values)
                summary[f"{metric}_last"] = values[-1]
                if len(values) > 1:
                    summary[f"{metric}_std"] = torch.tensor(values).std().item()
        return summary
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def compute_model_sparsity(model) -> float:
    """Compute overall model sparsity."""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param.abs() < 1e-8).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0

def compute_gradient_norm(model) -> float:
    """Compute gradient norm for monitoring."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def memory_usage_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0