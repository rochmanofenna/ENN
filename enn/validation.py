"""
Input validation and error handling utilities for ENN.
"""
import torch
from typing import Optional, Tuple, Union

def validate_tensor_dimensions(tensor: torch.Tensor, 
                             expected_dims: int, 
                             name: str = "tensor") -> None:
    """Validate tensor has expected number of dimensions."""
    if tensor.dim() != expected_dims:
        raise ValueError(f"{name} must have {expected_dims} dimensions, got {tensor.dim()}")

def validate_tensor_shape(tensor: torch.Tensor, 
                         expected_shape: Tuple[int, ...], 
                         name: str = "tensor") -> None:
    """Validate tensor has expected shape."""
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tensor.shape}")

def validate_positive_int(value: int, name: str = "value") -> None:
    """Validate value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")

def validate_probability(value: float, name: str = "probability") -> None:
    """Validate value is a probability between 0 and 1."""
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")

def validate_config(config) -> None:
    """Validate ENN configuration parameters."""
    validate_positive_int(config.num_layers, "num_layers")
    validate_positive_int(config.num_neurons, "num_neurons") 
    validate_positive_int(config.num_states, "num_states")
    validate_positive_int(config.buffer_size, "buffer_size")
    validate_positive_int(config.epochs, "epochs")
    validate_positive_int(config.batch_size, "batch_size")
    
    validate_probability(config.decay_rate, "decay_rate")
    validate_probability(config.recency_factor, "recency_factor")
    validate_probability(config.activation_probability, "activation_probability")
    
    if config.low_power_k > config.num_states:
        raise ValueError(f"low_power_k ({config.low_power_k}) cannot exceed num_states ({config.num_states})")

def safe_topk(tensor: torch.Tensor, k: int, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safely compute topk, handling edge cases."""
    actual_k = min(k, tensor.size(dim))
    if actual_k <= 0:
        return torch.zeros_like(tensor), torch.zeros(tensor.shape[:-1] + (0,), dtype=torch.long, device=tensor.device)
    return torch.topk(tensor, k=actual_k, dim=dim, largest=True)

def handle_device_mismatch(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure tensors are on the same device."""
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    return tensor1, tensor2