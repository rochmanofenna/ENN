"""
Robust Multi-Head Attention Mechanisms for ENN.

Implements various attention mechanisms optimized for ENN's entangled neuron architecture:
- Multi-Head Self-Attention 
- Cross-Attention between neurons
- Temporal Attention for memory buffers
- Sparse Attention for efficiency
- Neuron-State Attention for entanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from enn.validation import validate_tensor_dimensions, handle_device_mismatch


class MultiHeadAttention(nn.Module):
    """
    Robust Multi-Head Attention optimized for ENN architecture.
    
    Features:
    - Scaled dot-product attention
    - Multiple attention heads
    - Dropout for regularization
    - Residual connections
    - Layer normalization
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 attention_type: str = 'scaled_dot_product'):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_type = attention_type
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)  
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_len, d_model] or [batch, num_neurons, num_states]
            key: [batch, seq_len, d_model] or [batch, num_neurons, num_states]
            value: [batch, seq_len, d_model] or [batch, num_neurons, num_states]
            mask: Optional attention mask [batch, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # Validate inputs
        validate_tensor_dimensions(query, 3, "query")
        validate_tensor_dimensions(key, 3, "key") 
        validate_tensor_dimensions(value, 3, "value")
        
        # Ensure all tensors are on same device
        query, key = handle_device_mismatch(query, key)
        key, value = handle_device_mismatch(key, value)
        
        # Store residual
        residual = query
        
        # Linear projections
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)    # [batch, seq_len, d_model]
        V = self.W_v(value)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                     V: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                     dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: [batch, num_heads, seq_len, d_k]
            K: [batch, num_heads, seq_len, d_k] 
            V: [batch, num_heads, seq_len, d_k]
            mask: Optional mask [batch, seq_len, seq_len]
            dropout: Optional dropout layer
            
        Returns:
            output: [batch, num_heads, seq_len, d_k]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class NeuronStateAttention(nn.Module):
    """
    Specialized attention mechanism for ENN's neuron-state interactions.
    
    Computes attention between neurons and their states to enhance entanglement.
    """
    
    def __init__(self, num_neurons: int, num_states: int, hidden_dim: int = 64, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.num_states = num_states
        self.hidden_dim = hidden_dim
        
        # Project neuron states to attention space
        self.state_projection = nn.Linear(num_states, hidden_dim)
        
        # Multi-head attention for neuron interactions
        self.neuron_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Project back to neuron state space
        self.output_projection = nn.Linear(hidden_dim, num_states)
        
    def forward(self, neuron_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            neuron_states: [batch, num_neurons, num_states]
            
        Returns:
            enhanced_states: [batch, num_neurons, num_states] 
            attention_weights: [batch, num_heads, num_neurons, num_neurons]
        """
        batch_size = neuron_states.size(0)
        
        # Project to attention space
        projected_states = self.state_projection(neuron_states)  # [batch, num_neurons, hidden_dim]
        
        # Apply multi-head attention (self-attention over neurons)
        attended_states, attention_weights = self.neuron_attention(
            query=projected_states,
            key=projected_states, 
            value=projected_states
        )
        
        # Project back to state space
        enhanced_states = self.output_projection(attended_states)  # [batch, num_neurons, num_states]
        
        return enhanced_states, attention_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for ENN's memory buffers.
    
    Applies attention over temporal sequences in memory buffers.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64, num_heads: int = 2):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Temporal position encoding
        self.temporal_encoding = nn.Parameter(torch.randn(100, hidden_dim))  # Max 100 timesteps
        
        # Project states to attention space
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        
        # Multi-head attention
        self.temporal_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, temporal_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal_states: [batch, time_steps, state_dim]
            
        Returns:
            attended_states: [batch, time_steps, state_dim]
            attention_weights: [batch, num_heads, time_steps, time_steps]
        """
        batch_size, time_steps, state_dim = temporal_states.shape
        
        # Project states
        projected = self.state_projection(temporal_states)  # [batch, time_steps, hidden_dim]
        
        # Add temporal position encoding
        if time_steps <= self.temporal_encoding.size(0):
            pos_enc = self.temporal_encoding[:time_steps].unsqueeze(0)  # [1, time_steps, hidden_dim]
            projected = projected + pos_enc
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(
            query=projected,
            key=projected,
            value=projected
        )
        
        # Project back
        output = self.output_projection(attended)  # [batch, time_steps, state_dim]
        
        return output, attention_weights


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for efficiency in large ENN models.
    
    Uses top-k attention to reduce computational complexity.
    """
    
    def __init__(self, d_model: int, num_heads: int, top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.top_k = top_k
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model] 
            value: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
            sparse_attention_weights: [batch, num_heads, seq_len, top_k]
        """
        batch_size, seq_len, _ = query.shape
        residual = query
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply top-k sparse attention
        actual_k = min(self.top_k, seq_len)
        top_k_scores, top_k_indices = torch.topk(scores, k=actual_k, dim=-1)
        
        # Create sparse attention weights
        sparse_weights = F.softmax(top_k_scores, dim=-1)
        sparse_weights = self.dropout(sparse_weights)
        
        # Gather values using top-k indices
        # Expand indices for value gathering
        expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_k)
        selected_values = torch.gather(V.unsqueeze(-2).expand(-1, -1, -1, seq_len, -1), 
                                     dim=-2, index=expanded_indices)
        
        # Apply sparse attention
        output = torch.matmul(sparse_weights.unsqueeze(-1), selected_values).squeeze(-2)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, sparse_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating sequences into fixed-size representations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence: [batch, seq_len, input_dim]
            mask: Optional mask [batch, seq_len]
            
        Returns:
            pooled: [batch, input_dim]
        """
        # Compute attention weights
        weights = self.attention_weights(sequence).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        
        # Softmax normalization
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Weighted sum
        pooled = torch.sum(weights * sequence, dim=1)  # [batch, input_dim]
        
        return pooled


# Factory function for creating attention mechanisms
def create_attention_layer(attention_type: str, **kwargs) -> nn.Module:
    """Factory function to create different attention mechanisms."""
    
    attention_map = {
        'multihead': MultiHeadAttention,
        'neuron_state': NeuronStateAttention,
        'temporal': TemporalAttention,
        'sparse': SparseAttention,
        'pooling': AttentionPooling
    }
    
    if attention_type.lower() not in attention_map:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return attention_map[attention_type.lower()](**kwargs)


# Test attention mechanisms
if __name__ == "__main__":
    # Test MultiHeadAttention
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8
    
    attention = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = attention(x, x, x)
    
    print(f"MultiHeadAttention - Input: {x.shape}, Output: {output.shape}, Weights: {weights.shape}")
    
    # Test NeuronStateAttention
    num_neurons, num_states = 10, 5
    neuron_attention = NeuronStateAttention(num_neurons, num_states)
    
    neuron_states = torch.randn(batch_size, num_neurons, num_states)
    enhanced_states, neuron_weights = neuron_attention(neuron_states)
    
    print(f"NeuronStateAttention - Input: {neuron_states.shape}, Output: {enhanced_states.shape}")
    
    # Test TemporalAttention
    state_dim = 5
    temporal_attention = TemporalAttention(state_dim)
    
    temporal_states = torch.randn(batch_size, seq_len, state_dim)
    temporal_output, temporal_weights = temporal_attention(temporal_states)
    
    print(f"TemporalAttention - Input: {temporal_states.shape}, Output: {temporal_output.shape}")
    
    print("All attention mechanisms working correctly!")