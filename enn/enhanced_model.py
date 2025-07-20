"""
Enhanced ENN Model with Multi-Head Attention Integration.

This module extends the original ENN architecture with robust attention mechanisms:
- Multi-head self-attention between neurons
- Temporal attention for memory buffers  
- Cross-attention for entanglement enhancement
- Sparse attention for computational efficiency
"""

import asyncio
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from enn.memory import ShortTermBuffer, state_decay, reset_neuron_state, temporal_proximity_scaling
from enn.state_collapse import StateAutoEncoder, advanced_state_collapse
from enn.sparsity_control import dynamic_sparsity_control, low_power_state_collapse
from enn.scheduler import PriorityTaskScheduler
from enn.validation import validate_config, validate_tensor_dimensions, handle_device_mismatch
from enn.multihead_attention import (
    MultiHeadAttention, NeuronStateAttention, TemporalAttention, 
    SparseAttention, AttentionPooling
)


class ENNWithAttention(nn.Module):
    """
    Enhanced ENN Model with integrated multi-head attention mechanisms.
    
    Features:
    - Original ENN entangled neuron dynamics
    - Multi-head attention between neurons
    - Temporal attention for memory processing
    - Sparse attention for efficiency
    - Configurable attention types and parameters
    """
    
    def __init__(self, cfg, attention_config: Optional[dict] = None):
        super().__init__()
        
        # Validate configuration
        validate_config(cfg)
        
        # ── hyper-params ──────────────────────────────────────────
        self.num_layers = cfg.num_layers
        self.num_neurons = cfg.num_neurons
        self.num_states = cfg.num_states
        self.decay_rate = cfg.decay_rate
        self.recency_fact = cfg.recency_factor
        self.buffer_size = cfg.buffer_size
        self.low_power_k = cfg.low_power_k
        self.sparsity_thr = cfg.sparsity_threshold
        self.l1_lambda = getattr(cfg, "l1_lambda", 1e-4)
        
        # ── attention configuration ──────────────────────────────
        self.attention_config = attention_config or {
            'use_neuron_attention': True,
            'use_temporal_attention': True,
            'use_sparse_attention': False,
            'num_heads': 4,
            'attention_dropout': 0.1,
            'attention_hidden_dim': 64
        }
        
        # ── persistent state ─────────────────────────────────────
        self.register_buffer("neuron_states",
                             torch.zeros(self.num_neurons, self.num_states))
        
        # ── learnable parameters ─────────────────────────────────
        self.entanglement = nn.Parameter(torch.randn(self.num_neurons,
                                                     self.num_states))
        self.mixing = nn.Parameter(torch.eye(self.num_neurons))
        self.readout = nn.Linear(self.num_states, self.num_states, bias=False)
        
        # ── input projection for temporal data ───────────────────
        self.input_projection = nn.Linear(
            cfg.input_dim if hasattr(cfg, 'input_dim') else self.num_states,
            self.num_neurons * self.num_states
        )
        
        # ── attention mechanisms ─────────────────────────────────
        self._setup_attention_layers()
        
        # ── helpers ──────────────────────────────────────────────
        self.short_buffers = [ShortTermBuffer(self.buffer_size)
                              for _ in range(self.num_neurons)]
        self.autoencoder = StateAutoEncoder(self.num_states,
                                          cfg.compressed_dim)
        self.scheduler = PriorityTaskScheduler()
        
        # ── attention pooling for output ─────────────────────────
        self.attention_pooling = AttentionPooling(
            input_dim=self.num_states,
            hidden_dim=self.attention_config['attention_hidden_dim']
        )
    
    def _setup_attention_layers(self):
        """Initialize attention mechanisms based on configuration."""
        config = self.attention_config
        
        # Neuron-state attention for entanglement enhancement
        if config.get('use_neuron_attention', True):
            self.neuron_attention = NeuronStateAttention(
                num_neurons=self.num_neurons,
                num_states=self.num_states,
                hidden_dim=config['attention_hidden_dim'],
                num_heads=config['num_heads'],
                dropout=config['attention_dropout']
            )
        else:
            self.neuron_attention = None
        
        # Temporal attention for memory buffer processing
        if config.get('use_temporal_attention', True):
            self.temporal_attention = TemporalAttention(
                state_dim=self.num_states,
                hidden_dim=config['attention_hidden_dim'],
                num_heads=config['num_heads'] // 2  # Use fewer heads for temporal
            )
        else:
            self.temporal_attention = None
        
        # Sparse attention for efficiency (optional)
        if config.get('use_sparse_attention', False):
            self.sparse_attention = SparseAttention(
                d_model=self.num_states,
                num_heads=config['num_heads'],
                top_k=config.get('sparse_top_k', 8),
                dropout=config['attention_dropout']
            )
        else:
            self.sparse_attention = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with attention mechanisms.
        
        Args:
            x: Input tensor - can be:
               - [batch, num_neurons, num_states] (direct ENN format)
               - [batch, time_steps, features] (temporal data)
               - [batch, features] (single timestep)
        
        Returns:
            output: [batch, num_neurons, num_states]
        """
        dev = x.device
        self.neuron_states = self.neuron_states.to(dev)
        
        # ── input validation and reshaping ───────────────────────
        if x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D")
        
        # Handle different input formats
        x = self._process_input(x)
        
        # L1 regulariser on mask
        self.mask_l1 = self.l1_lambda * torch.sigmoid(self.entanglement).mean()
        
        # ── enhanced processing with attention ───────────────────
        for layer_idx in range(self.num_layers):
            # 1) Apply neuron-state attention if enabled
            if self.neuron_attention is not None:
                x_attended, neuron_attention_weights = self.neuron_attention(x)
                # Residual connection with attention
                x = x + 0.5 * x_attended
            
            # 2) Original ENN processing: prune + decay
            self.neuron_states = dynamic_sparsity_control(
                self.neuron_states, self.sparsity_thr)
            self.neuron_states = state_decay(
                self.neuron_states, self.decay_rate)
            
            # 3) Enhanced entangle + mix with attention
            mask = torch.sigmoid(self.entanglement).unsqueeze(0)  # [1,N,S]
            x = x * mask
            
            # Apply sparse attention if enabled
            if self.sparse_attention is not None:
                # Reshape for sparse attention: treat neurons as sequence
                x_flat = x.view(x.size(0), self.num_neurons, self.num_states)
                x_sparse, sparse_weights = self.sparse_attention(x_flat, x_flat, x_flat)
                x = x_sparse.view_as(x)
            
            # Original mixing operation
            x = torch.einsum("bns,nm->bms", x, self.mixing)
            self.neuron_states = x.mean(0)  # update memory
            
            # 4) Enhanced memory processing with temporal attention
            x = self._process_memory_with_attention(x, layer_idx)
            
            # 5) Original state collapse + low-power
            self.neuron_states = advanced_state_collapse(
                self.neuron_states, self.autoencoder, importance_threshold=0.)
            self.neuron_states = low_power_state_collapse(
                self.neuron_states, top_k=self.low_power_k)
        
        return self.readout(x)
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Process and reshape input to ENN format."""
        # Handle temporal input: [batch, time, features] -> [batch, num_neurons, num_states]
        if x.dim() == 3 and x.size(1) != self.num_neurons:
            batch_size, time_steps, features = x.shape
            if time_steps == 0:
                raise ValueError("Time dimension cannot be zero")
            # Take the last timestep and project to neuron space
            x_last = x[:, -1, :]  # [batch, features]
            x_projected = self.input_projection(x_last)
            x = x_projected.view(batch_size, self.num_neurons, self.num_states)
        elif x.dim() == 2:
            # Assume [batch, features] format
            batch_size, features = x.shape
            x_projected = self.input_projection(x)
            x = x_projected.view(batch_size, self.num_neurons, self.num_states)
        
        return x
    
    def _process_memory_with_attention(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Enhanced memory processing with temporal attention."""
        device = x.device
        
        # Original recency-weighted memory (vectorised)
        buf_stack = []
        for i, buf in enumerate(self.short_buffers):
            buf.add_to_buffer(self.neuron_states[i])
            acts = buf.get_recent_activations()
            if acts and all(a.size(-1) == self.num_states for a in acts):
                buf_stack.append(torch.stack(acts, 0))  # [L,S]
            else:
                buf_stack.append(self.neuron_states[i].unsqueeze(0))
        
        if buf_stack:
            buf_stack = torch.nn.utils.rnn.pad_sequence(buf_stack, batch_first=True)
            L = buf_stack.size(1)
            
            # Apply temporal attention to memory buffers if enabled
            if self.temporal_attention is not None and L > 1:
                # Process each neuron's buffer with temporal attention
                enhanced_buffers = []
                for neuron_idx in range(len(buf_stack)):
                    neuron_buffer = buf_stack[neuron_idx].unsqueeze(0)  # [1, L, S]
                    if neuron_buffer.size(1) > 1:  # Only apply if temporal dimension > 1
                        attended_buffer, _ = self.temporal_attention(neuron_buffer)
                        enhanced_buffers.append(attended_buffer.squeeze(0))  # [L, S]
                    else:
                        enhanced_buffers.append(neuron_buffer.squeeze(0))
                
                if enhanced_buffers:
                    buf_stack = torch.stack(enhanced_buffers, 0)  # [num_neurons, L, S]
            
            # Original recency weighting
            weights = self.recency_fact ** torch.arange(
                L - 1, -1, -1, device=device).view(1, L, 1)
            self.neuron_states = (buf_stack * weights).sum(1) / weights.sum(1)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """
        Get attention weights for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            attention_weights: Dictionary containing attention weights from different mechanisms
        """
        weights = {}
        
        # Process input
        x = self._process_input(x)
        
        # Get neuron attention weights
        if self.neuron_attention is not None:
            _, neuron_weights = self.neuron_attention(x)
            weights['neuron_attention'] = neuron_weights
        
        # Get temporal attention weights (from memory buffers)
        if self.temporal_attention is not None:
            # Create sample temporal sequence for demonstration
            sample_temporal = torch.randn(x.size(0), 5, self.num_states, device=x.device)
            _, temporal_weights = self.temporal_attention(sample_temporal)
            weights['temporal_attention'] = temporal_weights
        
        return weights
    
    async def async_process_event(self, neuron_state: torch.Tensor,
                                 input_data: torch.Tensor, priority: int):
        """Enhanced async processing with attention mechanisms."""
        import asyncio
        
        # Add event to scheduler with priority
        await self.scheduler.add_task(
            task_id=f"neuron_update_{id(neuron_state)}",
            priority=priority,
            data=input_data
        )
        
        # Process the update asynchronously
        await asyncio.sleep(0.001)  # Simulate async processing
        
        # Apply attention-enhanced update if available
        if self.neuron_attention is not None and input_data.dim() >= 2:
            # Reshape input for attention processing
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0).unsqueeze(0)  # [1, 1, states]
            elif input_data.dim() == 2:
                input_data = input_data.unsqueeze(1)  # [batch, 1, states]
            
            # Apply neuron attention
            enhanced_input, _ = self.neuron_attention(input_data)
            input_data = enhanced_input.squeeze()
        
        # Update neuron state with decay and enhanced input
        decayed_state = state_decay(neuron_state, self.decay_rate)
        updated_state = decayed_state + 0.1 * input_data
        
        # Apply sparsity control
        sparse_state = dynamic_sparsity_control(updated_state, self.sparsity_thr)
        
        # Update the global neuron states (ensure proper dimensions)
        if sparse_state.dim() == 1 and self.neuron_states.dim() == 2:
            self.neuron_states[0] = sparse_state
        else:
            # If dimensions don't match, take the mean over batch/time dimensions
            if sparse_state.dim() > 1:
                sparse_state = sparse_state.mean(0)
            self.neuron_states[0] = sparse_state[:self.num_states]
        
        return sparse_state
    
    def reset_memory(self):
        """Reset all neuron states and memory buffers."""
        self.neuron_states.zero_()
        for buffer in self.short_buffers:
            buffer.buffer.clear()


def create_attention_enn(cfg, attention_type: str = 'full') -> ENNWithAttention:
    """
    Factory function to create ENN models with different attention configurations.
    
    Args:
        cfg: ENN configuration
        attention_type: Type of attention configuration
            - 'full': All attention mechanisms enabled
            - 'neuron_only': Only neuron-state attention
            - 'temporal_only': Only temporal attention  
            - 'sparse': Sparse attention for efficiency
            - 'minimal': Minimal attention overhead
    
    Returns:
        Enhanced ENN model with specified attention configuration
    """
    
    attention_configs = {
        'full': {
            'use_neuron_attention': True,
            'use_temporal_attention': True,
            'use_sparse_attention': False,
            'num_heads': 8,
            'attention_dropout': 0.1,
            'attention_hidden_dim': 128
        },
        'neuron_only': {
            'use_neuron_attention': True,
            'use_temporal_attention': False,
            'use_sparse_attention': False,
            'num_heads': 4,
            'attention_dropout': 0.1,
            'attention_hidden_dim': 64
        },
        'temporal_only': {
            'use_neuron_attention': False,
            'use_temporal_attention': True,
            'use_sparse_attention': False,
            'num_heads': 4,
            'attention_dropout': 0.1,
            'attention_hidden_dim': 64
        },
        'sparse': {
            'use_neuron_attention': True,
            'use_temporal_attention': True,
            'use_sparse_attention': True,
            'num_heads': 4,
            'attention_dropout': 0.1,
            'attention_hidden_dim': 64,
            'sparse_top_k': 8
        },
        'minimal': {
            'use_neuron_attention': True,
            'use_temporal_attention': False,
            'use_sparse_attention': False,
            'num_heads': 2,
            'attention_dropout': 0.05,
            'attention_hidden_dim': 32
        }
    }
    
    if attention_type not in attention_configs:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return ENNWithAttention(cfg, attention_configs[attention_type])


# Test the enhanced model
if __name__ == "__main__":
    from enn.config import Config
    
    config = Config()
    config.num_neurons = 8
    config.num_states = 4
    config.input_dim = 4
    
    # Test different attention configurations
    attention_types = ['full', 'neuron_only', 'temporal_only', 'sparse', 'minimal']
    
    for att_type in attention_types:
        model = create_attention_enn(config, att_type)
        
        # Test forward pass
        test_input = torch.randn(2, 10, 4)  # [batch, time, features]
        output = model(test_input)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{att_type.upper()}: {params:,} parameters, output shape: {output.shape}")
        
        # Test attention weights extraction
        attention_weights = model.get_attention_weights(test_input)
        print(f"  Attention mechanisms: {list(attention_weights.keys())}")
    
    print("\\nEnhanced ENN with attention working correctly!")