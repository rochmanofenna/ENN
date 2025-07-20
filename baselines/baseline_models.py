"""
Baseline neural network models for comprehensive benchmarking against ENN.

Includes: LSTM, Transformer, CNN, MLP, Liquid Neural Networks (LNN), and ENN-BICEP hybrids.
All models are designed to be comparable with ENN for fair evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


class BaselineConfig:
    """Configuration for baseline models to match ENN complexity."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 output_dim: int = 1, seq_len: int = 20, 
                 num_layers: int = 2, num_heads: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = 0.1
        self.batch_size = 32
        self.epochs = 114


class LSTMBaseline(nn.Module):
    """LSTM baseline model with comparable complexity to ENN."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            output: [batch, output_dim]
        """
        if x.dim() == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Project to output dimension
        output = self.output_projection(last_output)  # [batch, output_dim]
        
        return output


class TransformerBaseline(nn.Module):
    """Transformer baseline with positional encoding and multi-head attention."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, config.dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, config.output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            output: [batch, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer(x)  # [batch, seq_len, d_model]
        
        # Global average pooling over sequence
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        
        # Output projection
        output = self.output_projection(pooled)  # [batch, output_dim]
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class CNNBaseline(nn.Module):
    """1D CNN baseline for temporal data."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        # Multiple CNN blocks with different kernel sizes
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(config.input_dim, config.hidden_dim // 4, 3),
            self._make_conv_block(config.hidden_dim // 4, config.hidden_dim // 2, 5),
            self._make_conv_block(config.hidden_dim // 2, config.hidden_dim, 7)
        ])
        
        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int):
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            output: [batch, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # Transpose for CNN: [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling: [batch, hidden_dim, 1] -> [batch, hidden_dim]
        x = self.global_pool(x).squeeze(-1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class MLPBaseline(nn.Module):
    """Multi-Layer Perceptron baseline."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        # Calculate input size (flatten if temporal)
        self.input_size = config.input_dim * config.seq_len
        
        layers = []
        current_dim = self.input_size
        hidden_dims = [config.hidden_dim * 2, config.hidden_dim, config.hidden_dim // 2]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            output: [batch, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # Flatten: [batch, seq_len * input_dim]
        x = x.flatten(1)
        
        # Forward through MLP
        output = self.network(x)
        
        return output


class LiquidNeuralNetwork(nn.Module):
    """Simplified Liquid Neural Network implementation."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        
        # Time constants (learnable)
        self.time_constants = nn.Parameter(torch.rand(self.hidden_size) * 0.5 + 0.1)
        
        # Input weights
        self.W_ih = nn.Linear(config.input_dim, self.hidden_size)
        self.W_hh = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, config.output_dim)
        
    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
            dt: Time step for differential equation
        Returns:
            output: [batch, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_dim]
            
            # Compute derivatives
            input_current = self.W_ih(x_t)
            recurrent_current = self.W_hh(torch.tanh(h))
            
            # Liquid dynamics: dh/dt = -h/tau + input + recurrent
            dhdt = (-h / self.time_constants.abs()) + input_current + recurrent_current
            
            # Euler integration
            h = h + dt * dhdt
        
        # Output projection
        output = self.output_projection(torch.tanh(h))
        
        return output


def create_baseline_model(model_type: str, config: BaselineConfig) -> nn.Module:
    """Factory function to create baseline models."""
    model_map = {
        'lstm': LSTMBaseline,
        'transformer': TransformerBaseline,
        'cnn': CNNBaseline,
        'mlp': MLPBaseline,
        'lnn': LiquidNeuralNetwork
    }
    
    if model_type.lower() not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type.lower()](config)


# Model parameter counting utility
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test all models
if __name__ == "__main__":
    config = BaselineConfig(input_dim=5, hidden_dim=64, output_dim=1)
    
    models = ['lstm', 'transformer', 'cnn', 'mlp', 'lnn']
    
    for model_name in models:
        model = create_baseline_model(model_name, config)
        params = count_parameters(model)
        
        # Test forward pass
        test_input = torch.randn(4, 20, 5)  # [batch, seq_len, input_dim]
        output = model(test_input)
        
        print(f"{model_name.upper()}: {params:,} parameters, output shape: {output.shape}")