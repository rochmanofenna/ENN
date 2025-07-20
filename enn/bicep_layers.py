"""
BICEP Integration Layers for ENN Benchmarking
Importable BICEP layers designed for seamless integration with ENN benchmark framework
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Optional, Dict, Any

# Import BICEP from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'BICEP'))

try:
    from metal_bicep_benchmark import metal_brownian_sde_kernel
    from metal_kernel_bicep import MinimalOverheadBICEP
    BICEP_AVAILABLE = True
except ImportError:
    print("Warning: BICEP not available, using mock implementation")
    BICEP_AVAILABLE = False

class BICEPNeuralLayer(nn.Module):
    """
    Optimized BICEP layer designed for ENN benchmark integration
    Compatible with ENN's config system and benchmark framework
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 n_paths: int = 100,
                 n_steps: int = 50,
                 device: str = 'cpu',
                 learnable_stochastic: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.n_paths = min(n_paths, 1000)  # Limit for benchmark performance
        self.n_steps = n_steps
        self.device = device
        
        # Learnable parameters for stochastic control
        if learnable_stochastic:
            self.feedback_controller = nn.Linear(input_size, 1)
            self.decay_controller = nn.Parameter(torch.tensor(0.1))
            self.variance_controller = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('feedback_scale', torch.tensor(0.5))
            self.register_buffer('decay_scale', torch.tensor(0.1))
            self.register_buffer('variance_scale', torch.tensor(1.0))
        
        # Input/output projection layers
        self.input_projection = nn.Linear(input_size, self.n_paths)
        # Fix temporal aggregator dimensions
        self.temporal_aggregator = nn.Conv1d(self.n_paths, self.n_paths, kernel_size=3, padding=1)
        self.output_projection = nn.Linear(self.n_paths, output_size)
        
        # Initialize BICEP engine if available
        if BICEP_AVAILABLE:
            self.bicep_engine = MinimalOverheadBICEP(device=device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BICEP layer
        
        Args:
            x: Input tensor [batch_size, input_size] or [batch_size, seq_len, input_size]
        
        Returns:
            Output tensor [batch_size, output_size]
        """
        # Handle temporal inputs by taking last timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch_size, input_size]
        
        batch_size = x.size(0)
        
        if not BICEP_AVAILABLE:
            # Mock implementation for testing when BICEP unavailable
            return self._mock_bicep_forward(x)
        
        # Project input to path control parameters
        path_controls = self.input_projection(x)  # [batch_size, n_paths]
        
        # Generate BICEP paths for each batch element
        batch_outputs = []
        
        for i in range(batch_size):
            # Use neural network to control feedback
            if hasattr(self, 'feedback_controller'):
                feedback_val = torch.sigmoid(self.feedback_controller(x[i:i+1])).item()
                decay_val = torch.sigmoid(self.decay_controller).item() * 0.5
            else:
                feedback_val = self.feedback_scale.item()
                decay_val = self.decay_scale.item()
            
            # Generate stochastic paths
            try:
                paths = metal_brownian_sde_kernel(
                    n_paths=self.n_paths,
                    n_steps=self.n_steps,
                    T=1.0,
                    feedback_value=feedback_val,
                    decay_rate=decay_val,
                    device=self.device
                )
                
                # Temporal aggregation to reduce time dimension
                # Transpose paths for conv1d: [n_paths, n_steps+1] -> [1, n_paths, n_steps+1]
                paths_for_conv = paths.unsqueeze(0).transpose(1, 2)  # [1, n_paths, n_steps+1]
                aggregated = self.temporal_aggregator(paths_for_conv)  # [1, n_paths, n_steps+1]
                aggregated = aggregated.mean(dim=2).squeeze(0)  # [n_paths]
                
            except Exception as e:
                print(f"BICEP generation failed: {e}, using fallback")
                # Fallback to deterministic computation
                aggregated = torch.randn(self.n_paths, device=x.device)
            
            batch_outputs.append(aggregated)
        
        # Stack and project to output
        stacked_paths = torch.stack(batch_outputs)  # [batch_size, n_paths]
        output = self.output_projection(stacked_paths)  # [batch_size, output_size]
        
        return output
    
    def _mock_bicep_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mock BICEP computation for testing"""
        projected = self.input_projection(x)
        # Simple nonlinear transformation as proxy for stochastic computation
        mock_paths = torch.tanh(projected) + 0.1 * torch.randn_like(projected)
        return self.output_projection(mock_paths)


class ENNBICEPHybrid(nn.Module):
    """
    Hybrid model combining ENN's entangled neurons with BICEP's stochastic computation
    Designed for revolutionary benchmarking against standard architectures
    """
    
    def __init__(self, config, bicep_integration_mode: str = 'parallel'):
        super().__init__()
        
        self.config = config
        self.bicep_integration_mode = bicep_integration_mode
        
        # Import ENN components
        from enn.model import ENNModelWithSparsityControl
        
        # Core ENN model
        self.enn_core = ENNModelWithSparsityControl(config)
        
        # BICEP enhancement layers
        if bicep_integration_mode == 'parallel':
            # Parallel processing: ENN + BICEP paths computed independently
            self.bicep_layer = BICEPNeuralLayer(
                input_size=config.input_dim if hasattr(config, 'input_dim') else config.num_states,
                output_size=config.num_neurons * config.num_states,
                n_paths=config.num_neurons * 2,
                n_steps=config.num_states * 10,
                device='cpu'  # Will be moved to proper device later
            )
            self.fusion_layer = nn.Linear(
                config.num_neurons * config.num_states * 2, 
                config.num_neurons * config.num_states
            )
            
        elif bicep_integration_mode == 'sequential':
            # Sequential: BICEP enhances ENN's neuron states
            self.bicep_layer = BICEPNeuralLayer(
                input_size=config.num_neurons * config.num_states,
                output_size=config.num_neurons * config.num_states,
                n_paths=config.num_neurons,
                n_steps=config.num_states * 5,
                device='cpu'
            )
            
        elif bicep_integration_mode == 'entangled':
            # Entangled: BICEP paths directly influence neuron entanglement
            self.entanglement_bicep = BICEPNeuralLayer(
                input_size=config.input_dim if hasattr(config, 'input_dim') else config.num_states,
                output_size=config.num_neurons ** 2,  # Entanglement matrix
                n_paths=config.num_neurons,
                n_steps=config.num_states * 3,
                device='cpu'
            )
        
        # Output projection maintains ENN interface
        self.output_layer = nn.Linear(
            config.num_neurons * config.num_states,
            config.num_neurons * config.num_states
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid ENN-BICEP architecture
        
        Args:
            x: Input tensor [batch_size, seq_len, features] or [batch_size, features]
        
        Returns:
            Output tensor matching ENN interface
        """
        
        if self.bicep_integration_mode == 'parallel':
            # Compute ENN and BICEP features in parallel
            enn_output = self.enn_core(x)
            bicep_features = self.bicep_layer(x)
            
            # Ensure compatible shapes
            if enn_output.dim() == 3:
                enn_output = enn_output.reshape(enn_output.size(0), -1)
            if bicep_features.dim() == 3:
                bicep_features = bicep_features.reshape(bicep_features.size(0), -1)
            
            # Fuse features
            combined = torch.cat([enn_output, bicep_features], dim=-1)
            fused = self.fusion_layer(combined)
            output = self.output_layer(fused)
            
        elif self.bicep_integration_mode == 'sequential':
            # ENN first, then BICEP enhancement
            enn_output = self.enn_core(x)
            
            # Flatten for BICEP processing
            if enn_output.dim() == 3:
                enn_flattened = enn_output.reshape(enn_output.size(0), -1)
            else:
                enn_flattened = enn_output
                
            # BICEP enhancement
            bicep_enhanced = self.bicep_layer(enn_flattened)
            output = self.output_layer(bicep_enhanced)
            
        elif self.bicep_integration_mode == 'entangled':
            # BICEP directly influences entanglement
            stochastic_entanglement = self.entanglement_bicep(x)
            
            # Reshape to entanglement matrix
            batch_size = x.size(0)
            entanglement_matrix = stochastic_entanglement.view(
                batch_size, self.config.num_neurons, self.config.num_neurons
            )
            
            # Modify ENN's entanglement parameter temporarily
            original_entanglement = self.enn_core.entanglement.data.clone()
            
            # Apply stochastic entanglement (average across batch for now)
            avg_stochastic = entanglement_matrix.mean(dim=0)
            self.enn_core.entanglement.data = (
                self.enn_core.entanglement.data + 0.1 * avg_stochastic[:, :self.config.num_states]
            )
            
            # Forward through modified ENN
            enn_output = self.enn_core(x)
            
            # Restore original entanglement
            self.enn_core.entanglement.data = original_entanglement
            
            output = self.output_layer(enn_output.reshape(enn_output.size(0), -1))
        
        else:
            raise ValueError(f"Unknown integration mode: {self.bicep_integration_mode}")
        
        # Reshape to match ENN output format
        if output.size(-1) == self.config.num_neurons * self.config.num_states:
            output = output.view(
                output.size(0), self.config.num_neurons, self.config.num_states
            )
        
        return output


def create_bicep_enhanced_model(config, integration_mode: str = 'parallel'):
    """
    Factory function to create BICEP-enhanced models for benchmarking
    
    Args:
        config: ENN configuration object
        integration_mode: 'parallel', 'sequential', or 'entangled'
    
    Returns:
        ENNBICEPHybrid model instance
    """
    return ENNBICEPHybrid(config, integration_mode)


def benchmark_bicep_integration(config, device='cpu'):
    """
    Quick benchmark of BICEP integration performance
    
    Returns:
        Dict with timing and parameter statistics
    """
    models = {}
    results = {}
    
    # Test different integration modes
    modes = ['parallel', 'sequential', 'entangled']
    
    for mode in modes:
        print(f"Benchmarking {mode} integration...")
        
        model = create_bicep_enhanced_model(config, mode).to(device)
        models[mode] = model
        
        # Generate test data
        batch_size = 16
        if hasattr(config, 'input_dim'):
            test_input = torch.randn(batch_size, config.input_dim, device=device)
        else:
            test_input = torch.randn(batch_size, config.num_states, device=device)
        
        # Warm up
        with torch.no_grad():
            _ = model(test_input)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
            end = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
            
            if device.startswith('cuda'):
                start.record()
            else:
                import time
                start_time = time.time()
            
            with torch.no_grad():
                output = model(test_input)
            
            if device.startswith('cuda'):
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                elapsed = (time.time() - start_time) * 1000
            
            times.append(elapsed)
        
        avg_time = np.mean(times)
        param_count = sum(p.numel() for p in model.parameters())
        
        results[mode] = {
            'avg_forward_time_ms': avg_time,
            'parameter_count': param_count,
            'output_shape': list(output.shape),
            'throughput_samples_per_sec': (batch_size / avg_time) * 1000
        }
        
        print(f"  {mode}: {avg_time:.2f}ms, {param_count:,} params")
    
    return results


if __name__ == "__main__":
    # Quick test
    from enn.config import Config
    
    config = Config()
    config.num_neurons = 8
    config.num_states = 4
    config.input_dim = 5
    
    print("Testing BICEP integration layers...")
    results = benchmark_bicep_integration(config)
    
    print("\nBenchmark Results:")
    for mode, stats in results.items():
        print(f"{mode}: {stats['avg_forward_time_ms']:.2f}ms, {stats['parameter_count']:,} params")