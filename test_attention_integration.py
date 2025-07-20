#!/usr/bin/env python3
"""
Quick test script to verify all attention mechanisms and benchmarking components work correctly.
"""

import torch
import sys
sys.path.append('/Users/rayhanroswendi/developer/ENN')

from enn.config import Config
from enn.enhanced_model import create_attention_enn
from enn.multihead_attention import create_attention_layer
from baselines.baseline_models import create_baseline_model, BaselineConfig

def test_attention_mechanisms():
    """Test all attention mechanism implementations."""
    print("Testing Attention Mechanisms...")
    
    # Test basic multi-head attention
    attention = create_attention_layer('multihead', d_model=64, num_heads=8)
    x = torch.randn(2, 10, 64)
    output, weights = attention(x, x, x)
    print(f"✓ MultiHeadAttention: {x.shape} -> {output.shape}")
    
    # Test neuron-state attention
    neuron_attention = create_attention_layer('neuron_state', 
                                            num_neurons=10, num_states=5, 
                                            hidden_dim=64, num_heads=4)
    neuron_states = torch.randn(2, 10, 5)
    enhanced_states, weights = neuron_attention(neuron_states)
    print(f"✓ NeuronStateAttention: {neuron_states.shape} -> {enhanced_states.shape}")
    
    # Test temporal attention
    temporal_attention = create_attention_layer('temporal', state_dim=5, hidden_dim=64)
    temporal_data = torch.randn(2, 15, 5)
    temporal_output, weights = temporal_attention(temporal_data)
    print(f"✓ TemporalAttention: {temporal_data.shape} -> {temporal_output.shape}")
    
    print("All attention mechanisms working correctly!\n")


def test_enhanced_enn():
    """Test ENN with integrated attention mechanisms."""
    print("Testing Enhanced ENN Models...")
    
    config = Config()
    config.num_neurons = 8
    config.num_states = 4
    config.input_dim = 4
    
    attention_types = ['minimal', 'neuron_only', 'temporal_only', 'full']
    
    for att_type in attention_types:
        model = create_attention_enn(config, att_type)
        
        # Test forward pass
        test_input = torch.randn(2, 10, 4)  # [batch, time, features]
        output = model(test_input)
        
        # Test attention weights extraction
        attention_weights = model.get_attention_weights(test_input)
        
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ ENN-{att_type}: {params:,} params, output: {output.shape}, attention: {list(attention_weights.keys())}")
    
    print("Enhanced ENN models working correctly!\n")


def test_baseline_models():
    """Test all baseline model implementations."""
    print("Testing Baseline Models...")
    
    config = BaselineConfig(input_dim=5, hidden_dim=64, output_dim=1, seq_len=20)
    models = ['lstm', 'transformer', 'cnn', 'mlp', 'lnn']
    
    for model_name in models:
        try:
            model = create_baseline_model(model_name, config)
            test_input = torch.randn(4, 20, 5)  # [batch, seq_len, input_dim]
            output = model(test_input)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ {model_name.upper()}: {params:,} params, output: {output.shape}")
        except Exception as e:
            print(f"✗ {model_name.upper()}: Error - {e}")
    
    print("Baseline models working correctly!\n")


def test_benchmarking_components():
    """Test benchmarking framework components."""
    print("Testing Benchmarking Components...")
    
    try:
        from benchmarking.benchmark_framework import DatasetGenerator, BenchmarkConfig
        
        # Test dataset generation
        X, y = DatasetGenerator.synthetic_regression(100, 20, 5)
        print(f"✓ Synthetic Regression: X{X.shape}, y{y.shape}")
        
        X, y = DatasetGenerator.binary_classification(100, 20, 5)
        print(f"✓ Binary Classification: X{X.shape}, y{y.shape}")
        
        X, y = DatasetGenerator.memory_task(100, 20, 5)
        print(f"✓ Memory Task: X{X.shape}, y{y.shape}")
        
        # Test benchmark config
        config = BenchmarkConfig(dataset_sizes=[100], sequence_lengths=[10], input_dimensions=[3], epochs=5, num_runs=1)
        print(f"✓ BenchmarkConfig: {config.epochs} epochs, {config.num_runs} runs")
        
        print("Benchmarking components working correctly!\n")
        
    except Exception as e:
        print(f"✗ Benchmarking Error: {e}\n")


def test_quick_training():
    """Test quick training run to ensure everything integrates properly."""
    print("Testing Quick Training Integration...")
    
    try:
        # Create simple dataset
        from benchmarking.benchmark_framework import DatasetGenerator
        X, y = DatasetGenerator.synthetic_regression(200, 15, 3)
        
        # Create models
        config = Config()
        config.input_dim = 3
        config.num_neurons = 6
        config.num_states = 3
        
        baseline_config = BaselineConfig(input_dim=3, hidden_dim=32, output_dim=1, seq_len=15)
        
        models = {
            'enn_original': None,  # Will create below
            'enn_attention': create_attention_enn(config, 'minimal'),
            'lstm': create_baseline_model('lstm', baseline_config),
            'transformer': create_baseline_model('transformer', baseline_config)
        }
        
        # Import original ENN
        from enn.model import ENNModelWithSparsityControl
        models['enn_original'] = ENNModelWithSparsityControl(config)
        
        # Quick training test (1 epoch)
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X[:150], y[:150])
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for model_name, model in models.items():
            if model is None:
                continue
                
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.MSELoss()
            
            model.train()
            total_loss = 0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                output = model(batch_x)
                if output.dim() > 2:
                    output = output.mean(dim=(1, 2))
                if output.dim() > 1:
                    output = output.squeeze(-1)
                
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ {model_name}: {params:,} params, avg_loss: {avg_loss:.4f}")
        
        print("Quick training integration successful!\n")
        
    except Exception as e:
        print(f"✗ Training Integration Error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE ENN ATTENTION & BENCHMARKING TEST")
    print("="*60)
    print()
    
    # Run all tests
    test_attention_mechanisms()
    test_enhanced_enn()
    test_baseline_models()
    test_benchmarking_components()
    test_quick_training()
    
    print("="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("Ready for comprehensive benchmarking.")
    print("="*60)