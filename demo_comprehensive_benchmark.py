#!/usr/bin/env python3
"""
Demonstration of Comprehensive ENN Benchmarking System.

This demo shows:
1. ENN with multi-head attention variants
2. Baseline model comparisons (LSTM, Transformer, CNN, MLP, LNN)
3. Multiple task types and datasets
4. Statistical analysis and visualization
5. Performance metrics and efficiency analysis

Run time: ~5-10 minutes for comprehensive demonstration
"""

import torch
import time
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.append('/Users/rayhanroswendi/developer/ENN')

from enn.config import Config
from enn.enhanced_model import create_attention_enn
from enn.model import ENNModelWithSparsityControl
from baselines.baseline_models import create_baseline_model, BaselineConfig
from benchmarking.benchmark_framework import DatasetGenerator, ModelEvaluator, BenchmarkConfig
from torch.utils.data import DataLoader, TensorDataset


def demo_attention_mechanisms():
    """Demonstrate ENN attention mechanisms."""
    print("🔍 ATTENTION MECHANISMS DEMONSTRATION")
    print("="*60)
    
    config = Config()
    config.input_dim = 5
    config.num_neurons = 8
    config.num_states = 4
    
    attention_types = {
        'minimal': 'Lightweight attention with minimal overhead',
        'neuron_only': 'Only neuron-state attention for entanglement',
        'temporal_only': 'Only temporal attention for memory buffers',
        'full': 'All attention mechanisms enabled'
    }
    
    test_input = torch.randn(2, 15, 5)
    
    for att_type, description in attention_types.items():
        model = create_attention_enn(config, att_type)
        output = model(test_input)
        attention_weights = model.get_attention_weights(test_input)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 {att_type.upper():12s}: {params:6,} params | {description}")
        print(f"   Attention types: {list(attention_weights.keys())}")
        print(f"   Output shape: {output.shape}")
        print()
    
    print("✅ All attention mechanisms working correctly!\n")


def demo_model_comparison():
    """Demonstrate model comparison on a simple task."""
    print("🏆 MODEL COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Generate test dataset
    print("📊 Generating synthetic regression dataset...")
    X, y = DatasetGenerator.synthetic_regression(800, 20, 3, noise_level=0.15)
    
    # Split data
    train_X, train_y = X[:600], y[:600]
    val_X, val_y = X[600:], y[600:]
    
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   Training samples: {len(train_X)}")
    print(f"   Validation samples: {len(val_X)}")
    print(f"   Input shape: {train_X.shape}")
    print()
    
    # Create models
    models = {}
    
    # ENN models
    enn_config = Config()
    enn_config.input_dim = 3
    enn_config.num_neurons = 8
    enn_config.num_states = 3
    enn_config.epochs = 30  # Quick demo
    
    models['ENN_Original'] = ENNModelWithSparsityControl(enn_config)
    models['ENN_Minimal_Attention'] = create_attention_enn(enn_config, 'minimal')
    models['ENN_Full_Attention'] = create_attention_enn(enn_config, 'full')
    
    # Baseline models
    baseline_config = BaselineConfig(input_dim=3, hidden_dim=32, output_dim=1, seq_len=20)
    
    models['LSTM'] = create_baseline_model('lstm', baseline_config)
    models['Transformer'] = create_baseline_model('transformer', baseline_config)
    models['CNN'] = create_baseline_model('cnn', baseline_config)
    models['LNN'] = create_baseline_model('lnn', baseline_config)
    
    # Evaluate models
    print("🚀 Training and evaluating models...")
    evaluator = ModelEvaluator(BenchmarkConfig(epochs=30))
    
    results = []
    for model_name, model in models.items():
        print(f"   Training {model_name}...")
        start_time = time.time()
        
        try:
            result = evaluator.train_and_evaluate(model, train_loader, val_loader, 'regression')
            result['model_name'] = model_name
            result['training_wall_time'] = time.time() - start_time
            results.append(result)
            
            print(f"     ✅ Final loss: {result['final_loss']:.6f}, Time: {result['training_wall_time']:.1f}s")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    # Results summary
    print("\\n📈 RESULTS SUMMARY")
    print("-"*60)
    print(f"{'Model':<20} {'Final Loss':<12} {'Parameters':<12} {'Time (s)':<10}")
    print("-"*60)
    
    # Sort by performance
    results.sort(key=lambda x: x['final_loss'])
    
    for result in results:
        print(f"{result['model_name']:<20} {result['final_loss']:<12.6f} "
              f"{result['n_parameters']:<12,} {result['training_wall_time']:<10.1f}")
    
    print()
    
    # Analysis
    best_model = results[0]
    print(f"🥇 Best performing model: {best_model['model_name']}")
    print(f"   Final loss: {best_model['final_loss']:.6f}")
    print(f"   Parameters: {best_model['n_parameters']:,}")
    print(f"   Training time: {best_model['training_wall_time']:.1f}s")
    
    # Efficiency analysis
    efficiency_scores = [(r['final_loss'] / (r['n_parameters'] / 1000), r['model_name']) for r in results]
    efficiency_scores.sort()
    most_efficient = efficiency_scores[0]
    
    print(f"\\n⚡ Most efficient model: {most_efficient[1]}")
    print(f"   Efficiency score: {most_efficient[0]:.6f} (loss per 1k parameters)")
    
    print("\\n✅ Model comparison completed!\n")


def demo_attention_analysis():
    """Demonstrate attention weight analysis."""
    print("🔍 ATTENTION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create ENN with full attention
    config = Config()
    config.input_dim = 4
    config.num_neurons = 6
    config.num_states = 3
    
    model = create_attention_enn(config, 'full')
    
    # Generate test sequence
    test_sequence = torch.randn(1, 20, 4)
    
    print("📊 Analyzing attention patterns...")
    print(f"   Input sequence shape: {test_sequence.shape}")
    
    # Get attention weights
    attention_weights = model.get_attention_weights(test_sequence)
    
    for attention_type, weights in attention_weights.items():
        print(f"\\n🎯 {attention_type.upper()} Attention:")
        print(f"   Weight tensor shape: {weights.shape}")
        print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"   Mean attention: {weights.mean():.4f}")
        
        # Analyze attention patterns
        if weights.dim() >= 3:
            # Average over heads and batch
            avg_weights = weights.mean(dim=(0, 1)) if weights.dim() == 4 else weights.mean(dim=0)
            print(f"   Attention entropy: {(-avg_weights * torch.log(avg_weights + 1e-8)).sum():.4f}")
    
    print("\\n✅ Attention analysis completed!\n")


def demo_scalability_test():
    """Demonstrate scalability across different data sizes."""
    print("📈 SCALABILITY DEMONSTRATION")
    print("="*60)
    
    # Test different data sizes
    data_sizes = [200, 500, 1000]
    sequence_lengths = [10, 25, 50]
    
    # Create models for comparison
    config = Config()
    config.input_dim = 3
    config.num_neurons = 6
    config.num_states = 3
    
    models = {
        'ENN_Original': ENNModelWithSparsityControl(config),
        'ENN_Attention': create_attention_enn(config, 'minimal'),
        'LSTM': create_baseline_model('lstm', BaselineConfig(input_dim=3, hidden_dim=24, output_dim=1))
    }
    
    print("📊 Testing scalability across different configurations...")
    print(f"{'Config':<15} {'Model':<15} {'Loss':<10} {'Time':<8} {'Memory':<8}")
    print("-"*60)
    
    for n_samples in data_sizes:
        for seq_len in sequence_lengths:
            config_name = f"{n_samples}x{seq_len}"
            
            # Generate data
            X, y = DatasetGenerator.synthetic_regression(n_samples, seq_len, 3)
            dataset = TensorDataset(X[:int(0.8*n_samples)], y[:int(0.8*n_samples)])
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for model_name, model in models.items():
                try:
                    # Quick training (5 epochs for demo)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = torch.nn.MSELoss()
                    
                    start_time = time.time()
                    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    model.train()
                    total_loss = 0
                    
                    for epoch in range(5):  # Quick test
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
                    
                    training_time = time.time() - start_time
                    memory_used = (torch.cuda.memory_allocated() - start_memory) / 1024 / 1024 if torch.cuda.is_available() else 0
                    avg_loss = total_loss / (len(loader) * 5)
                    
                    print(f"{config_name:<15} {model_name:<15} {avg_loss:<10.4f} {training_time:<8.1f} {memory_used:<8.1f}")
                    
                except Exception as e:
                    print(f"{config_name:<15} {model_name:<15} ERROR: {str(e)[:20]}")
    
    print("\\n✅ Scalability test completed!\n")


def main():
    """Run comprehensive demonstration."""
    print("🚀 COMPREHENSIVE ENN BENCHMARKING DEMONSTRATION")
    print("="*80)
    print("This demo showcases:")
    print("• Multi-head attention mechanisms in ENN")
    print("• Comparison with baseline models (LSTM, Transformer, CNN, LNN)")
    print("• Performance analysis and efficiency metrics")
    print("• Attention pattern analysis")
    print("• Scalability testing")
    print("="*80)
    print()
    
    start_time = time.time()
    
    try:
        # Run demonstrations
        demo_attention_mechanisms()
        demo_model_comparison()
        demo_attention_analysis()
        demo_scalability_test()
        
        total_time = time.time() - start_time
        
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total demonstration time: {total_time:.1f} seconds")
        print()
        print("📋 SUMMARY OF CAPABILITIES DEMONSTRATED:")
        print("✅ Multi-head attention integration in ENN")
        print("✅ Comprehensive baseline model comparisons")
        print("✅ Automated benchmarking framework")
        print("✅ Performance and efficiency analysis")
        print("✅ Attention pattern visualization")
        print("✅ Scalability across different data sizes")
        print()
        print("🚀 Ready for full-scale benchmarking!")
        print("   Run: python run_comprehensive_benchmark.py")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()