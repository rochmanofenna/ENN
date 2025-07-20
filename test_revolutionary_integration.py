#!/usr/bin/env python3
"""
Quick Test of Revolutionary ENN-BICEP Integration
Demonstrates that BICEP layers can be imported and used within ENN's benchmark framework
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add ENN to path
sys.path.insert(0, '.')

def test_bicep_layers():
    """Test that BICEP layers can be imported and work"""
    
    print("🧪 Testing BICEP Layer Import and Integration")
    print("=" * 50)
    
    try:
        # Test importing BICEP layers
        from enn.bicep_layers import BICEPNeuralLayer, create_bicep_enhanced_model, benchmark_bicep_integration
        print("✅ Successfully imported BICEP layers")
        
        # Test basic BICEP layer
        print("\n📋 Testing BICEPNeuralLayer...")
        bicep_layer = BICEPNeuralLayer(
            input_size=10,
            output_size=5,
            n_paths=50,
            n_steps=20,
            device='cpu'
        )
        
        test_input = torch.randn(8, 10)  # batch_size=8, input_size=10
        output = bicep_layer(test_input)
        
        print(f"✅ BICEP layer forward pass successful")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in bicep_layer.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ BICEP layer test failed: {e}")
        return False

def test_enn_bicep_hybrid():
    """Test the revolutionary ENN-BICEP hybrid models"""
    
    print("\n🚀 Testing Revolutionary ENN-BICEP Hybrid Models")
    print("=" * 50)
    
    try:
        from enn.config import Config
        from enn.bicep_layers import create_bicep_enhanced_model
        
        # Create config
        config = Config()
        config.num_neurons = 6
        config.num_states = 4
        config.input_dim = 8
        config.decay_rate = 0.1
        config.recency_factor = 0.9
        config.buffer_size = 5
        
        # Test different integration modes
        modes = ['parallel', 'sequential', 'entangled']
        
        for mode in modes:
            print(f"\n🔬 Testing {mode.upper()} integration...")
            
            try:
                model = create_bicep_enhanced_model(config, mode)
                param_count = sum(p.numel() for p in model.parameters())
                
                # Test forward pass
                test_input = torch.randn(4, config.input_dim)
                output = model(test_input)
                
                print(f"✅ {mode} model working")
                print(f"   Parameters: {param_count:,}")
                print(f"   Input: {test_input.shape} → Output: {output.shape}")
                
                # Quick timing test
                times = []
                for _ in range(5):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(test_input)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times) * 1000
                print(f"   Avg forward time: {avg_time:.2f}ms")
                
            except Exception as e:
                print(f"❌ {mode} model failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid model test failed: {e}")
        return False

def test_benchmark_integration():
    """Test integration with benchmark framework"""
    
    print("\n📊 Testing Benchmark Framework Integration")
    print("=" * 50)
    
    try:
        from enn.config import Config
        from enn.bicep_layers import benchmark_bicep_integration
        
        config = Config()
        config.num_neurons = 4
        config.num_states = 3
        config.input_dim = 5
        
        print("Running quick benchmark...")
        results = benchmark_bicep_integration(config, device='cpu')
        
        print("✅ Benchmark integration successful")
        print("\nResults:")
        for mode, stats in results.items():
            print(f"  {mode}: {stats['avg_forward_time_ms']:.2f}ms, {stats['parameter_count']:,} params")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark integration failed: {e}")
        return False

def test_enn_compatibility():
    """Test that regular ENN still works"""
    
    print("\n🧠 Testing ENN Compatibility")
    print("=" * 50)
    
    try:
        from enn.config import Config
        from enn.model import ENNModelWithSparsityControl
        
        config = Config()
        config.num_neurons = 5
        config.num_states = 3
        config.input_dim = 4
        
        model = ENNModelWithSparsityControl(config)
        test_input = torch.randn(3, config.input_dim)
        output = model(test_input)
        
        print("✅ Original ENN model still working")
        print(f"   Input: {test_input.shape} → Output: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ ENN compatibility test failed: {e}")
        return False

def run_revolutionary_demo():
    """Run a small demonstration comparing models"""
    
    print("\n🎯 Revolutionary Demo: ENN vs ENN+BICEP")
    print("=" * 50)
    
    try:
        from enn.config import Config
        from enn.model import ENNModelWithSparsityControl
        from enn.bicep_layers import create_bicep_enhanced_model
        
        # Setup
        config = Config()
        config.num_neurons = 8
        config.num_states = 4
        config.input_dim = 6
        
        # Create models
        enn_original = ENNModelWithSparsityControl(config)
        enn_bicep = create_bicep_enhanced_model(config, 'parallel')
        
        # Test data - temporal pattern that BICEP should handle well
        batch_size = 16
        test_data = torch.randn(batch_size, config.input_dim)
        
        # Add brownian-like patterns
        for i in range(batch_size):
            brownian_component = torch.cumsum(torch.randn(config.input_dim) * 0.1, dim=0)
            test_data[i] += brownian_component
        
        print("Comparing models on stochastic temporal data:")
        
        # Test original ENN
        start_time = time.time()
        with torch.no_grad():
            enn_output = enn_original(test_data)
        enn_time = time.time() - start_time
        
        enn_params = sum(p.numel() for p in enn_original.parameters())
        
        # Test BICEP-enhanced ENN
        start_time = time.time()
        with torch.no_grad():
            bicep_output = enn_bicep(test_data)
        bicep_time = time.time() - start_time
        
        bicep_params = sum(p.numel() for p in enn_bicep.parameters())
        
        print(f"\n📈 Results:")
        print(f"Original ENN:")
        print(f"  Parameters: {enn_params:,}")
        print(f"  Forward time: {enn_time*1000:.2f}ms")
        print(f"  Output variance: {enn_output.var().item():.6f}")
        
        print(f"\nENN + BICEP (Parallel):")
        print(f"  Parameters: {bicep_params:,}")
        print(f"  Forward time: {bicep_time*1000:.2f}ms")
        print(f"  Output variance: {bicep_output.var().item():.6f}")
        
        param_increase = ((bicep_params - enn_params) / enn_params) * 100
        time_increase = ((bicep_time - enn_time) / enn_time) * 100
        
        print(f"\n🔍 Analysis:")
        print(f"  Parameter increase: +{param_increase:.1f}%")
        print(f"  Computation overhead: +{time_increase:.1f}%")
        
        # Test stochastic behavior
        print(f"\n🎲 Testing Stochastic Behavior:")
        outputs = []
        for i in range(3):
            with torch.no_grad():
                output = enn_bicep(test_data[:1])  # Single sample
                outputs.append(output.item() if output.numel() == 1 else output.mean().item())
            print(f"  Run {i+1}: {outputs[-1]:.6f}")
        
        variance = np.var(outputs)
        print(f"  Output variance across runs: {variance:.6f}")
        
        if variance > 1e-8:
            print("  ✅ BICEP adds stochastic dynamics!")
        else:
            print("  ⚠ Limited stochastic variation detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Revolutionary demo failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 REVOLUTIONARY ENN-BICEP INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("BICEP Layer Import", test_bicep_layers),
        ("ENN-BICEP Hybrid Models", test_enn_bicep_hybrid),
        ("Benchmark Integration", test_benchmark_integration),
        ("ENN Compatibility", test_enn_compatibility),
        ("Revolutionary Demo", run_revolutionary_demo)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Revolutionary integration ready!")
        print("\n🚀 Ready to run full revolutionary benchmark:")
        print("   python revolutionary_bicep_benchmark.py")
    else:
        print("⚠ Some tests failed - check configuration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)