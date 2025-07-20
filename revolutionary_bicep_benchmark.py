#!/usr/bin/env python3
"""
Revolutionary ENN-BICEP Benchmark Suite
Tests groundbreaking combinations of Entangled Neural Networks with BICEP stochastic computation

This benchmark demonstrates how BICEP's brownian motion dynamics can supercharge
ENN's entangled neuron architecture to create revolutionary neural computation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import ENN components
from enn.config import Config
from enn.model import ENNModelWithSparsityControl
from enn.enhanced_model import create_attention_enn
from baselines.baseline_models import create_baseline_model, BaselineConfig

# Import our BICEP integration
from enn.bicep_layers import create_bicep_enhanced_model, benchmark_bicep_integration

# Import benchmarking framework
from benchmarking.benchmark_framework import BenchmarkConfig, DatasetGenerator, BenchmarkFramework


class RevolutionaryBenchmark:
    """
    Revolutionary benchmark comparing:
    1. Standard baselines (LSTM, Transformer, CNN)
    2. Original ENN
    3. ENN + Attention
    4. ENN + BICEP (Parallel)
    5. ENN + BICEP (Sequential) 
    6. ENN + BICEP (Entangled) - REVOLUTIONARY
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        self.timing_results = {}
        
    def create_revolutionary_models(self, config: Config) -> Dict[str, nn.Module]:
        """Create all model variants for revolutionary comparison"""
        
        models = {}
        
        # 1. Baseline models
        baseline_config = BaselineConfig(
            input_dim=config.input_dim if hasattr(config, 'input_dim') else config.num_states,
            hidden_dim=64,
            output_dim=config.num_neurons * config.num_states,
            num_layers=2,
            seq_len=20
        )
        
        models['LSTM'] = create_baseline_model('lstm', baseline_config)
        models['Transformer'] = create_baseline_model('transformer', baseline_config)
        models['CNN'] = create_baseline_model('cnn', baseline_config)
        
        # 2. ENN variants
        models['ENN_Original'] = ENNModelWithSparsityControl(config)
        models['ENN_Attention_Minimal'] = create_attention_enn(config, 'minimal')
        models['ENN_Attention_Full'] = create_attention_enn(config, 'full')
        
        # 3. REVOLUTIONARY ENN-BICEP HYBRIDS
        models['ENN_BICEP_Parallel'] = create_bicep_enhanced_model(config, 'parallel')
        models['ENN_BICEP_Sequential'] = create_bicep_enhanced_model(config, 'sequential')
        models['ENN_BICEP_Entangled'] = create_bicep_enhanced_model(config, 'entangled')  # REVOLUTIONARY!
        
        return models
    
    def generate_revolutionary_datasets(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate datasets that highlight revolutionary capabilities"""
        
        datasets = {}
        
        # 1. Temporal Memory Challenge - Tests long-term dependencies
        print("Generating Temporal Memory Challenge...")
        X_memory, y_memory = DatasetGenerator.synthetic_regression(
            n_samples=1000,
            seq_len=50,  # Long sequences
            input_dim=5,
            noise_level=0.15
        )
        datasets['Temporal_Memory'] = (X_memory, y_memory)
        
        # 2. Stochastic Pattern Recognition - Leverages BICEP's strengths  
        print("Generating Stochastic Pattern Recognition...")
        X_stochastic = torch.randn(1000, 20, 3)
        # Add brownian-like patterns that BICEP should excel at
        for i in range(X_stochastic.size(0)):
            brownian_component = torch.cumsum(torch.randn(20, 3) * 0.1, dim=0)
            X_stochastic[i] += brownian_component
        
        # Target: predict final brownian displacement
        y_stochastic = X_stochastic[:, -1, :].sum(dim=1, keepdim=True)
        datasets['Stochastic_Pattern'] = (X_stochastic, y_stochastic)
        
        # 3. Entanglement Challenge - Tests neuron interactions
        print("Generating Entanglement Challenge...")
        X_entangle = torch.randn(1000, 30, 4)
        # Create entangled relationships between different time points
        y_entangle = (X_entangle[:, ::3, :].mean(dim=1) * 
                     X_entangle[:, 1::3, :].mean(dim=1) +
                     X_entangle[:, 2::3, :].mean(dim=1))
        datasets['Entanglement_Challenge'] = (X_entangle, y_entangle)
        
        # 4. Multi-Scale Temporal Dynamics
        print("Generating Multi-Scale Temporal...")
        X_multiscale = torch.randn(1000, 40, 6)
        # Add patterns at different time scales
        short_pattern = torch.sin(torch.arange(40).float() * 0.5).unsqueeze(0).unsqueeze(-1)
        long_pattern = torch.sin(torch.arange(40).float() * 0.1).unsqueeze(0).unsqueeze(-1)
        X_multiscale += short_pattern + long_pattern
        
        y_multiscale = (X_multiscale[:, -5:, :].mean(dim=1) - 
                       X_multiscale[:, :5, :].mean(dim=1)).mean(dim=1, keepdim=True)
        datasets['MultiScale_Temporal'] = (X_multiscale, y_multiscale)
        
        return datasets
    
    def train_and_evaluate_model(self, model: nn.Module, model_name: str,
                               X: torch.Tensor, y: torch.Tensor, 
                               dataset_name: str) -> Dict[str, Any]:
        """Train and evaluate a single model"""
        
        model = model.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        
        # Split data
        n_train = int(0.8 * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Training metrics
        train_losses = []
        val_losses = []
        epoch_times = []
        
        print(f"Training {model_name} on {dataset_name}...")
        
        # Training loop
        epochs = 114 if 'ENN' in model_name else 50  # ENN gets more epochs as in original
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Handle different output formats
            try:
                output = model(X_train)
                if isinstance(output, dict):
                    output = output['output']
                elif output.dim() > 2:
                    output = output.reshape(output.size(0), -1)
                    
                # Match target dimensions
                if output.size(-1) != y_train.size(-1):
                    output = output[:, :y_train.size(-1)]
                    
                loss = criterion(output, y_train)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
                
            except Exception as e:
                print(f"Training error for {model_name}: {e}")
                return {'error': str(e)}
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        val_output = model(X_val)
                        if isinstance(val_output, dict):
                            val_output = val_output['output']
                        elif val_output.dim() > 2:
                            val_output = val_output.reshape(val_output.size(0), -1)
                            
                        if val_output.size(-1) != y_val.size(-1):
                            val_output = val_output[:, :y_val.size(-1)]
                            
                        val_loss = criterion(val_output, y_val)
                        val_losses.append(val_loss.item())
                        
                    except Exception as e:
                        val_losses.append(float('inf'))
            
            epoch_times.append(time.time() - start_time)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            try:
                final_output = model(X_val)
                if isinstance(final_output, dict):
                    final_output = final_output['output']
                elif final_output.dim() > 2:
                    final_output = final_output.reshape(final_output.size(0), -1)
                    
                if final_output.size(-1) != y_val.size(-1):
                    final_output = final_output[:, :y_val.size(-1)]
                    
                final_loss = criterion(final_output, y_val).item()
                
            except Exception as e:
                final_loss = float('inf')
        
        # Model statistics
        param_count = sum(p.numel() for p in model.parameters())
        avg_epoch_time = np.mean(epoch_times)
        
        return {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'final_validation_loss': final_loss,
            'best_validation_loss': min(val_losses) if val_losses else final_loss,
            'parameter_count': param_count,
            'avg_epoch_time': avg_epoch_time,
            'total_training_time': sum(epoch_times),
            'convergence_speed': len([l for l in train_losses if l > train_losses[-1] * 2]),
            'train_losses': train_losses[-10:],  # Last 10 for analysis
            'val_losses': val_losses[-3:] if val_losses else [final_loss]
        }
    
    def run_revolutionary_benchmark(self):
        """Run the complete revolutionary benchmark suite"""
        
        print("🚀 STARTING REVOLUTIONARY ENN-BICEP BENCHMARK 🚀")
        print("=" * 60)
        
        # Configuration
        config = Config()
        config.num_neurons = 10
        config.num_states = 5
        config.input_dim = 6  # Will be adjusted per dataset
        config.decay_rate = 0.1
        config.recency_factor = 0.9
        config.buffer_size = 5
        
        # Generate datasets
        datasets = self.generate_revolutionary_datasets()
        
        # Create models
        print("\nCreating revolutionary model architectures...")
        models = self.create_revolutionary_models(config)
        
        print(f"Created {len(models)} models:")
        for name in models.keys():
            param_count = sum(p.numel() for p in models[name].parameters())
            print(f"  - {name}: {param_count:,} parameters")
        
        # Run benchmarks
        all_results = []
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\n📊 BENCHMARKING ON {dataset_name.upper()}")
            print("-" * 50)
            
            # Adjust config for dataset
            current_config = Config()
            current_config.num_neurons = config.num_neurons
            current_config.num_states = config.num_states
            current_config.input_dim = X.size(-1)
            current_config.decay_rate = config.decay_rate
            current_config.recency_factor = config.recency_factor
            current_config.buffer_size = config.buffer_size
            
            # Recreate models with correct input dimension
            dataset_models = self.create_revolutionary_models(current_config)
            
            for model_name, model in dataset_models.items():
                try:
                    result = self.train_and_evaluate_model(
                        model, model_name, X, y, dataset_name
                    )
                    if 'error' not in result:
                        all_results.append(result)
                        print(f"✓ {model_name}: Loss={result['final_validation_loss']:.6f}, "
                              f"Params={result['parameter_count']:,}")
                    else:
                        print(f"✗ {model_name}: {result['error']}")
                        
                except Exception as e:
                    print(f"✗ {model_name}: Failed with {e}")
        
        # Analyze results
        self.analyze_revolutionary_results(all_results)
        return all_results
    
    def analyze_revolutionary_results(self, results: List[Dict]):
        """Analyze and visualize revolutionary benchmark results"""
        
        if not results:
            print("No results to analyze!")
            return
            
        df = pd.DataFrame(results)
        
        print("\n🏆 REVOLUTIONARY BENCHMARK RESULTS 🏆")
        print("=" * 60)
        
        # Performance analysis by dataset
        for dataset in df['dataset_name'].unique():
            dataset_results = df[df['dataset_name'] == dataset].copy()
            dataset_results = dataset_results.sort_values('final_validation_loss')
            
            print(f"\n📈 {dataset} Results (Lower Loss = Better):")
            print("-" * 40)
            
            for _, row in dataset_results.iterrows():
                model = row['model_name']
                loss = row['final_validation_loss']
                params = row['parameter_count']
                time_per_epoch = row['avg_epoch_time']
                
                if loss != float('inf'):
                    efficiency = params / (1 + loss)  # Higher = better efficiency
                    print(f"  {model:20} | Loss: {loss:.6f} | Params: {params:6,} | "
                          f"Time: {time_per_epoch:.3f}s | Efficiency: {efficiency:.0f}")
                else:
                    print(f"  {model:20} | FAILED")
        
        # Overall champion analysis
        print(f"\n🥇 OVERALL CHAMPIONS")
        print("-" * 30)
        
        # Best by loss
        valid_results = df[df['final_validation_loss'] != float('inf')]
        if not valid_results.empty:
            best_overall = valid_results.loc[valid_results['final_validation_loss'].idxmin()]
            print(f"🏆 Best Overall Performance: {best_overall['model_name']}")
            print(f"   Loss: {best_overall['final_validation_loss']:.6f}")
            print(f"   Dataset: {best_overall['dataset_name']}")
            
            # Best efficiency (performance per parameter)
            valid_results['efficiency'] = (
                valid_results['parameter_count'] / 
                (1 + valid_results['final_validation_loss'])
            )
            best_efficiency = valid_results.loc[valid_results['efficiency'].idxmax()]
            print(f"\n⚡ Most Efficient: {best_efficiency['model_name']}")
            print(f"   Efficiency Score: {best_efficiency['efficiency']:.0f}")
            print(f"   Loss: {best_efficiency['final_validation_loss']:.6f}")
            print(f"   Parameters: {best_efficiency['parameter_count']:,}")
        
        # Revolutionary insights
        self.generate_revolutionary_insights(df)
        
        # Save results
        results_dir = Path("revolutionary_benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        df.to_csv(results_dir / "revolutionary_results.csv", index=False)
        
        # Create visualizations
        self.create_revolutionary_visualizations(df, results_dir)
        
        print(f"\n💾 Results saved to {results_dir}/")
        print("🎯 Revolutionary benchmark complete!")
    
    def generate_revolutionary_insights(self, df: pd.DataFrame):
        """Generate insights about revolutionary combinations"""
        
        print(f"\n🔬 REVOLUTIONARY INSIGHTS")
        print("-" * 30)
        
        valid_results = df[df['final_validation_loss'] != float('inf')]
        if valid_results.empty:
            print("No valid results for analysis")
            return
        
        # BICEP vs non-BICEP comparison
        bicep_models = valid_results[valid_results['model_name'].str.contains('BICEP')]
        non_bicep_models = valid_results[~valid_results['model_name'].str.contains('BICEP')]
        
        if not bicep_models.empty and not non_bicep_models.empty:
            bicep_avg_loss = bicep_models['final_validation_loss'].mean()
            non_bicep_avg_loss = non_bicep_models['final_validation_loss'].mean()
            
            improvement = ((non_bicep_avg_loss - bicep_avg_loss) / non_bicep_avg_loss) * 100
            print(f"🚀 BICEP Enhancement: {improvement:.1f}% average improvement")
            
            if improvement > 0:
                print("   ✓ BICEP integration shows positive impact!")
            else:
                print("   ⚠ BICEP integration needs optimization")
        
        # Entangled BICEP analysis
        entangled_results = valid_results[
            valid_results['model_name'].str.contains('Entangled')
        ]
        if not entangled_results.empty:
            entangled_performance = entangled_results['final_validation_loss'].mean()
            overall_avg = valid_results['final_validation_loss'].mean()
            
            entangled_improvement = ((overall_avg - entangled_performance) / overall_avg) * 100
            print(f"🧬 Entangled BICEP: {entangled_improvement:.1f}% vs average")
            
            if entangled_improvement > 10:
                print("   🎯 REVOLUTIONARY: Entangled BICEP shows breakthrough performance!")
            elif entangled_improvement > 0:
                print("   ✓ Entangled BICEP shows promise")
            else:
                print("   🔧 Entangled BICEP needs further optimization")
        
        # Parameter efficiency
        enn_models = valid_results[valid_results['model_name'].str.contains('ENN')]
        baseline_models = valid_results[valid_results['model_name'].isin(['LSTM', 'Transformer', 'CNN'])]
        
        if not enn_models.empty and not baseline_models.empty:
            enn_avg_params = enn_models['parameter_count'].mean()
            baseline_avg_params = baseline_models['parameter_count'].mean()
            
            param_reduction = ((baseline_avg_params - enn_avg_params) / baseline_avg_params) * 100
            print(f"📉 Parameter Efficiency: {param_reduction:.1f}% fewer parameters with ENN")
    
    def create_revolutionary_visualizations(self, df: pd.DataFrame, save_dir: Path):
        """Create visualizations of revolutionary results"""
        
        valid_results = df[df['final_validation_loss'] != float('inf')]
        if valid_results.empty:
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance by model type
        ax1 = axes[0, 0]
        model_performance = valid_results.groupby('model_name')['final_validation_loss'].mean()
        colors = ['red' if 'BICEP' in name else 'blue' if 'ENN' in name else 'gray' 
                 for name in model_performance.index]
        
        bars = ax1.bar(range(len(model_performance)), model_performance.values, color=colors)
        ax1.set_xticks(range(len(model_performance)))
        ax1.set_xticklabels(model_performance.index, rotation=45, ha='right')
        ax1.set_ylabel('Validation Loss (Lower = Better)')
        ax1.set_title('Revolutionary Model Performance')
        ax1.set_yscale('log')
        
        # 2. Parameter efficiency
        ax2 = axes[0, 1]
        scatter_colors = ['red' if 'BICEP' in name else 'blue' if 'ENN' in name else 'gray' 
                         for name in valid_results['model_name']]
        
        ax2.scatter(valid_results['parameter_count'], valid_results['final_validation_loss'], 
                   c=scatter_colors, alpha=0.7, s=100)
        ax2.set_xlabel('Parameter Count')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Parameter Efficiency')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        # 3. Performance by dataset
        ax3 = axes[1, 0]
        dataset_pivot = valid_results.pivot_table(
            values='final_validation_loss', 
            index='dataset_name', 
            columns='model_name', 
            aggfunc='mean'
        )
        
        sns.heatmap(dataset_pivot, ax=ax3, cmap='viridis_r', annot=True, fmt='.4f')
        ax3.set_title('Performance by Dataset')
        
        # 4. Training efficiency
        ax4 = axes[1, 1]
        if 'total_training_time' in valid_results.columns:
            ax4.scatter(valid_results['total_training_time'], valid_results['final_validation_loss'],
                       c=scatter_colors, alpha=0.7, s=100)
            ax4.set_xlabel('Total Training Time (s)')
            ax4.set_ylabel('Validation Loss')
            ax4.set_title('Training Efficiency')
            ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / "revolutionary_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to {save_dir}/revolutionary_analysis.png")


def main():
    """Run the revolutionary benchmark"""
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Run benchmark
    benchmark = RevolutionaryBenchmark(device=device)
    results = benchmark.run_revolutionary_benchmark()
    
    return results


if __name__ == "__main__":
    main()