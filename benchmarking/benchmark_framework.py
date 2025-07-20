"""
Comprehensive Benchmarking Framework for ENN vs Baseline Models.

This framework provides:
- Standardized datasets (synthetic, time series, classification)
- Fair comparison metrics (accuracy, loss, training time, memory usage)
- Statistical significance testing
- Visualization and reporting
- Model complexity analysis
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from pathlib import Path
import json

# Import models
from baselines.baseline_models import create_baseline_model, BaselineConfig, count_parameters
from enn.model import ENNModelWithSparsityControl
from enn.enhanced_model import create_attention_enn
from enn.config import Config
from enn.enhanced_utils import MetricsTracker, compute_model_sparsity, compute_gradient_norm


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    
    # Dataset parameters
    dataset_sizes: List[int] = None
    sequence_lengths: List[int] = None
    input_dimensions: List[int] = None
    
    # Training parameters
    epochs: int = 114
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_runs: int = 3  # For statistical significance
    
    # Model parameters
    hidden_dim: int = 64
    num_layers: int = 2
    
    # Evaluation parameters
    test_size: float = 0.2
    validation_size: float = 0.1
    
    def __post_init__(self):
        if self.dataset_sizes is None:
            self.dataset_sizes = [500, 1000, 2000]
        if self.sequence_lengths is None:
            self.sequence_lengths = [10, 20, 50]
        if self.input_dimensions is None:
            self.input_dimensions = [1, 3, 5]


class DatasetGenerator:
    """Generate standardized datasets for benchmarking."""
    
    @staticmethod
    def synthetic_regression(n_samples: int, seq_len: int, input_dim: int, 
                           noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic regression dataset.
        
        Args:
            n_samples: Number of samples
            seq_len: Sequence length
            input_dim: Input feature dimension
            noise_level: Noise level for complexity
            
        Returns:
            X: [n_samples, seq_len, input_dim]
            y: [n_samples] - regression targets
        """
        # Multi-frequency sinusoidal patterns
        t = np.linspace(0, 4*np.pi, seq_len)
        base_pattern = np.sin(t)[None, :, None]
        high_freq = 0.3 * np.sin(3*t + np.pi/4)[None, :, None]
        
        # Add noise and replicate across dimensions
        noise = noise_level * np.random.randn(n_samples, seq_len, 1)
        X = base_pattern + high_freq + noise
        X = np.tile(X, (1, 1, input_dim))
        
        # Target: weighted sum of last timestep features + nonlinear component
        last_step = X[:, -1, :]  # [n_samples, input_dim]
        linear_component = last_step.sum(axis=1)
        nonlinear_component = 0.1 * (last_step ** 2).sum(axis=1)
        y = linear_component + nonlinear_component
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    @staticmethod
    def binary_classification(n_samples: int, seq_len: int, input_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate binary classification dataset.
        
        Args:
            n_samples: Number of samples
            seq_len: Sequence length  
            input_dim: Input feature dimension
            
        Returns:
            X: [n_samples, seq_len, input_dim]
            y: [n_samples] - binary labels
        """
        # Generate sequences with different patterns for each class
        X = np.random.randn(n_samples, seq_len, input_dim)
        
        # Class 0: Increasing trend in first dimension
        # Class 1: Decreasing trend in first dimension
        class_labels = np.random.randint(0, 2, n_samples)
        
        for i, label in enumerate(class_labels):
            if label == 0:
                # Increasing trend
                trend = np.linspace(-1, 1, seq_len)
                X[i, :, 0] += trend
            else:
                # Decreasing trend  
                trend = np.linspace(1, -1, seq_len)
                X[i, :, 0] += trend
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(class_labels, dtype=torch.long)
    
    @staticmethod
    def memory_task(n_samples: int, seq_len: int, input_dim: int, 
                   delay: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate memory task dataset (sequence recall with delay).
        
        Args:
            n_samples: Number of samples
            seq_len: Sequence length
            input_dim: Input feature dimension  
            delay: Delay between input and required recall
            
        Returns:
            X: [n_samples, seq_len, input_dim]
            y: [n_samples, input_dim] - values to recall
        """
        X = np.random.randn(n_samples, seq_len, input_dim)
        
        # Target: recall values from (seq_len - delay) position
        recall_position = max(0, seq_len - delay)
        y = X[:, recall_position, :]
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    model_name: str
    dataset_type: str
    n_samples: int
    seq_len: int
    input_dim: int
    
    # Performance metrics
    final_loss: float
    best_loss: float
    final_accuracy: Optional[float]
    best_accuracy: Optional[float]
    
    # Training metrics
    training_time: float
    convergence_epoch: int
    
    # Model metrics
    n_parameters: int
    model_size_mb: float
    sparsity: float
    
    # Memory and efficiency
    peak_memory_mb: float
    avg_epoch_time: float
    
    # Statistical metrics
    loss_std: float
    accuracy_std: Optional[float]


class ModelEvaluator:
    """Evaluate models on standardized tasks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_and_evaluate(self, model: nn.Module, train_loader: DataLoader, 
                          val_loader: DataLoader, task_type: str) -> Dict[str, float]:
        """
        Train and evaluate a model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of evaluation metrics
        """
        model = model.to(self.device)
        
        # Setup loss function and optimizer
        if task_type == 'regression':
            criterion = nn.MSELoss()
            metric_fn = lambda pred, target: torch.sqrt(criterion(pred, target)).item()
        else:  # classification
            criterion = nn.CrossEntropyLoss()
            metric_fn = lambda pred, target: (pred.argmax(1) == target).float().mean().item()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        # Training loop
        model.train()
        start_time = time.time()
        train_losses = []
        val_metrics = []
        convergence_epoch = self.config.epochs
        best_val_metric = float('inf') if task_type == 'regression' else 0.0
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            # Training
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Handle different output formats
                output = model(batch_x)
                if output.dim() > 2:
                    output = output.mean(dim=(1, 2)) if task_type == 'regression' else output.view(output.size(0), -1)
                if task_type == 'regression' and output.dim() > 1:
                    output = output.squeeze(-1)
                
                loss = criterion(output, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_metric = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = model(batch_x)
                    
                    # Handle output format
                    if output.dim() > 2:
                        output = output.mean(dim=(1, 2)) if task_type == 'regression' else output.view(output.size(0), -1)
                    if task_type == 'regression' and output.dim() > 1:
                        output = output.squeeze(-1)
                    
                    val_loss += criterion(output, batch_y).item()
                    val_metric += metric_fn(output, batch_y)
            
            model.train()
            
            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_metric = val_metric / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_metrics.append(avg_val_metric)
            
            # Check for convergence
            if task_type == 'regression':
                if avg_val_metric < best_val_metric:
                    best_val_metric = avg_val_metric
                    convergence_epoch = epoch
            else:
                if avg_val_metric > best_val_metric:
                    best_val_metric = avg_val_metric
                    convergence_epoch = epoch
            
            # Early stopping check (optional)
            if epoch - convergence_epoch > 20:  # 20 epochs without improvement
                break
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        final_loss = train_losses[-1]
        best_loss = min(train_losses)
        final_metric = val_metrics[-1] if val_metrics else 0.0
        best_metric = best_val_metric
        
        # Model analysis
        n_params = count_parameters(model)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        sparsity = compute_model_sparsity(model) if hasattr(model, 'parameters') else 0.0
        
        return {
            'final_loss': final_loss,
            'best_loss': best_loss,
            'final_metric': final_metric,
            'best_metric': best_metric,
            'training_time': training_time,
            'convergence_epoch': convergence_epoch,
            'n_parameters': n_params,
            'model_size_mb': model_size_mb,
            'sparsity': sparsity,
            'avg_epoch_time': training_time / len(train_losses)
        }


class BenchmarkSuite:
    """Main benchmarking suite for comprehensive model comparison."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.evaluator = ModelEvaluator(config)
        self.results: List[BenchmarkResult] = []
        
    def create_models(self, input_dim: int, seq_len: int) -> Dict[str, nn.Module]:
        """Create all models for comparison."""
        models = {}
        
        # Baseline configuration
        baseline_config = BaselineConfig(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=1,
            seq_len=seq_len,
            num_layers=self.config.num_layers,
            num_heads=4
        )
        
        # Baseline models
        baseline_types = ['lstm', 'transformer', 'cnn', 'mlp', 'lnn']
        for model_type in baseline_types:
            try:
                models[model_type] = create_baseline_model(model_type, baseline_config)
            except Exception as e:
                print(f"Warning: Could not create {model_type} model: {e}")
        
        # ENN models
        enn_config = Config()
        enn_config.input_dim = input_dim
        enn_config.num_neurons = self.config.hidden_dim // 4  # Comparable complexity
        enn_config.num_states = max(3, min(8, input_dim * 2))  # Ensure at least 3 states
        enn_config.low_power_k = min(3, enn_config.num_states)  # Ensure k <= num_states
        enn_config.epochs = self.config.epochs
        enn_config.batch_size = self.config.batch_size
        
        # Original ENN
        models['enn_original'] = ENNModelWithSparsityControl(enn_config)
        
        # ENN with attention variants
        attention_types = ['minimal', 'neuron_only', 'temporal_only', 'full']
        for att_type in attention_types:
            try:
                models[f'enn_{att_type}'] = create_attention_enn(enn_config, att_type)
            except Exception as e:
                print(f"Warning: Could not create ENN {att_type} model: {e}")
        
        return models
    
    def run_benchmark(self, dataset_type: str = 'regression') -> pd.DataFrame:
        """
        Run comprehensive benchmark across all configurations.
        
        Args:
            dataset_type: 'regression', 'classification', or 'memory'
            
        Returns:
            DataFrame with all benchmark results
        """
        print(f"Starting {dataset_type} benchmark...")
        
        for n_samples in self.config.dataset_sizes:
            for seq_len in self.config.sequence_lengths:
                for input_dim in self.config.input_dimensions:
                    
                    print(f"\\nConfiguration: {n_samples} samples, {seq_len} seq_len, {input_dim} input_dim")
                    
                    # Generate dataset
                    if dataset_type == 'regression':
                        X, y = DatasetGenerator.synthetic_regression(n_samples, seq_len, input_dim)
                        task_type = 'regression'
                    elif dataset_type == 'classification':
                        X, y = DatasetGenerator.binary_classification(n_samples, seq_len, input_dim)
                        task_type = 'classification'
                    else:  # memory
                        X, y = DatasetGenerator.memory_task(n_samples, seq_len, input_dim)
                        task_type = 'regression'
                    
                    # Split data
                    n_train = int(n_samples * (1 - self.config.test_size - self.config.validation_size))
                    n_val = int(n_samples * self.config.validation_size)
                    
                    train_dataset = TensorDataset(X[:n_train], y[:n_train])
                    val_dataset = TensorDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val])
                    
                    train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
                    
                    # Create models
                    models = self.create_models(input_dim, seq_len)
                    
                    # Evaluate each model
                    for model_name, model in models.items():
                        print(f"  Evaluating {model_name}...")
                        
                        # Run multiple times for statistical significance
                        run_results = []
                        for run in range(self.config.num_runs):
                            # Reinitialize model
                            if hasattr(model, 'reset_parameters'):
                                model.reset_parameters()
                            else:
                                # Reinitialize weights
                                for layer in model.modules():
                                    if hasattr(layer, 'reset_parameters'):
                                        layer.reset_parameters()
                            
                            try:
                                result = self.evaluator.train_and_evaluate(model, train_loader, val_loader, task_type)
                                run_results.append(result)
                            except Exception as e:
                                print(f"    Error in run {run}: {e}")
                                continue
                        
                        if run_results:
                            # Aggregate results
                            avg_result = self._aggregate_results(run_results)
                            
                            # Create benchmark result
                            benchmark_result = BenchmarkResult(
                                model_name=model_name,
                                dataset_type=dataset_type,
                                n_samples=n_samples,
                                seq_len=seq_len,
                                input_dim=input_dim,
                                final_loss=avg_result['final_loss'],
                                best_loss=avg_result['best_loss'],
                                final_accuracy=avg_result.get('final_metric'),
                                best_accuracy=avg_result.get('best_metric'),
                                training_time=avg_result['training_time'],
                                convergence_epoch=avg_result['convergence_epoch'],
                                n_parameters=avg_result['n_parameters'],
                                model_size_mb=avg_result['model_size_mb'],
                                sparsity=avg_result['sparsity'],
                                peak_memory_mb=0.0,  # TODO: Implement memory tracking
                                avg_epoch_time=avg_result['avg_epoch_time'],
                                loss_std=np.std([r['final_loss'] for r in run_results]),
                                accuracy_std=np.std([r['final_metric'] for r in run_results]) if task_type != 'regression' else None
                            )
                            
                            self.results.append(benchmark_result)
        
        # Convert to DataFrame
        return self._results_to_dataframe()
    
    def _aggregate_results(self, run_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results from multiple runs."""
        aggregated = {}
        for key in run_results[0].keys():
            values = [r[key] for r in run_results]
            aggregated[key] = np.mean(values)
        return aggregated
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in self.results:
            data.append({
                'model': result.model_name,
                'dataset': result.dataset_type,
                'n_samples': result.n_samples,
                'seq_len': result.seq_len,
                'input_dim': result.input_dim,
                'final_loss': result.final_loss,
                'best_loss': result.best_loss,
                'final_accuracy': result.final_accuracy,
                'best_accuracy': result.best_accuracy,
                'training_time': result.training_time,
                'convergence_epoch': result.convergence_epoch,
                'n_parameters': result.n_parameters,
                'model_size_mb': result.model_size_mb,
                'sparsity': result.sparsity,
                'avg_epoch_time': result.avg_epoch_time,
                'loss_std': result.loss_std,
                'accuracy_std': result.accuracy_std
            })
        return pd.DataFrame(data)
    
    def generate_report(self, results_df: pd.DataFrame, output_dir: str = "./benchmark_results"):
        """Generate comprehensive benchmark report."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save raw results
        results_df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)
        
        # Generate visualizations
        self._plot_performance_comparison(results_df, output_dir)
        self._plot_efficiency_analysis(results_df, output_dir)
        self._plot_scalability_analysis(results_df, output_dir)
        
        # Generate summary report
        self._generate_summary_report(results_df, output_dir)
        
        print(f"Benchmark report generated in {output_dir}")
    
    def _plot_performance_comparison(self, df: pd.DataFrame, output_dir: str):
        """Plot performance comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss comparison
        sns.boxplot(data=df, x='model', y='final_loss', ax=axes[0,0])
        axes[0,0].set_title('Final Loss Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        sns.boxplot(data=df, x='model', y='training_time', ax=axes[0,1])
        axes[0,1].set_title('Training Time Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Parameter count comparison
        sns.boxplot(data=df, x='model', y='n_parameters', ax=axes[1,0])
        axes[1,0].set_title('Model Complexity (Parameters)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Convergence epoch comparison
        sns.boxplot(data=df, x='model', y='convergence_epoch', ax=axes[1,1])
        axes[1,1].set_title('Convergence Speed (Epochs)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self, df: pd.DataFrame, output_dir: str):
        """Plot efficiency analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance vs Parameters
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            axes[0].scatter(model_data['n_parameters'], model_data['final_loss'], 
                          label=model, alpha=0.7)
        axes[0].set_xlabel('Number of Parameters')
        axes[0].set_ylabel('Final Loss')
        axes[0].set_title('Performance vs Model Complexity')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_xscale('log')
        
        # Training Time vs Performance
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            axes[1].scatter(model_data['training_time'], model_data['final_loss'], 
                          label=model, alpha=0.7)
        axes[1].set_xlabel('Training Time (seconds)')
        axes[1].set_ylabel('Final Loss')
        axes[1].set_title('Training Efficiency')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficiency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, df: pd.DataFrame, output_dir: str):
        """Plot scalability analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Performance vs Dataset Size
        for model in df['model'].unique():
            model_data = df[df['model'] == model].groupby('n_samples')['final_loss'].mean()
            axes[0].plot(model_data.index, model_data.values, marker='o', label=model)
        axes[0].set_xlabel('Dataset Size')
        axes[0].set_ylabel('Average Final Loss')
        axes[0].set_title('Scalability: Dataset Size')
        axes[0].legend()
        
        # Performance vs Sequence Length
        for model in df['model'].unique():
            model_data = df[df['model'] == model].groupby('seq_len')['final_loss'].mean()
            axes[1].plot(model_data.index, model_data.values, marker='o', label=model)
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Average Final Loss')
        axes[1].set_title('Scalability: Sequence Length')
        axes[1].legend()
        
        # Performance vs Input Dimension
        for model in df['model'].unique():
            model_data = df[df['model'] == model].groupby('input_dim')['final_loss'].mean()
            axes[2].plot(model_data.index, model_data.values, marker='o', label=model)
        axes[2].set_xlabel('Input Dimension')
        axes[2].set_ylabel('Average Final Loss')
        axes[2].set_title('Scalability: Input Dimension')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scalability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, df: pd.DataFrame, output_dir: str):
        """Generate text summary report."""
        report = []
        report.append("# ENN Comprehensive Benchmark Report\\n")
        
        # Overall summary
        report.append("## Overall Performance Summary\\n")
        summary_stats = df.groupby('model').agg({
            'final_loss': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'n_parameters': 'mean',
            'convergence_epoch': 'mean'
        }).round(4)
        
        report.append(summary_stats.to_string())
        report.append("\\n\\n")
        
        # Best performing models
        report.append("## Best Performing Models\\n")
        best_loss = df.loc[df['final_loss'].idxmin()]
        fastest_training = df.loc[df['training_time'].idxmin()]
        most_efficient = df.loc[(df['final_loss'] / df['n_parameters']).idxmin()]
        
        report.append(f"Best Loss: {best_loss['model']} (Loss: {best_loss['final_loss']:.4f})\\n")
        report.append(f"Fastest Training: {fastest_training['model']} ({fastest_training['training_time']:.2f}s)\\n")
        report.append(f"Most Efficient: {most_efficient['model']}\\n\\n")
        
        # ENN Analysis
        report.append("## ENN Variants Analysis\\n")
        enn_models = df[df['model'].str.contains('enn')]
        if not enn_models.empty:
            enn_summary = enn_models.groupby('model')['final_loss'].mean().sort_values()
            report.append("ENN Performance Ranking (by average loss):\\n")
            for i, (model, loss) in enumerate(enn_summary.items()):
                report.append(f"{i+1}. {model}: {loss:.4f}\\n")
        
        # Save report
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write(''.join(report))


# Main execution script
if __name__ == "__main__":
    # Configure benchmark
    config = BenchmarkConfig(
        dataset_sizes=[500, 1000],  # Smaller for testing
        sequence_lengths=[10, 20],
        input_dimensions=[3, 5],
        epochs=50,  # Reduced for testing
        num_runs=2
    )
    
    # Run benchmark
    suite = BenchmarkSuite(config)
    
    # Test on regression task
    results_df = suite.run_benchmark('regression')
    
    # Generate report
    suite.generate_report(results_df)
    
    print("Benchmark completed successfully!")