#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script for ENN vs All Baseline Models.

This script runs a complete evaluation suite including:
- All baseline models (LSTM, Transformer, CNN, MLP, LNN)
- Original ENN and ENN with various attention mechanisms
- Multiple datasets and task types
- Statistical analysis and reporting
- Visualization and comparison charts

Usage:
    python run_comprehensive_benchmark.py [--quick] [--output-dir PATH]
"""

import argparse
import time
import warnings
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append('/Users/rayhanroswendi/developer/ENN')

from benchmarking.benchmark_framework import BenchmarkSuite, BenchmarkConfig
from enn.enhanced_utils import ENNLogger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run comprehensive ENN benchmark')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick benchmark with reduced parameters')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--task-types', nargs='+', 
                       choices=['regression', 'classification', 'memory'],
                       default=['regression', 'classification'],
                       help='Task types to benchmark')
    parser.add_argument('--epochs', type=int, default=114,
                       help='Number of training epochs')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs for statistical significance')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def create_benchmark_config(args) -> BenchmarkConfig:
    """Create benchmark configuration based on arguments."""
    if args.quick:
        # Quick benchmark configuration
        config = BenchmarkConfig(
            dataset_sizes=[500, 1000],
            sequence_lengths=[10, 25],
            input_dimensions=[3, 5],
            epochs=50,  # Reduced for quick testing
            batch_size=32,
            learning_rate=1e-3,
            num_runs=2,  # Reduced runs
            hidden_dim=64,
            num_layers=2
        )
    else:
        # Full benchmark configuration
        config = BenchmarkConfig(
            dataset_sizes=[500, 1000, 2000, 4000],
            sequence_lengths=[10, 25, 50, 100],
            input_dimensions=[1, 3, 5, 8],
            epochs=args.epochs,
            batch_size=32,
            learning_rate=1e-3,
            num_runs=args.runs,
            hidden_dim=64,
            num_layers=2
        )
    
    return config


def main():
    """Main benchmarking execution."""
    args = parse_arguments()
    
    # Setup logging
    logger = ENNLogger("Benchmark", 
                       log_file=f"{args.output_dir}/benchmark.log" if args.verbose else None)
    
    logger.info("Starting Comprehensive ENN Benchmark")
    logger.info("Configuration", 
                quick=args.quick, 
                output_dir=args.output_dir,
                task_types=args.task_types,
                epochs=args.epochs,
                runs=args.runs)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create benchmark configuration
    config = create_benchmark_config(args)
    
    logger.info("Benchmark Configuration",
                dataset_sizes=config.dataset_sizes,
                sequence_lengths=config.sequence_lengths,
                input_dimensions=config.input_dimensions,
                epochs=config.epochs,
                runs=config.num_runs)
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(config)
    
    # Run benchmarks for each task type
    all_results = []
    
    for task_type in args.task_types:
        logger.info(f"Running {task_type} benchmark...")
        
        start_time = time.time()
        
        try:
            # Run benchmark
            results_df = suite.run_benchmark(task_type)
            
            # Save task-specific results
            task_output_dir = f"{args.output_dir}/{task_type}"
            Path(task_output_dir).mkdir(exist_ok=True)
            
            # Generate report for this task
            suite.generate_report(results_df, task_output_dir)
            
            # Add to combined results
            results_df['task_type'] = task_type
            all_results.append(results_df)
            
            task_time = time.time() - start_time
            logger.info(f"Completed {task_type} benchmark", 
                       duration=f"{task_time:.2f}s",
                       total_experiments=len(results_df))
            
        except Exception as e:
            logger.error(f"Error in {task_type} benchmark: {e}")
            continue
    
    # Combine all results
    if all_results:
        import pandas as pd
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_results.to_csv(f"{args.output_dir}/combined_results.csv", index=False)
        
        # Generate combined analysis
        generate_combined_analysis(combined_results, args.output_dir, logger)
        
        # Print summary
        print_benchmark_summary(combined_results, logger)
        
        logger.info("Comprehensive benchmark completed successfully",
                   total_experiments=len(combined_results),
                   output_directory=args.output_dir)
    else:
        logger.error("No benchmark results generated")


def generate_combined_analysis(results_df, output_dir: str, logger):
    """Generate cross-task analysis."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("Generating combined analysis...")
    
    # Model ranking across all tasks
    model_performance = results_df.groupby(['model', 'task_type'])['final_loss'].mean().reset_index()
    model_ranking = model_performance.groupby('model')['final_loss'].mean().sort_values()
    
    # Save model ranking
    with open(f"{output_dir}/model_ranking.txt", 'w') as f:
        f.write("Overall Model Ranking (by average loss across all tasks):\\n\\n")
        for i, (model, avg_loss) in enumerate(model_ranking.items()):
            f.write(f"{i+1:2d}. {model:20s}: {avg_loss:.6f}\\n")
    
    # Cross-task performance visualization
    plt.figure(figsize=(12, 8))
    pivot_table = model_performance.pivot(index='model', columns='task_type', values='final_loss')
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r')
    plt.title('Model Performance Across Task Types (Lower is Better)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cross_task_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ENN variants analysis
    enn_results = results_df[results_df['model'].str.contains('enn')]
    if not enn_results.empty:
        plt.figure(figsize=(12, 6))
        
        # ENN performance by attention type
        enn_performance = enn_results.groupby(['model', 'task_type'])['final_loss'].mean().reset_index()
        
        for task in enn_performance['task_type'].unique():
            task_data = enn_performance[enn_performance['task_type'] == task]
            plt.subplot(1, 2, list(enn_performance['task_type'].unique()).index(task) + 1)
            
            sns.barplot(data=task_data, x='model', y='final_loss')
            plt.title(f'ENN Variants - {task.title()} Task')
            plt.xticks(rotation=45)
            plt.ylabel('Final Loss')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/enn_variants_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Combined analysis completed")


def print_benchmark_summary(results_df, logger):
    """Print comprehensive benchmark summary."""
    import pandas as pd
    
    print("\\n" + "="*80)
    print("COMPREHENSIVE ENN BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\\nTotal Experiments: {len(results_df)}")
    print(f"Models Tested: {results_df['model'].nunique()}")
    print(f"Task Types: {', '.join(results_df['task_type'].unique())}")
    print(f"Dataset Configurations: {len(results_df[['n_samples', 'seq_len', 'input_dim']].drop_duplicates())}")
    
    # Best performing models overall
    print("\\n" + "-"*50)
    print("TOP 5 MODELS (by average loss across all tasks)")
    print("-"*50)
    
    overall_ranking = results_df.groupby('model')['final_loss'].mean().sort_values().head(5)
    for i, (model, avg_loss) in enumerate(overall_ranking.items()):
        print(f"{i+1}. {model:25s}: {avg_loss:.6f}")
    
    # Task-specific best models
    print("\\n" + "-"*50)
    print("BEST MODEL PER TASK TYPE")
    print("-"*50)
    
    for task in results_df['task_type'].unique():
        task_data = results_df[results_df['task_type'] == task]
        best_model = task_data.loc[task_data['final_loss'].idxmin()]
        print(f"{task.title():15s}: {best_model['model']:25s} (loss: {best_model['final_loss']:.6f})")
    
    # ENN variants performance
    enn_results = results_df[results_df['model'].str.contains('enn')]
    if not enn_results.empty:
        print("\\n" + "-"*50)
        print("ENN VARIANTS RANKING")
        print("-"*50)
        
        enn_ranking = enn_results.groupby('model')['final_loss'].mean().sort_values()
        for i, (model, avg_loss) in enumerate(enn_ranking.items()):
            print(f"{i+1}. {model:25s}: {avg_loss:.6f}")
    
    # Efficiency analysis
    print("\\n" + "-"*50)
    print("EFFICIENCY ANALYSIS")
    print("-"*50)
    
    # Fastest training
    fastest = results_df.loc[results_df['training_time'].idxmin()]
    print(f"Fastest Training: {fastest['model']:25s} ({fastest['training_time']:.2f}s)")
    
    # Most parameter efficient
    results_df['efficiency_score'] = results_df['final_loss'] / (results_df['n_parameters'] / 1000)
    most_efficient = results_df.loc[results_df['efficiency_score'].idxmin()]
    print(f"Most Efficient:  {most_efficient['model']:25s} (loss/1k_params: {most_efficient['efficiency_score']:.6f})")
    
    # Smallest model
    smallest = results_df.loc[results_df['n_parameters'].idxmin()]
    print(f"Smallest Model:  {smallest['model']:25s} ({smallest['n_parameters']:,} parameters)")
    
    print("\\n" + "="*80)
    
    # Log summary
    logger.info("Benchmark Summary Generated",
               total_experiments=len(results_df),
               models_tested=results_df['model'].nunique(),
               best_overall=overall_ranking.index[0])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()