# Entangled Neural Networks (ENN)

A novel neural architecture featuring entangled neuron dynamics, multi-head attention mechanisms, and adaptive sparsity control for sequence modeling tasks.

## Quick Start

```python
from enn.enhanced_model import create_attention_enn
from enn.config import Config

config = Config()
config.input_dim = 5
model = create_attention_enn(config, 'full')
output = model(torch.randn(32, 20, 5))  # [batch, time, features]
```

## Performance Overview

| Model | Parameters | Validation Loss | Relative Performance |
|-------|------------|-----------------|---------------------|
| **ENN Original** | **431** | **0.000016** | **1.0x** |
| ENN + Attention | 148,152 | 0.000066 | 0.24x |
| Transformer | 56,881 | 0.000710 | 0.023x |
| LSTM | 30,193 | 0.001869 | 0.009x |
| CNN | 11,065 | 0.020181 | 0.0008x |

*Tested on synthetic regression (1000 samples, 20 timesteps, 3 features)*

## Key Features

- **Entangled Neuron Dynamics**: Novel neuron interaction mechanism with shared state evolution
- **Multi-Head Attention**: Specialized attention for neuron-state and temporal memory processing  
- **Adaptive Sparsity**: Dynamic pruning and low-power state collapse
- **Memory Architecture**: Short-term buffers with recency weighting and temporal proximity scaling
- **Parameter Efficiency**: Achieves superior performance with 100x fewer parameters than comparable models

## Architecture Variants

```python
# Lightweight attention (5K parameters)
model = create_attention_enn(config, 'minimal')

# Neuron-focused attention (18K parameters)  
model = create_attention_enn(config, 'neuron_only')

# Full attention mechanisms (148K parameters)
model = create_attention_enn(config, 'full')
```

## Benchmarking

Run comprehensive comparison against LSTM, Transformer, CNN, and LNN baselines:

```bash
# Quick benchmark (5 minutes)
python run_comprehensive_benchmark.py --quick

# Focused comparison
python fast_benchmark_demo.py

# Full academic benchmark
python run_comprehensive_benchmark.py --epochs 114 --runs 3
```

## Installation

```bash
pip install torch numpy matplotlib pandas seaborn
git clone [repository]
cd ENN
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Core Components

### Entangled Model
```python
from enn.model import ENNModelWithSparsityControl
model = ENNModelWithSparsityControl(config)
```

### Attention Mechanisms
```python
from enn.multihead_attention import MultiHeadAttention, NeuronStateAttention
attention = NeuronStateAttention(num_neurons=10, num_states=5)
```

### Memory System
```python
from enn.memory import ShortTermBuffer, temporal_proximity_scaling
buffer = ShortTermBuffer(max_len=30)
```

## Technical Details

### Input Handling
- **Temporal sequences**: `[batch, time_steps, features]`
- **Direct format**: `[batch, num_neurons, num_states]`
- **Single timestep**: `[batch, features]`

### Memory Dynamics
- Exponential state decay with configurable rate
- Short-term buffers with FIFO eviction
- Recency-weighted temporal proximity scaling
- Advanced state collapse with autoencoder compression

### Attention Architecture
- **Neuron-State Attention**: Cross-attention between entangled neurons
- **Temporal Attention**: Memory buffer processing with positional encoding
- **Sparse Attention**: Top-k attention for computational efficiency
- **Attention Pooling**: Sequence-to-vector aggregation

## Configuration

```python
config = Config()
config.num_layers = 3        # Processing layers
config.num_neurons = 10      # Entangled neurons
config.num_states = 5        # States per neuron
config.input_dim = 5         # Input feature dimension
config.decay_rate = 0.1      # Memory decay rate
config.recency_factor = 0.9  # Temporal weighting
config.buffer_size = 5       # Short-term memory
config.epochs = 114          # Training epochs
config.batch_size = 32       # Optimal batch size
```

## Validation Results

### Regression Task (Synthetic Data)
- **Dataset**: 1000 samples, 20 timesteps, 3 features
- **Metric**: MSE validation loss
- **Training**: 20 epochs, AdamW optimizer

### Classification Task  
- **Dataset**: Binary temporal classification
- **Metric**: Cross-entropy loss, accuracy
- **Evaluation**: 3-fold cross-validation

### Memory Task
- **Dataset**: Sequence recall with delay
- **Metric**: Reconstruction loss
- **Challenge**: Variable delay lengths

## Research Applications

- Time series forecasting and anomaly detection
- Sequence classification and regression
- Memory-intensive temporal modeling
- Parameter-efficient deep learning
- Novel attention mechanism research

## Implementation Notes

- **Device Support**: Automatic CPU/GPU detection
- **Memory Management**: Efficient buffer cleanup and device synchronization
- **Numerical Stability**: Gradient clipping and regularization
- **Validation**: Comprehensive input checking and error handling
- **Logging**: Structured training metrics and checkpointing

## File Structure

```
enn/
├── model.py              # Core ENN architecture
├── enhanced_model.py     # ENN with attention mechanisms
├── multihead_attention.py # Attention implementations
├── memory.py             # Memory and buffer systems
├── config.py             # Configuration management
├── sparsity_control.py   # Dynamic sparsity mechanisms
└── validation.py         # Input validation utilities

baselines/
└── baseline_models.py    # LSTM, Transformer, CNN, LNN implementations

benchmarking/
└── benchmark_framework.py # Comprehensive evaluation suite
```

## Citation

```bibtex
@misc{enn2024,
  title={Entangled Neural Networks: Multi-Head Attention for Efficient Sequence Modeling},
  author={[Author]},
  year={2024},
  note={Research implementation with comprehensive benchmarking}
}
```

## License

[License details]

---

*Benchmarks run on Python 3.9, PyTorch 2.0+. Results may vary based on hardware and software configuration.*