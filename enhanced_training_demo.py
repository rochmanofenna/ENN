#!/usr/bin/env python3
"""
Enhanced ENN Training Demo with Robust Features

Demonstrates:
- Input validation and error handling
- Comprehensive logging and metrics tracking  
- Model checkpointing and state management
- Performance monitoring and optimization
- Extended training with 114 epochs and optimal batch size
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from enn.config import Config
from enn.model import ENNModelWithSparsityControl
from enn.enhanced_utils import ENNLogger, ENNCheckpointer, MetricsTracker, compute_model_sparsity, compute_gradient_norm, memory_usage_mb

def create_enhanced_synthetic_data(N=2000, T=25, S=3):
    """Create more challenging synthetic dataset."""
    # Multi-frequency sine waves with noise
    t = np.linspace(0, 4*np.pi, T)
    base_freq = np.sin(t)[None, :, None]
    high_freq = 0.3 * np.sin(3*t)[None, :, None]
    noise = 0.1 * np.random.randn(N, T, 1)
    
    X = base_freq + high_freq + noise
    X = np.tile(X, (1, 1, S))  # Replicate across features
    
    # Target: predict sum of features at next timestep
    y = X[:, -1, :].sum(axis=1)  # Sum of last timestep features
    
    return X.astype(np.float32), y.astype(np.float32)

def enhanced_training_loop():
    """Comprehensive training with all enhanced features."""
    
    # Initialize enhanced components
    logger = ENNLogger("ENN_Enhanced", "enn_training.log")
    checkpointer = ENNCheckpointer("./enhanced_checkpoints")
    metrics = MetricsTracker()
    
    logger.info("Starting Enhanced ENN Training", framework="PyTorch")
    
    # Configuration
    config = Config("enhanced")
    config.num_layers = 4
    config.num_neurons = 15
    config.num_states = 3
    config.input_dim = 3
    config.epochs = 114
    config.batch_size = 32  # Optimal for ENN
    config.base_lr = 5e-4
    config.max_grad_norm = 0.5
    
    logger.info("Configuration", **{
        'neurons': config.num_neurons,
        'states': config.num_states, 
        'layers': config.num_layers,
        'epochs': config.epochs,
        'batch_size': config.batch_size
    })
    
    # Data preparation
    logger.info("Generating enhanced synthetic dataset")
    X, y = create_enhanced_synthetic_data(N=2000, T=25, S=config.input_dim)
    
    # Normalization
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()
    X = (X - X_mean) / (X_std + 1e-8)
    y = (y - y_mean) / (y_std + 1e-8)
    
    # DataLoader
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # Model and optimization
    model = ENNModelWithSparsityControl(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.MSELoss()
    
    logger.info("Model initialized", 
                parameters=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Training loop with enhanced monitoring
    model.train()
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch_idx, (batch_X, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_X)
            predictions = output.mean(dim=(1, 2))  # Average over neurons and states
            loss = loss_fn(predictions, batch_y)
            
            # Add L1 regularization from model
            if hasattr(model, 'mask_l1'):
                loss += model.mask_l1
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(loader)
        
        # Compute metrics
        sparsity = compute_model_sparsity(model)
        memory_mb = memory_usage_mb()
        
        # Update metrics tracker
        metrics.update(
            loss=avg_loss,
            sparsity=sparsity,
            gradient_norm=grad_norm.item(),
            memory_usage=memory_mb,
            epoch_time=epoch_time
        )
        
        # Logging
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{config.epochs}",
                       loss=f"{avg_loss:.6f}",
                       sparsity=f"{sparsity:.3f}",
                       grad_norm=f"{grad_norm:.4f}",
                       lr=f"{scheduler.get_last_lr()[0]:.6f}",
                       memory_mb=f"{memory_mb:.1f}",
                       time_s=f"{epoch_time:.2f}")
        
        # Checkpointing
        if epoch % 20 == 0 or epoch == config.epochs:
            metadata = {
                'sparsity': sparsity,
                'gradient_norm': grad_norm.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            }
            checkpointer.save_checkpoint(model, optimizer, epoch, avg_loss, config, metadata)
            logger.info(f"Checkpoint saved", epoch=epoch)
    
    # Training summary
    total_time = time.time() - start_time
    summary = metrics.get_summary()
    
    logger.info("Training completed",
               total_time=f"{total_time:.2f}s",
               avg_epoch_time=f"{total_time/config.epochs:.2f}s",
               final_loss=f"{avg_loss:.6f}",
               final_sparsity=f"{sparsity:.3f}")
    
    # Save metrics
    metrics.save_metrics("enn_training_metrics.json")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_data = torch.from_numpy(X[:100])  # Use first 100 samples for quick test
        test_output = model(test_data)
        test_preds = test_output.mean(dim=(1, 2))
        test_loss = loss_fn(test_preds, torch.from_numpy(y[:100]))
        
        logger.info("Final evaluation",
                   test_loss=f"{test_loss.item():.6f}",
                   test_samples=100)
    
    logger.info("Enhanced training demo completed successfully!")
    
    return model, metrics, summary

if __name__ == "__main__":
    model, metrics, summary = enhanced_training_loop()
    print("\\n=== Training Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")