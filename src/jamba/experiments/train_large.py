#!/usr/bin/env python3
import sys
from pathlib import Path
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
from jamba.utils.dataset_generator import generate_balanced_dataset
from jamba.train import ModelTrainer
from jamba.config import ModelConfig as Config

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model import JambaThreatModel
from jamba.model_config import ModelConfig, DEFAULT_CPU_CONFIG
from jamba.utils.training_tracker import TrainingTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_with_size(size):
    """Create a balanced dataset with the specified total size."""
    n_samples = size // 2  # Split evenly between normal and threat samples
    X, y = generate_balanced_dataset(n_samples, n_samples)
    return X, y

def create_data_loaders(X, y, batch_size=32, train_split=0.8, val_split=0.1):
    """Create data loaders with the given dataset."""
    # Convert to tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Calculate split indices
    n_samples = len(X)
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    test_size = n_samples - train_size - val_size
    
    # Split indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    test_dataset = TensorDataset(X[test_indices], y[test_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_model_with_size(size):
    # Create dataset with specified size
    X, y = create_dataset_with_size(size)
    
    # Create config dictionary
    config_dict = {
        'version': '1.0.0',
        'input_dim': X.shape[1],
        'hidden_dim': 256,
        'output_dim': 1,
        'dropout_rate': 0.3,
        'n_heads': 4,
        'feature_layers': 2,
        'use_mixed_precision': False,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'cpu',
        'epochs': 20
    }
    
    # Create save directory
    save_dir = f'models/size_{size}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize trainer
    config = Config(**config_dict)
    trainer = ModelTrainer(config=config, save_dir=save_dir)
    trainer.setup_training()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X, y, batch_size=config.batch_size)
    
    # Train model
    metrics = trainer.train(train_loader, val_loader, epochs=config.epochs)
    
    # Convert metrics to dictionary if it's a tuple
    if isinstance(metrics, tuple) and len(metrics) >= 2:
        metrics = metrics[1]
    
    # Save metrics
    metrics_file = os.path.join(save_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    return metrics

def main():
    sizes = [45000, 70000, 90000, 120000]
    results = []
    for size in sizes:
        print(f"\nTraining model with dataset size: {size}")
        metrics = train_model_with_size(size)
        
        print("\nFinal metrics for dataset size {}:".format(size))
        if isinstance(metrics, dict):
            accuracy = metrics.get('accuracy', 0.0)
            f1_score = metrics.get('f1', 0.0)
            training_time = metrics.get('training_time', 0.0)
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"F1 Score: {f1_score:.4f}")
            print(f"Training Time: {training_time:.2f} seconds")
            results.append({'size': size, 'accuracy': accuracy, 'f1': f1_score, 'training_time': training_time})
        else:
            print("No metrics available")

    if results:
        df = pd.DataFrame(results)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Accuracy and F1 Score on the same plot
        sns.lineplot(x='size', y='accuracy', data=df, marker='o', ax=ax1)
        sns.lineplot(x='size', y='f1', data=df, marker='o', ax=ax1)
        ax1.set_title("Model Performance vs Dataset Size")
        ax1.set_xlabel("Dataset Size")
        ax1.set_ylabel("Metric Value")
        ax1.legend(['Accuracy (%)', 'F1 Score'])

        # Plot Training Time
        sns.lineplot(x='size', y='training_time', data=df, marker='o', ax=ax2)
        ax2.set_title("Training Time vs Dataset Size")
        ax2.set_xlabel("Dataset Size")
        ax2.set_ylabel("Time (seconds)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 