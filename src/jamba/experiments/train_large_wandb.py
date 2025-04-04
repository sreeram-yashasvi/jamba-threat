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
import wandb
from torch.cuda.amp import autocast, GradScaler

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.utils.dataset_generator import generate_balanced_dataset
from jamba.jamba_model import JambaThreatModel
from jamba.model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_large_dataset(size=200000):
    """Create a balanced dataset with 200k samples."""
    logger.info(f"Generating balanced dataset with {size} samples...")
    n_samples = size // 2  # Split evenly between normal and threat samples
    X, y = generate_balanced_dataset(n_samples, n_samples)
    return X, y

def train_large_model():
    # Initialize wandb
    wandb.login(key="82562f53305dea64f084ebd4faca54ec75e1536e")
    
    # Create dataset
    X, y = create_large_dataset()
    
    # Create config dictionary
    config_dict = {
        'version': '1.0.0',
        'input_dim': X.shape[1],
        'hidden_dim': 512,  # Larger hidden dimension for more capacity
        'output_dim': 1,
        'dropout_rate': 0.3,
        'n_heads': 8,
        'feature_layers': 4,
        'use_mixed_precision': True,
        'batch_size': 256,  # Larger batch size for faster training
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 30
    }
    
    # Initialize wandb run
    run = wandb.init(
        project="jamba-threat-detection",
        config=config_dict,
        name=f"large_scale_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Create save directory
    save_dir = f'models/large_scale_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model configuration
    config = ModelConfig(**config_dict)
    
    # Initialize model
    model = JambaThreatModel(config).to(config.device)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            
            if config.use_mixed_precision:
                with autocast():
                    output = model(data)
                    loss = criterion(output.squeeze(), target.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output.squeeze(), target.float())
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pred = (output.squeeze() > 0).float()
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += criterion(output.squeeze(), target.float()).item()
                pred = (output.squeeze() > 0).float()
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config_dict
            }, os.path.join(save_dir, 'best_model.pt'))
            wandb.save(os.path.join(save_dir, 'best_model.pt'))
        
        logger.info(f'Epoch {epoch}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Final test evaluation
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += criterion(output.squeeze(), target.float()).item()
            pred = (output.squeeze() > 0).float()
            test_correct += (pred == target).sum().item()
            test_total += target.size(0)
    
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    
    # Log final test metrics
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_acc
    })
    
    logger.info(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    wandb.finish()

if __name__ == '__main__':
    train_large_model() 