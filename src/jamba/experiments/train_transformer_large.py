#!/usr/bin/env python3
import sys
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import os
import wandb
from torch.cuda.amp import autocast, GradScaler

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model_transformer import JambaThreatTransformerModel, ThreatDataset
from jamba.model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=200000, n_features=20, random_state=42):
    """Generate synthetic dataset with balanced classes."""
    np.random.seed(random_state)
    
    # Generate normal samples
    n_normal = n_samples // 2
    normal_samples = np.random.randn(n_normal, n_features)
    normal_labels = np.zeros(n_normal)
    
    # Generate threat samples with a different distribution
    n_threat = n_samples - n_normal
    threat_samples = np.random.randn(n_threat, n_features) * 1.5 + 0.5
    threat_labels = np.ones(n_threat)
    
    # Combine and shuffle
    X = np.vstack([normal_samples, threat_samples])
    y = np.hstack([normal_labels, threat_labels])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def train_transformer_model():
    # Initialize wandb
    wandb.login(key="82562f53305dea64f084ebd4faca54ec75e1536e")
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    X, y = generate_synthetic_data()
    logger.info(f"Generated dataset with {len(X)} samples and {X.shape[1]} features")
    
    # Create model config
    config_dict = {
        'version': '1.0.0',
        'input_dim': X.shape[1],
        'hidden_dim': 256,
        'output_dim': 1,
        'dropout_rate': 0.2,
        'n_heads': 8,
        'feature_layers': 4,
        'use_mixed_precision': True,
        'batch_size': 128,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize wandb run
    run = wandb.init(
        project="jamba-threat-detection",
        config=config_dict,
        name=f"transformer_large_scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Create save directory
    save_dir = f'models/transformer_large_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model configuration and model
    config = ModelConfig(**config_dict)
    model = JambaThreatTransformerModel(config).to(config.device)
    
    # Split data into train, validation, and test sets (60/20/20)
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Create datasets
    train_dataset = ThreatDataset(X_train, y_train)
    val_dataset = ThreatDataset(X_val, y_val)
    test_dataset = ThreatDataset(X_test, y_test)
    
    # Create data loaders
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
    
    # Initialize training components
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    num_epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            
            if scaler is not None:
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
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
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
    train_transformer_model() 