#!/usr/bin/env python3
import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.metrics import f1_score
import json

from jamba.model_config import ModelConfig, DEFAULT_GPU_CONFIG, DEFAULT_CPU_CONFIG
from jamba.data_preprocessing import DataPreprocessor, create_preprocessor
from jamba.jamba_model import JambaThreatModel, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Unified training manager for Jamba Threat Model"""
    
    def __init__(self, config=None, save_dir='models'):
        """Initialize the trainer with model configuration."""
        self.config = config or DEFAULT_CPU_CONFIG
        self.device = torch.device(self.config.device)
        self.save_dir = save_dir
        self.model = JambaThreatModel(self.config).to(self.device)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def setup_training(self, learning_rate=None):
        """Setup training components."""
        if learning_rate is not None:
            self.config.learning_rate = learning_rate
            
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        self.scaler = GradScaler(enabled=self.config.use_mixed_precision)
        
    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1).float()  # Convert to float for BCE loss
                
                with autocast(enabled=self.config.use_mixed_precision):
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item()
                
                # Convert logits to predictions
                pred = torch.sigmoid(output) >= 0.5
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions and targets for F1 score
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate F1 score
        f1 = f1_score(
            np.array(all_targets).ravel(),
            np.array(all_preds).ravel(),
            zero_division=0
        )
        
        return val_loss, accuracy, f1

    def save_checkpoint(self, val_loss: float, is_best: bool = False):
        """Save training checkpoint"""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': val_loss,
                'config': self.config.to_dict()
            }
            
            # Save latest checkpoint
            checkpoint_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = os.path.join(self.save_dir, 'model_best.pt')
                torch.save(checkpoint, best_path)
                logger.info(f'Saved best model to {best_path}')
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_model(self, filename: str):
        """Save model state to a file."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Saved model to {filepath}")

    def train(self, train_loader, val_loader, epochs):
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Initialize metrics dictionary
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'final_accuracy': None,
            'final_f1': None,
            'training_time': None
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1)  # Reshape target to match output dimension
                
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                              f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            train_loss /= len(train_loader)
            metrics['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            metrics['val_loss'].append(val_loss)
            metrics['val_accuracy'].append(val_accuracy)
            metrics['val_f1'].append(val_f1)
            
            # Log epoch results
            logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, '
                       f'Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%, '
                       f'Val F1: {val_f1:.4f}, Time: {time.time() - start_time:.2f}s')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(val_loss, is_best=True)
                logger.info("Saved best model checkpoint")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final metrics
        metrics['final_accuracy'] = val_accuracy
        metrics['final_f1'] = val_f1
        metrics['training_time'] = training_time
        
        # Save metrics to file
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics, {
            'accuracy': val_accuracy,
            'f1': val_f1,
            'training_time': training_time
        }

def main():
    parser = argparse.ArgumentParser(description='Train Jamba Threat Model')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--target', default='is_threat', help='Target column name')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Batch size (default: auto)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, 
                       help='Early stopping patience')
    parser.add_argument('--save-dir', default='models', 
                       help='Directory to save models')
    parser.add_argument('--checkpoint', help='Path to checkpoint to resume from')
    parser.add_argument('--validate-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    try:
        # Create preprocessor
        preprocessor = create_preprocessor(args.data, args.target, args.save_dir)
        
        # Load and preprocess data
        df = pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_parquet(args.data)
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=args.validate_split, random_state=42, stratify=df[args.target]
        )
        
        # Preprocess training data
        X_train, y_train = preprocessor.preprocess(train_df, args.target, is_training=True)
        X_val, y_val = preprocessor.preprocess(val_df, args.target, is_training=False)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.values)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.FloatTensor(y_val.values)
        )
        
        # Determine batch size
        if args.batch_size is None:
            args.batch_size = 128 if torch.cuda.is_available() else 32
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), 4),
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=min(os.cpu_count(), 4),
            pin_memory=torch.cuda.is_available()
        )
        
        # Initialize trainer
        trainer = ModelTrainer(config=DEFAULT_GPU_CONFIG if torch.cuda.is_available() else DEFAULT_CPU_CONFIG, save_dir=args.save_dir)
        trainer.train_loader = train_loader
        trainer.setup_training(learning_rate=args.lr)
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Train model
        metrics, history = trainer.train(train_loader, val_loader, args.epochs)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 