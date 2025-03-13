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
from typing import Optional, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from jamba.model_config import ModelConfig, DEFAULT_GPU_CONFIG, DEFAULT_CPU_CONFIG
from jamba.data_preprocessing import DataPreprocessor, create_preprocessor
from jamba.jamba_model import JambaThreatModel, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Unified training manager for Jamba Threat Model"""
    
    def __init__(self, config: Optional[ModelConfig] = None, 
                 save_dir: str = "models"):
        self.config = config or (DEFAULT_GPU_CONFIG if torch.cuda.is_available() 
                               else DEFAULT_CPU_CONFIG)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_model(self.config).to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # Initialize preprocessor
        self.preprocessor = None
    
    def setup_training(self, learning_rate: float = 0.001):
        """Setup training components"""
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=1,  # Will be updated when data is loaded
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        )
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=self.config.use_mixed_precision)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=self.config.use_mixed_precision):
                    output = self.model(data)
                    loss = self.criterion(output, target.unsqueeze(1))
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Train Epoch: {self.current_epoch} '
                              f'[{batch_idx}/{len(train_loader)} '
                              f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                              f'Loss: {loss.item():.6f}')
                              
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    with autocast(enabled=self.config.use_mixed_precision):
                        output = self.model(data)
                        val_loss += self.criterion(output, target.unsqueeze(1)).item()
                    
                    pred = (output > 0).float()
                    total += target.size(0)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    
                except Exception as e:
                    logger.error(f"Error in validation: {str(e)}")
                    continue
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        logger.info(f'Validation set: Average loss: {val_loss:.4f}, '
                   f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
        
        return val_loss
    
    def save_checkpoint(self, val_loss: float, is_best: bool = False):
        """Save training checkpoint"""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'config': self.config.to_dict()
            }
            
            # Save latest checkpoint
            checkpoint_path = self.save_dir / f'checkpoint_latest.pt'
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = self.save_dir / f'model_best.pt'
                torch.save(checkpoint, best_path)
                logger.info(f'Saved best model to {best_path}')
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                return
            
            checkpoint = torch.load(checkpoint_path)
            
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['val_loss']
            
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, patience: int = 5):
        """Full training loop with early stopping"""
        try:
            # Update scheduler steps
            self.scheduler.total_steps = epochs * len(train_loader)
            
            start_time = time.time()
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training phase
                train_loss = self.train_epoch(train_loader)
                
                # Validation phase
                val_loss = self.validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step()
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                self.save_checkpoint(val_loss, is_best)
                
                # Early stopping
                if self.early_stop_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                epoch_time = time.time() - epoch_start
                logger.info(f'Epoch {epoch + 1}/{epochs} - '
                          f'Train Loss: {train_loss:.6f}, '
                          f'Val Loss: {val_loss:.6f}, '
                          f'Time: {epoch_time:.2f}s')
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

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
        trainer = ModelTrainer(save_dir=args.save_dir)
        trainer.setup_training(learning_rate=args.lr)
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Train model
        trainer.train(train_loader, val_loader, args.epochs, args.patience)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 