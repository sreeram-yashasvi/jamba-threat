import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import time
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class JambaThreatModel(nn.Module):
    def __init__(self, input_dim):
        super(JambaThreatModel, self).__init__()
        
        # OPTIMIZATION: Replace dynamic head calculation with efficient padding
        self.embed_dim = ((input_dim + 3) // 4) * 4  # Round up to nearest multiple of 4
        self.projection = nn.Linear(input_dim, self.embed_dim) if input_dim != self.embed_dim else nn.Identity()
        
        # OPTIMIZATION: Use fixed 4 heads for better GPU utilization
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4)
        
        # OPTIMIZATION: Use nn.Sequential for better JIT optimization
        self.feature_layers = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.SiLU(),  # Replace ReLU with SiLU for better gradient flow
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),  # Add batch norm for faster convergence
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128)  # Add batch norm
        )
        
        # OPTIMIZATION: Use GRU instead of LSTM (faster, similar performance)
        self.temporal = nn.GRU(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        
        # OPTIMIZATION: Add residual connections
        self.output_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Apply projection if needed
        x = self.projection(x)
        
        # OPTIMIZATION: Use batch attention for larger batches
        batch_size = x.size(0)
        if batch_size > 1:
            # Process entire batch at once
            x_att, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            x_att = x_att.squeeze(1)
        else:
            # Process single sample
            x_att, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x_att = x_att.squeeze(0)
        
        # Extract features
        features = self.feature_layers(x_att)
        
        # Process temporal information
        temporal_out, _ = self.temporal(features.unsqueeze(1))
        temporal_out = temporal_out.squeeze(1)
        
        # Generate predictions
        output = self.output_layers(temporal_out)
        return output

class ThreatModelTrainer:
    def __init__(self, model_save_dir="models"):
        """Initialize the Jamba Threat Model trainer for local training.
        
        Args:
            model_save_dir: Directory to save trained models
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        self.model_save_dir = model_save_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
    def prepare_data(self, data_path, target_column='is_threat', batch_size=32):
        """Optimized data loading and preparation."""
        logger.info(f"Loading data from {data_path}")
        
        # OPTIMIZATION: Use memory mapping for large files
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, low_memory=True)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path, engine='pyarrow')
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        # OPTIMIZATION: Convert to more efficient types
        for col in df.select_dtypes('float64').columns:
            df[col] = df[col].astype('float32')
        
        # Split features and target
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # OPTIMIZATION: Use stratified split for imbalanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # OPTIMIZATION: Prefetch and pin memory
        train_dataset = ThreatDataset(X_train, y_train)
        test_dataset = ThreatDataset(X_test, y_test)
        
        # OPTIMIZATION: Calculate optimal number of workers
        optimal_workers = min(os.cpu_count(), 8)  # Limit to 8 workers max
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=optimal_workers,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=3  # Prefetch 3 batches per worker
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size * 2,  # Double batch size for validation
            pin_memory=True,
            num_workers=optimal_workers,
            persistent_workers=True
        )
        
        return train_loader, test_loader, X_train, X_test, y_train, y_test
        
    def train_model(self, data_path, target_column='is_threat', epochs=50, learning_rate=0.001, batch_size=32):
        """Optimized training loop."""
        try:
            # Prepare data
            train_loader, test_loader, X_train, X_test, y_train, y_test = self.prepare_data(
                data_path, target_column, batch_size)
            
            # Initialize model
            input_dim = X_train.shape[1]
            self.model = JambaThreatModel(input_dim).to(self.device)
            logger.info(f"Initialized Jamba model with input dimension: {input_dim}")
            
            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            
            # OPTIMIZATION: Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # OPTIMIZATION: Use AdamW instead of Adam
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            
            # OPTIMIZATION: Use OneCycleLR scheduler
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,  # Warm up for 30% of training
                div_factor=25,
                final_div_factor=1000
            )
            
            # OPTIMIZATION: Use AMP for mixed precision
            scaler = GradScaler()
            
            # OPTIMIZATION: Track time per epoch
            epoch_times = []
            
            # Training loop
            logger.info("Starting training...")
            for epoch in range(epochs):
                start_time = time.time()
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)  # Non-blocking transfers
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                    
                    with autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y.unsqueeze(1))
                    
                    scaler.scale(loss).backward()
                    
                    # OPTIMIZATION: Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()  # Step per batch with OneCycleLR
                    
                    total_loss += loss.item()
                
                avg_train_loss = total_loss / len(train_loader)
                
                # Validation
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        with autocast():
                            outputs = self.model(batch_X)
                            val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
                        
                        predicted = (outputs > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted.squeeze() == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(test_loader)
                accuracy = correct / total
                
                # OPTIMIZATION: Log training stats
                epoch_time = time.time() - start_time
                epoch_times.append(epoch_time)
                logger.info(f'Epoch [{epoch+1}/{epochs}], Time: {epoch_time:.2f}s, '
                          f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                          f'Val Accuracy: {accuracy:.4f}')
            
            # OPTIMIZATION: Log training stats
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
            logger.info(f"Total training time: {sum(epoch_times):.2f}s")
            
            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_save_dir, f"jamba_threat_model_{timestamp}.pth")
            
            model_info = {
                'model_state': self.model.state_dict(),
                'input_dim': input_dim,
                'timestamp': timestamp,
                'epochs': epochs,
                'final_accuracy': accuracy,
                'history': {
                    'train_loss': [avg_train_loss],
                    'val_loss': [avg_val_loss],
                    'val_accuracy': [accuracy]
                }
            }
            
            torch.save(model_info, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Plot training curves
            self.plot_training_history(model_info['history'], save_path=os.path.join(self.model_save_dir, f"training_history_{timestamp}.png"))
            
            return self.model, model_info['history']
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self, test_loader):
        """Evaluate the model on test data."""
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                predicted = (outputs > 0.5).float()
                
                # Store predictions and true labels for metrics calculation
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())
                
                total += batch_y.size(0)
                correct += (predicted.squeeze() == batch_y).sum().item()
        
        accuracy = correct / total
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # More detailed metrics could be calculated here using sklearn.metrics
        
        return accuracy, predictions, true_labels
    
    def load_model(self, model_path):
        """Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            loaded model
        """
        model_info = torch.load(model_path, map_location=self.device)
        input_dim = model_info['input_dim']
        
        # Initialize the model
        model = JambaThreatModel(input_dim).to(self.device)
        
        # Load the model weights
        model.load_state_dict(model_info['model_state'])
        
        self.model = model
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model accuracy: {model_info['final_accuracy']:.4f}")
        
        return model

    def plot_training_history(self, history, save_path=None):
        """Plot training history curves.
        
        Args:
            history: Dictionary containing training metrics
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot training & validation loss
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        
        # Plot validation accuracy
        plt.subplot(2, 1, 2)
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()

def optimize_memory():
    """Apply memory optimizations."""
    # Free memory
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # OPTIMIZATION: Set memory fraction for GPU
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory
        
        # OPTIMIZATION: Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # OPTIMIZATION: Set allocator config
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        
    # OPTIMIZATION: Set number of threads for CPU operations
    torch.set_num_threads(4)  # Limit CPU threads
    torch.set_num_interop_threads(4)  # Limit interop threads

def main():
    """Main function to demonstrate local training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Jamba Threat Detection Model')
    parser.add_argument('--data', required=True, help='Path to dataset file (.csv or .parquet)')
    parser.add_argument('--target', default='is_threat', help='Target column name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-dir', default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Initialize the trainer
    trainer = ThreatModelTrainer(model_save_dir=args.model_dir)
    
    # Train the model
    trainer.train_model(
        data_path=args.data,
        target_column=args.target,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 