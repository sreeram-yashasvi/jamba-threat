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
        
        # Determine number of attention heads - ensure input_dim is divisible by num_heads
        for num_heads in [4, 2, 7, 1]:  # Try 4 first, then fallback to other divisors
            if input_dim % num_heads == 0:
                self.num_heads = num_heads
                break
        else:
            # If no divisor found, pad the input dimension to make it divisible by 4
            pad_size = 4 - (input_dim % 4)
            input_dim += pad_size
            self.num_heads = 4
            
        logger.info(f"Using {self.num_heads} attention heads with input dimension {input_dim}")
        
        # Add embedding layer if we had to pad the input
        self.need_embedding = (self.num_heads == 4 and input_dim != self.num_heads * (input_dim // self.num_heads))
        if self.need_embedding:
            self.embedding = nn.Linear(input_dim - pad_size, input_dim)
            
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=self.num_heads)
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Temporal processing layers
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(128, 64),  # 128 because of bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply embedding if needed
        if hasattr(self, 'need_embedding') and self.need_embedding:
            x = self.embedding(x)
            
        # Apply self-attention
        x_att, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x_att = x_att.squeeze(0)
        
        # Extract features
        features = self.feature_layers(x_att)
        
        # Process temporal information
        lstm_out, _ = self.lstm(features.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        
        # Generate predictions
        output = self.output_layers(lstm_out)
        return output

class ThreatModelTrainer:
    def __init__(self, model_save_dir="models"):
        """Initialize the Jamba Threat Model trainer for local training.
        
        Args:
            model_save_dir: Directory to save trained models
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model_save_dir = model_save_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
    def prepare_data(self, data_path, target_column='is_threat', batch_size=32):
        """Load and prepare data for training.
        
        Args:
            data_path: Path to the dataset file (.csv, .parquet)
            target_column: Name of the target column
            batch_size: Batch size for DataLoader
            
        Returns:
            train_loader, test_loader, X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Split features and target
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Create data loaders
        train_dataset = ThreatDataset(X_train, y_train)
        test_dataset = ThreatDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, X_train, X_test, y_train, y_test
        
    def train_model(self, data_path, target_column='is_threat', epochs=50, learning_rate=0.001, batch_size=32):
        """Train the Jamba threat detection model locally.
        
        Args:
            data_path: Path to the dataset file
            target_column: Name of the target column
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            
        Returns:
            trained model, training history
        """
        try:
            # Prepare data
            train_loader, test_loader, X_train, X_test, y_train, y_test = self.prepare_data(
                data_path, target_column, batch_size)
            
            # Initialize model
            input_dim = X_train.shape[1]
            self.model = JambaThreatModel(input_dim).to(self.device)
            logger.info(f"Initialized Jamba model with input dimension: {input_dim}")
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # For learning rate scheduling
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Training loop
            logger.info("Starting training...")
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_train_loss = total_loss / len(train_loader)
                history['train_loss'].append(avg_train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
                        
                        predicted = (outputs > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted.squeeze() == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(test_loader)
                accuracy = correct / total
                
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(accuracy)
                
                # Update learning rate based on validation loss
                scheduler.step(avg_val_loss)
                
                logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
            
            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_save_dir, f"jamba_threat_model_{timestamp}.pth")
            
            model_info = {
                'model_state': self.model.state_dict(),
                'input_dim': input_dim,
                'timestamp': timestamp,
                'epochs': epochs,
                'final_accuracy': history['val_accuracy'][-1],
                'history': history
            }
            
            torch.save(model_info, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Plot training curves
            self.plot_training_history(history, save_path=os.path.join(self.model_save_dir, f"training_history_{timestamp}.png"))
            
            return self.model, history
            
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