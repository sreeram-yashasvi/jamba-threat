import sys
from pathlib import Path
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model import JambaThreatModel
from jamba.model_config import ModelConfig
from jamba.utils.dataset_generator import generate_balanced_dataset
from jamba.utils.training_tracker import TrainingTracker
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_balanced_dataset():
    """Generate a new balanced dataset."""
    logger.info("Generating balanced dataset...")
    X, y = generate_balanced_dataset(
        n_samples=35000,
        n_features=20,
        threat_ratio=0.5,
        random_state=42,
        output_dir="data/balanced"
    )
    return X, y

def create_data_loaders(X, y, batch_size=32):
    """Create train/val/test data loaders."""
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.values)
    
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_dataset = TensorDataset(
        X_tensor[train_size:train_size+val_size],
        y_tensor[train_size:train_size+val_size]
    )
    test_dataset = TensorDataset(
        X_tensor[train_size+val_size:],
        y_tensor[train_size+val_size:]
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )

def train_model(model, train_loader, val_loader, config, device):
    """Train the model and return metrics."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    start_time = datetime.now()
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_metrics = {'loss': 0, 'correct': 0, 'total': 0}
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_metrics['loss'] += loss.item()
            predictions = (outputs.squeeze() > 0.5).float()
            train_metrics['correct'] += (predictions == batch_y).sum().item()
            train_metrics['total'] += len(batch_y)
        
        # Validation
        model.eval()
        val_metrics = {'loss': 0, 'correct': 0, 'total': 0}
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                val_metrics['loss'] += loss.item()
                predictions = (outputs.squeeze() > 0.5).float()
                val_metrics['correct'] += (predictions == batch_y).sum().item()
                val_metrics['total'] += len(batch_y)
        
        # Calculate epoch metrics
        train_loss = train_metrics['loss'] / len(train_loader)
        train_acc = train_metrics['correct'] / train_metrics['total']
        val_loss = val_metrics['loss'] / len(val_loader)
        val_acc = val_metrics['correct'] / val_metrics['total']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
            
        logger.info(f"Epoch {epoch+1}/{config.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        'accuracy': val_acc,
        'f1_score': val_acc,  # Simplified
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'training_time': (datetime.now() - start_time).total_seconds(),
        'early_stopped': patience_counter >= patience
    }, history

def run_experiments():
    """Run training experiments with different configurations."""
    # Create dataset
    logger.info("Generating balanced dataset...")
    X, y = generate_balanced_dataset()
    
    # Initialize training tracker
    tracker = TrainingTracker(log_dir="experiments/training_logs")
    
    # Define experiment configurations
    configurations = [
        {'learning_rate': 0.001, 'batch_size': 32, 'n_heads': 4, 'feature_layers': 2},
        {'learning_rate': 0.0005, 'batch_size': 64, 'n_heads': 8, 'feature_layers': 3},
        {'learning_rate': 0.0001, 'batch_size': 128, 'n_heads': 4, 'feature_layers': 4},
        {'learning_rate': 0.001, 'batch_size': 256, 'n_heads': 8, 'feature_layers': 2},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    for i, config_params in enumerate(configurations):
        logger.info(f"\nRunning experiment {i+1}/{len(configurations)}")
        logger.info(f"Configuration: {config_params}")
        
        # Create config and data loaders
        config = ModelConfig(
            input_dim=20,
            hidden_dim=64,
            output_dim=1,
            dropout_rate=0.3,
            epochs=30,
            **config_params
        )
        
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, batch_size=config.batch_size
        )
        
        # Train model
        model = JambaThreatModel(config)
        metrics, history = train_model(model, train_loader, val_loader, config, device)
        
        # Log the run
        tracker.log_run(
            config=config.__dict__,
            metrics=metrics,
            training_history=history
        )
    
    # Generate analysis
    tracker.plot_run_comparison(metric='accuracy')
    tracker.plot_run_comparison(metric='f1')
    
    impact_analysis = tracker.analyze_parameter_impact()
    logger.info("\nParameter Impact Analysis:")
    logger.info(impact_analysis)
    
    best_runs = tracker.get_best_runs(metric='accuracy', top_n=3)
    logger.info("\nTop 3 Best Runs:")
    logger.info(best_runs)

if __name__ == "__main__":
    run_experiments() 