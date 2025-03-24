import os
import logging
from pathlib import Path
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jamba.model_config import ModelConfig
from jamba.train import ModelTrainer
from jamba.data.sample_data import generate_sample_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    # Initialize model configuration
    config = ModelConfig(
        version='1.0.0',
        input_dim=9,
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3,
        n_heads=4,
        feature_layers=3,
        use_mixed_precision=False,
        batch_size=64,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epochs=20
    )

    # Initialize model trainer
    trainer = ModelTrainer(config=config)

    # Load and prepare data
    logger.info("Loading and preparing data...")
    data = generate_sample_data()
    
    # Create data loaders
    train_data = torch.FloatTensor(data['train'][0].values)
    train_labels = torch.FloatTensor(data['train'][1])
    val_data = torch.FloatTensor(data['val'][0].values)
    val_labels = torch.FloatTensor(data['val'][1])
    test_data = torch.FloatTensor(data['test'][0].values)
    test_labels = torch.FloatTensor(data['test'][1])

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    trainer.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    trainer.val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    trainer.test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Set up training components
    trainer.setup_training(learning_rate=config.learning_rate)

    # Train model
    logger.info(f"Training model on {config.device}...")
    trainer.train(
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        epochs=config.epochs,
        patience=5
    )

    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    trainer.load_checkpoint('models/model_best.pt')
    test_loss, test_accuracy = trainer.evaluate(trainer.test_loader)
    logger.info(f"Test set: Average loss: {test_loss:.4f}")

if __name__ == '__main__':
    main() 