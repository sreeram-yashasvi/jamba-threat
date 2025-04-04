#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import logging

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.utils.checkpoint_manager import CheckpointManager, save_pretrained, load_pretrained
from jamba.jamba_model_transformer import JambaThreatTransformerModel
from jamba.model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize model
    config = ModelConfig(
        version='1.0.0',
        input_dim=20,
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3,
        n_heads=4,
        feature_layers=2,
        learning_rate=0.001,
        batch_size=16,
        device='cpu'
    )
    
    model = JambaThreatTransformerModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(base_dir='checkpoints/jamba_threat', max_checkpoints=3)
    
    # Example training loop with checkpoints
    logger.info("Starting training simulation...")
    for epoch in range(3):
        # Simulate training
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': 0.85 + epoch * 0.05
        }
        
        is_best = epoch == 2  # For demonstration, consider last epoch as best
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best,
            name=f"epoch_{epoch}"
        )
    
    # List all checkpoints
    logger.info("\nAvailable checkpoints:")
    checkpoints = ckpt_manager.list_checkpoints()
    for ckpt in checkpoints:
        logger.info(f"- {ckpt['name']}: epoch {ckpt['epoch']}, metrics: {ckpt['metrics']}")
    
    # Load the best checkpoint
    logger.info("\nLoading best checkpoint...")
    best_checkpoint = ckpt_manager.get_best_checkpoint()
    if best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])
        optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {best_checkpoint['epoch']}")
        logger.info(f"Metrics: {best_checkpoint['metrics']}")
    
    # Save as pretrained model
    logger.info("\nSaving as pretrained model...")
    pretrained_dir = save_pretrained(
        model=model,
        save_dir='pretrained/jamba_threat_v1',
        optimizer=optimizer,
        epoch=best_checkpoint['epoch'],
        metrics=best_checkpoint['metrics']
    )
    
    # Load pretrained model
    logger.info("\nLoading pretrained model...")
    loaded_model, loaded_checkpoint = load_pretrained(pretrained_dir)
    logger.info(f"Loaded pretrained model from epoch {loaded_checkpoint.get('epoch')}")
    logger.info(f"Config: {loaded_model.config.__dict__}")

if __name__ == '__main__':
    main() 