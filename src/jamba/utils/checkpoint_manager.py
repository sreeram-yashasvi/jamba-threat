#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import logging
import json
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints with versioning and metadata."""
    
    def __init__(self, base_dir='checkpoints', max_checkpoints=5):
        """
        Initialize the checkpoint manager.
        
        Args:
            base_dir (str): Base directory for storing checkpoints
            max_checkpoints (int): Maximum number of checkpoints to keep
        """
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint index file if it doesn't exist
        self.index_file = self.base_dir / 'checkpoint_index.json'
        if not self.index_file.exists():
            self._save_index({
                'checkpoints': [],
                'latest': None,
                'best': None
            })
    
    def _save_index(self, index_data):
        """Save checkpoint index to file."""
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _load_index(self):
        """Load checkpoint index from file."""
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False, name=None):
        """
        Save a model checkpoint.
        
        Args:
            model: The PyTorch model
            optimizer: The optimizer
            epoch (int): Current epoch
            metrics (dict): Dictionary of metrics
            is_best (bool): Whether this is the best model so far
            name (str): Optional name for the checkpoint
        
        Returns:
            str: Path to the saved checkpoint
        """
        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = name or f"checkpoint_{timestamp}"
        checkpoint_dir = self.base_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': model.config.__dict__,
            'timestamp': timestamp
        }
        
        checkpoint_path = checkpoint_dir / 'model.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata = {
            'name': checkpoint_name,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best
        }
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update index
        index = self._load_index()
        index['checkpoints'].append(checkpoint_name)
        index['latest'] = checkpoint_name
        if is_best:
            index['best'] = checkpoint_name
        
        # Remove old checkpoints if exceeding max_checkpoints
        if len(index['checkpoints']) > self.max_checkpoints:
            to_remove = index['checkpoints'][:-self.max_checkpoints]
            for checkpoint_name in to_remove:
                if checkpoint_name != index['best']:  # Don't remove best checkpoint
                    self._remove_checkpoint(checkpoint_name)
                    index['checkpoints'].remove(checkpoint_name)
        
        self._save_index(index)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, name='latest', device='cpu'):
        """
        Load a checkpoint.
        
        Args:
            name (str): Name of checkpoint or 'latest'/'best'
            device (str): Device to load the model to
        
        Returns:
            dict: Checkpoint data
        """
        index = self._load_index()
        
        if name == 'latest':
            name = index['latest']
        elif name == 'best':
            name = index['best']
        
        if not name or name not in index['checkpoints']:
            raise ValueError(f"Checkpoint {name} not found")
        
        checkpoint_path = self.base_dir / name / 'model.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    
    def _remove_checkpoint(self, name):
        """Remove a checkpoint directory."""
        checkpoint_dir = self.base_dir / name
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Removed checkpoint: {name}")
    
    def list_checkpoints(self):
        """List all available checkpoints."""
        index = self._load_index()
        checkpoints = []
        
        for name in index['checkpoints']:
            metadata_path = self.base_dir / name / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
        
        return checkpoints
    
    def get_best_checkpoint(self):
        """Get the best checkpoint information."""
        index = self._load_index()
        if index['best']:
            return self.load_checkpoint('best')
        return None

def save_pretrained(model, save_dir, optimizer=None, epoch=None, metrics=None):
    """
    Utility function to save a pretrained model with its configuration.
    
    Args:
        model: The PyTorch model
        save_dir (str): Directory to save the model
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state and config
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    model_path = save_dir / 'model.pt'
    torch.save(checkpoint, model_path)
    
    # Save config separately for easy access, handling device object
    config_dict = model.config.__dict__.copy()
    if 'device' in config_dict:
        config_dict['device'] = str(config_dict['device'])
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved pretrained model to: {save_dir}")
    return str(save_dir)

def load_pretrained(model_dir, device='cpu'):
    """
    Utility function to load a pretrained model.
    
    Args:
        model_dir (str): Directory containing the saved model
        device (str): Device to load the model to
    
    Returns:
        tuple: (model, checkpoint_data)
    """
    from ..jamba_model_transformer import JambaThreatTransformerModel
    from ..model_config import ModelConfig
    
    model_dir = Path(model_dir)
    model_path = model_dir / 'model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = ModelConfig(**checkpoint['config'])
    
    model = JambaThreatTransformerModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Loaded pretrained model from: {model_dir}")
    return model, checkpoint 