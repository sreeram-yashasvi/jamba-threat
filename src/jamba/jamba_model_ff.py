#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import logging
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import json
from .model_config import ModelConfig, VERSION_COMPATIBILITY

# Default configuration for CPU training
DEFAULT_CPU_CONFIG = ModelConfig(
    input_dim=20,  # Default input dimension
    hidden_dim=64,  # Default hidden dimension
    output_dim=1,  # Binary classification
    dropout_rate=0.2,
    learning_rate=0.001,
    batch_size=32,
    device="cpu"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Set seeds for deterministic behavior

def set_seed(seed=42):
    """Set seeds for deterministic behavior"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds for reproducibility
set_seed()

logger = logging.getLogger(__name__)

class JambaThreatModel(nn.Module):
    """Neural network model for threat detection."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        logger.info(f"Initializing JambaThreatModel v{config.version}")
        logger.info(f"Config: {config.to_dict()}")
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Feature processing layers
        self.feature_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
            for _ in range(config.feature_layers)
        ])
        
        # Output classifier
        self.classifier = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        """Forward pass through the model."""
        try:
            # Feature extraction
            x = self.feature_extractor(x)
            
            # Process through feature layers
            for layer in self.feature_processor:
                x = layer(x)
            
            # Output layer (no sigmoid - will use BCEWithLogitsLoss)
            x = self.classifier(x)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from input data."""
        with torch.no_grad():
            return self.feature_extractor(x)


def create_model(config: Optional[Dict[str, Any]] = None) -> JambaThreatModel:
    """Create a new instance of the model with optional config override."""
    from .model_config import ModelConfig
    if config is None:
        config = {}
    return JambaThreatModel(ModelConfig(**config))


def load_model(path: Union[str, Path], device: str = 'cpu') -> JambaThreatModel:
    """Load a model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
        
    state_dict = torch.load(path, map_location=device)
    config = ModelConfig(**state_dict['config'])
    model = JambaThreatModel(config)
    model.load_state_dict(state_dict['model'])
    return model


def save_model(model: JambaThreatModel, path: Union[str, Path]) -> None:
    """Save a model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = {
        'model': model.state_dict(),
        'config': model.config.__dict__
    }
    torch.save(state_dict, path)


class ThreatDataset(Dataset):
    """Dataset for loading threat detection data"""
    def __init__(self, data, targets=None, input_dim=None):
        """
        Initialize the dataset
        
        Args:
            data: Features data (DataFrame or array)
            targets: Target labels (if None, assumes data is a DataFrame with targets)
            input_dim: Input dimension to pad/truncate to (optional)
        """
        # Handle different input types
        if hasattr(data, 'values'):
            # It's a DataFrame
            if targets is None and 'target' in data.columns:
                self.targets = torch.tensor(data['target'].values, dtype=torch.long)
                self.features = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
            elif targets is None and 'is_threat' in data.columns:
                self.targets = torch.tensor(data['is_threat'].values, dtype=torch.long)
                self.features = torch.tensor(data.drop('is_threat', axis=1).values, dtype=torch.float32)
            else:
                if targets is None:
                    raise ValueError("No target column found in DataFrame and no targets provided")
                self.targets = torch.tensor(targets, dtype=torch.long)
                self.features = torch.tensor(data.values, dtype=torch.float32)
        else:
            # It's a numpy array or similar
            self.features = torch.tensor(data, dtype=torch.float32)
            if targets is not None:
                self.targets = torch.tensor(targets, dtype=torch.long)
            else:
                # Create dummy targets if none provided
                self.targets = torch.zeros(len(self.features), dtype=torch.long)
        
        # Handle dimensionality
        if input_dim and self.features.shape[1] != input_dim:
            if self.features.shape[1] < input_dim:
                # Pad with zeros
                padding = torch.zeros(self.features.shape[0], input_dim - self.features.shape[1])
                self.features = torch.cat([self.features, padding], dim=1)
            else:
                # Truncate
                self.features = self.features[:, :input_dim]
                
        logging.info(f"Created dataset with {len(self.features)} samples, feature dimension: {self.features.shape[1]}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx] 