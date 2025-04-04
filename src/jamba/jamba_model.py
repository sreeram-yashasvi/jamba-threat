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
        """
        Initialize the model.
        
        Args:
            config: ModelConfig instance containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        layers = []
        current_dim = config.input_dim
        
        for i in range(config.feature_layers):
            next_dim = config.hidden_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = next_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Reshape for attention (batch_size, sequence_length=1, hidden_dim)
        features = features.unsqueeze(1)
        
        # Apply self-attention
        attended_features, _ = self.attention(features, features, features)
        
        # Reshape back
        attended_features = attended_features.squeeze(1)
        
        # Classification
        output = self.classifier(attended_features)
        
        return output

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

# For testing the model directly
if __name__ == "__main__":
    logger.info("Testing JambaThreatModel...")
    # Create a sample input tensor
    batch_size = 4
    input_dim = 28  # Match expected dimension for dataset features
    sample_input = torch.randn(batch_size, input_dim)
    
    # Initialize the model
    model = JambaThreatModel(input_dim)
    
    # Set eval mode for consistent output
    model.eval()
    
    # Test serialization
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    # Load the model in a new instance
    model2 = JambaThreatModel(input_dim)
    model2.load_state_dict(torch.load(buffer))
    model2.eval()
    
    # Compare outputs
    with torch.no_grad():
        output1 = model(sample_input)
        output2 = model2(sample_input)
    
    # Verify outputs match
    match = torch.allclose(output1, output2)
    logger.info(f"Serialization test {'passed' if match else 'failed'}")
    logger.info(f"Model output shape: {output1.shape}")
    logger.info("Model test completed successfully") 