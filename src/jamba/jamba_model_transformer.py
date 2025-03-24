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

# Default configuration for CPU training (with transformer-specific arguments)
DEFAULT_CPU_CONFIG = ModelConfig(
    input_dim=20,            # Input feature dimension
    hidden_dim=64,           # Embedding and hidden dimension
    output_dim=1,            # Binary classification
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

class JambaThreatTransformerModel(nn.Module):
    """Transformer based model for threat detection using self-attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        logger.info(f"Initializing JambaThreatTransformerModel v{config.version}")
        logger.info(f"Config: {config.to_dict()}")
        
        # Instead of projecting the entire vector, we treat each feature as a token.
        # Input x: [batch, input_dim] -> unsqueeze to [batch, input_dim, 1]
        # Then embed each token (scalar) into hidden_dim using a linear layer.
        self.token_embedding = nn.Linear(1, config.hidden_dim)

        # Learnable positional encoding for each token position (input_dim tokens)
        self.positional_encoding = nn.Parameter(torch.randn(1, config.input_dim, config.hidden_dim))

        # Transformer encoder layers, using defaults if not provided in config
        nheads = getattr(config, 'num_heads', 4)
        n_layers = getattr(config, 'transformer_layers', 2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_dim, nhead=nheads, dropout=config.dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classifier head: from pooled transformer output to output dimension
        self.classifier = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        # x shape: [batch, input_dim]
        try:
            # Convert input to tokens: [batch, input_dim, 1]
            x = x.unsqueeze(-1)
            # Embed each token
            tokens = self.token_embedding(x)  # [batch, input_dim, hidden_dim]
            
            # Add learnable positional encoding
            tokens = tokens + self.positional_encoding
            
            # Pass through the transformer encoder
            encoded = self.transformer_encoder(tokens)  # [batch, input_dim, hidden_dim]
            
            # Pooling: mean over the token dimension
            pooled = encoded.mean(dim=1)  # [batch, hidden_dim]
            
            # Classifier for binary prediction; note: no sigmoid as BCEWithLogitsLoss is used
            out = self.classifier(pooled)
            return out
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.unsqueeze(-1)
            tokens = self.token_embedding(x)
            tokens = tokens + self.positional_encoding
            encoded = self.transformer_encoder(tokens)
            pooled = encoded.mean(dim=1)
            return pooled


def create_model(config: Optional[Dict[str, Any]] = None) -> JambaThreatTransformerModel:
    """Create a new instance of the transformer model with optional config override."""
    from .model_config import ModelConfig
    if config is None:
        config = {}
    return JambaThreatTransformerModel(ModelConfig(**config))


def load_model(path: Union[str, Path], device: str = 'cpu') -> JambaThreatTransformerModel:
    """Load a transformer model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    state_dict = torch.load(path, map_location=device)
    config = ModelConfig(**state_dict['config'])
    model = JambaThreatTransformerModel(config)
    model.load_state_dict(state_dict['model'])
    return model


def save_model(model: JambaThreatTransformerModel, path: Union[str, Path]) -> None:
    """Save the transformer model to disk."""
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
        # Similar implementation as in the feed-forward model for consistency
        if hasattr(data, 'values'):
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
            self.features = torch.tensor(data, dtype=torch.float32)
            if targets is not None:
                self.targets = torch.tensor(targets, dtype=torch.long)
            else:
                self.targets = torch.zeros(len(self.features), dtype=torch.long)
        
        if input_dim and self.features.shape[1] != input_dim:
            if self.features.shape[1] < input_dim:
                padding = torch.zeros(self.features.shape[0], input_dim - self.features.shape[1])
                self.features = torch.cat([self.features, padding], dim=1)
            else:
                self.features = self.features[:, :input_dim]
        
        logging.info(f"Created dataset with {len(self.features)} samples, feature dimension: {self.features.shape[1]}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

if __name__ == "__main__":
    logger.info("Testing JambaThreatTransformerModel...")
    batch_size = 4
    input_dim = 20
    sample_input = torch.randn(batch_size, input_dim)
    
    model = JambaThreatTransformerModel(DEFAULT_CPU_CONFIG)
    model.eval()
    
    with torch.no_grad():
        output = model(sample_input)
    
    logger.info(f"Sample output shape: {output.shape}")
    logger.info("Transformer model test completed.") 