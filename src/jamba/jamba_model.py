#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple
from .model_config import ModelConfig, VERSION_COMPATIBILITY

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
    """Jamba Threat Detection Model with versioning and configuration management"""
    
    def __init__(self, config: ModelConfig):
        super(JambaThreatModel, self).__init__()
        self.config = config
        
        # Log initialization
        logger.info(f"Initializing JambaThreatModel v{config.version}")
        logger.info(f"Config: {config.to_dict()}")
        
        # Calculate dimensions
        self.hidden_dim = config.hidden_dim or max(128, min(512, int(config.input_dim * 1.5)))
        self.n_heads = config.n_heads or max(4, min(8, config.input_dim // 64))
        self.feature_layers = config.feature_layers or max(2, min(4, config.input_dim // 128))
        
        # Input projection
        if config.input_dim != self.hidden_dim:
            self.projection = nn.Linear(config.input_dim, self.hidden_dim)
            self.use_projection = True
        else:
            self.projection = nn.Identity()
            self.use_projection = False
        
        # Feature extraction layers
        feat_layers = []
        current_dim = self.hidden_dim
        for i in range(self.feature_layers):
            layer_dim = self.hidden_dim // (2 ** min(i, 2))
            feat_layers.extend([
                nn.Linear(current_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = layer_dim
        
        self.feature_extraction = nn.Sequential(*feat_layers)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=current_dim,
            num_heads=self.n_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Temporal processing
        self.gru = nn.GRU(
            current_dim,
            current_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate if self.feature_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers
        self.out_linear1 = nn.Linear(current_dim * 2, self.hidden_dim)
        self.out_bn = nn.BatchNorm1d(self.hidden_dim)
        self.out_linear2 = nn.Linear(self.hidden_dim, config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Input validation
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            batch_size = x.shape[0]
            
            # Project input if needed
            x = self.projection(x)
            
            # Feature extraction
            x = self.feature_extraction(x)
            
            # Prepare for attention
            x_seq = x.unsqueeze(1) if len(x.shape) == 2 else x
            
            # Self-attention with error handling
            try:
                attn_output, _ = self.attention(x_seq, x_seq, x_seq)
                
                # GRU processing
                gru_out, _ = self.gru(attn_output)
                
                # Get final state
                if len(gru_out.shape) == 3:
                    gru_out = gru_out[:, -1, :]
                
            except RuntimeError as e:
                logger.warning(f"Attention mechanism failed, using fallback: {str(e)}")
                gru_out = x
            
            # Output layers
            out = self.out_linear1(gru_out)
            out = self.out_bn(out)
            out = F.silu(out)
            out = self.out_linear2(out)
            
            return out
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings"""
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            x = self.projection(x)
            return self.feature_extraction(x)
    
    def save(self, path: str):
        """Save model with configuration"""
        save_dict = {
            'model_state': self.state_dict(),
            'config': self.config.to_dict(),
            'version': self.config.version
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'JambaThreatModel':
        """Load model with version compatibility check"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        save_dict = torch.load(path)
        
        # Version compatibility check
        model_version = save_dict.get('version', '0.9.0')  # Default for old models
        current_version = ModelConfig().version
        
        if current_version not in VERSION_COMPATIBILITY.get(model_version, []):
            raise ValueError(
                f"Model version {model_version} is not compatible with current version {current_version}"
            )
        
        # Create config and model
        config = ModelConfig.from_dict(save_dict['config'])
        model = cls(config)
        model.load_state_dict(save_dict['model_state'])
        
        logger.info(f"Loaded model version {model_version}")
        return model

def create_model(config: Optional[ModelConfig] = None) -> JambaThreatModel:
    """Factory function to create model with proper configuration"""
    if config is None:
        config = ModelConfig()
        
    # Determine if GPU is available and select appropriate config
    if torch.cuda.is_available():
        config = ModelConfig(**{**config.to_dict(), **DEFAULT_GPU_CONFIG.to_dict()})
    else:
        config = ModelConfig(**{**config.to_dict(), **DEFAULT_CPU_CONFIG.to_dict()})
    
    return JambaThreatModel(config)

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