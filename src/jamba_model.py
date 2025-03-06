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

class JambaThreatModel(nn.Module):
    """
    Jamba Threat Detection Model - Neural network for identifying threats in network traffic
    """
    def __init__(self, input_dim=512, hidden_dim=None, output_dim=2, dropout_rate=0.3):
        super(JambaThreatModel, self).__init__()
        
        # Log initialization parameters
        logging.info(f"Initializing JambaThreatModel with input_dim={input_dim}, output_dim={output_dim}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Store input dimension for serialization checks
        self.input_dim = input_dim
        
        # Calculate optimal dimensions based on input size
        if hidden_dim is None:
            hidden_dim = max(128, min(512, int(input_dim * 1.5)))
        
        # Calculate optimal number of attention heads
        n_heads = max(4, min(8, input_dim // 64))
        
        # Calculate feature extraction layers
        feature_layers = max(2, min(4, input_dim // 128))
        
        logging.info(f"Derived architecture parameters: hidden_dim={hidden_dim}, n_heads={n_heads}, feature_layers={feature_layers}")
        
        # Input projection layer (only if needed)
        if input_dim != hidden_dim:
            self.projection = nn.Linear(input_dim, hidden_dim)
            self.use_projection = True
        else:
            self.use_projection = False
        
        # Feature extraction layers with batch normalization
        feat_layers = []
        current_dim = hidden_dim
        for i in range(feature_layers):
            layer_dim = hidden_dim // (2 ** min(i, 2))
            feat_layers.append(nn.Linear(current_dim, layer_dim))
            feat_layers.append(nn.BatchNorm1d(layer_dim))
            feat_layers.append(nn.SiLU())  # SiLU (Swish) for better gradient flow
            feat_layers.append(nn.Dropout(dropout_rate))
            current_dim = layer_dim
        
        self.feature_extraction = nn.Sequential(*feat_layers)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=current_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Temporal processing with GRU
        self.gru = nn.GRU(
            current_dim,
            current_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if feature_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers with residual connection
        self.out_linear1 = nn.Linear(current_dim * 2, hidden_dim)
        self.out_bn = nn.BatchNorm1d(hidden_dim)
        self.out_linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Ensure input is proper tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Reshape if needed (handles both single samples and batches)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size = x.shape[0]
        
        # Log input shape for debugging
        logging.debug(f"Input shape: {x.shape}")
        
        # Apply projection if configured
        if self.use_projection:
            x = self.projection(x)
        
        # Feature extraction
        x = self.feature_extraction(x)
        
        # Prepare for attention (need sequence dimension)
        if len(x.shape) == 2:
            # For non-sequential data, create a sequence of length 1
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x
        
        # Self-attention
        try:
            attn_output, _ = self.attention(x_seq, x_seq, x_seq)
            
            # GRU processing
            gru_out, _ = self.gru(attn_output)
            
            # Get final state
            if len(gru_out.shape) == 3:
                # Take the last output for sequence data
                gru_out = gru_out[:, -1, :]
                
            # Output layers with residual connection
            out = self.out_linear1(gru_out)
            out = self.out_bn(out)
            out = F.silu(out)  # SiLU activation
            out = self.out_linear2(out)
            
            return out
            
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            # Fallback to simpler processing if attention fails
            out = self.out_linear1(x)
            out = self.out_bn(out)
            out = F.silu(out)
            out = self.out_linear2(out)
            return out
    
    def get_embedding(self, x):
        """Extract feature embeddings from the model"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        if self.use_projection:
            x = self.projection(x)
            
        return self.feature_extraction(x)
    
    def __getstate__(self):
        """Custom state management for improved serialization"""
        state = self.__dict__.copy()
        # Add model metadata
        state['_model_version'] = '1.0.0'
        state['_serialization_date'] = torch.tensor(
            [torch.cuda.current_device() if torch.cuda.is_available() else -1]
        )
        return state
    
    def __setstate__(self, state):
        """Custom state loading for improved deserialization"""
        version = state.pop('_model_version', None)
        if version:
            logging.info(f"Loading model version: {version}")
        
        self.__dict__.update(state)

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