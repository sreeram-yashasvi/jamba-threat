import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log model initialization
logger.info(f"Initializing Jamba Threat Model module")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

class ThreatDataset(Dataset):
    """Dataset for threat data."""
    
    def __init__(self, features, targets):
        """
        Initialize the dataset.
        
        Args:
            features: Feature dataframe or array
            targets: Target array
        """
        if isinstance(features, pd.DataFrame):
            self.X = features.values.astype(np.float32)
        else:
            self.X = features.astype(np.float32)
            
        if isinstance(targets, pd.Series):
            self.y = targets.values.astype(np.float32)
        else:
            self.y = targets.astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class JambaThreatModel(nn.Module):
    """Jamba model for threat detection."""
    
    def __init__(self, input_dim):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
        """
        super(JambaThreatModel, self).__init__()
        
        logger.info(f"Initializing JambaThreatModel with input dimension: {input_dim}")
        
        try:
            # Simplify the attention head logic by using a fixed embedding dimension
            self.orig_input_dim = input_dim
            self.embed_dim = 128  # Fixed embedding dimension that is divisible by common head counts
            self.num_heads = 4    # Fixed number of attention heads
            
            # Always use an embedding layer to project to the fixed dimension
            self.embedding = nn.Linear(input_dim, self.embed_dim)
            logger.info(f"Using embedding layer: {input_dim} -> {self.embed_dim} with {self.num_heads} attention heads")
                
            # Multi-head self-attention layer
            self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
            
            # Feature extraction
            self.fc1 = nn.Linear(self.embed_dim, 256)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            
            # Further feature processing
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            
            # Temporal processing using GRU (faster than LSTM)
            self.temporal = nn.GRU(128, 64, batch_first=True, bidirectional=True)
            
            # Output layers
            self.fc3 = nn.Linear(128, 64)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(0.1)
            self.fc4 = nn.Linear(64, 32)
            self.relu4 = nn.ReLU()
            self.output = nn.Linear(32, 1)
            
            logger.info("JambaThreatModel initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing JambaThreatModel: {str(e)}")
            raise
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        try:
            # Apply embedding if needed
            x = self.embedding(x)
            
            # Self-attention
            # Reshape for attention: (batch_size, seq_len=1, embed_dim)
            x_reshaped = x.unsqueeze(1)
            
            # Apply attention - PyTorch expects (seq_len, batch_size, embed_dim)
            x_att, _ = self.attention(
                x_reshaped.transpose(0, 1),
                x_reshaped.transpose(0, 1),
                x_reshaped.transpose(0, 1)
            )
            
            # Convert back to (batch_size, embed_dim)
            x_att = x_att.transpose(0, 1).squeeze(1)
            
            # Feature extraction
            x = self.fc1(x_att)
            x = self.relu1(x)
            x = self.dropout1(x)
            
            # Further feature processing
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            
            # Temporal processing - needs (batch_size, seq_len, features)
            x_temporal = x.unsqueeze(1)  # Add sequence dimension
            temporal_out, _ = self.temporal(x_temporal)
            
            # Flatten the output (batch_size, seq_len=1, 2*hidden_size)
            # The *2 is because we're using bidirectional
            temporal_out = temporal_out.reshape(temporal_out.size(0), -1)
            
            # Output layers
            x = self.fc3(temporal_out)
            x = self.relu3(x)
            x = self.dropout3(x)
            x = self.fc4(x)
            x = self.relu4(x)
            x = self.output(x)
            
            return x
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

# For testing the model directly
if __name__ == "__main__":
    logger.info("Testing JambaThreatModel...")
    # Create a sample input tensor
    batch_size = 4
    input_dim = 18  # Number of features
    sample_input = torch.randn(batch_size, input_dim)
    
    # Initialize the model
    model = JambaThreatModel(input_dim)
    
    # Run a forward pass
    output = model(sample_input)
    
    logger.info(f"Model output shape: {output.shape}")
    logger.info("Model test completed successfully") 