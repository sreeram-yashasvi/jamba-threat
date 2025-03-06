import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import logging
import sys
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set deterministic behavior for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        
        try:
            logger.info(f"Initializing JambaThreatModel with input dimension: {input_dim}")
            
            # Important: Use a fixed embed_dim for consistency across all models
            self.embed_dim = 28  # Match the input dimension exactly for consistency
            
            # Fixed number of attention heads
            num_heads = 4
            logger.info(f"Using {num_heads} attention heads")
            
            # Multi-head attention layer
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim, 
                num_heads=num_heads,
                batch_first=True
            )
            
            # Feature extraction layers
            self.fc1 = nn.Linear(self.embed_dim, 256)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            
            # Bidirectional GRU for temporal processing
            self.temporal = nn.GRU(
                input_size=128,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # Output layers
            self.fc3 = nn.Linear(128, 64)  # 128 = 64*2 (bidirectional)
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
            # Reshape for attention: (batch_size, seq_len=1, embed_dim)
            x_reshaped = x.unsqueeze(1)
            
            # Apply attention (with batch_first=True)
            x_att, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x_att = x_att.squeeze(1)
            
            # Feature extraction
            x = self.fc1(x_att)
            x = self.relu1(x)
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            
            # Temporal processing (batch_size, seq_len=1, features)
            x_temporal = x.unsqueeze(1)
            temporal_out, _ = self.temporal(x_temporal)
            
            # Flatten the output (bidirectional doubles the last dimension)
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