import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .config import ModelConfig, DEFAULT_CPU_CONFIG
from .model.model_factory import ModelFactory

logger = logging.getLogger(__name__)

class JambaThreatModel(nn.Module):
    """
    Jamba Threat Detection Model
    A neural network model for detecting security threats based on system metrics and network behavior.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        logger.info(f"Initializing JambaThreatModel v{config.version}")
        logger.info(f"Config: {config.to_dict()}")
        
        self.config = config
        self.version = config.version
        
        # Set dimensions from validated config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout_rate = config.dropout_rate
        self.n_heads = config.n_heads
        self.feature_layers = config.feature_layers
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Feature processing layers
        self.feature_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ) for _ in range(self.feature_layers)
        ])
        
        # Output layer
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        try:
            batch_size = x.size(0)
            
            # Feature extraction
            features = self.feature_extractor(x)
            
            # Reshape for attention
            features = features.unsqueeze(1)  # Add sequence length dimension
            
            # Self-attention
            attended_features, _ = self.attention(features, features, features)
            
            # Remove sequence dimension and process features
            features = attended_features.squeeze(1)
            
            # Process through feature layers
            for layer in self.feature_processor:
                features = layer(features)
            
            # Classification
            output = self.classifier(features)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            x = self.feature_extractor(x)
            return x

# Create model factory instance
model_factory = ModelFactory(JambaThreatModel)

def create_model(config: Optional[ModelConfig] = None) -> JambaThreatModel:
    """Create a new model instance with the given configuration."""
    return model_factory.create_model(config)

def load_model(path: str, config: Optional[ModelConfig] = None) -> JambaThreatModel:
    """Load a model from file with version compatibility check."""
    return model_factory.load_model(path, config)

def save_model(model: JambaThreatModel, path: str):
    """Save model with its configuration."""
    model_factory.save_model(model, path) 