#!/usr/bin/env python3
import dataclasses
from typing import Optional, Dict, Any
import json
import os
from dataclasses import dataclass, asdict

# Version compatibility mapping
VERSION_COMPATIBILITY = {
    '1.0.0': {
        'min_input_dim': 1,
        'max_input_dim': 1024,
        'min_hidden_dim': 16,
        'max_hidden_dim': 512,
        'min_output_dim': 1,
        'max_output_dim': 100
    }
}

@dataclass
class ModelConfig:
    """Configuration for the Jamba Threat Detection model."""
    
    version: str = "1.0.0"
    input_dim: int = 9  # Updated to match latest checkpoint
    hidden_dim: int = 128
    output_dim: int = 1
    dropout_rate: float = 0.3
    n_heads: int = 4
    feature_layers: int = 3
    use_mixed_precision: bool = False
    batch_size: int = 64
    learning_rate: float = 0.001
    device: str = "cpu"
    epochs: int = 20
    
    def __post_init__(self):
        if self.version not in VERSION_COMPATIBILITY:
            raise ValueError(f"Unsupported model version: {self.version}")
        
        # Validate dimensions if provided
        compat = VERSION_COMPATIBILITY[self.version]
        
        if self.input_dim is not None:
            if not (compat['min_input_dim'] <= self.input_dim <= compat['max_input_dim']):
                raise ValueError(f"input_dim must be between {compat['min_input_dim']} and {compat['max_input_dim']}")
        
        if not (compat['min_hidden_dim'] <= self.hidden_dim <= compat['max_hidden_dim']):
            raise ValueError(f"hidden_dim must be between {compat['min_hidden_dim']} and {compat['max_hidden_dim']}")
        
        if not (compat['min_output_dim'] <= self.output_dim <= compat['max_output_dim']):
            raise ValueError(f"output_dim must be between {compat['min_output_dim']} and {compat['max_output_dim']}")
        
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in {f.name for f in dataclasses.fields(cls)}
        })
    
    def save(self, path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

# Default configurations for different scenarios
DEFAULT_CPU_CONFIG = ModelConfig(
    input_dim=512,
    hidden_dim=256,
    output_dim=2,
    dropout_rate=0.3,
    n_heads=4,
    feature_layers=2,
    use_mixed_precision=False,
    batch_size=32
)

DEFAULT_GPU_CONFIG = ModelConfig(
    input_dim=512,
    hidden_dim=512,
    output_dim=2,
    dropout_rate=0.3,
    n_heads=8,
    feature_layers=4,
    use_mixed_precision=True,
    batch_size=128
) 