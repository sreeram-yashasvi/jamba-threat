#!/usr/bin/env python3
import dataclasses
from typing import Optional, Dict, Any
import json
import os

@dataclasses.dataclass
class ModelConfig:
    """Configuration for Jamba Threat Model"""
    version: str = "1.0.0"
    input_dim: int = 512
    hidden_dim: Optional[int] = None
    output_dim: int = 2
    dropout_rate: float = 0.3
    n_heads: Optional[int] = None
    feature_layers: Optional[int] = None
    use_mixed_precision: bool = True
    batch_size: int = 128
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return dataclasses.asdict(self)
    
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
    version="1.0.0",
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
    version="1.0.0",
    input_dim=512,
    hidden_dim=512,
    output_dim=2,
    dropout_rate=0.3,
    n_heads=8,
    feature_layers=4,
    use_mixed_precision=True,
    batch_size=128
)

# Version compatibility mapping
VERSION_COMPATIBILITY = {
    "1.0.0": ["1.0.0"],  # Compatible with itself
    "0.9.0": ["0.9.0", "1.0.0"],  # Old version compatible with new
} 