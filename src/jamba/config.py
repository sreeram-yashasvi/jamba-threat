from dataclasses import dataclass
from typing import Optional
from .version_compatibility import version_manager

@dataclass
class ModelConfig:
    """Configuration for the JambaThreatModel."""
    version: str = "1.0.0"
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    output_dim: int = 1
    dropout_rate: float = 0.2
    n_heads: Optional[int] = None
    feature_layers: Optional[int] = None
    use_mixed_precision: bool = False
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cpu"
    epochs: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        is_valid, message = version_manager.validate_version(self.version, "1.0.0")
        if not is_valid:
            raise ValueError(message)
        
        # Set default input dimension if not provided
        if self.input_dim is None:
            self.input_dim = 20
        
        # Set derived parameters
        if self.n_heads is None:
            self.n_heads = max(1, self.input_dim // 8)
        
        if self.feature_layers is None:
            self.feature_layers = 2
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "version": self.version,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "n_heads": self.n_heads,
            "feature_layers": self.feature_layers,
            "use_mixed_precision": self.use_mixed_precision,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "epochs": self.epochs
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

# Default configuration for CPU training
DEFAULT_CPU_CONFIG = ModelConfig() 