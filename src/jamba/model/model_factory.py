import os
import logging
import torch
from typing import Optional, Type
from ..model_config import ModelConfig, DEFAULT_CPU_CONFIG
from .config_validator import ConfigValidator

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating and loading models."""
    
    def __init__(self, model_class: Type):
        """
        Initialize the factory.
        
        Args:
            model_class: The model class to create instances of
        """
        self.model_class = model_class
        self.config_validator = ConfigValidator()
    
    def create_model(self, config: Optional[ModelConfig] = None) -> 'model_class':
        """Create a new model instance with validated configuration."""
        logger.info("Creating new model instance...")
        
        # Merge with default config
        config = self.config_validator.merge_with_defaults(config, DEFAULT_CPU_CONFIG)
        
        # Validate configuration
        is_valid, error_msg, processed_config = self.config_validator.validate_and_process(config)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Create and return model instance
        model = self.model_class(config)
        logger.info(f"Created model with config: {processed_config}")
        
        return model
    
    def load_model(self, path: str, config: Optional[ModelConfig] = None) -> 'model_class':
        """Load a model from file with version compatibility check."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading model from {path}...")
        
        # Load saved state
        save_dict = torch.load(path)
        
        # Get model version and validate compatibility
        model_version = save_dict.get('version', '0.9.0')  # Default for old models
        if not self.config_validator.validate_version(model_version):
            raise ValueError(f"Incompatible model version: {model_version}")
        
        # Create config and model
        saved_config = ModelConfig(**save_dict['config'])
        if config is not None:
            # Merge saved config with provided config
            config = self.config_validator.merge_with_defaults(config, saved_config)
        else:
            config = saved_config
        
        # Create model and load state
        model = self.create_model(config)
        model.load_state_dict(save_dict['model_state'])
        
        logger.info(f"Loaded model version {model_version}")
        return model
    
    @staticmethod
    def save_model(model: 'model_class', path: str):
        """Save model with its configuration."""
        save_dict = {
            'model_state': model.state_dict(),
            'config': model.config.__dict__,
            'version': model.version
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}") 