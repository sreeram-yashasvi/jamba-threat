from typing import Optional
from dataclasses import asdict
from ..model_config import ModelConfig, VERSION_COMPATIBILITY

class ConfigValidator:
    """Validates and processes model configurations."""
    
    @staticmethod
    def validate_version(version: str) -> bool:
        """Validate model version compatibility."""
        return version in VERSION_COMPATIBILITY
    
    @staticmethod
    def calculate_dimensions(config: ModelConfig) -> dict:
        """Calculate and validate model dimensions."""
        dimensions = {}
        
        # Input dimension
        dimensions['input_dim'] = config.input_dim if config.input_dim is not None else 20
        
        # Hidden dimension
        dimensions['hidden_dim'] = config.hidden_dim
        
        # Number of attention heads
        dimensions['n_heads'] = (
            config.n_heads if config.n_heads is not None 
            else max(1, dimensions['input_dim'] // 8)
        )
        
        # Feature layers
        dimensions['feature_layers'] = (
            config.feature_layers if config.feature_layers is not None 
            else 2
        )
        
        return dimensions
    
    @staticmethod
    def merge_with_defaults(config: Optional[ModelConfig], default_config: ModelConfig) -> ModelConfig:
        """Merge provided config with default config."""
        if config is None:
            return default_config
        
        # Create dictionaries excluding version
        config_dict = asdict(config)
        default_dict = asdict(default_config)
        
        # Remove version from both dictionaries
        config_dict.pop('version', None)
        default_dict.pop('version', None)
        
        # Create new config with merged parameters
        merged_dict = {**default_dict, **config_dict}
        merged_dict['version'] = config.version  # Restore version from input config
        
        return ModelConfig(**merged_dict)
    
    @classmethod
    def validate_and_process(cls, config: ModelConfig) -> tuple[bool, str, dict]:
        """
        Validate and process the configuration.
        
        Returns:
            tuple: (is_valid, error_message, processed_config)
        """
        # Version validation
        if not cls.validate_version(config.version):
            return False, f"Unsupported model version: {config.version}", {}
        
        try:
            # Calculate dimensions
            dimensions = cls.calculate_dimensions(config)
            
            # Add other config parameters
            processed_config = {
                'version': config.version,
                'output_dim': config.output_dim,
                'dropout_rate': config.dropout_rate,
                'device': config.device,
                **dimensions
            }
            
            return True, "", processed_config
            
        except Exception as e:
            return False, f"Configuration processing error: {str(e)}", {} 