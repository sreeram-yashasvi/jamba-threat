"""
Utility for uploading models to Hugging Face Hub.
"""

import os
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    ModelCard,
    ModelCardData,
)
from ..jamba_model import JambaThreatModel
from ..model_config import ModelConfig
from .model_converter import ModelConverter

logger = logging.getLogger(__name__)

class HuggingFaceUploader:
    """Handles uploading models to Hugging Face Hub."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        models_dir: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize the uploader.
        
        Args:
            token: Hugging Face API token
            models_dir: Directory containing models
            organization: Hugging Face organization name
        """
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "Hugging Face token not provided. Either pass it directly or "
                "set the HF_TOKEN environment variable."
            )
        
        self.models_dir = models_dir or os.environ.get("MODEL_DIR", "models")
        self.organization = organization
        self.api = HfApi(token=self.token)
    
    def upload_model(
        self,
        model: JambaThreatModel,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        include_gguf: bool = True,
        private: bool = False,
        quantization: str = "q4_k_m"
    ) -> str:
        """
        Upload a model to Hugging Face Hub.
        
        Args:
            model: The model to upload
            model_name: Name for the model on Hugging Face
            metrics: Optional performance metrics to include
            tags: Optional tags for the model
            include_gguf: Whether to include GGUF versions
            private: Whether to create a private repository
            quantization: GGUF quantization type to use
            
        Returns:
            URL of the uploaded model
        """
        # Create the repository
        repo_name = f"{model_name}-threat-detection"
        if self.organization:
            repo_id = f"{self.organization}/{repo_name}"
        else:
            repo_id = repo_name
        
        # Create temporary directory for upload
        with tempfile.TemporaryDirectory() as upload_dir:
            try:
                # Create repository
                create_repo(
                    repo_id,
                    token=self.token,
                    private=private,
                    repo_type="model",
                    exist_ok=True
                )
                
                # Save PyTorch model
                pytorch_path = os.path.join(upload_dir, "pytorch_model.pt")
                torch.save({
                    'model_state': model.state_dict(),
                    'config': model.config.to_dict()
                }, pytorch_path)
                
                # Save config
                config_path = os.path.join(upload_dir, "config.json")
                with open(config_path, 'w') as f:
                    json.dump(model.config.to_dict(), f, indent=2)
                
                # Convert to GGUF if requested
                if include_gguf:
                    converter = ModelConverter(models_dir=upload_dir)
                    gguf_path = converter.convert_to_gguf(
                        model=model,
                        model_name=model_name,
                        quantization=quantization
                    )
                    # Copy GGUF model to upload directory
                    shutil.copy2(gguf_path, os.path.join(upload_dir, f"{model_name}.gguf"))
                
                # Create and save model card
                model_card = self._create_model_card(
                    model_name=model_name,
                    config=model.config.to_dict(),
                    metrics=metrics,
                    tags=tags
                )
                model_card.save(os.path.join(upload_dir, "README.md"))
                
                # Upload to Hugging Face
                logger.info(f"Uploading model to {repo_id}")
                upload_folder(
                    folder_path=upload_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=self.token
                )
                
                model_url = f"https://huggingface.co/{repo_id}"
                logger.info(f"Model uploaded successfully: {model_url}")
                
                return model_url
                
            except Exception as e:
                logger.error(f"Failed to upload model: {e}")
                raise
    
    def _create_model_card(
        self,
        model_name: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelCard:
        """Create a model card for Hugging Face."""
        if tags is None:
            tags = ["threat-detection", "security", "pytorch"]
        
        if metrics is None:
            metrics = {}
        
        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="pytorch",
            tags=tags
        )
        
        # Create model card content
        content = f"""
# Jamba Threat Detection Model

This is a deep learning model for detecting security threats using system metrics and network behavior patterns.

## Model Description

- **Model Type:** Neural network with self-attention
- **Input Features:** {config.get('input_dim', 20)} system metrics
- **Output:** Binary threat classification
- **Architecture:**
  - Hidden dimension: {config.get('hidden_dim', 64)}
  - Attention heads: {config.get('n_heads', 4)}
  - Feature layers: {config.get('feature_layers', 2)}
  - Dropout rate: {config.get('dropout_rate', 0.3)}

## Performance Metrics

"""
        if metrics:
            content += "| Metric | Value |\n|--------|-------|\n"
            for metric, value in metrics.items():
                content += f"| {metric} | {value:.4f} |\n"
        
        content += """
## Usage

```python
from jamba.utils.model_converter import ModelConverter

# Load the model (PyTorch version)
checkpoint = torch.load("pytorch_model.pt")
model = JambaThreatModel(ModelConfig(**checkpoint['config']))
model.load_state_dict(checkpoint['model_state'])

# Or load the GGUF version for efficient inference
converter = ModelConverter()
model = converter.load_gguf_model("model.gguf")

# Make predictions
predictions = model(input_data)
```

## Training

This model was trained on a balanced dataset of normal and threat samples, using:
- Batch size: {config.get('batch_size', 32)}
- Learning rate: {config.get('learning_rate', 0.001)}
- Epochs: {config.get('epochs', 30)}

## Formats

The model is available in two formats:
1. PyTorch format (`pytorch_model.pt`)
2. GGUF format (`model.gguf`) - Optimized for efficient deployment
"""
        
        return ModelCard(content=content, data=card_data) 