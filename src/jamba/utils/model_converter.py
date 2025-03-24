"""
Utility for converting models between different formats.
"""

import os
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import numpy as np
import struct

from ..jamba_model import JambaThreatModel
from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelConverter:
    """Handles conversion between different model formats."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the converter.
        
        Args:
            models_dir: Directory for model files
        """
        self.models_dir = models_dir or os.environ.get("MODEL_DIR", "models")
        
        # Ensure models directory exists
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def convert_to_gguf(
        self,
        model: Union[str, JambaThreatModel],
        model_name: str,
        quantization: str = "q4_k_m"
    ) -> str:
        """
        Convert a PyTorch model to GGUF format.
        
        Args:
            model: Either a model instance or path to PyTorch model
            model_name: Name for the converted model
            quantization: GGUF quantization type
            
        Returns:
            Path to the converted model
        """
        # Load model if path provided
        if isinstance(model, str):
            checkpoint = torch.load(model)
            config = ModelConfig(**checkpoint['config'])
            model = JambaThreatModel(config)
            model.load_state_dict(checkpoint['model_state'])
        
        # Get model state
        state_dict = model.state_dict()
        
        # Create output path
        output_path = os.path.join(self.models_dir, f"{model_name}.gguf")
        
        # Write GGUF file
        with open(output_path, 'wb') as f:
            # Write magic number and version
            f.write(b'GGUF')
            f.write(struct.pack('<I', 1))  # Version 1
            
            # Write metadata
            metadata = {
                'model_type': 'jamba',
                'input_dim': model.config.input_dim,
                'hidden_dim': model.config.hidden_dim,
                'output_dim': model.config.output_dim,
                'n_heads': model.config.n_heads,
                'feature_layers': model.config.feature_layers,
                'quantization': quantization
            }
            
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            f.write(struct.pack('<I', len(metadata_bytes)))
            f.write(metadata_bytes)
            
            # Write tensors
            for name, tensor in state_dict.items():
                # Convert to numpy and quantize if needed
                tensor_np = tensor.detach().cpu().numpy()
                
                if quantization == 'q4_k_m':
                    # 4-bit quantization with k-means
                    tensor_np = self._quantize_4bit(tensor_np)
                
                # Write tensor name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                
                # Write tensor shape
                f.write(struct.pack('<I', len(tensor_np.shape)))
                for dim in tensor_np.shape:
                    f.write(struct.pack('<I', dim))
                
                # Write tensor data
                tensor_bytes = tensor_np.tobytes()
                f.write(struct.pack('<I', len(tensor_bytes)))
                f.write(tensor_bytes)
        
        logger.info(f"Model converted to GGUF: {output_path}")
        return output_path
    
    def _quantize_4bit(self, tensor: np.ndarray) -> np.ndarray:
        """Quantize tensor to 4 bits using k-means."""
        original_shape = tensor.shape
        flattened = tensor.reshape(-1)
        
        # Use 16 centroids for 4-bit quantization
        n_clusters = 16
        
        # Simple k-means implementation
        min_val, max_val = flattened.min(), flattened.max()
        centroids = np.linspace(min_val, max_val, n_clusters)
        
        for _ in range(10):  # 10 iterations of k-means
            # Assign points to nearest centroid
            distances = np.abs(flattened[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    centroids[i] = flattened[mask].mean()
        
        # Quantize using nearest centroid
        distances = np.abs(flattened[:, np.newaxis] - centroids)
        labels = np.argmin(distances, axis=1)
        
        # Pack 4-bit values into bytes
        packed = np.zeros(len(labels) // 2, dtype=np.uint8)
        packed[...] = labels[::2] | (labels[1::2] << 4)
        
        return packed.reshape(-1)
    
    def load_gguf_model(self, model_path: str) -> JambaThreatModel:
        """
        Load a GGUF model.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            Loaded model instance
        """
        with open(model_path, 'rb') as f:
            # Read magic number and version
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF file")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != 1:
                raise ValueError(f"Unsupported GGUF version: {version}")
            
            # Read metadata
            metadata_size = struct.unpack('<I', f.read(4))[0]
            metadata = json.loads(f.read(metadata_size).decode('utf-8'))
            
            # Create model from metadata
            config = ModelConfig(
                input_dim=metadata['input_dim'],
                hidden_dim=metadata['hidden_dim'],
                output_dim=metadata['output_dim'],
                n_heads=metadata['n_heads'],
                feature_layers=metadata['feature_layers']
            )
            model = JambaThreatModel(config)
            
            # Read tensors
            state_dict = {}
            while True:
                try:
                    # Read tensor name
                    name_size = struct.unpack('<I', f.read(4))[0]
                    name = f.read(name_size).decode('utf-8')
                    
                    # Read tensor shape
                    ndim = struct.unpack('<I', f.read(4))[0]
                    shape = []
                    for _ in range(ndim):
                        dim = struct.unpack('<I', f.read(4))[0]
                        shape.append(dim)
                    
                    # Read tensor data
                    data_size = struct.unpack('<I', f.read(4))[0]
                    data = f.read(data_size)
                    
                    # Convert to tensor
                    if metadata['quantization'] == 'q4_k_m':
                        # Dequantize 4-bit values
                        tensor = self._dequantize_4bit(
                            np.frombuffer(data, dtype=np.uint8),
                            tuple(shape)
                        )
                    else:
                        tensor = np.frombuffer(data, dtype=np.float32).reshape(shape)
                    
                    state_dict[name] = torch.from_numpy(tensor)
                    
                except EOFError:
                    break
            
            # Load state dict
            model.load_state_dict(state_dict)
            return model
    
    def _dequantize_4bit(self, packed: np.ndarray, shape: tuple) -> np.ndarray:
        """Dequantize 4-bit values."""
        # Unpack 4-bit values
        unpacked = np.zeros(len(packed) * 2, dtype=np.uint8)
        unpacked[::2] = packed & 0x0F
        unpacked[1::2] = (packed >> 4) & 0x0F
        
        # Convert to float32
        float_vals = unpacked.astype(np.float32)
        
        # Scale back to original range
        min_val, max_val = float_vals.min(), float_vals.max()
        float_vals = (float_vals - min_val) / (max_val - min_val)
        
        return float_vals.reshape(shape) 