#!/usr/bin/env python3
"""
Script to convert a trained PyTorch model to GGUF format.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from ..jamba_model import JambaThreatModel
from ..model_config import ModelConfig
from ..utils.model_converter import ModelConverter, QuantizationType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF format")
    
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to input PyTorch model (.pt file)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save GGUF model (default: /app/models/gguf)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="jamba_threat_model",
        help="Name for the converted model"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["q4_k_m", "q5_k_m", "q8_0"],
        default="q4_k_m",
        help="Quantization type to use"
    )
    
    parser.add_argument(
        "--show-info",
        action="store_true",
        help="Show information about available quantization types"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize converter
    converter = ModelConverter(gguf_dir=args.output_dir)
    
    # Show quantization info if requested
    if args.show_info:
        logger.info("Available quantization types:")
        for quant_type in converter.get_available_quantizations():
            info = converter.get_quantization_info(quant_type)
            logger.info(f"\n{quant_type}:")
            for key, value in info.items():
                logger.info(f"  {key}: {value}")
        return
    
    # Load PyTorch model
    try:
        logger.info(f"Loading PyTorch model from {args.input_model}")
        checkpoint = torch.load(args.input_model)
        
        config = ModelConfig()
        if 'config' in checkpoint:
            config = ModelConfig(**checkpoint['config'])
        
        model = JambaThreatModel(config)
        model.load_state_dict(checkpoint['model_state'])
        
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        sys.exit(1)
    
    # Convert to GGUF
    try:
        logger.info(f"Converting model to GGUF format with {args.quantization} quantization")
        output_path = converter.convert_to_gguf(
            model=model,
            model_name=args.model_name,
            quantization=args.quantization
        )
        logger.info(f"Successfully converted model to: {output_path}")
        
        # Print size comparison
        pt_size = os.path.getsize(args.input_model) / (1024 * 1024)  # MB
        gguf_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression = (1 - gguf_size/pt_size) * 100
        
        logger.info("\nSize comparison:")
        logger.info(f"PyTorch model: {pt_size:.2f} MB")
        logger.info(f"GGUF model: {gguf_size:.2f} MB")
        logger.info(f"Compression ratio: {compression:.1f}%")
        
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 