#!/usr/bin/env python3
"""
Script to upload models to Hugging Face Hub.
"""

import argparse
import json
import logging
import os
import sys
import torch

from ..jamba_model import JambaThreatModel
from ..model_config import ModelConfig
from ..utils.hf_uploader import HuggingFaceUploader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to input PyTorch model (.pt file)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name for the model on Hugging Face"
    )
    
    parser.add_argument(
        "--organization",
        type=str,
        help="Hugging Face organization name"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--metrics-file",
        type=str,
        help="JSON file containing model metrics"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Additional tags for the model"
    )
    
    parser.add_argument(
        "--include-gguf",
        action="store_true",
        default=True,
        help="Include GGUF version of the model"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["q4_k_m", "q5_k_m", "q8_0"],
        default="q4_k_m",
        help="GGUF quantization type"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load metrics if provided
    metrics = None
    if args.metrics_file and os.path.exists(args.metrics_file):
        try:
            with open(args.metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metrics file: {e}")
    
    # Load model
    try:
        logger.info(f"Loading model from {args.input_model}")
        checkpoint = torch.load(args.input_model)
        
        config = ModelConfig()
        if 'config' in checkpoint:
            config = ModelConfig(**checkpoint['config'])
        elif 'input_dim' in checkpoint:
            config = ModelConfig(input_dim=checkpoint['input_dim'])
        
        model = JambaThreatModel(config)
        
        # Handle different state dict keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            raise ValueError("No model state found in checkpoint")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Initialize uploader
    try:
        uploader = HuggingFaceUploader(
            token=args.token,
            organization=args.organization
        )
        
        # Upload model
        model_url = uploader.upload_model(
            model=model,
            model_name=args.model_name,
            metrics=metrics,
            tags=args.tags,
            include_gguf=args.include_gguf,
            private=args.private,
            quantization=args.quantization
        )
        
        logger.info(f"\nModel uploaded successfully!")
        logger.info(f"View your model at: {model_url}")
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 