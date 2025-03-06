#!/usr/bin/env python3
"""
Example script for running Jamba Threat model training on RunPod GPUs.

This script demonstrates how to use the RunPodTrainer to train a model
using a RunPod GPU environment.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Make sure the src directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train_runpod import RunPodTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Jamba Threat model training on RunPod")
    
    parser.add_argument(
        "--data-path",
        default="data/jamba_training_data.csv",
        help="Path to training data file (.csv or .parquet)"
    )
    parser.add_argument(
        "--output-model",
        default="models/jamba_model.pth",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--config-file",
        help="Path to JSON config file with training parameters"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--api-key",
        help="RunPod API key (defaults to RUNPOD_API_KEY environment variable)"
    )
    parser.add_argument(
        "--endpoint-id",
        help="RunPod endpoint ID (defaults to RUNPOD_ENDPOINT_ID environment variable)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Check for config file
        params = {}
        if args.config_file:
            logger.info(f"Loading training parameters from {args.config_file}")
            with open(args.config_file, 'r') as f:
                params = json.load(f)
        else:
            # Use command-line parameters
            params = {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size
            }
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
        
        # Initialize trainer
        trainer = RunPodTrainer(api_key=args.api_key, endpoint_id=args.endpoint_id)
        
        # Run training process
        logger.info("Starting training process")
        
        # Load data
        data = trainer.prepare_data(args.data_path)
        
        # Submit training job
        job_result = trainer.submit_training_job(data, params)
        job_id = job_result.get("id")
        logger.info(f"Training job submitted with ID: {job_id}")
        
        # Wait for job completion
        result = trainer.wait_for_completion(job_id)
        
        # Save model
        trainer.save_model(result, args.output_model)
        
        # Print metrics
        metrics = result.get("metrics", {})
        logger.info(f"Training complete with accuracy: {metrics.get('accuracy', 'N/A')}")
        logger.info(f"Training time: {metrics.get('training_time', 'N/A')} seconds")
        
        logger.info(f"Model saved to {args.output_model}")
        logger.info("Training completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 