#!/usr/bin/env python3
"""
Local Training Script for Jamba Threat Detection Model

This script provides a way to train the model locally on a sample of data
for quick testing and verification. This is useful when you don't want to
wait for RunPod or just want to verify your data is properly formatted.
"""

import os
import sys
import logging
import argparse
import json
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure the src directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import from appropriate modules
try:
    from handler import train_model
    from jamba_model import JambaThreatModel, ThreatDataset
except ImportError:
    logger.error("Failed to import required modules. Make sure jamba_model and handler are available.")
    sys.exit(1)

def prepare_data(data_path, sample_size=1000):
    """Load and prepare data for training.
    
    Args:
        data_path: Path to the training data file (.csv or .parquet)
        sample_size: Number of samples to use (default: 1000)
        
    Returns:
        DataFrame with sampled data
    """
    logger.info(f"Loading data from {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Check if sample size is smaller than dataset
    if sample_size < len(df):
        # Try to use stratified sampling if we have a target column
        target_columns = [col for col in df.columns if col.lower() in ['is_threat', 'target', 'label']]
        if target_columns:
            target_col = target_columns[0]
            logger.info(f"Using stratified sampling on column: {target_col}")
            
            # Get class distribution for balanced sampling
            class_counts = df[target_col].value_counts()
            minority_class_count = class_counts.min()
            
            # Take equal samples from each class
            samples = []
            for class_val in class_counts.index:
                class_data = df[df[target_col] == class_val]
                class_sample = class_data.sample(min(minority_class_count, sample_size // len(class_counts)))
                samples.append(class_sample)
            
            sampled_data = pd.concat(samples).sample(frac=1).reset_index(drop=True)
            logger.info(f"Created balanced sample with {len(sampled_data)} records")
        else:
            # Random sampling if no target column
            sampled_data = df.sample(sample_size).reset_index(drop=True)
            logger.info(f"Created random sample with {len(sampled_data)} records")
        
        return sampled_data
    else:
        logger.info(f"Using entire dataset ({len(df)} records)")
        return df

def train_local_model(data, params, output_path):
    """Train a model locally.
    
    Args:
        data: DataFrame with training data
        params: Dictionary of training parameters
        output_path: Path to save the trained model
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Starting local training with {len(data)} records")
    logger.info(f"Parameters: {params}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Train the model
    start_time = time.time()
    result = train_model(data, params)
    training_time = time.time() - start_time
    
    if result.get("success", False):
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        
        # Save metrics
        metrics = result.get("metrics", {})
        metrics_path = f"{os.path.splitext(output_path)[0]}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_path}")
        
        # Check if the model was saved
        if os.path.exists(output_path):
            logger.info(f"Model saved to {output_path}")
        else:
            logger.warning(f"Model file not found at {output_path}")
    else:
        logger.error(f"Training failed: {result.get('error')}")
    
    return result

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Local Training for Jamba Threat Detection")
    
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the training data file (.csv or .parquet)"
    )
    parser.add_argument(
        "--output-path",
        default="models/jamba_model_local.pth",
        help="Path to save the trained model (default: models/jamba_model_local.pth)"
    )
    parser.add_argument(
        "--config-file",
        help="Path to JSON config file with training parameters"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to use (default: 1000)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
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
        
        # Prepare the data
        data = prepare_data(args.data_path, args.sample_size)
        
        # Train the model
        result = train_local_model(data, params, args.output_path)
        
        # Print metrics
        metrics = result.get("metrics", {})
        logger.info(f"Training complete with accuracy: {metrics.get('accuracy', 'N/A')}")
        logger.info(f"Training time: {metrics.get('training_time', 'N/A')} seconds")
        
        logger.info("Local training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 