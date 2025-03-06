#!/usr/bin/env python3
"""
RunPod Training Script for Jamba Threat Detection Model

This script facilitates running model training on RunPod GPUs by submitting
a training job to a RunPod endpoint. It handles data preparation, submission,
and monitoring of the training job.
"""

import os
import sys
import logging
import argparse
import json
import time
import pandas as pd
import base64
import tempfile
from pathlib import Path
import requests
from typing import Dict, Any, Optional, List, Tuple

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

# Constants
MAX_PAYLOAD_SIZE = 9 * 1024 * 1024  # 9MB to stay safely under 10MiB limit

class RunPodTrainer:
    def __init__(self, api_key: Optional[str] = None, endpoint_id: Optional[str] = None):
        """Initialize the RunPod trainer.
        
        Args:
            api_key: RunPod API key (defaults to RUNPOD_API_KEY environment variable)
            endpoint_id: RunPod endpoint ID (defaults to RUNPOD_ENDPOINT_ID environment variable)
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RunPod API key is required. Set RUNPOD_API_KEY environment variable or pass api_key parameter.")
        
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID is required. Set RUNPOD_ENDPOINT_ID environment variable or pass endpoint_id parameter.")
        
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"RunPod trainer initialized for endpoint {self.endpoint_id}")
    
    def prepare_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data for training.
        
        Args:
            data_path: Path to the training data file (.csv or .parquet)
            
        Returns:
            DataFrame with prepared data
        """
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    
    def get_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the training data.
        
        Args:
            data: DataFrame with training data
            
        Returns:
            Dictionary with data statistics
        """
        stats = {
            "num_samples": len(data),
            "num_features": len(data.columns),
            "feature_names": list(data.columns),
        }
        
        # Add class distribution if target column exists
        target_columns = [col for col in data.columns if col.lower() in ['is_threat', 'target', 'label']]
        if target_columns:
            target_col = target_columns[0]
            stats["class_distribution"] = data[target_col].value_counts().to_dict()
        
        return stats
    
    def estimate_payload_size(self, data_chunk: pd.DataFrame) -> int:
        """Estimate the size of the payload with the given data chunk.
        
        Args:
            data_chunk: DataFrame with a subset of the training data
            
        Returns:
            Estimated size in bytes
        """
        # Convert to JSON to estimate size
        data_records = data_chunk.to_dict(orient='records')
        payload = {
            "input": {
                "operation": "train",
                "data": data_records,
                "params": {}  # Empty params for estimation
            }
        }
        return len(json.dumps(payload).encode('utf-8'))
    
    def split_dataframe(self, df: pd.DataFrame, max_size: int) -> List[pd.DataFrame]:
        """Split a DataFrame into chunks that will fit within the payload size limit.
        
        Args:
            df: DataFrame to split
            max_size: Maximum payload size in bytes
            
        Returns:
            List of DataFrame chunks
        """
        chunks = []
        start_idx = 0
        total_rows = len(df)
        
        # Start with a conservative estimate (100 rows per chunk)
        chunk_size = min(100, total_rows)
        
        while start_idx < total_rows:
            # Try to find the largest chunk that fits
            end_idx = start_idx + chunk_size
            if end_idx > total_rows:
                end_idx = total_rows
            
            current_chunk = df.iloc[start_idx:end_idx]
            current_size = self.estimate_payload_size(current_chunk)
            
            # If too large, reduce chunk size
            while current_size > max_size and len(current_chunk) > 1:
                chunk_size = max(1, chunk_size // 2)
                end_idx = start_idx + chunk_size
                current_chunk = df.iloc[start_idx:end_idx]
                current_size = self.estimate_payload_size(current_chunk)
            
            chunks.append(current_chunk)
            start_idx = end_idx
            
            # Adaptively adjust chunk size based on what worked
            chunk_size = len(current_chunk)
            
            logger.info(f"Created chunk with {len(current_chunk)} rows ({current_size/1024:.2f} KB)")
        
        logger.info(f"Split dataset into {len(chunks)} chunks")
        return chunks
    
    def create_temp_dataset(self, data: pd.DataFrame) -> Tuple[str, str]:
        """Save the dataset to a temporary file.
        
        Args:
            data: DataFrame with training data
            
        Returns:
            Tuple of (file_path, file_format)
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "training_data.csv")
        
        # Save data to CSV
        data.to_csv(temp_file_path, index=False)
        logger.info(f"Saved dataset to temporary file: {temp_file_path}")
        
        return temp_file_path, "csv"
    
    def submit_training_job(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a training job to RunPod.
        
        For large datasets, this method uses a more efficient approach:
        1. For small datasets: Send data directly in the request
        2. For large datasets: Split into chunks or use temporary file reference
        
        Args:
            data: DataFrame with training data
            params: Dictionary of training parameters
            
        Returns:
            Dictionary with job submission result
        """
        # Add data statistics to parameters
        data_stats = self.get_data_stats(data)
        params["data_stats"] = data_stats
        
        # Estimate payload size with all data
        data_records = data.to_dict(orient='records')
        full_payload = {
            "input": {
                "operation": "train",
                "data": data_records,
                "params": params
            }
        }
        payload_size = len(json.dumps(full_payload).encode('utf-8'))
        logger.info(f"Full payload size would be: {payload_size/1024:.2f} KB")
        
        # Check if payload exceeds size limit
        if payload_size < MAX_PAYLOAD_SIZE:
            # Small enough to send directly
            logger.info(f"Submitting training job with {len(data_records)} records in a single request")
            payload = full_payload
            
            # Submit the job
            response = requests.post(f"{self.base_url}/run", headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Job submitted successfully. Job ID: {result.get('id')}")
                return result
            else:
                logger.error(f"Failed to submit job. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                raise RuntimeError(f"Failed to submit job: {response.text}")
        else:
            # Too large to send directly
            logger.info("Dataset too large for a single request")
            
            # Method 1: Use a representative sample for initial training
            sample_size = min(1000, len(data))
            stratify_col = None
            
            # Try to use stratified sampling if we have a target column
            target_columns = [col for col in data.columns if col.lower() in ['is_threat', 'target', 'label']]
            if target_columns:
                stratify_col = target_columns[0]
                logger.info(f"Using stratified sampling on column: {stratify_col}")
                
                # Get class distribution for balanced sampling
                class_counts = data[stratify_col].value_counts()
                minority_class_count = class_counts.min()
                
                # Take equal samples from each class
                samples = []
                for class_val in class_counts.index:
                    class_data = data[data[stratify_col] == class_val]
                    class_sample = class_data.sample(min(minority_class_count, sample_size // len(class_counts)))
                    samples.append(class_sample)
                
                sampled_data = pd.concat(samples).sample(frac=1).reset_index(drop=True)
            else:
                # Random sampling if no target column
                sampled_data = data.sample(sample_size).reset_index(drop=True)
            
            # Add flag to indicate this is a sample
            params["is_sample"] = True
            params["sample_size"] = len(sampled_data)
            params["original_size"] = len(data)
            
            # Convert sample to records
            sample_records = sampled_data.to_dict(orient='records')
            
            # Create payload with sample
            sample_payload = {
                "input": {
                    "operation": "train",
                    "data": sample_records,
                    "params": params
                }
            }
            
            sample_size = len(json.dumps(sample_payload).encode('utf-8'))
            logger.info(f"Sample payload size: {sample_size/1024:.2f} KB")
            
            # Check if sample fits
            if sample_size < MAX_PAYLOAD_SIZE:
                logger.info(f"Submitting training job with sampled data: {len(sampled_data)} records")
                response = requests.post(f"{self.base_url}/run", headers=self.headers, json=sample_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Job submitted successfully. Job ID: {result.get('id')}")
                    return result
                else:
                    logger.error(f"Failed to submit job. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    raise RuntimeError(f"Failed to submit job: {response.text}")
            else:
                logger.error("Even sampled data is too large for request")
                raise ValueError(
                    "Dataset is too large to send via API. Consider using a smaller dataset "
                    "or implementing a file-based approach."
                )
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a training job.
        
        Args:
            job_id: Job ID returned by submit_training_job
            
        Returns:
            Dictionary with job status
        """
        response = requests.get(f"{self.base_url}/status/{job_id}", headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to check job status. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise RuntimeError(f"Failed to check job status: {response.text}")
    
    def wait_for_completion(self, job_id: str, check_interval: int = 10, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for a training job to complete.
        
        Args:
            job_id: Job ID returned by submit_training_job
            check_interval: Interval in seconds between status checks
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary with job result
        """
        start_time = time.time()
        logger.info(f"Waiting for job {job_id} to complete (timeout: {timeout}s)")
        
        while time.time() - start_time < timeout:
            status = self.check_job_status(job_id)
            status_value = status.get("status")
            
            logger.info(f"Job status: {status_value}")
            
            if status_value == "COMPLETED":
                logger.info("Job completed successfully")
                return self.get_job_output(job_id)
            elif status_value in ["FAILED", "CANCELLED"]:
                logger.error(f"Job failed or was cancelled: {status}")
                raise RuntimeError(f"Job failed or was cancelled: {status}")
            
            logger.info(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within timeout period of {timeout} seconds")
    
    def get_job_output(self, job_id: str) -> Dict[str, Any]:
        """Get the output of a completed job.
        
        Args:
            job_id: Job ID returned by submit_training_job
            
        Returns:
            Dictionary with job output
        """
        response = requests.get(f"{self.base_url}/output/{job_id}", headers=self.headers)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Retrieved job output successfully")
            return result
        else:
            logger.error(f"Failed to get job output. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise RuntimeError(f"Failed to get job output: {response.text}")
    
    def save_model(self, result: Dict[str, Any], output_path: str) -> None:
        """Save the trained model from the job result.
        
        Args:
            result: Job result from get_job_output
            output_path: Path to save the model
        """
        if not result.get("success", False):
            logger.error(f"Training was not successful: {result.get('error')}")
            raise RuntimeError(f"Training was not successful: {result.get('error')}")
        
        # Extract model data from result
        model_data = result.get("model")
        if not model_data:
            logger.error("No model data found in job result")
            raise ValueError("No model data found in job result")
        
        # Decode the base64-encoded model
        model_bytes = base64.b64decode(model_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the model
        with open(output_path, "wb") as f:
            f.write(model_bytes)
        
        logger.info(f"Model saved to {output_path}")
        
        # Save metrics
        metrics = result.get("metrics", {})
        metrics_path = f"{os.path.splitext(output_path)[0]}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RunPod Training for Jamba Threat Detection")
    
    parser.add_argument(
        "data_path",
        help="Path to the training data file (.csv or .parquet)"
    )
    parser.add_argument(
        "--output-path",
        default="models/jamba_model.pth",
        help="Path to save the trained model (default: models/jamba_model.pth)"
    )
    parser.add_argument(
        "--api-key",
        help="RunPod API key (defaults to RUNPOD_API_KEY environment variable)"
    )
    parser.add_argument(
        "--endpoint-id",
        help="RunPod endpoint ID (defaults to RUNPOD_ENDPOINT_ID environment variable)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)"
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
        "--timeout",
        type=int,
        default=3600,
        help="Maximum time to wait for job completion in seconds (default: 3600)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to use if dataset is too large (default: 1000)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        # Initialize the trainer
        trainer = RunPodTrainer(api_key=args.api_key, endpoint_id=args.endpoint_id)
        
        # Prepare the data
        data = trainer.prepare_data(args.data_path)
        
        # Prepare training parameters
        params = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size
        }
        
        # Submit the training job
        job_result = trainer.submit_training_job(data, params)
        job_id = job_result.get("id")
        
        # Wait for the job to complete
        result = trainer.wait_for_completion(job_id, timeout=args.timeout)
        
        # Save the trained model
        trainer.save_model(result, args.output_path)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 