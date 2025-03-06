#!/usr/bin/env python3
import os
import argparse
import logging
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run the Jamba Threat Intelligence Pipeline')
    parser.add_argument('--entries', type=int, default=15000, 
                        help='Number of threat intel entries to generate')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--regenerate', action='store_true', 
                        help='Regenerate data even if it exists')
    parser.add_argument('--no-train', action='store_true', 
                        help='Skip training phase')
    parser.add_argument('--no-predict', action='store_true', 
                        help='Skip prediction phase')
    parser.add_argument('--runpod', action='store_true',
                        help='Use RunPod for training (GPU accelerated)')
    parser.add_argument('--api-key', 
                        help='RunPod API key (required with --runpod, or set RUNPOD_API_KEY env var)')
    parser.add_argument('--endpoint-id', 
                        help='RunPod endpoint ID (required with --runpod, or set RUNPOD_ENDPOINT_ID env var)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (larger for GPU training)')
    
    args = parser.parse_args()
    
    # Validate RunPod arguments if using RunPod
    if args.runpod:
        api_key = args.api_key or os.environ.get('RUNPOD_API_KEY')
        endpoint_id = args.endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID')
        
        if not api_key:
            logger.error("RunPod API key must be provided via --api-key or RUNPOD_API_KEY environment variable")
            return 1
        
        if not endpoint_id:
            logger.error("RunPod endpoint ID must be provided via --endpoint-id or RUNPOD_ENDPOINT_ID environment variable")
            return 1
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Generate and preprocess data
    logger.info("=== STEP 1: GENERATING AND PREPROCESSING DATA ===")
    use_existing = not args.regenerate
    data_cmd = (f"python src/generate_threat_data.py "
                f"--entries {args.entries} "
                f"{'' if args.regenerate else '--use-existing'}")
    
    logger.info(f"Running: {data_cmd}")
    start_time = time.time()
    os.system(data_cmd)
    logger.info(f"Data generation completed in {time.time() - start_time:.2f} seconds")
    
    if not args.no_train:
        # Step 2: Train the model
        logger.info("\n=== STEP 2: TRAINING JAMBA THREAT MODEL ===")
        
        if args.runpod:
            # Train on RunPod with GPU
            logger.info("Using RunPod with GPU acceleration for training")
            api_key = args.api_key or os.environ.get('RUNPOD_API_KEY')
            endpoint_id = args.endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID')
            
            train_cmd = (f"python src/train_with_runpod.py "
                        f"--data data/processed_threat_data.csv "
                        f"--target is_threat "
                        f"--epochs {args.epochs} "
                        f"--batch-size {args.batch_size} "
                        f"--lr 0.001 "
                        f"--api-key {api_key} "
                        f"--endpoint-id {endpoint_id}")
        else:
            # Train locally
            train_cmd = (f"python src/train_with_intel.py "
                        f"--epochs {args.epochs} "
                        f"--entries {args.entries} "
                        f"{'' if args.regenerate else '--use-existing'}")
        
        logger.info(f"Running: {train_cmd}")
        start_time = time.time()
        os.system(train_cmd)
        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    if not args.no_predict and not args.no_train:
        # Step 3: Make predictions
        logger.info("\n=== STEP 3: MAKING PREDICTIONS ===")
        
        # Find the most recent model
        model_dir = "models"
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if model_files:
            # Sort by timestamp in filename
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(model_dir, latest_model)
            
            predict_cmd = (f"python src/predict.py "
                          f"--model {model_path} "
                          f"--data data/threat_intel_feed.csv "
                          f"--output data/predictions_{timestamp}.csv")
            
            logger.info(f"Running: {predict_cmd}")
            start_time = time.time()
            os.system(predict_cmd)
            logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        else:
            logger.warning("No trained models found for prediction")
    
    logger.info("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main() 