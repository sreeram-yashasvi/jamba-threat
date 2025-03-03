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
    parser.add_argument('--entries', type=int, default=1000, 
                        help='Number of threat intel entries to generate')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--regenerate', action='store_true', 
                        help='Regenerate data even if it exists')
    parser.add_argument('--no-train', action='store_true', 
                        help='Skip training phase')
    parser.add_argument('--no-predict', action='store_true', 
                        help='Skip prediction phase')
    
    args = parser.parse_args()
    
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