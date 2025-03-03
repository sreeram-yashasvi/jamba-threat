import os
import logging
import argparse
from model_training import ThreatModelTrainer
from generate_threat_data import generate_threat_intel_feed, preprocess_threat_intel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate threat intelligence data and train Jamba model')
    parser.add_argument('--entries', type=int, default=1000, help='Number of threat intel entries to generate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--model-dir', default='models', help='Directory to save models')
    parser.add_argument('--data-dir', default='data', help='Directory to save data')
    parser.add_argument('--use-existing', action='store_true', help='Use existing data instead of generating new data')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Define file paths
    raw_data_path = os.path.join(args.data_dir, 'threat_intel_feed.csv')
    processed_data_path = os.path.join(args.data_dir, 'processed_threat_data.csv')
    
    # Step 1: Generate or use existing threat intel data
    if not args.use_existing or not os.path.exists(raw_data_path):
        logger.info(f"Generating {args.entries} threat intelligence entries")
        generate_threat_intel_feed(raw_data_path, args.entries)
    else:
        logger.info(f"Using existing threat intelligence data from {raw_data_path}")
    
    # Step 2: Preprocess the data
    if not args.use_existing or not os.path.exists(processed_data_path):
        logger.info("Preprocessing threat intelligence data")
        preprocess_threat_intel(raw_data_path, processed_data_path)
    else:
        logger.info(f"Using existing preprocessed data from {processed_data_path}")
    
    # Step 3: Initialize model trainer
    logger.info("Initializing Jamba Threat Model trainer")
    trainer = ThreatModelTrainer(model_save_dir=args.model_dir)
    
    # Step 4: Train the model
    logger.info(f"Training model with {args.epochs} epochs, learning rate {args.lr}, batch size {args.batch_size}")
    model, history = trainer.train_model(
        data_path=processed_data_path,
        target_column='is_threat',
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    logger.info("Training completed successfully")
    
    # Step 5: Print final metrics
    final_accuracy = history['val_accuracy'][-1]
    logger.info(f"Final model accuracy: {final_accuracy:.4f}")
    
    return final_accuracy

if __name__ == "__main__":
    main() 