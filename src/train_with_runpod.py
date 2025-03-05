import os
import argparse
import logging
import base64
from runpod_client import RunPodClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Jamba Threat Detection Model using RunPod')
    parser.add_argument('--data', required=True, help='Path to dataset file (.csv or .parquet)')
    parser.add_argument('--target', default='is_threat', help='Target column name')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--api-key', required=True, help='RunPod API key')
    parser.add_argument('--endpoint-id', required=True, help='RunPod endpoint ID')
    
    args = parser.parse_args()
    
    # Initialize RunPod client
    client = RunPodClient(args.api_key, args.endpoint_id)
    
    # Train the model
    logger.info(f"Sending training job to RunPod with {args.epochs} epochs, batch size {args.batch_size}")
    result = client.train_model(
        data_path=args.data,
        target_column=args.target,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    # Save the model locally
    if 'model' in result:
        os.makedirs('models', exist_ok=True)
        model_path = f"models/jamba_runpod_model.pth"
        # Decode base64 model data
        model_data = base64.b64decode(result['model'])
        with open(model_path, 'wb') as f:
            f.write(model_data)
        logger.info(f"Model saved to {model_path}")
    
    # Show training metrics
    if 'metrics' in result:
        logger.info(f"Final accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"Training time: {result['metrics']['training_time']:.2f} seconds")
        logger.info(f"Final training loss: {result['metrics'].get('final_train_loss', 'N/A')}")
        logger.info(f"Final validation loss: {result['metrics'].get('final_val_loss', 'N/A')}")
    
    return result

if __name__ == "__main__":
    main()
