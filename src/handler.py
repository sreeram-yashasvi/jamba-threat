import io
import os
import json
import time
import torch
import base64
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.cuda.amp as amp
import logging
import runpod
import sys
import traceback
from pathlib import Path
import importlib.util

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger("jamba-threat-handler")

# Check and set environment variables with defaults
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
LOGS_DIR = os.environ.get("LOGS_DIR", "./logs")
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the features used by the model
FEATURES = [
    'device_age', 'connection_count', 'packet_size_variance', 
    'bandwidth_usage', 'protocol_anomaly_score', 'encryption_level',
    'auth_failures', 'unusual_ports', 'traffic_pattern_change',
    'geographic_anomaly', 'signature_matches', 'privilege_escalation',
    'packet_manipulation', 'data_exfiltration_attempt', 'api_abuse',
    'dns_tunneling', 'file_integrity', 'process_injection'
]

# Print environment info
logger.info(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Files in current directory: {os.listdir('.')}")
if os.path.exists('/app/jamba_model'):
    logger.info(f"Files in jamba_model directory: {os.listdir('/app/jamba_model')}")
    if os.path.exists('/app/jamba_model/__init__.py'):
        with open('/app/jamba_model/__init__.py', 'r') as f:
            logger.info(f"Contents of __init__.py: {f.read()}")

# Global model cache to avoid reloading
model_cache = {}

# Import model classes with better error handling
try:
    logger.info("Importing JambaThreatModel from jamba_model")
    from jamba_model import JambaThreatModel, ThreatDataset
    logger.info("Successfully imported JambaThreatModel and ThreatDataset")
except ImportError as e:
    logger.warning(f"Failed to import from jamba_model directly: {e}")
    logger.info("Attempting to import from alternative paths...")
    
    # Try adjusting sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_paths = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.join(os.path.dirname(current_dir), "src")
    ]
    
    for path in potential_paths:
        if path not in sys.path:
            logger.info(f"Adding {path} to sys.path")
            sys.path.append(path)
    
    try:
        logger.info("Trying import after path adjustment")
        from jamba_model import JambaThreatModel, ThreatDataset
        logger.info("Successfully imported after path adjustment")
    except ImportError as e:
        logger.error(f"Failed to import from jamba_model after path adjustment: {e}")
        logger.info("Current sys.path: " + str(sys.path))
        logger.info("Current directory: " + os.getcwd())
        logger.info("Files in current directory: " + str(os.listdir('.')))
        if os.path.exists('./src'):
            logger.info("Files in ./src: " + str(os.listdir('./src')))
        raise

# Train the model function
def train_model(data, params):
    """Train the Jamba threat detection model.
    
    Args:
        data: DataFrame with training data
        params: Dictionary with training parameters
        
    Returns:
        Dictionary with training results
    """
    try:
        logger.info("Starting model training")
        
        # Extract parameters
        target_column = params.get('target_column', 'is_threat')
        epochs = params.get('epochs', 30)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 128)
        
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Prepare data
        logger.info(f"Preparing data with shape: {data.shape}")
        X = data.drop([target_column], axis=1)
        y = data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = ThreatDataset(X_train, y_train)
        test_dataset = ThreatDataset(X_test, y_test)
        
        try:
            # Reduced worker count for stability in container
            num_workers = 2 if torch.cuda.is_available() else 0
            logger.info(f"Using {num_workers} dataloader workers")
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers
            )
            
            # Initialize model
            input_dim = X_train.shape[1]
            logger.info(f"Initializing model with input dimension: {input_dim}")
            model = JambaThreatModel(input_dim).to(device)
            
            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Use mixed precision if available
            scaler = GradScaler() if torch.cuda.is_available() else None
            
            # Training loop
            logger.info(f"Starting training for {epochs} epochs...")
            start_time = time.time()
            
            training_history = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for batch_idx, (features, targets) in enumerate(train_loader):
                    features, targets = features.to(device), targets.to(device)
                    
                    # Mixed precision training
                    if scaler is not None:
                        with autocast():
                            outputs = model(features)
                            loss = criterion(outputs.squeeze(), targets)
                        
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(features)
                        loss = criterion(outputs.squeeze(), targets)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
                avg_train_loss = train_loss / len(train_loader)
                training_history['train_loss'].append(avg_train_loss)
                
                # Validation
                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0
                
                with torch.no_grad():
                    for features, targets in test_loader:
                        features, targets = features.to(device), targets.to(device)
                        outputs = model(features)
                        loss = criterion(outputs.squeeze(), targets)
                        val_loss += loss.item()
                        
                        predicted = (outputs.squeeze() > 0.5).float()
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                accuracy = correct / total
                avg_val_loss = val_loss / len(test_loader)
                
                training_history['val_loss'].append(avg_val_loss)
                training_history['val_accuracy'].append(accuracy)
                
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final accuracy: {accuracy:.4f}")
            
            # Serialize model
            model_buffer = io.BytesIO()
            torch.save(model.state_dict(), model_buffer)  # Save state_dict instead of full model
            model_buffer.seek(0)
            model_data = base64.b64encode(model_buffer.getvalue()).decode('utf-8')
            
            # Return results
            return {
                'model': model_data,
                'metrics': {
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'training_history': training_history
                }
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc()}
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

def predict(model_data, data):
    """Make predictions using a trained model.
    
    Args:
        model_data: Serialized model
        data: DataFrame with features
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Deserialize model
        model_buffer = io.BytesIO(base64.b64decode(model_data))
        
        # Get input dimensions from data
        input_dim = data.shape[1]
        logger.info(f"Creating model with input dimension: {input_dim}")
        
        # Initialize model architecture
        model = JambaThreatModel(input_dim).to(device)
        
        # Load model state
        try:
            state_dict = torch.load(model_buffer, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': f"Model loading failed: {str(e)}"}
        
        # Prepare data
        dataset = ThreatDataset(data, np.zeros(len(data)))  # Dummy targets
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Run predictions
        model.eval()
        all_predictions = []
        all_scores = []
        
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(device)
                outputs = model(features)
                scores = torch.sigmoid(outputs.squeeze())
                predictions = (scores > 0.5).int()
                
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_scores.extend(scores.cpu().numpy().tolist())
        
        # Return predictions
        return {
            'predictions': all_predictions,
            'scores': all_scores,
            'metadata': {
                'model_input_dim': input_dim,
                'device': str(device),
                'num_samples': len(data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

def health_check():
    """
    Perform a health check on the model.
    
    Returns:
        Dictionary with health check results
    """
    try:
        # Check if PyTorch is available
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check if model can be initialized
        input_dim = len(FEATURES)
        model = JambaThreatModel(input_dim)
        
        # Create a sample input
        sample_input = torch.randn(4, input_dim)
        
        # Test forward pass
        output = model(sample_input)
        
        return {
            "success": True,
            "status": "healthy",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "model_initialized": True,
            "forward_pass_successful": True,
            "output_shape": list(output.shape)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }

def _handler(event):
    """Handler function for RunPod.
    
    Args:
        event: RunPod event object
        
    Returns:
        Response object
    """
    try:
        logger.info(f"Received request: {json.dumps(event)[:1000]}...")
        
        # Extract input data
        input_data = event.get("input", {})
        operation = input_data.get("operation")
        data_dict = input_data.get("data", {})
        
        # Deserialize dataset if present
        if "dataset" in data_dict:
            serialized_df = data_dict["dataset"]
            if isinstance(serialized_df, dict):
                df_type = serialized_df.get("type")
                df_format = serialized_df.get("format")
                
                if df_type == "dataframe":
                    if df_format == "parquet":
                        # Deserialize parquet
                        parquet_bytes = base64.b64decode(serialized_df["data"])
                        data = pd.read_parquet(io.BytesIO(parquet_bytes))
                    else:
                        # Fallback to pickle
                        pickle_bytes = base64.b64decode(serialized_df["data"])
                        data = pickle.loads(pickle_bytes)
                else:
                    data = pd.DataFrame()
            else:
                data = pd.DataFrame()
        else:
            data = pd.DataFrame()
        
        # Process operations
        if operation == "train":
            logger.info(f"Training model with {len(data)} samples")
            params = data_dict.get("params", {})
            result = train_model(data, params)
            logger.info("Training completed")
            return result
            
        elif operation == "predict":
            logger.info(f"Making predictions for {len(data)} samples")
            model_data = data_dict.get("model")
            result = predict(model_data, data)
            logger.info("Prediction completed")
            return result
        
        elif operation == "health":
            # Health check operation
            logger.info("Processing health check operation")
            return health_check()
        
        else:
            # Unknown operation
            logger.error(f"Unknown operation: {operation}")
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}

def handler(event):
    """Wrapper function for the handler to ensure consistent error handling.
    
    Args:
        event: RunPod event object
        
    Returns:
        Response object
    """
    try:
        logger.info("Starting Jamba Threat Handler...")
        
        if not event:
            logger.error("Received empty event")
            return {"error": "Empty event received"}
        
        if not isinstance(event, dict):
            logger.error(f"Received non-dict event: {type(event)}")
            return {"error": f"Expected dict event, got {type(event)}"}
        
        # For debugging
        logger.info(f"Event keys: {list(event.keys())}")
        
        result = _handler(event)
        return result
        
    except Exception as e:
        logger.error(f"Critical error in handler wrapper: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Critical error: {str(e)}", "traceback": traceback.format_exc()}

# Only register and start the server if run directly
if __name__ == "__main__":
    logger.info("Starting Jamba Threat Handler")
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"Failed to start serverless function: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
else:
    # When imported as a module by runpod.serverless.start
    # we need to explicitly register our handler
    logger.info("Registering handler with RunPod serverless system")
    # This is what runpod.serverless.start will look for
    run_model = {"handler": handler}
