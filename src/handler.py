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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jamba-threat-handler")

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

# Import model classes
try:
    # Direct import from the properly installed module
    from jamba_model import JambaThreatModel, ThreatDataset
    logger.info("Successfully imported model classes from jamba_model module")
except ImportError as e:
    logger.error(f"Failed to import model classes: {e}")
    logger.error("This is a critical error. Please check your installation and module structure.")
    raise

# Global cache for loaded models to avoid reloading the same model
_model_cache = {}

def train_model(data, params):
    """Train the Jamba threat detection model.
    
    Args:
        data: DataFrame with training data
        params: Dictionary with training parameters
        
    Returns:
        Dictionary with training results
    """
    # Extract parameters
    target_column = params.get('target_column', 'is_threat')
    epochs = params.get('epochs', 30)
    learning_rate = params.get('learning_rate', 0.001)
    batch_size = params.get('batch_size', 128)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = ThreatDataset(X_train, y_train)
    test_dataset = ThreatDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
    
    # Initialize model
    input_dim = X.shape[1]
    model = JambaThreatModel(input_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # OPTIMIZATION: Use automatic mixed precision
    # In PyTorch 2.0.1, we need to be careful with dtype settings
    amp_enabled = torch.cuda.is_available()
    scaler = amp.GradScaler(enabled=amp_enabled)
    
    # OPTIMIZATION: JIT compile model if not training on GPU
    if not amp_enabled:
        model = torch.jit.script(model)  # JIT for CPU

    # Log training configuration
    logger.info(f"Training on device: {device}, Mixed precision: {amp_enabled}")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    # OPTIMIZATION: Set cuda high water mark once
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # This line causes errors with PyTorch 2.0.1 in the Docker container
        # torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        
        # Use these more compatible memory settings instead
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory
        logger.info(f"Set CUDA memory fraction to 0.9")
        
        # Report available GPU memory
        if hasattr(torch.cuda, 'mem_get_info'):
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            logger.info(f"GPU memory: {free_mem/1e9:.2f} GB free, {total_mem/1e9:.2f} GB total")
    
    # Training loop
    start_time = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            if amp_enabled:
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training without mixed precision
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                if amp_enabled:
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.unsqueeze(1))
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted.squeeze() == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        accuracy = correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(accuracy)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
    
    training_time = time.time() - start_time
    
    # Save the model to a byte buffer
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)  # Save state_dict instead of full model
    model_buffer.seek(0)
    
    # Return the serialized model and metrics
    return {
        "model": base64.b64encode(model_buffer.getvalue()).decode('utf-8'),
        "metrics": {
            "accuracy": float(accuracy),
            "training_time": training_time
        }
    }

def predict(model_data, data):
    """
    Make predictions with a trained model.
    
    Args:
        model_data (str): Base64 encoded model data
        data (dict): Dictionary containing the input data for prediction
    
    Returns:
        dict: Prediction results
    """
    try:
        logger.info("Starting prediction process")
        
        # Deserialize the model
        try:
            model_binary = base64.b64decode(model_data)
            model_buffer = io.BytesIO(model_binary)
            
            # Load the model state dict
            state_dict = torch.load(model_buffer, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Create a new model instance and load the state dict
            model = JambaThreatModel(len(FEATURES))
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"error": error_msg}
        
        # Deserialize the dataset
        try:
            if isinstance(data, dict) and "type" in data and data["type"] == "dataframe":
                if data.get("format") == "parquet":
                    binary_data = base64.b64decode(data["data"])
                    buffer = io.BytesIO(binary_data)
                    df = pd.read_parquet(buffer)
                else:
                    binary_data = base64.b64decode(data["data"])
                    df = pickle.loads(binary_data)
                
                logger.info(f"Prediction data loaded, shape: {df.shape}")
            else:
                return {"error": "Invalid data format for prediction"}
        except Exception as e:
            error_msg = f"Error loading prediction data: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"error": error_msg}
        
        # Preprocess the data for prediction
        try:
            # Ensure all features are present
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                return {"error": f"Missing features in prediction data: {', '.join(missing_features)}"}
            
            # Convert data to tensor
            X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
            dataset = ThreatDataset(X, None)  # No labels for prediction
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
            
            logger.info("Data prepared for prediction")
        except Exception as e:
            error_msg = f"Error preprocessing prediction data: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"error": error_msg}
        
        # Make predictions
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            all_predictions = []
            
            with torch.no_grad():
                for batch_x in dataloader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    
                    # Apply sigmoid to convert logits to probabilities
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_predictions.extend(probs.flatten().tolist())
            
            logger.info(f"Predictions generated for {len(all_predictions)} samples")
            
            # Return predictions
            return {
                "predictions": all_predictions,
                "threshold": 0.5,  # Default threshold for binary classification
                "num_samples": len(all_predictions)
            }
            
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Unexpected error in prediction: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return {"error": error_msg}

# Add RunPod serverless handler wrapper
def _handler(event):
    """
    Handler function for RunPod serverless.
    """
    try:
        logger.info("RunPod handler started")
        
        # Initialize response
        response = {"error": None}
        
        # Handle both synchronous/asynchronous job
        job_input = event["input"]
        
        if not job_input:
            response["error"] = "No input provided"
            return response
            
        operation = job_input.get("operation", "predict")
        
        if operation == "train":
            logger.info("Starting training job")
            
            # Get data and parameters from input
            serialized_data = job_input.get("data", {})
            
            if not serialized_data:
                response["error"] = "No data provided for training"
                return response
                
            try:
                # Extract dataset and parameters
                dataset_data = serialized_data.get("dataset", {})
                params = serialized_data.get("params", {})
                
                # Deserialize the dataset if needed
                if isinstance(dataset_data, dict) and "type" in dataset_data and dataset_data["type"] == "dataframe":
                    logger.info("Deserializing DataFrame from serialized data")
                    try:
                        if dataset_data.get("format") == "parquet":
                            # Deserialize from parquet
                            binary_data = base64.b64decode(dataset_data["data"])
                            buffer = io.BytesIO(binary_data)
                            df = pd.read_parquet(buffer)
                        else:
                            # Deserialize from pickle
                            binary_data = base64.b64decode(dataset_data["data"])
                            df = pickle.loads(binary_data)
                        
                        logger.info(f"Successfully deserialized DataFrame with shape {df.shape}")
                    except Exception as e:
                        error_msg = f"Error deserializing DataFrame: {str(e)}"
                        logger.error(error_msg)
                        logger.exception(e)
                        response["error"] = error_msg
                        return response
                else:
                    logger.error("Invalid dataset format in request")
                    response["error"] = "Invalid dataset format in request"
                    return response
                
                # Process the training request
                result = train_model(df, params)
                
                # Make sure the accuracy is a Python float, not numpy or torch type
                if "metrics" in result and "accuracy" in result["metrics"]:
                    result["metrics"]["accuracy"] = float(result["metrics"]["accuracy"])
                    
                # Log successful completion and result size
                model_size = len(result.get("model", "")) if isinstance(result.get("model"), str) else 0
                logger.info(f"Training completed. Result size: {model_size / 1024:.2f} KB")
                
                # Set the output
                response = result
                
            except Exception as e:
                error_msg = f"Error in training: {str(e)}"
                logger.error(error_msg)
                logger.exception(e)
                response["error"] = error_msg
                
        elif operation == "predict":
            logger.info("Starting prediction job")
            
            # Get data and model from input
            serialized_data = job_input.get("data", {})
            model_data = job_input.get("model")
            
            if not serialized_data or not model_data:
                response["error"] = "Missing data or model for prediction"
                return response
            
            try:
                # Make predictions
                result = predict(model_data, serialized_data)
                response = result
            except Exception as e:
                error_msg = f"Error in prediction: {str(e)}"
                logger.error(error_msg)
                logger.exception(e)
                response["error"] = error_msg
        else:
            response["error"] = f"Unsupported operation: {operation}"
            
        return response
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.exception(e)
        return {"error": f"Handler error: {str(e)}"}

# This is the function that will be called by the RunPod serverless system
def handler(event):
    logger.info(f"Received event: {event}")
    try:
        return _handler(event)
    except Exception as e:
        logger.error(f"Unhandled exception in wrapper: {str(e)}")
        logger.exception(e)
        return {"error": f"Critical error: {str(e)}"}

# Only register and start the server if run directly
if __name__ == "__main__":
    # Start server
    print("Starting Jamba Threat Model server")
    # Register the handler function with RunPod
    runpod.serverless.start({"handler": handler})
else:
    # When imported as a module by runpod.serverless.start
    # we need to explicitly register our handler
    logger.info("Registering handler with RunPod serverless system")
    # This is what runpod.serverless.start will look for
    run_model = {"handler": handler}
