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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print the python path and list directories to debug
logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Files in current directory: {os.listdir('.')}")
if os.path.exists('/app/jamba_model'):
    logger.info(f"Files in /app/jamba_model: {os.listdir('/app/jamba_model')}")

# Try importing model classes
try:
    logger.info("Attempting to import model classes...")
    # Try different import approaches
    try:
        from jamba_model.model import JambaThreatModel, ThreatDataset
        logger.info("Successfully imported from jamba_model.model")
    except ImportError:
        logger.info("Trying alternative import path...")
        # Try direct import if module structure fails
        import sys
        sys.path.append('/app')
        from jamba_model.model import JambaThreatModel, ThreatDataset
        logger.info("Successfully imported with sys.path.append")
    
    logger.info("Successfully imported model classes")
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    try:
        # Check if file exists
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists('/app/jamba_model'):
            logger.error(f"Files in jamba_model directory: {os.listdir('/app/jamba_model')}")
        else:
            logger.error("/app/jamba_model directory not found")
            
        # Try to write the model classes directly
        logger.info("Attempting to create model classes directly...")
        
        # Define model classes inline as a last resort
        class ThreatDataset(torch.utils.data.Dataset):
            def __init__(self, features, targets):
                if isinstance(features, pd.DataFrame):
                    self.X = features.values.astype(np.float32)
                else:
                    self.X = features.astype(np.float32)
                    
                if isinstance(targets, pd.Series):
                    self.y = targets.values.astype(np.float32)
                else:
                    self.y = targets.astype(np.float32)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
                
        class JambaThreatModel(torch.nn.Module):
            def __init__(self, input_dim):
                super(JambaThreatModel, self).__init__()
                
                # Make embed_dim divisible by num_heads
                num_heads = 4
                self.embed_dim = input_dim
                
                # Find a suitable number of heads that divides embed_dim
                found_valid_heads = False
                possible_heads = [4, 2, 7, 1]
                
                for heads in possible_heads:
                    if input_dim % heads == 0:
                        num_heads = heads
                        found_valid_heads = True
                        logger.info(f"Using {num_heads} attention heads")
                        break
                
                # If no suitable head count found, pad the input dimension
                if not found_valid_heads:
                    self.embed_dim = ((input_dim + 3) // 4) * 4
                    logger.info(f"Padding input dimension from {input_dim} to {self.embed_dim}")
                    self.embedding = torch.nn.Linear(input_dim, self.embed_dim)
                else:
                    self.embedding = torch.nn.Identity()
                    
                self.attention = torch.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads)
                self.fc1 = torch.nn.Linear(self.embed_dim, 256)
                self.relu1 = torch.nn.ReLU()
                self.dropout1 = torch.nn.Dropout(0.3)
                self.fc2 = torch.nn.Linear(256, 128)
                self.relu2 = torch.nn.ReLU()
                self.dropout2 = torch.nn.Dropout(0.2)
                self.temporal = torch.nn.GRU(128, 64, batch_first=True, bidirectional=True)
                self.fc3 = torch.nn.Linear(128, 64)
                self.relu3 = torch.nn.ReLU()
                self.dropout3 = torch.nn.Dropout(0.1)
                self.fc4 = torch.nn.Linear(64, 32)
                self.relu4 = torch.nn.ReLU()
                self.output = torch.nn.Linear(32, 1)
                self.sigmoid = torch.nn.Sigmoid()
            
            def forward(self, x):
                x = self.embedding(x)
                x_reshaped = x.unsqueeze(1)
                x_att, _ = self.attention(
                    x_reshaped.transpose(0, 1),
                    x_reshaped.transpose(0, 1),
                    x_reshaped.transpose(0, 1)
                )
                x_att = x_att.transpose(0, 1).squeeze(1)
                x = self.fc1(x_att)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                x_temporal = x.unsqueeze(1)
                temporal_out, _ = self.temporal(x_temporal)
                temporal_out = temporal_out.reshape(temporal_out.size(0), -1)
                x = self.fc3(temporal_out)
                x = self.relu3(x)
                x = self.dropout3(x)
                x = self.fc4(x)
                x = self.relu4(x)
                x = self.output(x)
                x = self.sigmoid(x)
                return x
                
        logger.info("Successfully created model classes directly")
        
    except Exception as check_error:
        logger.error(f"Error creating model classes: {str(check_error)}")
    
    logger.error("Could not import or create model classes. Will attempt to proceed with inline definitions.")

# Optimization: Cache model
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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # OPTIMIZATION: Use automatic mixed precision
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    amp_enabled = torch.cuda.is_available()
    
    # OPTIMIZATION: JIT compile model if not training
    if not amp_enabled:
        model = torch.jit.script(model)  # JIT for CPU
    
    scaler = amp.GradScaler(enabled=amp_enabled)
    
    # OPTIMIZATION: Set cuda high water mark once
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    
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
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                
                with autocast():
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
                
                predicted = (outputs > 0.5).float()
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
    
    # Save model
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    model_buffer.seek(0)
    
    # OPTIMIZATION: Optimize model for inference after training
    model.eval()
    if torch.cuda.is_available():
        model = torch.jit.trace(model, torch.rand(1, input_dim, device=device))
    
    return {
        'model': model_buffer.read(),
        'metrics': {
            'accuracy': history['val_accuracy'][-1],
            'training_time': training_time,
            'history': history
        }
    }

def predict(model_data, data):
    """Make predictions using a trained model.
    
    Args:
        model_data: Binary model data
        data: DataFrame with prediction data
        
    Returns:
        DataFrame with predictions
    """
    # OPTIMIZATION: Cache model to avoid reloading
    model_hash = hash(model_data)
    
    if model_hash in _model_cache:
        model, input_dim = _model_cache[model_hash]
        print("Using cached model")
    else:
        # Load the model
        model_buffer = io.BytesIO(model_data)
        model_state = torch.load(model_buffer, map_location=device)
        
        # Initialize model
        input_dim = data.shape[1]
        model = JambaThreatModel(input_dim).to(device)
        model.load_state_dict(model_state)
        model.eval()
        
        _model_cache[model_hash] = (model, input_dim)
    
    # Convert data to tensor
    X_tensor = torch.FloatTensor(data.values).to(device)
    
    # OPTIMIZATION: Process in batches for large datasets
    batch_size = 1024
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        # Process batch...
        results.append(batch_result)
    
    # Combine results
    return pd.concat(results)

def handler(event):
    """RunPod handler function.
    
    Args:
        event: Dictionary with input data
        
    Returns:
        Dictionary with results
    """
    logger.info("Handler function called")
    logger.info(f"Received event: {event}")
    
    try:
        # Get input data
        input_data = event["input"]
        operation = input_data.get("operation", "predict")
        
        if operation == "train":
            # Process training request
            data = input_data.get("data")
            params = input_data.get("params", {})
            result = train_model(data, params)
            return result
        
        elif operation == "predict":
            # Process prediction request
            model_data = input_data.get("model")
            data = input_data.get("data")
            predictions = predict(model_data, data)
            return predictions
        
        elif operation == "test":
            # Simple test operation to verify the endpoint is working
            logger.info("Test operation received")
            return {"status": "success", "message": "Endpoint is working correctly"}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# Start server
if __name__ == "__main__":
    # For RunPod version 0.10.0
    import runpod
    runpod.serverless.start({"handler": handler})
    
    # Uncomment for local debugging if needed:
    # test_input = {
    #     "input": {
    #         "operation": "train",
    #         "data": {...}  # Add test data here if needed
    #     }
    # }
    # result = handler(test_input)
    # print(json.dumps(result, indent=2))
