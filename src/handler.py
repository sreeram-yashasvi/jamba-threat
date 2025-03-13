#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import logging
import traceback
import io
from functools import lru_cache
from typing import Dict, Any, Optional, List, Union, Tuple
from io import BytesIO
import importlib.util
from pathlib import Path
import base64

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Add utility modules
try:
    from utils import environment, validation
    environment_module_loaded = True
except ImportError:
    environment_module_loaded = False
    logging.warning("Could not import utility modules directly. Attempting path adjustment.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure paths and environment
if not environment_module_loaded:
    # If utils module not in path, add APP_DIR to path
    APP_DIR = os.environ.get('APP_DIR', '/app')
    if APP_DIR not in sys.path:
        sys.path.append(APP_DIR)
    if os.path.join(APP_DIR, 'src') not in sys.path:
        sys.path.append(os.path.join(APP_DIR, 'src'))
    
    try:
        from utils import environment, validation
        environment_module_loaded = True
    except ImportError:
        logging.error("Failed to import utility modules even after path adjustment.")
        MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(APP_DIR, 'models'))
        LOG_DIR = os.environ.get('LOG_DIR', os.path.join(APP_DIR, 'logs'))
        environment_module_loaded = False

# Initialize environment if module was loaded
if environment_module_loaded:
    environment.setup_environment(create_dirs=True)
    MODEL_DIR = environment.get_model_dir()
    LOG_DIR = environment.get_log_dir()
    DEBUG = environment.is_debug_mode()
else:
    MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
    LOG_DIR = os.environ.get('LOG_DIR', '/app/logs')
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# Global model cache
MODEL_CACHE = {}

# Import model classes
try:
    from jamba.jamba_model import JambaThreatModel, ThreatDataset
    from jamba.model_config import ModelConfig, DEFAULT_CPU_CONFIG, DEFAULT_GPU_CONFIG
    logging.info("Successfully imported JambaThreatModel, ThreatDataset, and ModelConfig")
except ImportError as e:
    logging.error(f"Failed to import model classes: {e}")
    logging.error(f"Python path: {sys.path}")
    logging.error("Attempting fallback import strategy...")
    
    # Try alternative import strategy
    try:
        sys.path.append('/app')
        sys.path.append('/app/src')
        
        # Try importing from jamba package first
        try:
            from jamba.jamba_model import JambaThreatModel, ThreatDataset
            from jamba.model_config import ModelConfig, DEFAULT_CPU_CONFIG, DEFAULT_GPU_CONFIG
            logging.info("Successfully imported model classes from jamba package")
        except ImportError:
            # Fallback to direct import
            from jamba_model import JambaThreatModel, ThreatDataset
            # Create a minimal ModelConfig class if not available
            class ModelConfig:
                def __init__(self, input_dim=512, hidden_dim=None, output_dim=2, dropout_rate=0.3,
                             n_heads=None, feature_layers=None, version="1.0.0",
                             use_mixed_precision=True, batch_size=128, learning_rate=0.001):
                    self.version = version
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.output_dim = output_dim
                    self.dropout_rate = dropout_rate
                    self.n_heads = n_heads
                    self.feature_layers = feature_layers
                    self.use_mixed_precision = use_mixed_precision
                    self.batch_size = batch_size
                    self.learning_rate = learning_rate
                
                def to_dict(self):
                    return self.__dict__
            
            # Create default configs
            DEFAULT_CPU_CONFIG = ModelConfig(
                input_dim=512, hidden_dim=256, output_dim=2, dropout_rate=0.3,
                n_heads=4, feature_layers=2, use_mixed_precision=False, batch_size=32
            )
            
            DEFAULT_GPU_CONFIG = ModelConfig(
                input_dim=512, hidden_dim=512, output_dim=2, dropout_rate=0.3,
                n_heads=8, feature_layers=4, use_mixed_precision=True, batch_size=128
            )
            
            logging.info("Successfully imported model classes using fallback strategy with minimal ModelConfig")
    except ImportError as e:
        logging.error(f"Fatal error: Could not import model classes: {e}")
        logging.error(traceback.format_exc())
        raise

# Use lru_cache to avoid reloading the same model multiple times
@lru_cache(maxsize=5)
def load_model(model_path: str, force_cpu: bool = False):
    """
    Load a model from the specified path with caching
    """
    import torch
    
    if model_path in MODEL_CACHE:
        logging.info(f"Using cached model from {model_path}")
        return MODEL_CACHE[model_path]
    
    logging.info(f"Loading model from {model_path}")
    
    # Determine device
    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")
    
    try:
        # Load the model
        model_data = torch.load(model_path, map_location=device)
        
        # Check if it's the state dict or the full model
        if isinstance(model_data, dict) and "state_dict" in model_data:
            # Create model with the same parameters
            input_dim = model_data.get("input_dim", 512)
            logging.info(f"Creating model with input dimension: {input_dim}")
            
            # Initialize model with config
            config = ModelConfig(input_dim=input_dim)
            model = JambaThreatModel(config)
            model.load_state_dict(model_data["state_dict"])
        else:
            # Full model serialization
            model = model_data
            
        model.to(device)
        model.eval()
        
        # Cache the model
        MODEL_CACHE[model_path] = model
        logging.info(f"Model loaded successfully from {model_path}")
        
        # Test forward pass
        if environment_module_loaded:
            test_result = validation.run_health_check()
            if test_result.get('success', False):
                logging.info("Model initialization test passed")
            else:
                logging.warning(f"Model initialization test failed: {test_result.get('error', 'Unknown error')}")
        
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        logging.error(traceback.format_exc())
        raise

def train_model(df, params):
    """Train the model with support for chunked training and external storage."""
    try:
        # Check if this is a continuation of chunked training
        is_chunk = params.get('is_chunk', False)
        is_final = params.get('is_final', False)
        
        if is_final:
            # Final job just needs to retrieve and return the model
            temp_model_path = params.get('temp_model_path')
            if not temp_model_path or not os.path.exists(temp_model_path):
                raise ValueError("Final job requires valid temp_model_path")
            
            # Load the model from temp storage
            model = load_model_from_temp(temp_model_path)
            
            # For large models, store in external storage instead of returning directly
            # Get model size - need to check if it's a file path or a loaded model
            if isinstance(temp_model_path, str) and os.path.exists(temp_model_path):
                model_size = os.path.getsize(temp_model_path)
            else:
                # Estimate size by serializing to a buffer
                buffer = BytesIO()
                torch.save(model, buffer)
                model_size = len(buffer.getvalue())
            if model_size > MAX_RETURN_SIZE:
                # Store in external storage and return URL
                storage_url = store_model_in_external_storage(temp_model_path)
                return {
                    "success": True,
                    "model_storage_url": storage_url,
                    "metrics": load_metrics_from_temp(temp_model_path)
                }
            else:
                # Small enough to return directly
                buffer = BytesIO()
                torch.save(model.state_dict(), buffer)
                model_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return {
                    "success": True,
                    "model": model_data,
                    "metrics": load_metrics_from_temp(temp_model_path)
                }
        
        # Regular training or chunk processing
        logging.info("Starting model training")
        
        # Use provided temp model if continuation of chunked training
        if is_chunk and params.get('chunk_number', 1) > 1:
            temp_model_path = params.get('temp_model_path')
            model = load_model_from_temp(temp_model_path)
            logging.info(f"Continuing training from chunk {params.get('chunk_number')}/{params.get('total_chunks')}")
        else:
            # Initialize new model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_dim = len(feature_list)
            
            # Create config with appropriate device settings
            if torch.cuda.is_available():
                config = DEFAULT_GPU_CONFIG
                config.input_dim = input_dim
            else:
                config = DEFAULT_CPU_CONFIG
                config.input_dim = input_dim
                
            model = JambaThreatModel(config).to(device)
            logging.info(f"Initialized new model with input dimension: {input_dim}")
        
        # Extract parameters
        target_column = params.get('target_column', 'is_threat')
        epochs = params.get('epochs', 30)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 128)
        
        # Get feature list
        try:
            feature_list = environment.DEFAULT_FEATURES
        except (AttributeError, NameError):
            # Fallback to hardcoded features
            feature_list = df.columns.tolist()
            if target_column in feature_list:
                feature_list.remove(target_column)
        
        # Prepare data
        logging.info(f"Preparing data with shape: {df.shape}")
        X = df[feature_list].copy()
        y = df[target_column].copy()
        
        # Split data
        train_size = int(0.8 * len(df))
        test_size = len(df) - train_size
        logging.info(f"Splitting data into {train_size} training and {test_size} testing samples")
        
        # Create datasets
        train_dataset = ThreatDataset(
            X.iloc[:train_size], 
            y.iloc[:train_size]
        )
        test_dataset = ThreatDataset(
            X.iloc[train_size:], 
            y.iloc[train_size:]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size
        )
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Use mixed precision training if available
        use_amp = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training loop
        logging.info(f"Starting training for {epochs} epochs")
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'gpu_memory': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            try:
                # Record GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                    training_history['gpu_memory'].append({
                        'allocated': memory_allocated,
                        'reserved': memory_reserved
                    })
                    logging.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # Training phase with error recovery
                model.train()
                running_loss = 0.0
                failed_batches = 0
                
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    try:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                        
                        optimizer.zero_grad()
                        
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                        
                        running_loss += loss.item()
                        
                    except RuntimeError as e:
                        # Handle batch-specific errors
                        failed_batches += 1
                        if "out of memory" in str(e).lower():
                            logging.warning(f"OOM error in batch {batch_idx}, skipping")
                            torch.cuda.empty_cache()
                            # If too many OOM errors, reduce batch size for next epoch
                            if failed_batches > len(train_loader) // 4:
                                raise RuntimeError("Too many OOM errors, consider reducing batch size")
                        else:
                            raise  # Re-raise other runtime errors
                
                epoch_train_loss = running_loss / len(train_loader)
                training_history['train_loss'].append(epoch_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                epoch_val_loss = val_loss / len(test_loader)
                epoch_val_accuracy = correct / total
                
                training_history['val_loss'].append(epoch_val_loss)
                training_history['val_accuracy'].append(epoch_val_accuracy)
                
                logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, "
                          f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")
            
            except Exception as e:
                logging.error(f"Error in epoch {epoch+1}: {str(e)}")
                logging.error(traceback.format_exc())
                
                # Attempt recovery
                torch.cuda.empty_cache()
                
                # Save checkpoint to avoid losing progress
                checkpoint_path = f"/tmp/training_checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_history': training_history
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Continue to next epoch instead of failing completely
                continue
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_accuracy = 0.0
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                outputs = model(batch_X)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                final_accuracy += (predicted == batch_y).sum().item() / batch_y.size(0)
            
            final_accuracy /= len(test_loader)
        
        # Save model to temporary location if chunked training
        if is_chunk:
            temp_path = f"/tmp/model_chunk_{params.get('chunk_number')}.pt"
            torch.save(model.state_dict(), temp_path)
            logging.info(f"Saved intermediate model to {temp_path}")
            
            # Include temp path in result for next chunk
            return {
                "success": True,
                "temp_model_path": temp_path,
                "metrics": {
                    "accuracy": final_accuracy,
                    "training_time": training_time,
                    "history": training_history
                }
            }
        
        # Save model
        try:
            model_path = environment.get_model_path()
        except (AttributeError, NameError):
            # Fallback path
            model_path = os.path.join(os.environ.get("MODEL_DIR", "./models"), "jamba_model.pth")
            
        logging.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        
        # Serialize model for return
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        model_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "model": model_data,
            "metrics": {
                "accuracy": final_accuracy,
                "training_time": training_time,
                "history": training_history
            }
        }
    
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

def predict(model_path, input_data):
    """
    Make predictions using the trained model.
    
    Args:
        model_path: Path to the trained model
        input_data: DataFrame or dictionary of input data
        
    Returns:
        Dictionary with predictions
    """
    try:
        logging.info(f"Loading model from {model_path}")
        
        # Check if model is in cache
        if model_path in MODEL_CACHE:
            logging.info("Using cached model")
            model = MODEL_CACHE[model_path]
        else:
            # Load model
            if not os.path.exists(model_path):
                try:
                    model_path = environment.get_model_path()
                except (AttributeError, NameError):
                    # Fallback path
                    model_path = os.path.join(os.environ.get("MODEL_DIR", "./models"), "jamba_model.pth")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get feature list
            try:
                feature_list = environment.DEFAULT_FEATURES
                input_dim = len(feature_list)
            except (AttributeError, NameError):
                # Fallback to hardcoded input dimension
                input_dim = 28  # Standard input dimension for the model
            
            # Create config with appropriate device settings
            if torch.cuda.is_available():
                config = DEFAULT_GPU_CONFIG
                config.input_dim = input_dim
            else:
                config = DEFAULT_CPU_CONFIG
                config.input_dim = input_dim
                
            model = JambaThreatModel(config).to(device)
            
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise
            
            # Cache model
            MODEL_CACHE[model_path] = model
        
        # Prepare input data
        if isinstance(input_data, dict):
            # Convert dict to DataFrame
            logging.info("Converting input dict to DataFrame")
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        
        # Get feature list
        try:
            feature_list = environment.DEFAULT_FEATURES
        except (AttributeError, NameError):
            # Fallback to available features in the input data
            feature_list = [
                'dst_ip_domain_encoded', 'dst_port', 'src_port', 'protocol', 'packet_count',
                'byte_count', 'tcp_flags', 'duration', 'avg_bytes_per_packet', 'packets_per_second',
                'bytes_per_second', 'avg_packet_size', 'avg_packet_interval', 'packet_size_variance',
                'packet_interval_variance', 'network_time_variance', 'encryption_score',
                'destination_popularity', 'data_transfer_ratio', 'service_blacklist_score',
                'src_ip_reputation', 'dst_ip_reputation', 'data_exfiltration_score',
                'conn_established', 'scan_score', 'rare_domain_score', 'temporal_anomaly_score'
            ]
        
        # Ensure all features are present
        for feature in feature_list:
            if feature not in df.columns:
                logging.warning(f"Feature '{feature}' missing in input data, filling with zeros")
                df[feature] = 0
        
        # Keep only the required features
        df = df[feature_list]
        
        # Create dataset and loader
        dataset = ThreatDataset(df, np.zeros(len(df)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=len(df))
        
        # Make predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        logging.info("Making predictions")
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                scores = torch.sigmoid(outputs).cpu().numpy().flatten()
                predictions = (scores > 0.5).astype(int)
        
        logging.info(f"Predictions complete: {predictions.tolist()}")
        return {
            "success": True,
            "predictions": predictions.tolist(),
            "scores": scores.tolist()
        }
    
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

def health_check():
    """
    Perform a health check on the model.
    
    Returns:
        Dictionary with health check results
    """
    try:
        # Use the centralized validation module if available
        try:
            return validation.run_health_check()
        except (NameError, AttributeError):
            logging.info("Fallback to local health check implementation")
            
            # Check if PyTorch is available
            logging.info(f"PyTorch version: {torch.__version__}")
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            
            # Get feature list
            try:
                feature_list = environment.DEFAULT_FEATURES
                input_dim = len(feature_list)
            except (AttributeError, NameError):
                # Fallback to hardcoded input dimension
                input_dim = 28  # Standard input dimension for the model
            
            # Check if model can be initialized
            config = ModelConfig(input_dim=input_dim)
            model = JambaThreatModel(config)
            
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
        logging.error(f"Health check failed: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }

def auto_tune_batch_size(model, X_sample, y_sample, initial_batch_size, device):
    """Automatically tune batch size based on GPU memory.
    
    This function tries to find the largest batch size that fits in GPU memory
    by testing progressively smaller batch sizes until one works.
    
    Args:
        model: Model to use for testing
        X_sample: Sample features to use for testing
        y_sample: Sample labels to use for testing
        initial_batch_size: Initial batch size to try
        device: Device to use (cuda or cpu)
        
    Returns:
        The largest batch size that fits in GPU memory
    """
    logging.info("Auto-tuning batch size for optimal GPU memory usage")
    
    # Start with the initial batch size
    batch_size = initial_batch_size
    
    # Create a sample dataset
    sample_dataset = ThreatDataset(X_sample, y_sample)
    
    # Try to find the maximum batch size that fits in GPU memory
    while batch_size > 1:
        try:
            # Create a test loader with the current batch size
            test_loader = DataLoader(sample_dataset, batch_size=batch_size)
            
            # Try a forward and backward pass
            model.train()
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                
                # Forward pass
                outputs = model(batch_X)
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Clear gradients
                model.zero_grad()
                
                # If we got here without OOM error, this batch size works
                logging.info(f"Auto-tuned batch size: {batch_size}")
                return batch_size
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduce batch size and try again
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logging.info(f"Reducing batch size to {batch_size} due to OOM")
            else:
                # If it's not an OOM error, re-raise
                raise
    
    # If we got here, even batch_size=1 doesn't work
    logging.warning("Model may be too large for GPU memory even with batch_size=1")
    return 1

def enhanced_training_loop(model, train_loader, test_loader, optimizer, criterion, device, epochs, use_amp=False):
    """Enhanced training loop with better error handling and monitoring."""
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # History tracking
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'gpu_memory': []
    }
    
    # Training loop with robust error handling
    start_time = time.time()
    
    for epoch in range(epochs):
        try:
            # Record GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                training_history['gpu_memory'].append({
                    'allocated': memory_allocated,
                    'reserved': memory_reserved
                })
                logging.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            # Training phase with error recovery
            model.train()
            running_loss = 0.0
            failed_batches = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                try:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                    
                    optimizer.zero_grad()
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item()
                    
                except RuntimeError as e:
                    # Handle batch-specific errors
                    failed_batches += 1
                    if "out of memory" in str(e).lower():
                        logging.warning(f"OOM error in batch {batch_idx}, skipping")
                        torch.cuda.empty_cache()
                        # If too many OOM errors, reduce batch size for next epoch
                        if failed_batches > len(train_loader) // 4:
                            raise RuntimeError("Too many OOM errors, consider reducing batch size")
                    else:
                        raise  # Re-raise other runtime errors
            
            epoch_train_loss = running_loss / len(train_loader)
            training_history['train_loss'].append(epoch_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_accuracy = correct / total
            
            training_history['val_loss'].append(epoch_val_loss)
            training_history['val_accuracy'].append(epoch_val_accuracy)
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, "
                      f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")
        
        except Exception as e:
            logging.error(f"Error in epoch {epoch+1}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Attempt recovery
            torch.cuda.empty_cache()
            
            # Save checkpoint to avoid losing progress
            checkpoint_path = f"/tmp/training_checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_history': training_history
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Continue to next epoch instead of failing completely
            continue
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_accuracy = 0.0
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
            outputs = model(batch_X)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            final_accuracy += (predicted == batch_y).sum().item() / batch_y.size(0)
        
        final_accuracy /= len(test_loader)
    
    return {
        "success": True,
        "metrics": {
            "accuracy": final_accuracy,
            "training_time": training_time,
            "history": training_history
        }
    }

def _handler(event):
    """
    Handle RunPod events.
    
    Args:
        event: RunPod event
    
    Returns:
        Response dictionary
    """
    try:
        logging.info(f"Received event: {json.dumps(event)}")
        
        # Check operation type
        operation = event.get("operation", "predict")
        
        if operation == "train":
            # Training operation
            logging.info("Processing training operation")
            
            input_data = event.get("input", {})
            data = input_data.get("data")
            params = input_data.get("params", {})
            
            if not data:
                return {
                    "success": False,
                    "error": "Missing data for training"
                }
            
            # Convert data to DataFrame
            try:
                df = pd.DataFrame(data)
                logging.info(f"Training data loaded with shape: {df.shape}")
            except Exception as e:
                logging.error(f"Error loading training data: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error loading training data: {str(e)}"
                }
            
            # Train model
            result = train_model(df, params)
            return result
        
        elif operation == "predict":
            # Prediction operation
            logging.info("Processing prediction operation")
            
            input_data = event.get("input", {})
            data = input_data.get("data")
            
            # Get model path
            try:
                model_path = input_data.get("model_path", environment.get_model_path())
            except (AttributeError, NameError):
                # Fallback path
                model_path = input_data.get("model_path", os.path.join(
                    os.environ.get("MODEL_DIR", "./models"), "jamba_model.pth"))
            
            if not data:
                return {
                    "success": False,
                    "error": "Missing data for prediction"
                }
            
            # Convert data to DataFrame
            try:
                if isinstance(data, list) and isinstance(data[0], dict):
                    # List of dictionaries
                    df = pd.DataFrame(data)
                else:
                    # Single dictionary
                    df = pd.DataFrame([data])
                
                logging.info(f"Prediction data loaded with shape: {df.shape}")
            except Exception as e:
                logging.error(f"Error loading prediction data: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error loading prediction data: {str(e)}"
                }
            
            # Make predictions
            result = predict(model_path, df)
            return result
        
        elif operation == "health":
            # Health check operation
            logging.info("Processing health check operation")
            return health_check()
        
        else:
            # Unknown operation
            logging.error(f"Unknown operation: {operation}")
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
    
    except Exception as e:
        logging.error(f"Error in handler: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

def handler(event):
    """
    RunPod handler function.
    
    Args:
        event: RunPod event
    
    Returns:
        Response dictionary
    """
    try:
        logging.info(f"Handler triggered with event size: {len(str(event)) if event else 0} bytes")
        
        # Check job size for logging
        event_size = sys.getsizeof(str(event))
        if event_size > 10240:  # 10KB
            logging.info(f"Large job received: {event_size / 1024:.2f} KB")
        
        # Process the event
        result = _handler(event)
        
        # Log the size of the response
        result_size = sys.getsizeof(str(result))
        logging.info(f"Returning result of size: {result_size / 1024:.2f} KB")
        
        return result
    
    except Exception as e:
        logging.error(f"Error handling event: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

# Only register and start the server if run directly
if __name__ == "__main__":
    logging.info("Starting Jamba Threat Handler")
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logging.error(f"Failed to start serverless function: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
else:
    # When imported as a module by runpod.serverless.start
    # we need to explicitly register our handler
    logging.info("Registering handler with RunPod serverless system")
    # This is what runpod.serverless.start will look for
    run_model = {"handler": handler}
