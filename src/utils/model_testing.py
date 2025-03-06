"""
Model testing utilities for Jamba Threat Detection.

This module provides functions for testing the model, including
forward pass testing, serialization testing, and performance testing.
"""

import os
import io
import torch
import logging
import numpy as np
import pandas as pd
import traceback
from time import time

logger = logging.getLogger(__name__)

def create_test_input(batch_size=4, input_dim=28, seed=42):
    """
    Create a test input tensor with fixed random seed for reproducibility.
    
    Args:
        batch_size (int): Batch size
        input_dim (int): Input dimension
        seed (int): Random seed
        
    Returns:
        torch.Tensor: Test input tensor
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create random input tensor
    return torch.randn(batch_size, input_dim)

def create_test_dataset(num_samples=100, input_dim=28, seed=42):
    """
    Create a test dataset with random data.
    
    Args:
        num_samples (int): Number of samples
        input_dim (int): Input dimension
        seed (int): Random seed
        
    Returns:
        tuple: (features, targets) as numpy arrays
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create random features
    features = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Create random binary targets
    targets = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)
    
    return features, targets

def test_forward_pass(model, input_tensor=None, device=None):
    """
    Test a forward pass through the model.
    
    Args:
        model: PyTorch model
        input_tensor: Optional input tensor, will be created if None
        device: Optional device to use
        
    Returns:
        tuple: (output_tensor, inference_time_ms)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if input_tensor is None:
        # Get input dimension from the first layer
        input_dim = None
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                input_dim = param.shape[1]
                break
        
        if input_dim is None:
            input_dim = 28  # Default for JambaThreatModel
        
        input_tensor = create_test_input(batch_size=4, input_dim=input_dim)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move to device
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Time the forward pass
    start_time = time()
    with torch.no_grad():
        output = model(input_tensor)
    inference_time = (time() - start_time) * 1000  # Convert to ms
    
    logger.info(f"Forward pass successful. Output shape: {output.shape}")
    logger.info(f"Inference time: {inference_time:.2f} ms")
    
    return output, inference_time

def test_serialization(model, input_tensor=None, device=None):
    """
    Test model serialization and deserialization.
    
    Args:
        model: PyTorch model
        input_tensor: Optional input tensor, will be created if None
        device: Optional device to use
        
    Returns:
        tuple: (success, error_message, max_difference)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Create input tensor if not provided
        if input_tensor is None:
            # Get input dimension from the first layer
            input_dim = None
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    input_dim = param.shape[1]
                    break
            
            if input_dim is None:
                input_dim = 28  # Default for JambaThreatModel
            
            input_tensor = create_test_input(batch_size=4, input_dim=input_dim)
        
        # Move input tensor to device
        input_tensor = input_tensor.to(device)
        
        # Get output from original model
        with torch.no_grad():
            output1 = model(input_tensor)
        
        # Serialize model to buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Create new model instance and load state dict
        model_class = model.__class__
        new_model = model_class(input_dim=input_tensor.shape[1])
        new_model.to(device)
        new_model.load_state_dict(torch.load(buffer, map_location=device))
        new_model.eval()
        
        # Get output from new model
        with torch.no_grad():
            output2 = new_model(input_tensor)
        
        # Compare outputs
        match = torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)
        max_diff = (output1 - output2).abs().max().item()
        
        if match:
            logger.info("✓ Model serialization and loading successful")
            return True, None, max_diff
        else:
            error_msg = f"Model outputs don't match after serialization (max diff = {max_diff})"
            logger.error(f"✗ {error_msg}")
            return False, error_msg, max_diff
            
    except Exception as e:
        error_msg = f"Model serialization test failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg, None

def test_dataset_loading(dataset_class, features, targets):
    """
    Test dataset loading and iteration.
    
    Args:
        dataset_class: Dataset class to test
        features: Features array or DataFrame
        targets: Targets array or Series
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Create dataset
        dataset = dataset_class(features, targets)
        
        # Test dataset length
        length = len(dataset)
        logger.info(f"Dataset length: {length}")
        
        # Test item access
        x, y = dataset[0]
        logger.info(f"First item shapes: x={x.shape}, y={type(y)}")
        
        # Test iteration
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=min(32, length), shuffle=True)
        
        for batch_x, batch_y in loader:
            logger.info(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
            break
        
        return True, None
        
    except Exception as e:
        error_msg = f"Dataset test failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg

def benchmark_model(model, input_shape=(100, 28), device=None, num_runs=10):
    """
    Benchmark model inference performance.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to use
        num_runs: Number of runs to average
        
    Returns:
        dict: Benchmark results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Create input tensor
        input_tensor = torch.randn(*input_shape).to(device)
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Warm-up run
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Measure inference time
        inference_times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_times.append((time() - start_time) * 1000)  # ms
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        # Calculate throughput
        throughput = input_shape[0] / (avg_time / 1000)  # samples/s
        
        results = {
            "device": str(device),
            "input_shape": list(input_shape),
            "batch_size": input_shape[0],
            "num_runs": num_runs,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "throughput_samples_per_sec": throughput
        }
        
        logger.info(f"Benchmark results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test model import
    try:
        from jamba_model import JambaThreatModel
        
        # Create model
        model = JambaThreatModel(input_dim=28)
        
        # Test forward pass
        output, inference_time = test_forward_pass(model)
        
        # Test serialization
        success, error, _ = test_serialization(model)
        
        # Print results
        if success:
            logger.info("All tests passed!")
        else:
            logger.error(f"Test failed: {error}")
            
    except ImportError:
        logger.error("Could not import JambaThreatModel. Are you in the correct directory?")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error(traceback.format_exc()) 