"""
Validation utilities for Jamba Threat Detection.

This module provides functions for validating the environment, model,
and other components of the Jamba Threat Detection system.
"""

import os
import sys
import io
import logging
import importlib
import importlib.util
import torch
import traceback
import json
import requests
from time import sleep

# Set up logging
logger = logging.getLogger(__name__)

def check_python_modules(required_modules=None):
    """
    Check if required Python modules are available.
    
    Args:
        required_modules (list): List of module names to check
        
    Returns:
        tuple: (bool, list) - Success flag and list of missing modules
    """
    if required_modules is None:
        required_modules = ["torch", "numpy", "pandas", "runpod"]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            logger.info(f"✓ Module {module} is available")
        except ImportError:
            logger.warning(f"✗ Module {module} is NOT available")
            missing_modules.append(module)
    
    return len(missing_modules) == 0, missing_modules

def check_cuda_availability():
    """
    Check if CUDA is available and get GPU information.
    
    Returns:
        dict: CUDA and GPU information
    """
    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "device_capability": None,
        "total_memory_mb": None
    }
    
    if cuda_info["cuda_available"]:
        logger.info(f"✓ CUDA is available (version {cuda_info['cuda_version']})")
        logger.info(f"  Device count: {cuda_info['device_count']}")
        
        # Get information about the first device
        if cuda_info["device_count"] > 0:
            cuda_info["current_device"] = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_info["current_device"])
            cuda_info["device_name"] = props.name
            cuda_info["device_capability"] = f"{props.major}.{props.minor}"
            cuda_info["total_memory_mb"] = props.total_memory / (1024 * 1024)
            
            logger.info(f"  Device name: {cuda_info['device_name']}")
            logger.info(f"  Compute capability: {cuda_info['device_capability']}")
            logger.info(f"  Total memory: {cuda_info['total_memory_mb']:.2f} MB")
    else:
        logger.warning("✗ CUDA is NOT available - running in CPU mode")
    
    return cuda_info

def check_model_imports():
    """
    Check if Jamba model classes can be imported.
    
    Returns:
        tuple: (success_flag, error_message, import_path)
    """
    # Try different import strategies with informative logging
    import_attempts = [
        ("Direct import", lambda: importlib.import_module("jamba_model")),
        ("From src", lambda: importlib.import_module("src.jamba_model")),
        ("With path adjustment", lambda: _import_with_path_adjustment("jamba_model"))
    ]
    
    for description, import_func in import_attempts:
        try:
            logger.info(f"Attempting import using {description}")
            module = import_func()
            
            # Verify specific classes exist in the module
            if hasattr(module, "JambaThreatModel") and hasattr(module, "ThreatDataset"):
                logger.info(f"✓ Successfully imported model classes using {description}")
                return True, None, module.__file__
            else:
                logger.warning(f"✗ Module found but missing required classes using {description}")
        except Exception as e:
            logger.warning(f"✗ Import failed using {description}: {str(e)}")
    
    # If all attempts failed, provide diagnostic information
    logger.error("✗ All import attempts failed")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    return False, "Could not import Jamba model classes", None

def _import_with_path_adjustment(module_name):
    """
    Try to import a module after adjusting the Python path.
    
    Args:
        module_name (str): Name of the module to import
        
    Returns:
        module: Imported module object
    """
    # Add common paths to sys.path
    app_dir = os.environ.get("APP_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    src_dir = os.path.join(app_dir, "src")
    
    for path in [app_dir, src_dir]:
        if path not in sys.path:
            sys.path.append(path)
            logger.info(f"Added {path} to sys.path")
    
    # Try to find the module spec
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Could not find module {module_name} after path adjustment")
    
    # Load the module
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_model_initialization():
    """
    Test if the model can be initialized and performs a forward pass.
    
    Returns:
        tuple: (success_flag, error_message, model_info)
    """
    try:
        # First check if we can import the model
        import_success, import_error, _ = check_model_imports()
        if not import_success:
            return False, import_error, None
        
        # Import the model class
        from jamba_model import JambaThreatModel
        
        # Initialize the model
        input_dim = 28  # Default dimension for threat features
        logger.info(f"Initializing model with input_dim={input_dim}")
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        model = JambaThreatModel(input_dim)
        model.eval()  # Set to evaluation mode
        
        # Create a sample input and run a forward pass
        sample_input = torch.randn(4, input_dim)
        with torch.no_grad():
            output = model(sample_input)
        
        model_info = {
            "model_type": type(model).__name__,
            "input_dim": input_dim,
            "output_shape": list(output.shape),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "layers": str(model).count("\n") + 1
        }
        
        logger.info(f"✓ Model initialized successfully")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Parameter count: {model_info['parameter_count']}")
        
        # Log model structure information
        if logger.level <= logging.INFO:
            logger.info("Model structure overview:")
            for name, param in model.named_parameters():
                logger.info(f"  {name}: {param.shape}")
        
        return True, None, model_info
        
    except Exception as e:
        error_msg = f"Model initialization failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg, None

def test_model_serialization():
    """
    Test if the model can be serialized and deserialized correctly.
    
    Returns:
        tuple: (success_flag, error_message)
    """
    try:
        # First check if we can import the model
        import_success, import_error, _ = check_model_imports()
        if not import_success:
            return False, import_error
        
        # Import the model class
        from jamba_model import JambaThreatModel
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        # Create test model
        input_dim = 28
        model = JambaThreatModel(input_dim)
        model.eval()  # Set to evaluation mode
        
        # Serialize model
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Use different seed to ensure we're testing proper serialization
        torch.manual_seed(100)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(100)
        
        # Load state dictionary to new model
        state_dict = torch.load(buffer)
        new_model = JambaThreatModel(input_dim)
        new_model.load_state_dict(state_dict)
        new_model.eval()  # Set to evaluation mode
        
        # Create fixed test input and run inference
        torch.manual_seed(42)  # Reset seed for consistent input
        sample_input = torch.randn(4, input_dim)
        
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = new_model(sample_input)
        
        # Compare outputs with a tolerance
        match = torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)
        
        if match:
            logger.info("✓ Model serialization and loading successful")
            return True, None
        else:
            max_diff = (output1 - output2).abs().max().item()
            error_msg = f"Model outputs don't match after serialization (max diff = {max_diff})"
            logger.error(f"✗ {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Model serialization test failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg

def verify_runpod_endpoint(api_key, endpoint_id, max_retries=3):
    """
    Verify that a RunPod endpoint is reachable and functioning.
    
    Args:
        api_key (str): RunPod API key
        endpoint_id (str): RunPod endpoint ID
        max_retries (int): Maximum number of retries
        
    Returns:
        tuple: (success_flag, message, response)
    """
    if not api_key or not endpoint_id:
        logger.error("API key and endpoint ID are required")
        return False, "API key and endpoint ID are required", None
    
    logger.info(f"Verifying RunPod endpoint {endpoint_id}")
    
    # First, check if endpoint exists
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        health_url = f"https://api.runpod.io/v2/{endpoint_id}/health"
        
        logger.info(f"Checking endpoint health at {health_url}")
        response = requests.get(health_url, headers=headers)
        
        if response.status_code != 200:
            error_msg = f"Endpoint health check failed with status code {response.status_code}"
            logger.error(f"✗ {error_msg}")
            return False, error_msg, response.json() if response.text else None
            
        health_data = response.json()
        if not health_data.get("success", False):
            error_msg = f"Endpoint health check returned error: {health_data.get('error', 'Unknown error')}"
            logger.error(f"✗ {error_msg}")
            return False, error_msg, health_data
            
        logger.info(f"✓ Endpoint health check successful")
        
        # Now send a test job to verify functionality
        run_url = f"https://api.runpod.io/v2/{endpoint_id}/run"
        payload = {
            "input": {
                "operation": "health"
            }
        }
        
        logger.info(f"Sending test health check job to {run_url}")
        response = requests.post(run_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_msg = f"Test job submission failed with status code {response.status_code}"
            logger.error(f"✗ {error_msg}")
            return False, error_msg, response.json() if response.text else None
            
        job_data = response.json()
        job_id = job_data.get("id")
        
        if not job_id:
            error_msg = "Test job submission did not return a job ID"
            logger.error(f"✗ {error_msg}")
            return False, error_msg, job_data
            
        logger.info(f"✓ Test job submitted successfully with ID {job_id}")
        
        # Poll for job status
        status_url = f"https://api.runpod.io/v2/{endpoint_id}/status/{job_id}"
        
        for attempt in range(max_retries):
            logger.info(f"Checking job status (attempt {attempt + 1}/{max_retries})")
            response = requests.get(status_url, headers=headers)
            
            if response.status_code != 200:
                logger.warning(f"Status check failed with code {response.status_code}, retrying...")
                sleep(2)
                continue
                
            status_data = response.json()
            status = status_data.get("status")
            
            if status == "COMPLETED":
                logger.info(f"✓ Test job completed successfully")
                return True, "Endpoint verification successful", status_data
            elif status == "FAILED":
                error_msg = f"Test job failed: {status_data.get('error', 'Unknown error')}"
                logger.error(f"✗ {error_msg}")
                return False, error_msg, status_data
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                logger.info(f"Job status: {status}, waiting...")
                sleep(2)
            else:
                logger.warning(f"Unknown job status: {status}, waiting...")
                sleep(2)
        
        logger.error(f"✗ Test job timed out after {max_retries} attempts")
        return False, "Test job timed out", None
            
    except Exception as e:
        error_msg = f"Endpoint verification failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg, None

def health_check():
    """
    Perform a comprehensive health check of the system.
    
    Returns:
        dict: Health check results
    """
    results = {
        "timestamp": None,
        "status": "unhealthy",
        "success": False,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda": check_cuda_availability(),
        "model_imports": {"success": False},
        "model_initialization": {"success": False},
        "environment_variables": {"success": False}
    }
    
    try:
        # Check model imports
        import_success, import_error, import_path = check_model_imports()
        results["model_imports"] = {
            "success": import_success,
            "error": import_error,
            "path": import_path
        }
        
        # Check model initialization
        if import_success:
            init_success, init_error, model_info = test_model_initialization()
            results["model_initialization"] = {
                "success": init_success,
                "error": init_error,
                "info": model_info
            }
        
        # Check environment variables
        from src.utils.environment import check_environment_variables
        missing_vars, available_vars = check_environment_variables()
        results["environment_variables"] = {
            "success": len(missing_vars) == 0,
            "missing": missing_vars,
            "available": {k: "Set" for k, v in available_vars.items() if v}
        }
        
        # Determine overall status
        critical_checks = [
            results["model_imports"]["success"],
            results["model_initialization"]["success"]
        ]
        
        results["success"] = all(critical_checks)
        results["status"] = "healthy" if results["success"] else "unhealthy"
        
        import datetime
        results["timestamp"] = datetime.datetime.now().isoformat()
        
        return results
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        return results

def run_validation_checks(skip_endpoint_check=False, api_key=None, endpoint_id=None):
    """
    Run all validation checks and return a comprehensive report.
    
    Args:
        skip_endpoint_check (bool): Whether to skip endpoint verification
        api_key (str): RunPod API key
        endpoint_id (str): RunPod endpoint ID
        
    Returns:
        dict: Validation check results
    """
    import datetime
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "success": False,
        "checks": {}
    }
    
    # Check environment variables
    from src.utils.environment import check_environment_variables
    missing_vars, available_vars = check_environment_variables()
    results["checks"]["environment_variables"] = {
        "success": len(missing_vars) == 0,
        "missing": missing_vars,
        "available": {k: "Set" for k, v in available_vars.items() if v}
    }
    
    # Check required Python modules
    modules_success, missing_modules = check_python_modules()
    results["checks"]["python_modules"] = {
        "success": modules_success,
        "missing": missing_modules
    }
    
    # Check CUDA availability
    results["checks"]["cuda"] = check_cuda_availability()
    
    # Check model imports
    import_success, import_error, import_path = check_model_imports()
    results["checks"]["model_imports"] = {
        "success": import_success,
        "error": import_error,
        "path": import_path
    }
    
    # Check model initialization
    if import_success:
        init_success, init_error, model_info = test_model_initialization()
        results["checks"]["model_initialization"] = {
            "success": init_success,
            "error": init_error,
            "info": model_info
        }
        
        # Check model serialization
        if init_success:
            serial_success, serial_error = test_model_serialization()
            results["checks"]["model_serialization"] = {
                "success": serial_success,
                "error": serial_error
            }
    
    # Check RunPod endpoint
    if not skip_endpoint_check:
        if not api_key or not endpoint_id:
            # Try to get from environment
            api_key = os.environ.get("RUNPOD_API_KEY")
            endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
            
        if api_key and endpoint_id:
            endpoint_success, endpoint_msg, _ = verify_runpod_endpoint(api_key, endpoint_id)
            results["checks"]["runpod_endpoint"] = {
                "success": endpoint_success,
                "message": endpoint_msg
            }
        else:
            results["checks"]["runpod_endpoint"] = {
                "success": False,
                "message": "API key or endpoint ID not provided"
            }
    else:
        results["checks"]["runpod_endpoint"] = {
            "success": True,
            "message": "Endpoint check skipped"
        }
    
    # Determine overall success
    # Critical checks: model_imports, model_initialization, model_serialization
    critical_checks = [
        results["checks"].get("model_imports", {}).get("success", False),
        results["checks"].get("model_initialization", {}).get("success", False),
        results["checks"].get("model_serialization", {}).get("success", False)
    ]
    
    results["success"] = all(critical_checks)
    
    return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run a quick health check
    logger.info("Running health check...")
    health_results = health_check()
    
    # Print results in a readable format
    logger.info(f"Health check status: {health_results['status']}")
    for check, result in health_results.items():
        if isinstance(result, dict) and 'success' in result:
            status = "✓" if result["success"] else "✗"
            logger.info(f"{check}: {status}")
    
    # Exit with status code
    sys.exit(0 if health_results["success"] else 1) 