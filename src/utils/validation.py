"""
Validation utilities for Jamba Threat Detection model.
Centralizes health checks, model validation, and environment validation.
"""

import os
import sys
import json
import logging
import torch
import io
import importlib
import traceback
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import environment module (local import to avoid circular dependencies)
try:
    from utils import environment
except ImportError:
    # Adjust path if running as main script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        from utils import environment
    except ImportError:
        logger.error("Could not import environment module")
        environment = None

def check_environment_variables():
    """
    Check if required environment variables are set.
    
    Returns:
        tuple: (bool, list) - Success flag and list of missing variables
    """
    if environment:
        return environment.check_environment_variables()
    
    # Fallback if environment module not available
    required_vars = ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False, missing_vars
    
    return True, []

def check_cuda_availability():
    """
    Check if CUDA is available and get CUDA information.
    
    Returns:
        dict: CUDA availability information
    """
    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }
    
    if cuda_info["cuda_available"]:
        cuda_info.update({
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.device_count() > 0 else None
        })
    
    logger.info(f"CUDA available: {cuda_info['cuda_available']}")
    if cuda_info["cuda_available"]:
        logger.info(f"CUDA version: {cuda_info['cuda_version']}")
        logger.info(f"GPU count: {cuda_info['device_count']}")
    
    return cuda_info

def check_model_imports():
    """
    Check if the model classes can be imported.
    
    Returns:
        tuple: (bool, dict) - Success flag and import results
    """
    import_results = {
        "jamba_model": False,
        "ThreatDataset": False,
        "JambaThreatModel": False,
        "import_path": None,
        "error": None
    }
    
    try:
        # Try direct import
        import jamba_model
        import_results["jamba_model"] = True
        import_results["import_path"] = jamba_model.__file__
        
        # Check for required classes
        if hasattr(jamba_model, "ThreatDataset"):
            import_results["ThreatDataset"] = True
        
        if hasattr(jamba_model, "JambaThreatModel"):
            import_results["JambaThreatModel"] = True
        
        logger.info(f"Successfully imported jamba_model from {import_results['import_path']}")
        return all([import_results["ThreatDataset"], import_results["JambaThreatModel"]]), import_results
    
    except ImportError as e:
        import_results["error"] = str(e)
        logger.warning(f"Failed to import jamba_model directly: {e}")
        
        # Try adjusting sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        potential_paths = [
            parent_dir,
            os.path.dirname(parent_dir),
            os.path.join(os.path.dirname(parent_dir), "src")
        ]
        
        for path in potential_paths:
            if path not in sys.path:
                logger.info(f"Adding {path} to sys.path")
                sys.path.append(path)
        
        try:
            import jamba_model
            import_results["jamba_model"] = True
            import_results["import_path"] = jamba_model.__file__
            
            # Check for required classes
            if hasattr(jamba_model, "ThreatDataset"):
                import_results["ThreatDataset"] = True
            
            if hasattr(jamba_model, "JambaThreatModel"):
                import_results["JambaThreatModel"] = True
            
            logger.info(f"Successfully imported jamba_model after path adjustment from {import_results['import_path']}")
            return all([import_results["ThreatDataset"], import_results["JambaThreatModel"]]), import_results
        
        except ImportError as e:
            import_results["error"] = str(e)
            logger.error(f"Failed to import jamba_model after path adjustment: {e}")
            logger.info(f"Current sys.path: {sys.path}")
            return False, import_results

def test_model_initialization(input_dim=28):
    """
    Test if the model can be initialized.
    
    Args:
        input_dim: Input dimension for the model
        
    Returns:
        tuple: (bool, dict) - Success flag and initialization results
    """
    results = {
        "model_initialized": False,
        "forward_pass_successful": False,
        "output_shape": None,
        "error": None,
        "model_parameters": {}
    }
    
    try:
        # Check if model can be imported
        success, import_results = check_model_imports()
        if not success:
            results["error"] = f"Failed to import model classes: {import_results['error']}"
            return False, results
        
        # Import the model
        from jamba_model import JambaThreatModel
        
        # Initialize the model
        model = JambaThreatModel(input_dim)
        results["model_initialized"] = True
        
        # Log model parameters
        for name, param in model.named_parameters():
            results["model_parameters"][name] = list(param.shape)
        
        # Test forward pass
        sample_input = torch.randn(4, input_dim)
        output = model(sample_input)
        results["forward_pass_successful"] = True
        results["output_shape"] = list(output.shape)
        
        logger.info(f"Model initialized successfully. Output shape: {results['output_shape']}")
        return True, results
    
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error initializing model: {e}")
        logger.error(traceback.format_exc())
        return False, results

def test_model_serialization(input_dim=28):
    """
    Test if the model can be serialized and deserialized.
    
    Args:
        input_dim: Input dimension for the model
        
    Returns:
        tuple: (bool, dict) - Success flag and serialization results
    """
    results = {
        "serialization_successful": False,
        "deserialization_successful": False,
        "outputs_match": False,
        "error": None
    }
    
    try:
        # Check if model can be imported
        success, import_results = check_model_imports()
        if not success:
            results["error"] = f"Failed to import model classes: {import_results['error']}"
            return False, results
        
        # Import the model
        from jamba_model import JambaThreatModel
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Initialize the model
        model = JambaThreatModel(input_dim)
        model.eval()
        
        # Serialize the model
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        results["serialization_successful"] = True
        
        # Set different seed to ensure proper serialization test
        torch.manual_seed(100)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(100)
        
        # Deserialize the model
        new_model = JambaThreatModel(input_dim)
        new_model.load_state_dict(torch.load(buffer))
        new_model.eval()
        results["deserialization_successful"] = True
        
        # Test with fixed input
        torch.manual_seed(42)
        sample_input = torch.randn(4, input_dim)
        
        # Compare outputs
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = new_model(sample_input)
        
        # Check if outputs match
        results["outputs_match"] = torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)
        
        if not results["outputs_match"]:
            results["error"] = f"Outputs don't match. Max diff: {(output1 - output2).abs().max().item()}"
            logger.error(results["error"])
        else:
            logger.info("Model serialization test successful")
        
        return results["outputs_match"], results
    
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error testing model serialization: {e}")
        logger.error(traceback.format_exc())
        return False, results

def verify_runpod_endpoint(api_key=None, endpoint_id=None):
    """
    Verify that the RunPod endpoint is reachable and functioning.
    
    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        
    Returns:
        tuple: (bool, dict) - Success flag and verification results
    """
    results = {
        "endpoint_reachable": False,
        "health_check_successful": False,
        "error": None
    }
    
    # Use environment variables if not provided
    api_key = api_key or os.environ.get("RUNPOD_API_KEY")
    endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not api_key or not endpoint_id:
        results["error"] = "API key and endpoint ID are required"
        logger.error(results["error"])
        return False, results
    
    # Check if endpoint exists
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        health_url = f"https://api.runpod.io/v2/{endpoint_id}/health"
        
        logger.info(f"Checking endpoint health at {health_url}")
        response = requests.get(health_url, headers=headers)
        
        if response.status_code == 200:
            results["endpoint_reachable"] = True
            results["health_check_successful"] = True
            results["health_response"] = response.json()
            logger.info(f"Endpoint health check successful: {response.json()}")
        else:
            results["endpoint_reachable"] = True
            results["error"] = f"Endpoint health check failed: {response.status_code} - {response.text}"
            logger.error(results["error"])
        
        return results["health_check_successful"], results
    
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error checking endpoint health: {e}")
        return False, results

def run_health_check():
    """
    Run a comprehensive health check on the model and environment.
    
    Returns:
        dict: Health check results
    """
    health_results = {
        "timestamp": None,
        "environment_variables": None,
        "cuda_availability": None,
        "model_imports": None,
        "model_initialization": None,
        "model_serialization": None,
        "endpoint_verification": None,
        "status": "unhealthy",
        "issues": []
    }
    
    # Check environment variables
    env_ok, env_results = check_environment_variables()
    health_results["environment_variables"] = {
        "success": env_ok,
        "missing": env_results if not env_ok else []
    }
    
    if not env_ok:
        health_results["issues"].append(f"Missing environment variables: {', '.join(env_results)}")
    
    # Check CUDA availability
    health_results["cuda_availability"] = check_cuda_availability()
    
    # Check model imports
    import_ok, import_results = check_model_imports()
    health_results["model_imports"] = {
        "success": import_ok,
        "details": import_results
    }
    
    if not import_ok:
        health_results["issues"].append(f"Model import issues: {import_results['error']}")
    
    # Check model initialization
    init_ok, init_results = test_model_initialization()
    health_results["model_initialization"] = {
        "success": init_ok,
        "details": init_results
    }
    
    if not init_ok:
        health_results["issues"].append(f"Model initialization issues: {init_results['error']}")
    
    # Check model serialization
    serial_ok, serial_results = test_model_serialization()
    health_results["model_serialization"] = {
        "success": serial_ok,
        "details": serial_results
    }
    
    if not serial_ok:
        health_results["issues"].append(f"Model serialization issues: {serial_results['error']}")
    
    # Check endpoint if environment variables are set
    if env_ok:
        endpoint_ok, endpoint_results = verify_runpod_endpoint()
        health_results["endpoint_verification"] = {
            "success": endpoint_ok,
            "details": endpoint_results
        }
        
        if not endpoint_ok:
            health_results["issues"].append(f"Endpoint verification issues: {endpoint_results['error']}")
    
    # Determine overall status
    if not health_results["issues"]:
        health_results["status"] = "healthy"
    
    import datetime
    health_results["timestamp"] = datetime.datetime.now().isoformat()
    
    logger.info(f"Health check completed. Status: {health_results['status']}")
    if health_results["issues"]:
        logger.warning(f"Health check issues: {health_results['issues']}")
    
    return health_results

def generate_shell_validation_script():
    """
    Generate a shell script for validation that can be called from bash.
    
    Returns:
        str: Shell script content
    """
    script = """#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Running Jamba Threat Model validation..."

# Run Python validation script
python -c "
import sys
import os
import json

# Add paths
sys.path.append('/app')
sys.path.append('/app/src')
sys.path.append('.')
sys.path.append('./src')

try:
    from utils.validation import run_health_check
    
    # Run health check
    results = run_health_check()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Set exit code based on health status
    sys.exit(0 if results['status'] == 'healthy' else 1)
except Exception as e:
    print(f'Validation script error: {str(e)}')
    sys.exit(1)
"

# Store the exit code
validation_result=$?

if [ $validation_result -eq 0 ]; then
    log "✓ Validation completed successfully"
    exit 0
else
    log "✗ Validation failed"
    exit 1
fi
"""
    return script

if __name__ == "__main__":
    # When run directly, perform a health check
    results = run_health_check()
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results["status"] == "healthy" else 1) 