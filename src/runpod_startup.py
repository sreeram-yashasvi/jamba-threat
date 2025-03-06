#!/usr/bin/env python3
"""
RunPod Startup Script for Jamba Threat Detection System

This script performs all necessary startup checks and configuration
before the RunPod handler is started. It ensures that the environment
is properly set up, the model can be loaded, and the system is ready
to handle requests.
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import utility modules
try:
    from utils.environment import setup_environment, check_environment_variables, configure_logging
    from utils.validation import check_model_imports, test_model_initialization, check_cuda_availability
    from utils.model_testing import test_forward_pass
    utils_available = True
    logger.info("Successfully imported utility modules")
except ImportError as e:
    logger.error(f"Error importing utility modules: {e}")
    utils_available = False

def setup():
    """
    Set up the environment for the RunPod handler.
    
    Returns:
        dict: Configuration information
    """
    logger.info("Starting RunPod startup checks")
    
    config = {
        "startup_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {},
        "passed": True
    }
    
    # Set up environment
    if utils_available:
        try:
            env_config = setup_environment(create_dirs=True)
            config["env_config"] = env_config
            config["checks"]["environment_setup"] = {"passed": True}
            logger.info("Environment setup successful")
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            config["checks"]["environment_setup"] = {"passed": False, "error": str(e)}
            config["passed"] = False
    else:
        # Manual environment setup if utils not available
        logger.info("Setting up environment manually")
        try:
            # Create necessary directories
            os.makedirs("models", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("data", exist_ok=True)
            
            # Add current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            config["checks"]["environment_setup"] = {"passed": True}
        except Exception as e:
            logger.error(f"Error in manual environment setup: {e}")
            config["checks"]["environment_setup"] = {"passed": False, "error": str(e)}
            config["passed"] = False
    
    # Check environment variables
    if utils_available:
        missing_vars, available_vars = check_environment_variables()
        if not missing_vars:
            logger.info("All required environment variables are set")
            config["checks"]["environment_variables"] = {"passed": True}
        else:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            config["checks"]["environment_variables"] = {
                "passed": False,
                "missing": missing_vars,
                "available": list(available_vars.keys())
            }
            config["passed"] = False
    else:
        # Manual environment variable check
        required_vars = ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if not missing:
            logger.info("All required environment variables are set")
            config["checks"]["environment_variables"] = {"passed": True}
        else:
            logger.warning(f"Missing environment variables: {', '.join(missing)}")
            config["checks"]["environment_variables"] = {
                "passed": False,
                "missing": missing
            }
            config["passed"] = False
    
    # Check CUDA availability
    if utils_available:
        cuda_available, cuda_info = check_cuda_availability()
        config["checks"]["cuda"] = {
            "passed": True,
            "available": cuda_available,
            "info": cuda_info
        }
        logger.info(f"CUDA available: {cuda_available}")
    else:
        # Manual CUDA check
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        config["checks"]["cuda"] = {
            "passed": True,
            "available": cuda_available,
            "device_count": device_count
        }
        logger.info(f"CUDA available: {cuda_available}, Device count: {device_count}")
    
    # Check model imports
    if utils_available:
        import_success, import_error, import_path = check_model_imports()
        if import_success:
            logger.info(f"Successfully imported model from {import_path}")
            config["checks"]["model_imports"] = {"passed": True, "path": import_path}
        else:
            logger.error(f"Failed to import model: {import_error}")
            config["checks"]["model_imports"] = {"passed": False, "error": import_error}
            config["passed"] = False
    else:
        # Manual model import check
        try:
            # Try different import paths
            try:
                from jamba_model import JambaThreatModel, ThreatDataset
                import_path = "jamba_model"
            except ImportError:
                try:
                    from src.jamba_model import JambaThreatModel, ThreatDataset
                    import_path = "src.jamba_model"
                except ImportError:
                    try:
                        from jamba_model.model import JambaThreatModel, ThreatDataset
                        import_path = "jamba_model.model"
                    except ImportError:
                        raise ImportError("Could not import model from any known path")
            
            logger.info(f"Successfully imported model from {import_path}")
            config["checks"]["model_imports"] = {"passed": True, "path": import_path}
        except Exception as e:
            logger.error(f"Failed to import model: {e}")
            config["checks"]["model_imports"] = {"passed": False, "error": str(e)}
            config["passed"] = False
    
    # Test model initialization
    if config["checks"].get("model_imports", {}).get("passed", False):
        if utils_available:
            init_success, init_error, model_info = test_model_initialization()
            if init_success:
                logger.info("Model initialization successful")
                config["checks"]["model_initialization"] = {"passed": True, "info": model_info}
            else:
                logger.error(f"Model initialization failed: {init_error}")
                config["checks"]["model_initialization"] = {"passed": False, "error": init_error}
                config["passed"] = False
        else:
            # Manual model initialization test
            try:
                # Get the model classes based on the successful import path
                import_path = config["checks"]["model_imports"]["path"]
                if import_path == "jamba_model":
                    from jamba_model import JambaThreatModel
                elif import_path == "src.jamba_model":
                    from src.jamba_model import JambaThreatModel
                elif import_path == "jamba_model.model":
                    from jamba_model.model import JambaThreatModel
                
                # Initialize the model
                model = JambaThreatModel(input_dim=15)
                
                # Test a forward pass
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                
                # Create a test input
                test_input = torch.randn(1, 15).to(device)
                
                # Forward pass
                with torch.no_grad():
                    output = model(test_input)
                
                logger.info(f"Model initialization and forward pass successful, output shape: {output.shape}")
                config["checks"]["model_initialization"] = {"passed": True}
            except Exception as e:
                logger.error(f"Model initialization failed: {e}")
                config["checks"]["model_initialization"] = {"passed": False, "error": str(e)}
                config["passed"] = False
    
    return config

def print_summary(config):
    """
    Print a summary of the startup checks.
    
    Args:
        config: Configuration information from setup
    """
    logger.info("\n" + "=" * 50)
    logger.info("RUNPOD STARTUP SUMMARY")
    logger.info("=" * 50)
    
    for check_name, check_result in config["checks"].items():
        status = "PASSED" if check_result.get("passed", False) else "FAILED"
        error = f" - {check_result.get('error', '')}" if not check_result.get("passed", False) and "error" in check_result else ""
        logger.info(f"{check_name}: {status}{error}")
    
    logger.info("-" * 50)
    overall = "PASSED" if config["passed"] else "FAILED"
    logger.info(f"OVERALL RESULT: {overall}")
    logger.info("=" * 50)

def main():
    """
    Main function to run the startup checks.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="RunPod Startup Script for Jamba Threat Detection")
    parser.add_argument("--output", help="Path to save the startup report as JSON")
    parser.add_argument("--fail-fast", action="store_true", help="Exit immediately on first check failure")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run startup checks
    config = setup()
    
    # Print summary
    print_summary(config)
    
    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Startup report saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving report to file: {e}")
    
    # Return appropriate exit code
    return 0 if config["passed"] else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 