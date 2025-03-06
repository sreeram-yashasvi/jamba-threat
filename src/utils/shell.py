#!/usr/bin/env python3
"""
Shell interface for Jamba Threat Detection model utilities.
Provides command-line access to validation functions with simplified output.
Used primarily by shell scripts for easier integration.
"""

import os
import sys
import argparse
import logging
import json
import traceback

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utility modules
try:
    from src.utils import environment, validation
except ImportError:
    print("Error: Could not import utility modules.")
    print("Make sure you're running this script from the project root or src directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Shell interface for Jamba Threat Detection model utilities"
    )
    
    # Add main command argument
    parser.add_argument(
        "command",
        choices=["check-env", "check-model", "check-cuda", "check-runpod", "health"],
        help="Command to run"
    )
    
    # Add optional arguments
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except the final result"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    # Command-specific arguments
    parser.add_argument(
        "--api-key",
        help="RunPod API key (for check-runpod command)"
    )
    
    parser.add_argument(
        "--endpoint-id",
        help="RunPod endpoint ID (for check-runpod command)"
    )
    
    parser.add_argument(
        "--input-dim",
        type=int,
        default=validation.DEFAULT_INPUT_DIM,
        help="Input dimension for the model (for check-model command)"
    )
    
    return parser.parse_args()

def run_check_env(args):
    """Run environment check and return exit code."""
    try:
        # Initialize environment
        environment.initialize_environment()
        
        # Check environment variables
        result = validation.check_environment_variables()
        
        if args.json:
            print(json.dumps(result, indent=2))
        elif not args.quiet:
            missing = len(result["missing_vars"])
            if missing == 0:
                print("✓ All required environment variables are set")
            else:
                print(f"✗ Missing {missing} environment variable(s):")
                for var in result["missing_vars"]:
                    print(f"  - {var}")
        
        # Return 0 if all variables are set, otherwise number of missing variables
        return 0 if result["all_vars_set"] else len(result["missing_vars"])
        
    except Exception as e:
        if not args.quiet:
            print(f"Error checking environment: {e}")
            print(traceback.format_exc())
        return 1

def run_check_model(args):
    """Run model check and return exit code."""
    try:
        # Set up environment
        environment.setup_python_path()
        
        # Check model imports
        import_result = validation.check_model_imports()
        if not import_result["model_class_available"]:
            if not args.quiet:
                print("✗ Failed to import model classes")
                if import_result["error"]:
                    print(f"  Error: {import_result['error']}")
            return 1
        
        # Check model initialization
        init_result = validation.test_model_initialization(args.input_dim)
        
        # Check model serialization if initialization succeeded
        serial_result = None
        if init_result["initialization_successful"]:
            serial_result = validation.test_model_serialization(args.input_dim)
        
        # Prepare combined result
        result = {
            "imports": import_result,
            "initialization": init_result,
            "serialization": serial_result
        }
        
        if args.json:
            print(json.dumps(result, indent=2))
        elif not args.quiet:
            # Print import status
            import_status = "✓" if import_result["model_class_available"] else "✗"
            print(f"Model Import: {import_status}")
            
            # Print initialization status
            init_status = "✓" if init_result["initialization_successful"] else "✗"
            print(f"Model Initialization: {init_status}")
            
            if init_result["forward_pass_successful"]:
                print(f"Forward Pass: ✓")
            elif init_result["initialization_successful"]:
                print(f"Forward Pass: ✗")
                if init_result["error"]:
                    print(f"  Error: {init_result['error']}")
            
            # Print serialization status if tested
            if serial_result:
                serial_status = "✓" if serial_result["outputs_match"] else "✗"
                print(f"Model Serialization: {serial_status}")
                if not serial_result["outputs_match"] and serial_result["max_diff"] is not None:
                    print(f"  Max Difference: {serial_result['max_diff']}")
        
        # Success if all checks passed
        if (import_result["model_class_available"] and 
            init_result["initialization_successful"] and 
            init_result["forward_pass_successful"] and
            (serial_result is None or serial_result["outputs_match"])):
            return 0
        else:
            return 1
        
    except Exception as e:
        if not args.quiet:
            print(f"Error checking model: {e}")
            print(traceback.format_exc())
        return 1

def run_check_cuda(args):
    """Run CUDA check and return exit code."""
    try:
        # Check CUDA availability
        result = validation.check_cuda_availability()
        
        if args.json:
            print(json.dumps(result, indent=2))
        elif not args.quiet:
            if result["cuda_available"]:
                print(f"✓ CUDA is available (version {result['cuda_version']})")
                print(f"  GPU Count: {result['device_count']}")
                for i, device in enumerate(result["devices"]):
                    print(f"  GPU {i}: {device['name']} ({device['total_memory_mb']} MB)")
            else:
                print("⚠️ CUDA is not available, running in CPU mode")
                print(f"  PyTorch version: {result['pytorch_version']}")
        
        # Return 0 regardless of CUDA availability (not critical)
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"Error checking CUDA: {e}")
            print(traceback.format_exc())
        return 1

def run_check_runpod(args):
    """Run RunPod endpoint check and return exit code."""
    try:
        # Get API key and endpoint ID
        api_key = args.api_key or environment.get_runpod_api_key()
        endpoint_id = args.endpoint_id or environment.get_runpod_endpoint_id()
        
        if not api_key or not endpoint_id:
            if not args.quiet:
                print("✗ RunPod API key or endpoint ID is missing")
                if not api_key:
                    print("  - RUNPOD_API_KEY is not set")
                if not endpoint_id:
                    print("  - RUNPOD_ENDPOINT_ID is not set")
            return 1
        
        # Check RunPod endpoint
        result = validation.check_runpod_endpoint(api_key, endpoint_id)
        
        if args.json:
            print(json.dumps(result, indent=2))
        elif not args.quiet:
            if result["endpoint_reachable"]:
                print("✓ RunPod endpoint is reachable")
                
                if result["health_check_successful"]:
                    print("✓ Health check successful")
                else:
                    print("✗ Health check failed")
                    if result["error"]:
                        print(f"  Error: {result['error']}")
            else:
                print("✗ RunPod endpoint is not reachable")
                if result["error"]:
                    print(f"  Error: {result['error']}")
        
        # Return 0 if endpoint is reachable and health check passed
        if result["endpoint_reachable"] and result["health_check_successful"]:
            return 0
        else:
            return 1
        
    except Exception as e:
        if not args.quiet:
            print(f"Error checking RunPod endpoint: {e}")
            print(traceback.format_exc())
        return 1

def run_health_check(args):
    """Run comprehensive health check and return exit code."""
    try:
        # Initialize environment
        environment.initialize_environment()
        
        # Run health check
        result = validation.run_health_check()
        
        if args.json:
            print(json.dumps(result, indent=2))
        elif not args.quiet:
            print(validation.format_health_check_summary(result))
        
        # Return 0 if healthy, 1 otherwise
        return 0 if result["status"] == "healthy" else 1
        
    except Exception as e:
        if not args.quiet:
            print(f"Error running health check: {e}")
            print(traceback.format_exc())
        return 1

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Run the appropriate command
    if args.command == "check-env":
        return run_check_env(args)
    elif args.command == "check-model":
        return run_check_model(args)
    elif args.command == "check-cuda":
        return run_check_cuda(args)
    elif args.command == "check-runpod":
        return run_check_runpod(args)
    elif args.command == "health":
        return run_health_check(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 