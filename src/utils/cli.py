"""
Command-line interface for Jamba Threat Detection model utilities.
Provides access to validation and environment functions.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

from src.utils import environment, validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_parser():
    """
    Create the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: The argument parser
    """
    parser = argparse.ArgumentParser(
        description="Jamba Threat Detection Model Utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Health check command
    health_parser = subparsers.add_parser(
        "health",
        help="Run a comprehensive health check on the model deployment"
    )
    health_parser.add_argument(
        "--skip-endpoint",
        action="store_true",
        help="Skip RunPod endpoint check"
    )
    
    # Environment command
    env_parser = subparsers.add_parser(
        "env",
        help="Initialize and validate the environment"
    )
    env_parser.add_argument(
        "--initialize",
        action="store_true",
        help="Initialize the environment before checking"
    )
    
    # Model command
    model_parser = subparsers.add_parser(
        "model",
        help="Test model functionality"
    )
    model_parser.add_argument(
        "--input-dim",
        type=int,
        default=validation.DEFAULT_INPUT_DIM,
        help="Input dimension for the model"
    )
    model_parser.add_argument(
        "--test",
        choices=["import", "init", "serialization", "all"],
        default="all",
        help="Type of model test to run"
    )
    
    # RunPod command
    runpod_parser = subparsers.add_parser(
        "runpod",
        help="Check RunPod endpoint"
    )
    runpod_parser.add_argument(
        "--api-key",
        help="RunPod API key (defaults to environment variable)"
    )
    runpod_parser.add_argument(
        "--endpoint-id",
        help="RunPod endpoint ID (defaults to environment variable)"
    )
    
    return parser

def run_health_command(args):
    """
    Run the health check command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        dict: Health check results
    """
    logger.info("Running health check command...")
    environment.initialize_environment()
    
    health_result = validation.run_health_check()
    
    if not args.json:
        print(validation.format_health_check_summary(health_result))
    
    return health_result

def run_env_command(args):
    """
    Run the environment command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        dict: Environment status
    """
    logger.info("Running environment command...")
    
    if args.initialize:
        environment.initialize_environment()
    
    env_status = environment.get_env_status()
    
    if not args.json:
        print("\n=== Environment Status ===")
        print(f"APP_DIR: {env_status['app_dir']}")
        print(f"MODEL_DIR: {env_status['model_dir']}")
        print(f"LOGS_DIR: {env_status['logs_dir']}")
        print(f"DATA_DIR: {env_status['data_dir']}")
        print(f"Debug Mode: {env_status['debug_mode']}")
        print(f"RUNPOD_API_KEY set: {env_status['runpod_api_key_set']}")
        print(f"RUNPOD_ENDPOINT_ID set: {env_status['runpod_endpoint_id_set']}")
        print(f"\nPython Version: {env_status['python_version']}")
    
    return env_status

def run_model_command(args):
    """
    Run the model command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        dict: Model test results
    """
    logger.info(f"Running model command with test={args.test}, input_dim={args.input_dim}...")
    environment.setup_python_path()
    
    results = {}
    
    if args.test in ["import", "all"]:
        results["imports"] = validation.check_model_imports()
        
        if not results["imports"]["model_class_available"] and args.test != "all":
            logger.error("Model class not available, cannot proceed with other tests")
            return results
    
    if args.test in ["init", "all"] and (args.test == "init" or results["imports"]["model_class_available"]):
        results["initialization"] = validation.test_model_initialization(args.input_dim)
    
    if args.test in ["serialization", "all"] and (args.test == "serialization" or results["imports"]["model_class_available"]):
        results["serialization"] = validation.test_model_serialization(args.input_dim)
    
    if not args.json:
        print("\n=== Model Test Results ===")
        
        if "imports" in results:
            status = "✓" if results["imports"]["model_class_available"] else "✗"
            print(f"Model Import: {status}")
        
        if "initialization" in results:
            init_status = "✓" if results["initialization"]["initialization_successful"] else "✗"
            print(f"Model Initialization: {init_status}")
            
            if results["initialization"]["forward_pass_successful"]:
                print(f"  Forward Pass: ✓ (Output shape: {results['initialization']['output_shape']})")
            else:
                print(f"  Forward Pass: ✗")
                if results["initialization"]["error"]:
                    print(f"  Error: {results['initialization']['error']}")
        
        if "serialization" in results:
            serial_status = "✓" if results["serialization"]["outputs_match"] else "✗"
            print(f"Model Serialization: {serial_status}")
            
            if not results["serialization"]["outputs_match"]:
                print(f"  Max Difference: {results['serialization']['max_diff']}")
                if results["serialization"]["error"]:
                    print(f"  Error: {results['serialization']['error']}")
    
    return results

def run_runpod_command(args):
    """
    Run the RunPod command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        dict: RunPod endpoint status
    """
    logger.info("Running RunPod endpoint check...")
    
    api_key = args.api_key or environment.get_runpod_api_key()
    endpoint_id = args.endpoint_id or environment.get_runpod_endpoint_id()
    
    if not api_key or not endpoint_id:
        logger.error("RunPod API key or endpoint ID missing")
        if not args.json:
            print("\n=== RunPod Endpoint Check ===")
            print("Error: RunPod API key or endpoint ID missing")
            print("Please provide them as command-line arguments or set environment variables")
        
        return {
            "error": "API key or endpoint ID missing",
            "api_key_set": api_key is not None,
            "endpoint_id_set": endpoint_id is not None
        }
    
    result = validation.check_runpod_endpoint(api_key, endpoint_id)
    
    if not args.json:
        print("\n=== RunPod Endpoint Check ===")
        status = "✓" if result["endpoint_reachable"] else "✗"
        print(f"Endpoint Reachable: {status}")
        
        health_status = "✓" if result["health_check_successful"] else "✗"
        print(f"Health Check: {health_status}")
        
        if result["error"]:
            print(f"Error: {result['error']}")
    
    return result

def main():
    """
    Main entry point for the command-line interface.
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the appropriate command
    result = None
    
    if args.command == "health":
        result = run_health_command(args)
    elif args.command == "env":
        result = run_env_command(args)
    elif args.command == "model":
        result = run_model_command(args)
    elif args.command == "runpod":
        result = run_runpod_command(args)
    else:
        parser.print_help()
        return 1
    
    # Output JSON if requested
    if args.json and result:
        print(json.dumps(result, indent=2))
    
    # Return appropriate exit code
    if not result:
        return 1
    
    if args.command == "health":
        return 0 if result["status"] == "healthy" else 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 