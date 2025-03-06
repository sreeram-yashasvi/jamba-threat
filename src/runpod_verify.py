#!/usr/bin/env python3
"""
Verification tool for Jamba Threat Detection model deployment on RunPod.

This script verifies:
1. Model structure compatibility
2. Model serialization consistency
3. Environment variable configuration
4. RunPod endpoint connectivity

Usage:
    python src/runpod_verify.py [options]

Options:
    --skip-endpoint-check    Skip RunPod endpoint check
    --api-key KEY            RunPod API key (defaults to environment variable)
    --endpoint-id ID         RunPod endpoint ID (defaults to environment variable)
    --json                   Output results as JSON
    --verbose, -v            Enable verbose logging
"""

import os
import sys
import argparse
import logging
import json
import time

# Add the project root to the path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.utils import environment, validation
except ImportError:
    print("Error: Could not import utility modules.")
    print("Make sure you're running this script from the project root directory.")
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
        description="Verify the Jamba Threat Detection model deployment on RunPod"
    )
    
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help="Skip RunPod endpoint check"
    )
    
    parser.add_argument(
        "--api-key",
        help="RunPod API key (defaults to environment variable)"
    )
    
    parser.add_argument(
        "--endpoint-id",
        help="RunPod endpoint ID (defaults to environment variable)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def run_verification(args):
    """
    Run the verification process.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        dict: Verification results
    """
    logger.info("=== RunPod Deployment Verification ===")
    
    # Initialize environment
    environment.initialize_environment()
    
    # Check environment variables
    env_result = validation.check_environment_variables()
    if not env_result["all_vars_set"]:
        logger.warning(f"Missing environment variables: {', '.join(env_result['missing_vars'])}")
    
    # Check model structure
    logger.info("\n=== Model Structure Verification ===")
    model_imports = validation.check_model_imports()
    if not model_imports["model_class_available"]:
        logger.error("Model structure verification failed: Could not import model classes")
        model_init = {"initialization_successful": False, "error": "Model import failed"}
    else:
        model_init = validation.test_model_initialization()
    
    # Check model serialization
    logger.info("\n=== Model Serialization Verification ===")
    if not model_imports["model_class_available"]:
        logger.error("Model serialization verification skipped: Could not import model classes")
        model_serial = {"serialization_successful": False, "error": "Model import failed"}
    elif not model_init["initialization_successful"]:
        logger.error("Model serialization verification skipped: Model initialization failed")
        model_serial = {"serialization_successful": False, "error": "Model initialization failed"}
    else:
        model_serial = validation.test_model_serialization()
    
    # Check RunPod endpoint if not skipped
    runpod_result = None
    if not args.skip_endpoint_check:
        logger.info("\n=== RunPod Endpoint Verification ===")
        api_key = args.api_key or environment.get_runpod_api_key()
        endpoint_id = args.endpoint_id or environment.get_runpod_endpoint_id()
        
        if not api_key or not endpoint_id:
            logger.error("RunPod endpoint verification skipped: API key or endpoint ID missing")
            runpod_result = {"endpoint_reachable": False, "error": "API key or endpoint ID missing"}
        else:
            runpod_result = validation.check_runpod_endpoint(api_key, endpoint_id)
    else:
        logger.info("\n=== RunPod Endpoint Verification Skipped ===")
        runpod_result = {"endpoint_reachable": True, "health_check_successful": True, "skipped": True}
    
    # Compile results
    results = {
        "timestamp": time.time(),
        "environment_variables": {
            "status": "passed" if env_result["all_vars_set"] else "failed",
            "details": env_result
        },
        "model_structure": {
            "status": "passed" if model_init["initialization_successful"] and model_init["forward_pass_successful"] else "failed",
            "details": {
                "imports": model_imports,
                "initialization": model_init
            }
        },
        "model_serialization": {
            "status": "passed" if model_serial["serialization_successful"] and model_serial.get("outputs_match", False) else "failed",
            "details": model_serial
        },
        "runpod_endpoint": {
            "status": "passed" if runpod_result["endpoint_reachable"] and runpod_result.get("health_check_successful", False) else ("skipped" if args.skip_endpoint_check else "failed"),
            "details": runpod_result
        }
    }
    
    # Log summary
    logger.info("\n=== Verification Summary ===")
    logger.info(f"Environment Variables: {'✓' if results['environment_variables']['status'] == 'passed' else '✗'}")
    logger.info(f"Model Structure: {'✓' if results['model_structure']['status'] == 'passed' else '✗'}")
    logger.info(f"Model Serialization: {'✓' if results['model_serialization']['status'] == 'passed' else '✗'}")
    logger.info(f"RunPod Endpoint: {'✓' if results['runpod_endpoint']['status'] == 'passed' else ('✓' if args.skip_endpoint_check else '✗')}")
    
    # Determine overall status
    all_passed = (
        results["environment_variables"]["status"] == "passed" and
        results["model_structure"]["status"] == "passed" and
        results["model_serialization"]["status"] == "passed" and
        (results["runpod_endpoint"]["status"] == "passed" or results["runpod_endpoint"]["status"] == "skipped")
    )
    
    if all_passed:
        logger.info("\n✅ All verification checks passed.")
    else:
        logger.error("\n❌ Some verification checks failed. Please review the logs above.")
    
    return results, all_passed

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the verification
    results, all_passed = run_verification(args)
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Return 0 if all checks passed, 1 otherwise
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 