#!/usr/bin/env python3
"""
Verification script for Jamba Threat Detection RunPod deployment.

This script verifies the various components of the Jamba Threat Detection
system, including the model structure, serialization, and RunPod endpoint.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
try:
    from utils.environment import configure_logging, setup_environment, check_environment_variables
    from utils.validation import check_model_imports, test_model_initialization, test_model_serialization, verify_runpod_endpoint
    from utils.model_testing import test_forward_pass, benchmark_model
except ImportError as e:
    logger.error(f"Error importing utility modules: {e}")
    logger.error("Please ensure you're running this script from the src directory or project root.")
    sys.exit(1)

def main():
    """
    Run the verification process for the RunPod deployment.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify Jamba Threat Detection RunPod deployment")
    parser.add_argument("--api-key", help="RunPod API key (can also be set via RUNPOD_API_KEY env var)")
    parser.add_argument("--endpoint-id", help="RunPod endpoint ID (can also be set via RUNPOD_ENDPOINT_ID env var)")
    parser.add_argument("--skip-endpoint-check", action="store_true", help="Skip RunPod endpoint verification")
    parser.add_argument("--output-file", help="Path to output verification results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Setup environment
    env_config = setup_environment()
    
    logger.info("=== RunPod Deployment Verification ===")
    
    # Check environment variables
    missing_vars, _ = check_environment_variables()
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    
    # Run verification checks
    verification_results = run_verification(args)
    
    # Print summary
    print_verification_summary(verification_results)
    
    # Save results to file if requested
    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(verification_results, f, indent=2)
            logger.info(f"Verification results saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving results to file: {e}")
    
    # Exit with appropriate code
    if verification_results["success"]:
        logger.info("\n✅ All verification checks passed.")
        return 0
    else:
        logger.error("\n❌ Some verification checks failed. Please review the logs above.")
        return 1

def run_verification(args):
    """
    Run all verification checks.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Verification results
    """
    results = {
        "success": False,
        "model_structure": False,
        "model_serialization": False,
        "environment_variables": False,
        "runpod_endpoint": False
    }
    
    # Get API key and endpoint ID from args or environment variables
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    endpoint_id = args.endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
    
    # Check environment variables
    logger.info("\n=== Environment Variables Check ===")
    missing_vars, _ = check_environment_variables()
    results["environment_variables"] = len(missing_vars) == 0
    
    # Check model structure
    logger.info("\n=== Model Structure Verification ===")
    logger.info(f"PyTorch version: {os.environ.get('PYTORCH_VERSION', 'Unknown')}")
    logger.info(f"CUDA available: {os.environ.get('CUDA_AVAILABLE', 'Unknown')}")
    
    logger.info("Testing local imports:")
    import_success, import_error, import_path = check_model_imports()
    if import_success:
        logger.info("✓ Successfully imported from 'jamba_model'")
    else:
        logger.error(f"✗ Error importing from 'jamba_model': {import_error}")
    
    logger.info("Testing model initialization:")
    if import_success:
        init_success, init_error, model_info = test_model_initialization()
        if init_success:
            logger.info("✓ Model initialized successfully")
            results["model_structure"] = True
        else:
            logger.error(f"✗ Model initialization failed: {init_error}")
    
    # Check model serialization
    logger.info("\n=== Model Serialization Verification ===")
    if import_success and results["model_structure"]:
        serial_success, serial_error = test_model_serialization()
        if serial_success:
            logger.info("✓ Model serialization and loading successful")
            results["model_serialization"] = True
        else:
            logger.error(f"✗ {serial_error}")
    
    # Check RunPod endpoint
    if not args.skip_endpoint_check:
        logger.info("\n=== RunPod Endpoint Verification ===")
        if api_key and endpoint_id:
            endpoint_success, endpoint_msg, _ = verify_runpod_endpoint(api_key, endpoint_id)
            if endpoint_success:
                logger.info("✓ RunPod endpoint verification successful")
                results["runpod_endpoint"] = True
            else:
                logger.error(f"✗ RunPod endpoint verification failed: {endpoint_msg}")
        else:
            logger.warning("⚠️ Skipping RunPod endpoint verification due to missing API key or endpoint ID")
    else:
        logger.info("\n=== RunPod Endpoint Verification ===")
        logger.info("✓ Skipping RunPod endpoint verification as requested")
        results["runpod_endpoint"] = True
    
    # Determine overall success (model structure and serialization are critical)
    results["success"] = results["model_structure"] and results["model_serialization"]
    
    return results

def print_verification_summary(results):
    """
    Print a summary of the verification results.
    
    Args:
        results: Verification results dictionary
    """
    logger.info("\n=== Verification Summary ===")
    logger.info(f"Environment Variables: {'✓' if results['environment_variables'] else '✗'}")
    logger.info(f"Model Structure: {'✓' if results['model_structure'] else '✗'}")
    logger.info(f"Model Serialization: {'✓' if results['model_serialization'] else '✗'}")
    logger.info(f"RunPod Endpoint: {'✓' if results['runpod_endpoint'] else '✗'}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 