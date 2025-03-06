#!/usr/bin/env python
"""
RunPod Health Check Script for Jamba Threat Detection

This script checks if a RunPod endpoint for the Jamba Threat Detection model
is properly configured and operational.

Usage:
    python runpod_health_check.py [--api-key YOUR_API_KEY] [--endpoint-id YOUR_ENDPOINT_ID]

Environment Variables:
    RUNPOD_API_KEY: Your RunPod API key
    RUNPOD_ENDPOINT_ID: Your RunPod endpoint ID
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("health-check")

# Ensure we can find our modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

try:
    from utils import environment, validation
    utils_available = True
except ImportError:
    logger.warning("Could not import utility modules. Running in standalone mode.")
    utils_available = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RunPod Health Check for Jamba Threat Detection")
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--endpoint-id", help="RunPod endpoint ID")
    parser.add_argument("--skip-model-checks", action="store_true", 
                        help="Skip model structure and serialization checks")
    parser.add_argument("--skip-endpoint-checks", action="store_true",
                        help="Skip RunPod endpoint checks")
    return parser.parse_args()

def standalone_check_environment_variables(api_key=None, endpoint_id=None):
    """Check if required environment variables are set."""
    if not utils_available:
        # Standalone implementation
        env_vars = {}
        env_vars["RUNPOD_API_KEY"] = api_key or os.environ.get("RUNPOD_API_KEY")
        env_vars["RUNPOD_ENDPOINT_ID"] = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        
        missing_vars = [var for var, value in env_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            return False, missing_vars
        
        return True, None
    else:
        # Use centralized validation
        return validation.check_environment_variables(
            required_vars=["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"],
            override_values={
                "RUNPOD_API_KEY": api_key,
                "RUNPOD_ENDPOINT_ID": endpoint_id
            }
        )

def main():
    """Run the health check for the RunPod endpoint."""
    args = parse_args()
    
    logger.info("Starting Jamba Threat Detection RunPod health check")
    results = {"success": True, "checks": {}}
    
    # Setup environment if available
    if utils_available:
        environment.setup_environment()
        logger.info("Environment setup complete")
    
    # Check environment variables
    api_key = args.api_key
    endpoint_id = args.endpoint_id
    
    env_check, missing_vars = standalone_check_environment_variables(api_key, endpoint_id)
    results["checks"]["environment_variables"] = {
        "success": env_check,
        "missing_vars": missing_vars if not env_check else None
    }
    
    if not env_check:
        logger.error("Environment variable check failed")
        if not api_key:
            logger.error("API_KEY not provided. Use --api-key or set RUNPOD_API_KEY environment variable")
        if not endpoint_id:
            logger.error("ENDPOINT_ID not provided. Use --endpoint-id or set RUNPOD_ENDPOINT_ID environment variable")
        results["success"] = False
    
    # Use the API key and endpoint ID from arguments or environment variables
    api_key = api_key or os.environ.get("RUNPOD_API_KEY")
    endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
    
    # Run checks
    if not args.skip_model_checks:
        logger.info("Running model structure checks")
        if utils_available:
            model_check = validation.check_model_structure()
            results["checks"]["model_structure"] = model_check
            
            if not model_check.get("success", False):
                logger.error(f"Model structure check failed: {model_check.get('error')}")
                results["success"] = False
            else:
                logger.info("Model structure check passed")
                
            logger.info("Running model serialization checks")
            serialization_check = validation.test_model_serialization()
            results["checks"]["model_serialization"] = serialization_check
            
            if not serialization_check.get("success", False):
                logger.error(f"Model serialization check failed: {serialization_check.get('error')}")
                results["success"] = False
            else:
                logger.info("Model serialization check passed")
        else:
            logger.warning("Skipping model checks - utility modules not available")
            results["checks"]["model_checks_skipped"] = True
    else:
        logger.info("Skipping model checks as requested")
        results["checks"]["model_checks_skipped"] = True
    
    # Check if RunPod endpoint is working
    if not args.skip_endpoint_checks and api_key and endpoint_id:
        logger.info(f"Checking RunPod endpoint {endpoint_id}")
        if utils_available:
            endpoint_check = validation.verify_runpod_endpoint(api_key, endpoint_id)
            results["checks"]["endpoint"] = endpoint_check
            
            if not endpoint_check.get("success", False):
                logger.error(f"Endpoint check failed: {endpoint_check.get('error')}")
                results["success"] = False
            else:
                logger.info("Endpoint check passed")
        else:
            # Import RunPod client on-demand
            try:
                from runpod_client import RunPodClient
                
                # Test endpoint health with simple health check request
                logger.info("Testing endpoint with health check request")
                client = RunPodClient(api_key=api_key, endpoint_id=endpoint_id)
                
                start_time = time.time()
                response = client.health_check()
                duration = time.time() - start_time
                
                if response.get("success", False):
                    logger.info(f"Health check successful (took {duration:.2f}s)")
                    results["checks"]["endpoint"] = {
                        "success": True,
                        "response_time": f"{duration:.2f}s",
                        "details": response
                    }
                else:
                    logger.error(f"Health check failed: {response}")
                    results["checks"]["endpoint"] = {
                        "success": False,
                        "error": str(response),
                        "response_time": f"{duration:.2f}s"
                    }
                    results["success"] = False
            except Exception as e:
                logger.error(f"Error checking endpoint: {e}")
                results["checks"]["endpoint"] = {
                    "success": False,
                    "error": str(e)
                }
                results["success"] = False
    else:
        if args.skip_endpoint_checks:
            logger.info("Skipping endpoint checks as requested")
        else:
            logger.warning("Skipping endpoint checks - API key or endpoint ID not available")
        results["checks"]["endpoint_checks_skipped"] = True
    
    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("HEALTH CHECK SUMMARY")
    logger.info("=" * 40)
    
    for check_name, check_result in results["checks"].items():
        if isinstance(check_result, dict) and "success" in check_result:
            status = "✅ PASS" if check_result["success"] else "❌ FAIL"
            logger.info(f"{check_name}: {status}")
        elif check_name.endswith("_skipped") and check_result:
            logger.info(f"{check_name.replace('_skipped', '')}: ⏭️ SKIPPED")
    
    logger.info("=" * 40)
    logger.info(f"OVERALL RESULT: {'✅ PASS' if results['success'] else '❌ FAIL'}")
    logger.info("=" * 40)
    
    # Exit with appropriate status code
    return 0 if results["success"] else 1

if __name__ == "__main__":
    sys.exit(main()) 