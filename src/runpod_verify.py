#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import json
import torch
import io
import base64
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_model_structure():
    """Verify that model structures are consistent between local and RunPod environments."""
    
    # Log versions
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Check if jamba_model module exists and can be imported locally
        logger.info("Testing local imports:")
        try:
            # First try importing from jamba_model (container structure)
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import jamba_model
            from jamba_model import JambaThreatModel, ThreatDataset
            logger.info("✓ Successfully imported from 'jamba_model'")
            model_source = "jamba_model"
        except ImportError as e:
            logger.info(f"✗ Import from jamba_model failed: {e}")
            
            # Try importing directly from current directory
            try:
                from jamba_model import JambaThreatModel, ThreatDataset
                logger.info("✓ Successfully imported from current directory")
                model_source = "current_directory"
            except ImportError as e:
                logger.info(f"✗ Import from current directory failed: {e}")
                
                # Try importing from src directory
                try:
                    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
                    from jamba_model import JambaThreatModel, ThreatDataset
                    logger.info("✓ Successfully imported from src directory")
                    model_source = "src_directory"
                except ImportError as e:
                    logger.error(f"✗ All import attempts failed: {e}")
                    return False
        
        # Test model initialization
        try:
            logger.info("Testing model initialization:")
            model = JambaThreatModel(input_dim=28)
            logger.info("✓ Model initialized successfully")
            
            # Check model structure
            logger.info("Model structure overview:")
            for name, param in model.named_parameters():
                logger.info(f"  {name}: {param.shape}")
            
            # Verify the model's forward pass
            sample_input = torch.randn(4, 28)
            output = model(sample_input)
            logger.info(f"✓ Forward pass successful. Output shape: {output.shape}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Model initialization or forward pass failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def verify_runpod_endpoint(api_key, endpoint_id):
    """Verify that the RunPod endpoint is reachable and functioning."""
    
    if not api_key or not endpoint_id:
        logger.error("API key and endpoint ID are required")
        return False
    
    logger.info(f"Verifying RunPod endpoint {endpoint_id}")
    
    # Check if endpoint exists
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(
            f"https://api.runpod.io/v2/{endpoint_id}/health",
            headers=headers
        )
        
        if response.status_code == 200:
            logger.info("✓ Endpoint exists and is reachable")
        else:
            logger.error(f"✗ Endpoint health check failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"✗ Error connecting to RunPod API: {e}")
        return False
    
    # Submit a health check job
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input": {
                "operation": "health"
            }
        }
        response = requests.post(
            f"https://api.runpod.io/v2/{endpoint_id}/run",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            job_status = response.json()
            job_id = job_status.get("id")
            logger.info(f"✓ Health check job submitted: {job_id}")
            
            # Poll for job completion
            max_polls = 12
            poll_count = 0
            while poll_count < max_polls:
                poll_count += 1
                
                status_response = requests.get(
                    f"https://api.runpod.io/v2/{endpoint_id}/status/{job_id}",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    if status == "COMPLETED":
                        logger.info(f"✓ Health check job completed")
                        
                        # Parse job output
                        output = status_data.get("output")
                        if output:
                            logger.info("Health check results:")
                            if "system_info" in output:
                                system_info = output["system_info"]
                                logger.info(f"  Device: {system_info.get('device', 'unknown')}")
                                logger.info(f"  CUDA available: {system_info.get('cuda_available', 'unknown')}")
                                logger.info(f"  CUDA version: {system_info.get('cuda_version', 'unknown')}")
                                logger.info(f"  Python version: {system_info.get('python_version', 'unknown')}")
                                logger.info(f"  Model init success: {system_info.get('model_init_success', 'unknown')}")
                                
                                # Check for potential issues
                                env = system_info.get("environment", {})
                                if not env.get("model_directory_exists", True):
                                    logger.error("✗ Model directory does not exist in container")
                                    return False
                                
                                if not system_info.get("model_init_success", False):
                                    logger.error(f"✗ Model initialization failed: {system_info.get('model_init_error', 'Unknown error')}")
                                    return False
                            
                            return True
                        else:
                            logger.error("✗ No output from health check job")
                            return False
                    elif status == "FAILED":
                        logger.error(f"✗ Health check job failed: {status_data.get('error', 'Unknown error')}")
                        return False
                    else:
                        logger.info(f"Health check job status: {status}")
                        import time
                        time.sleep(5)
                else:
                    logger.error(f"✗ Error checking job status: {status_response.status_code} - {status_response.text}")
                    return False
            
            logger.error("✗ Timed out waiting for health check job completion")
            return False
        else:
            logger.error(f"✗ Error submitting health check job: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"✗ Error during health check: {e}")
        return False

def test_model_serialization():
    """Test model serialization to ensure compatibility between local and RunPod environments."""
    
    try:
        # Import model class
        from jamba_model import JambaThreatModel
        
        # Set deterministic mode for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create test model
        model = JambaThreatModel(input_dim=28)
        model.eval()  # Set to evaluation mode for reproducible results
        
        # Serialize model
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Set new seed to ensure we're testing proper serialization
        torch.manual_seed(100)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(100)
        
        # Load model back to verify
        state_dict = torch.load(buffer)
        
        # Create a new model and load state
        new_model = JambaThreatModel(input_dim=28)
        new_model.load_state_dict(state_dict)
        new_model.eval()  # Set to evaluation mode
        
        # Create fixed test input
        torch.manual_seed(42)
        sample_input = torch.randn(4, 28)
        
        # Test forward pass with no gradients
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = new_model(sample_input)
        
        # Compare outputs
        match = torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)
        if match:
            logger.info("✓ Model serialization and loading successful")
            return True
        else:
            logger.error(f"✗ Model outputs don't match after serialization: max diff = {(output1 - output2).abs().max().item()}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing model serialization: {e}")
        return False

def check_environment_variables():
    """Check for required environment variables."""
    
    missing_vars = []
    
    # Check for RunPod variables
    runpod_api_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_api_key:
        missing_vars.append("RUNPOD_API_KEY")
    
    runpod_endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not runpod_endpoint_id:
        missing_vars.append("RUNPOD_ENDPOINT_ID")
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        logger.info("✓ All required environment variables are set")
        return True

def main():
    parser = argparse.ArgumentParser(description='Verify RunPod deployment')
    parser.add_argument('--api-key', help='RunPod API key (or set RUNPOD_API_KEY env var)')
    parser.add_argument('--endpoint-id', help='RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)')
    parser.add_argument('--skip-model-check', action='store_true', help='Skip model structure check')
    parser.add_argument('--skip-endpoint-check', action='store_true', help='Skip endpoint check')
    parser.add_argument('--skip-serialization-check', action='store_true', help='Skip model serialization check')
    
    args = parser.parse_args()
    
    # Get API key and endpoint ID
    api_key = args.api_key or os.environ.get('RUNPOD_API_KEY')
    endpoint_id = args.endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID')
    
    # Run verification checks
    logger.info("=== RunPod Deployment Verification ===")
    
    # Check environment variables
    env_check = check_environment_variables()
    
    # Check model structure
    model_check = True
    if not args.skip_model_check:
        logger.info("\n=== Model Structure Verification ===")
        model_check = verify_model_structure()
    
    # Check model serialization
    serialization_check = True
    if not args.skip_serialization_check:
        logger.info("\n=== Model Serialization Verification ===")
        serialization_check = test_model_serialization()
    
    # Check RunPod endpoint
    endpoint_check = True
    if not args.skip_endpoint_check and api_key and endpoint_id:
        logger.info("\n=== RunPod Endpoint Verification ===")
        endpoint_check = verify_runpod_endpoint(api_key, endpoint_id)
    
    # Summary
    logger.info("\n=== Verification Summary ===")
    logger.info(f"Environment Variables: {'✓' if env_check else '✗'}")
    logger.info(f"Model Structure: {'✓' if model_check else '✗'}")
    logger.info(f"Model Serialization: {'✓' if serialization_check else '✗'}")
    logger.info(f"RunPod Endpoint: {'✓' if endpoint_check else '✗'}")
    
    if env_check and model_check and serialization_check and endpoint_check:
        logger.info("\n✅ All checks passed! The RunPod deployment is correctly configured.")
        return 0
    else:
        logger.error("\n❌ Some verification checks failed. Please review the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 