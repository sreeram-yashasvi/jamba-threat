#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_endpoint_health(api_key, endpoint_id, max_retries=5):
    """Check the health of a RunPod endpoint.
    
    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        max_retries: Maximum number of retries
        
    Returns:
        dict: Health check results
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    payload = {
        "input": {
            "operation": "health_check"
        }
    }
    
    logger.info(f"Sending health check request to endpoint {endpoint_id}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                job_id = response.json().get("id")
                logger.info(f"Health check request submitted. Job ID: {job_id}")
                
                # Poll for results
                status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                for _ in range(30):  # Wait up to 30 seconds for result
                    time.sleep(1)
                    status_response = requests.get(status_url, headers=headers)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("status") == "COMPLETED":
                            logger.info("Health check completed successfully")
                            return {
                                "status": "success",
                                "endpoint_id": endpoint_id,
                                "response": status_data.get("output", {})
                            }
                        elif status_data.get("status") == "FAILED":
                            logger.error(f"Health check failed: {status_data}")
                            return {
                                "status": "failed",
                                "endpoint_id": endpoint_id,
                                "error": status_data.get("error", "Unknown error")
                            }
                
                logger.warning("Health check timed out waiting for result")
                return {
                    "status": "timeout",
                    "endpoint_id": endpoint_id
                }
            else:
                logger.warning(f"Request failed with status code {response.status_code}: {response.text}")
                
                # Wait before retrying
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except Exception as e:
            logger.error(f"Error checking endpoint health: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
            
    return {
        "status": "error",
        "endpoint_id": endpoint_id,
        "error": "Max retries exceeded"
    }

def main():
    parser = argparse.ArgumentParser(description='Check the health of a RunPod endpoint')
    parser.add_argument('--api-key', help='RunPod API key (or set RUNPOD_API_KEY env var)')
    parser.add_argument('--endpoint-id', help='RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)')
    parser.add_argument('--retries', type=int, default=5, help='Maximum number of retries')
    parser.add_argument('--output', help='Output file for health check results (optional)')
    
    args = parser.parse_args()
    
    # Get API key and endpoint ID from environment variables if not provided
    api_key = args.api_key or os.environ.get('RUNPOD_API_KEY')
    endpoint_id = args.endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID')
    
    if not api_key:
        logger.error("API key must be provided via --api-key or RUNPOD_API_KEY environment variable")
        return 1
    
    if not endpoint_id:
        logger.error("Endpoint ID must be provided via --endpoint-id or RUNPOD_ENDPOINT_ID environment variable")
        return 1
    
    # Perform health check
    results = check_endpoint_health(api_key, endpoint_id, args.retries)
    
    # Print results
    logger.info(f"Health check results: {json.dumps(results, indent=2)}")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Return exit code based on health check status
    if results["status"] == "success":
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main()) 