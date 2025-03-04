import requests
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_runpod_connection(api_key, endpoint_id):
    """Test connection to RunPod endpoint.
    
    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
    """
    logger.info(f"Testing connection to RunPod endpoint: {endpoint_id}")
    
    # Check if endpoint exists (GET request)
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        response = requests.get(status_url, headers=headers)
        logger.info(f"Health check status code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Endpoint is active and responding!")
        elif response.status_code == 404:
            logger.error("❌ Endpoint not found. Verify your endpoint ID.")
        elif response.status_code == 401:
            logger.error("❌ Authentication failed. Verify your API key.")
        else:
            logger.error(f"❌ Unknown error. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error connecting to RunPod: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test RunPod connection')
    parser.add_argument('--api-key', required=True, help='RunPod API key')
    parser.add_argument('--endpoint-id', required=True, help='RunPod endpoint ID')
    
    args = parser.parse_args()
    
    test_runpod_connection(args.api_key, args.endpoint_id)

if __name__ == "__main__":
    main() 