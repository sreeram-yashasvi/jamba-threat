import requests
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_runpod_connection(api_key, endpoint_id):
    """Test RunPod connection with different URL formats."""
    
    # Try different URL formats
    url_formats = [
        f"https://api.runpod.io/v2/ub3lew01pzj80j/run",
        f"https://api.runpod.io/v2/ub3lew01pzj80j/runsync",
        f"https://api.runpod.io/v1/ub3lew01pzj80j/run",
        f"https://api.runpod.ai/v2/ub3lew01pzj80j/run"
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple test input
    input_data = {
        "input": {
            "operation": "test"
        }
    }
    
    for url in url_formats:
        logger.info(f"Testing URL: {url}")
        try:
            # First just check if the endpoint exists
            head_response = requests.head(url, headers=headers, timeout=10)
            logger.info(f"HEAD response: {head_response.status_code}")
            
            # Try a simple POST request
            response = requests.post(url, headers=headers, json=input_data, timeout=10)
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response: {response.text}")
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully connected to RunPod with URL: {url}")
                return url
                
        except Exception as e:
            logger.error(f"❌ Error connecting to {url}: {str(e)}")
    
    logger.error("Failed to connect to RunPod with any URL format.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RunPod serverless endpoint connection")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    
    args = parser.parse_args()
    
    working_url = test_runpod_connection(args.api_key, args.endpoint_id)
    
    if working_url:
        logger.info(f"Use this URL format in your RunPodClient: {working_url}")
    else:
        logger.info("Please verify your endpoint ID and API key.") 