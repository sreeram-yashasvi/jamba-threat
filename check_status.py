import requests
import time
import sys
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check the status of a RunPod job")
    parser.add_argument("--job-id", required=True, help="RunPod job ID to check")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", help="RunPod API key (or set RUNPOD_API_KEY env var)")
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum number of status check attempts")
    parser.add_argument("--interval", type=int, default=2, help="Interval between status checks in seconds")
    
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        print("Please set it with: export RUNPOD_API_KEY='your_api_key' or use --api-key")
        sys.exit(1)
    
    job_id = args.job_id
    endpoint_id = args.endpoint_id
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print(f"Checking status of job {job_id} on endpoint {endpoint_id}")
    
    for i in range(args.max_attempts):
        try:
            response = requests.get(status_url, headers=headers)
            print(f"Attempt {i+1}/{args.max_attempts}, Status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                status = response_data.get("status")
                print(f"Job status: {status}")
                
                if status == "COMPLETED":
                    print("Job completed successfully!")
                    print(f"Output: {response_data.get('output')}")
                    sys.exit(0)
                elif status == "FAILED":
                    print(f"Job failed: {response_data.get('error')}")
                    sys.exit(1)
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    print(f"Job is {status}, waiting...")
                else:
                    print(f"Unknown status: {status}")
            else:
                print(f"Error response: {response.text}")
        except Exception as e:
            print(f"Error checking status: {e}")
        
        time.sleep(args.interval)
    
    print(f"Timed out after {args.max_attempts} attempts waiting for job completion")
    sys.exit(1)

if __name__ == "__main__":
    main()