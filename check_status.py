import requests
import time
import sys
import os

# Get API key from environment variable
api_key = os.environ.get("RUNPOD_API_KEY")
if not api_key:
    print("Error: RUNPOD_API_KEY environment variable not set")
    print("Please set it with: export RUNPOD_API_KEY='your_api_key'")
    sys.exit(1)

job_id = "cb67e2d0-8132-4b29-8f42-7dc946835622-e1"
status_url = f"https://api.runpod.ai/v2/gzy7xg5d8xau03/status/{job_id}"
headers = {"Authorization": f"Bearer {api_key}"}

for i in range(10):
    response = requests.get(status_url, headers=headers)
    print(f"Attempt {i+1}, Status: {response.status_code}, Response: {response.text}")
    
    if "COMPLETED" in response.text:
        print("Job completed successfully!")
        sys.exit(0)
    
    time.sleep(2)

print("Timed out waiting for job completion") 