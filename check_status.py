import requests
import time
import sys

job_id = "cb67e2d0-8132-4b29-8f42-7dc946835622-e1"
status_url = f"https://api.runpod.ai/v2/gzy7xg5d8xau03/status/{job_id}"
headers = {"Authorization": "Bearer rpa_MJSZ50MNMRV3SWE886WEJX20N4F0CF9RM7ETE2YA1p6df0"}

for i in range(10):
    response = requests.get(status_url, headers=headers)
    print(f"Attempt {i+1}, Status: {response.status_code}, Response: {response.text}")
    
    if "COMPLETED" in response.text:
        print("Job completed successfully!")
        sys.exit(0)
    
    time.sleep(2)

print("Timed out waiting for job completion") 