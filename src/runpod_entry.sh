#!/bin/bash
set -e

echo "======================================="
echo "Jamba Threat Detection - RunPod Startup"
echo "======================================="

# Check if environment variables are set
echo "Checking environment variables..."
if [ -z "${RUNPOD_API_KEY}" ]; then
    echo "⚠️  WARNING: RUNPOD_API_KEY environment variable is not set"
    echo "    Some functionality may be limited"
fi

if [ -z "${RUNPOD_ENDPOINT_ID}" ]; then
    echo "⚠️  WARNING: RUNPOD_ENDPOINT_ID environment variable is not set"
    echo "    Some functionality may be limited"
fi

# Load environment variables from .env file if it exists
if [ -f "/app/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' /app/.env | xargs)
fi

# Run startup validation checks
echo "Running startup validation checks..."
/app/startup_check.sh
if [ $? -ne 0 ]; then
    echo "❌ Startup validation failed. Exiting."
    exit 1
fi

# Check for CUDA and GPU availability
echo "Checking CUDA and GPU availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Print Python and package versions
echo "Python and package versions:"
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import runpod; print('RunPod:', runpod.__version__)"

echo "======================================="
echo "Starting RunPod handler..."
echo "======================================="

# Start the actual handler
exec python -m runpod.serverless.start 