#!/bin/bash

# Jamba Threat Detection - RunPod Entry Script
# This script is executed when the container starts in RunPod

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting Jamba Threat Detection RunPod Server"
echo "$(date): Container startup initiated"

# Setup environment
export PYTHONPATH=${PYTHONPATH}:/app:/app/src
export MODEL_DIR=${MODEL_DIR:-/app/models}
export LOGS_DIR=${LOGS_DIR:-/app/logs}
export DEBUG_MODE=${DEBUG_MODE:-false}

# Create necessary directories
mkdir -p ${MODEL_DIR}
mkdir -p ${LOGS_DIR}

# Log environment information
echo "$(date): Environment setup:"
echo "- PYTHONPATH: ${PYTHONPATH}"
echo "- MODEL_DIR: ${MODEL_DIR}"
echo "- LOGS_DIR: ${LOGS_DIR}"
echo "- DEBUG_MODE: ${DEBUG_MODE}"
echo "- Python version: $(python --version)"
echo "- Directory structure:"
ls -la /app

# Verify system dependencies
echo "$(date): Checking system dependencies"
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found"
    exit 1
fi

# Check if required Python packages are installed
echo "$(date): Checking Python dependencies"
pip list | grep torch
pip list | grep pandas
pip list | grep numpy
pip list | grep runpod

# Verify GPU if CUDA is available
if [[ -n "${NVIDIA_VISIBLE_DEVICES}" ]] && [[ "${NVIDIA_VISIBLE_DEVICES}" != "none" ]]; then
    echo "$(date): GPU information:"
    nvidia-smi || echo "WARNING: nvidia-smi command failed, but continuing"
fi

# Run health check and environment setup using the utils modules
echo "$(date): Running pre-flight health checks"
echo "-------------------------"
python -c "
import sys
import traceback
try:
    # Add src to path if needed
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    if str(current_dir.parent) not in sys.path:
        sys.path.append(str(current_dir.parent))
    
    # Import and run environment setup
    from utils import environment, validation
    print('Setting up environment...')
    environment.setup_environment(create_dirs=True)
    
    # Run validation checks
    print('Running validation checks...')
    validation_result = validation.run_health_check()
    
    if validation_result.get('success', False):
        print('Health check passed!')
        for key, value in validation_result.items():
            if key != 'success':
                print(f'- {key}: {value}')
    else:
        print('Health check failed!')
        print(f'Error: {validation_result.get(\"error\", \"Unknown error\")}')
        # Continue anyway to not break existing deployments
        print('Continuing despite health check failure...')
except Exception as e:
    print(f'Error during health check: {e}')
    traceback.print_exc()
    print('Continuing despite health check failure...')
" || echo "WARNING: Health check script failed, but continuing"
echo "-------------------------"

# Start the RunPod handler
echo "$(date): Starting RunPod handler"
cd /app
runpod --handler-path /app/src/handler.py
# Keep the container running until it's stopped
echo "$(date): RunPod handler exited, keeping container alive for debugging"
tail -f /dev/null 