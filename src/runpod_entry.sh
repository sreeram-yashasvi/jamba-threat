#!/bin/bash
set -e

echo "======================================="
echo "Jamba Threat Detection - RunPod Startup"
echo "======================================="

# Function to log with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Define directories
export APP_DIR="${APP_DIR:-/app}"
export MODEL_DIR="${MODEL_DIR:-${APP_DIR}/models}"
export LOGS_DIR="${LOGS_DIR:-${APP_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${APP_DIR}/data}"

# Create necessary directories
log "Creating necessary directories..."
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${DATA_DIR}"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${APP_DIR}"

# Check if environment variables are set
log "Checking environment variables..."
if [ -z "${RUNPOD_API_KEY}" ]; then
    log "⚠️  WARNING: RUNPOD_API_KEY environment variable is not set"
    log "    Some functionality may be limited"
fi

if [ -z "${RUNPOD_ENDPOINT_ID}" ]; then
    log "⚠️  WARNING: RUNPOD_ENDPOINT_ID environment variable is not set"
    log "    Some functionality may be limited"
fi

# Load environment variables from .env file if it exists
if [ -f "${APP_DIR}/.env" ]; then
    log "Loading environment variables from .env file..."
    export $(grep -v '^#' "${APP_DIR}/.env" | xargs)
fi

# Print directory structure for debugging
log "Directory structure:"
ls -la "${APP_DIR}"
ls -la "${APP_DIR}/src" 2>/dev/null || log "src directory not found"

# Run startup validation checks if the script exists
if [ -f "${APP_DIR}/startup_check.sh" ]; then
    log "Running startup validation checks..."
    "${APP_DIR}/startup_check.sh"
    if [ $? -ne 0 ]; then
        log "❌ Startup validation failed. Exiting."
        exit 1
    fi
else
    log "No startup_check.sh found, skipping validation"
fi

# Check for CUDA and GPU availability
log "Checking CUDA and GPU availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Print Python and package versions
log "Python and package versions:"
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import runpod; print('RunPod:', runpod.__version__)"

# Print Python path and modules
log "Python path:"
python -c "import sys; print('\n'.join(sys.path))"
log "Installed modules:"
pip list

# Check if jamba_model module can be imported
log "Testing jamba_model import..."
python -c "import sys; sys.path.append('${APP_DIR}'); sys.path.append('${APP_DIR}/src'); import importlib; print('Import successful' if importlib.util.find_spec('jamba_model') else 'Import failed')" || log "Import test failed"

# Run a quick model health check if possible
log "Running model health check..."
python -c "
import sys
sys.path.append('${APP_DIR}')
sys.path.append('${APP_DIR}/src')
try:
    from jamba_model import JambaThreatModel
    model = JambaThreatModel(input_dim=28)
    print('✓ Model successfully initialized')
except Exception as e:
    print(f'✗ Model initialization failed: {e}')
" || log "Model health check failed"

echo "======================================="
log "Starting RunPod handler..."
echo "======================================="

# Start the actual handler with enhanced logging
exec python -m runpod.serverless.start --debug 