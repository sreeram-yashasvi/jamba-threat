#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a Python module can be imported
check_module() {
    local module=$1
    log "Checking for module: $module"
    python -c "import $module" 2>/dev/null
    if [ $? -eq 0 ]; then
        log "✓ Module $module is available"
        return 0
    else
        log "✗ Module $module is NOT available"
        return 1
    fi
}

# Check file structure
log "Checking file structure..."
for dir in "/app" "/app/src" "/app/models"; do
    if [ -d "$dir" ]; then
        log "✓ Directory $dir exists"
    else
        log "✗ Directory $dir does NOT exist"
        mkdir -p "$dir"
        log "  Created directory $dir"
    fi
done

# Check for required files
log "Checking for required files..."
for file in "/app/src/handler.py" "/app/src/jamba_model.py"; do
    if [ -f "$file" ]; then
        log "✓ File $file exists"
    else
        log "✗ File $file does NOT exist"
        exit 1
    fi
done

# Check for required Python modules
log "Checking for required Python modules..."
missing_modules=0
for module in "torch" "numpy" "pandas" "runpod"; do
    check_module $module
    if [ $? -ne 0 ]; then
        missing_modules=$((missing_modules+1))
    fi
done

# Check if we can import our model
log "Checking if we can import the model..."
python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/src')
try:
    from jamba_model import JambaThreatModel
    print('✓ Model class import successful')
except Exception as e:
    print(f'✗ Model class import failed: {e}')
    sys.exit(1)
"
model_import_result=$?

# Check CUDA
log "Checking CUDA availability..."
cuda_available=$(python -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$cuda_available" -eq 1 ]; then
    log "✓ CUDA is available"
    
    # Check CUDA version and device count
    cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    device_count=$(python -c "import torch; print(torch.cuda.device_count())")
    
    log "  CUDA Version: $cuda_version"
    log "  GPU Count: $device_count"
    
    # Check GPU memory
    free_memory=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory)")
    log "  Total GPU Memory: $((free_memory / (1024*1024))) MB"
else
    log "⚠️ CUDA is NOT available - running in CPU mode"
fi

# Check if environment variables are set
log "Checking environment variables..."
env_issues=0

if [ -z "${RUNPOD_API_KEY}" ]; then
    log "⚠️ RUNPOD_API_KEY environment variable is not set"
    env_issues=$((env_issues+1))
fi

if [ -z "${RUNPOD_ENDPOINT_ID}" ]; then
    log "⚠️ RUNPOD_ENDPOINT_ID environment variable is not set"
    env_issues=$((env_issues+1))
fi

# Check disk space
log "Checking disk space..."
df -h /app

# Summarize check results
log "Startup check summary:"

if [ $missing_modules -eq 0 ]; then
    log "✓ All required Python modules are available"
else
    log "✗ Missing $missing_modules required Python module(s)"
fi

if [ $model_import_result -eq 0 ]; then
    log "✓ Model class import successful"
else
    log "✗ Model class import failed"
fi

if [ $env_issues -eq 0 ]; then
    log "✓ All environment variables are set"
else
    log "⚠️ $env_issues environment variables are missing"
fi

# Exit with success if everything is good, or warning if there are non-critical issues
if [ $missing_modules -eq 0 ] && [ $model_import_result -eq 0 ]; then
    log "✓ System is ready for operation"
    exit 0
else
    log "✗ System has critical issues - see above for details"
    exit 1
fi 