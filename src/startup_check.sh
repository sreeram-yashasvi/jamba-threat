#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if the Python utility script is available
check_utility_script() {
    if [ -f "${APP_DIR}/src/utils/shell.py" ]; then
        log "✓ Utility script found at ${APP_DIR}/src/utils/shell.py"
        return 0
    else
        log "✗ Utility script not found at ${APP_DIR}/src/utils/shell.py"
        
        # Check if utils directory exists
        if [ ! -d "${APP_DIR}/src/utils" ]; then
            log "  Creating utils directory"
            mkdir -p "${APP_DIR}/src/utils"
        fi
        
        log "✗ Cannot proceed without utility script. Please ensure it's properly installed."
        return 1
    fi
}

# Define app directory
export APP_DIR="${APP_DIR:-/app}"
export PYTHONPATH="${PYTHONPATH}:${APP_DIR}"

log "Starting startup check..."
log "APP_DIR is set to ${APP_DIR}"

# Check for required directories
log "Checking file structure..."
for dir in "${APP_DIR}" "${APP_DIR}/src" "${APP_DIR}/models"; do
    if [ -d "$dir" ]; then
        log "✓ Directory $dir exists"
    else
        log "✗ Directory $dir does NOT exist"
        mkdir -p "$dir"
        log "  Created directory $dir"
    fi
done

# Check for basic required files
log "Checking for basic required files..."
essential_files=()

for file in "${APP_DIR}/src/handler.py" "${APP_DIR}/src/jamba_model.py"; do
    if [ -f "$file" ]; then
        log "✓ File $file exists"
    else
        log "✗ File $file does NOT exist"
        essential_files+=("$file")
    fi
done

if [ ${#essential_files[@]} -gt 0 ]; then
    log "✗ Missing essential files: ${essential_files[*]}"
    exit 1
fi

# Check if utility script is available
if ! check_utility_script; then
    log "Falling back to basic checks without utility script"
    
    # Basic Python module check
    log "Checking for required Python modules..."
    for module in "torch" "numpy" "pandas" "runpod"; do
        if python -c "import $module" 2>/dev/null; then
            log "✓ Module $module is available"
        else
            log "✗ Module $module is NOT available"
            exit 1
        fi
    done
    
    # Basic CUDA check
    log "Checking CUDA availability..."
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        log "✓ CUDA is available"
    else
        log "⚠️ CUDA is not available, running in CPU mode"
    fi
    
    # Basic model import check
    log "Checking if model can be imported..."
    if python -c "
import sys
sys.path.append('${APP_DIR}')
sys.path.append('${APP_DIR}/src')
try:
    from jamba_model import JambaThreatModel
    print('Model import successful')
    exit(0)
except Exception as e:
    print(f'Model import failed: {e}')
    exit(1)
" >/dev/null 2>&1; then
        log "✓ Model class import successful"
    else
        log "✗ Model class import failed"
        exit 1
    fi
else
    # Use utility script for complete checks
    log "Using utility script for comprehensive checks"
    
    # Check Python modules
    log "Checking Python modules..."
    if "${APP_DIR}/src/utils/shell.py" check-model --quiet; then
        log "✓ Model checks passed"
    else
        module_result=$?
        log "✗ Model checks failed (exit code $module_result)"
        
        # Get detailed error information
        "${APP_DIR}/src/utils/shell.py" check-model
        
        exit 1
    fi
    
    # Check CUDA
    log "Checking CUDA availability..."
    "${APP_DIR}/src/utils/shell.py" check-cuda
    
    # Environment variables are not critical for startup, just warn
    log "Checking environment variables..."
    if "${APP_DIR}/src/utils/shell.py" check-env --quiet; then
        log "✓ All environment variables are set"
    else
        log "⚠️ Some environment variables are missing (this is a warning only)"
        # Show which variables are missing
        "${APP_DIR}/src/utils/shell.py" check-env
    fi
fi

# Check disk space
log "Checking disk space..."
df -h "${APP_DIR}"

log "✓ System is ready for operation"
exit 0 