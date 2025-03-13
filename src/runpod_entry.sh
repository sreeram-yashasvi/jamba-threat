#!/bin/bash

# Jamba Threat Detection - RunPod Entry Script
# This script is executed when the container starts in RunPod

# Don't exit immediately on errors - handle them gracefully
set +e

echo "Starting Jamba Threat Detection RunPod Server"
echo "$(date): Container startup initiated"

# Determine application directory
if [ -d "/app" ]; then
    APP_DIR="/app"
elif [ -d "/workspace" ]; then
    APP_DIR="/workspace"
else
    APP_DIR=$(pwd)
fi
echo "Using APP_DIR: $APP_DIR"

# Setup environment
export PYTHONPATH=${PYTHONPATH}:${APP_DIR}:${APP_DIR}/src
export APP_DIR=${APP_DIR}
export MODEL_DIR=${MODEL_DIR:-${APP_DIR}/models}
export LOGS_DIR=${LOGS_DIR:-${APP_DIR}/logs}
export DEBUG_MODE=${DEBUG_MODE:-false}

# Create necessary directories
mkdir -p ${MODEL_DIR}
mkdir -p ${LOGS_DIR}
mkdir -p ${APP_DIR}/tmp

# Log environment information
echo "$(date): Environment setup:"
echo "- PYTHONPATH: ${PYTHONPATH}"
echo "- APP_DIR: ${APP_DIR}"
echo "- MODEL_DIR: ${MODEL_DIR}"
echo "- LOGS_DIR: ${LOGS_DIR}"
echo "- DEBUG_MODE: ${DEBUG_MODE}"
echo "- Python version: $(python --version)"
echo "- Directory structure:"
ls -la ${APP_DIR}

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
pip list | grep torch || echo "WARNING: PyTorch not found"
pip list | grep pandas || echo "WARNING: Pandas not found"
pip list | grep numpy || echo "WARNING: NumPy not found"
pip list | grep runpod || echo "WARNING: RunPod not found, will install"

# Verify GPU if CUDA is available
if [[ -n "${NVIDIA_VISIBLE_DEVICES}" ]] && [[ "${NVIDIA_VISIBLE_DEVICES}" != "none" ]]; then
    echo "$(date): GPU information:"
    nvidia-smi || echo "WARNING: nvidia-smi command failed, but continuing"
fi

# Create health check script in app's temp directory
HEALTH_CHECK_PATH="${APP_DIR}/tmp/health_check.py"
echo "$(date): Creating health check script at ${HEALTH_CHECK_PATH}"

# Ensure the directory exists
mkdir -p $(dirname ${HEALTH_CHECK_PATH})

# Create the health check script
cat > ${HEALTH_CHECK_PATH} << 'EOF'
#!/usr/bin/env python3
import sys
import os
import traceback

# Set up basic logging to console
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("health_check")

try:
    # Add paths explicitly
    app_dir = os.environ.get('APP_DIR', '/app')
    sys.path.append(app_dir)
    sys.path.append(os.path.join(app_dir, 'src'))
    
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # First try to import directly
    try:
        from utils import environment, validation
        logger.info("Successfully imported utils directly")
    except ImportError:
        logger.warning("Direct import failed, trying with absolute imports")
        # Try with explicit imports
        import importlib.util
        
        # Function to import a module from a file path
        def import_from_path(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                logger.error(f"Could not find module {module_name} at {file_path}")
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        
        # Try to find and import utils modules
        for path in [app_dir, os.path.join(app_dir, 'src')]:
            env_path = os.path.join(path, 'utils', 'environment.py')
            val_path = os.path.join(path, 'utils', 'validation.py')
            
            if os.path.exists(env_path) and os.path.exists(val_path):
                logger.info(f"Found utils modules at {path}")
                environment = import_from_path('environment', env_path)
                validation = import_from_path('validation', val_path)
                break
        else:
            raise ImportError("Could not find utils modules")
    
    # Setup environment
    logger.info('Setting up environment...')
    if hasattr(environment, 'setup_environment'):
        environment.setup_environment()
    else:
        logger.warning("setup_environment function not found, creating directories manually")
        os.makedirs(os.environ.get('MODEL_DIR', os.path.join(app_dir, 'models')), exist_ok=True)
        os.makedirs(os.environ.get('LOGS_DIR', os.path.join(app_dir, 'logs')), exist_ok=True)
    
    # Run validation
    logger.info('Running validation checks...')
    if hasattr(validation, 'run_health_check'):
        validation_result = validation.run_health_check()
    elif hasattr(validation, 'health_check'):
        validation_result = validation.health_check()
    else:
        validation_result = {"success": False, "error": "No health check function found"}
    
    if validation_result.get('success', False):
        logger.info('Health check passed!')
        for key, value in validation_result.items():
            if key != 'success':
                logger.info(f'- {key}: {value}')
    else:
        logger.warning('Health check failed!')
        logger.warning(f'Error: {validation_result.get("error", "Unknown error")}')
        logger.warning('Continuing despite health check failure...')
    
except Exception as e:
    logger.error(f'Error during health check: {str(e)}')
    traceback.print_exc()
    logger.warning('Continuing despite health check failure...')
EOF

# Make the health check script executable
chmod +x ${HEALTH_CHECK_PATH}

# Run the health check script
echo "$(date): Running pre-flight health checks"
echo "-------------------------"
python ${HEALTH_CHECK_PATH} || echo "WARNING: Health check script failed, but continuing"
echo "-------------------------"

# Make sure runpod is installed
if ! pip list | grep -q runpod; then
    echo "$(date): Installing RunPod SDK"
    pip install runpod==0.10.0
fi

# Create the RunPod fix script if it doesn't exist
RUNPOD_FIX_SCRIPT="${APP_DIR}/tmp/fix_runpod_command.py"
echo "$(date): Creating RunPod fix script at ${RUNPOD_FIX_SCRIPT}"

cat > ${RUNPOD_FIX_SCRIPT} << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess
import site
import shutil
import importlib.util

def check_runpod_installed():
    """Check if runpod is installed in the Python environment."""
    try:
        spec = importlib.util.find_spec("runpod")
        return spec is not None
    except ImportError:
        return False

def get_runpod_install_location():
    """Get the installation location of the runpod package."""
    try:
        import runpod
        return os.path.dirname(runpod.__file__)
    except ImportError:
        return None

def get_runpod_executable():
    """Find the runpod executable in common locations."""
    # Check if runpod is in PATH
    for path in os.environ.get("PATH", "").split(os.pathsep):
        runpod_path = os.path.join(path, "runpod")
        if os.path.isfile(runpod_path) and os.access(runpod_path, os.X_OK):
            return runpod_path
    
    # Check site-packages bin directory
    site_packages = site.getsitepackages()
    for site_pkg in site_packages:
        possible_paths = [
            os.path.join(site_pkg, "runpod", "bin", "runpod"),
            os.path.join(site_pkg, "bin", "runpod"),
            os.path.join(os.path.dirname(site_pkg), "bin", "runpod")
        ]
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    
    return None

def fix_runpod_command():
    """Fix the RunPod command by ensuring it's properly installed and accessible."""
    print("RunPod Command Fix Utility")
    print("==========================")
    
    app_dir = os.environ.get('APP_DIR', '/app')
    
    # Check if runpod package is installed
    if not check_runpod_installed():
        print("❌ RunPod package is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "runpod==0.10.0"])
            print("✅ RunPod package installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install RunPod package.")
            return False
    else:
        print("✅ RunPod package is installed.")
    
    # Get installation location
    install_location = get_runpod_install_location()
    if install_location:
        print(f"📁 RunPod package location: {install_location}")
    else:
        print("❌ Could not find RunPod package location.")
        return False
    
    # Find runpod executable
    runpod_exec = get_runpod_executable()
    if runpod_exec:
        print(f"🔍 Found RunPod executable at: {runpod_exec}")
        
        # Create symlink if needed
        for bin_dir in ["/usr/local/bin", "/usr/bin"]:
            if os.path.isdir(bin_dir) and os.access(bin_dir, os.W_OK):
                symlink_path = os.path.join(bin_dir, "runpod")
                if not os.path.exists(symlink_path):
                    try:
                        # Create symlink
                        os.symlink(runpod_exec, symlink_path)
                        print(f"✅ Created symlink at {symlink_path}")
                        break
                    except OSError:
                        print(f"❌ Failed to create symlink at {symlink_path}")
        
        # Test if runpod command works
        try:
            subprocess.check_call(["runpod", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ RunPod command is working correctly.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ RunPod command not found in PATH. Using direct path instead.")
            
        # Create a wrapper script
        wrapper_path = f"{app_dir}/runpod_wrapper.sh"
        with open(wrapper_path, "w") as f:
            f.write(f"""#!/bin/bash
# RunPod command wrapper
{runpod_exec} "$@"
""")
        os.chmod(wrapper_path, 0o755)
        print(f"✅ Created wrapper script at {wrapper_path}")
        
        return True
    else:
        print("❌ Could not find RunPod executable.")
        
        # Try installing with -m
        print("Attempting to run RunPod as a module...")
        try:
            cmd = [sys.executable, "-m", "runpod", "--version"]
            subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ RunPod can be run as a Python module.")
            
            # Create a wrapper script for module
            wrapper_path = f"{app_dir}/runpod_wrapper.sh"
            with open(wrapper_path, "w") as f:
                f.write(f"""#!/bin/bash
# RunPod module wrapper
{sys.executable} -m runpod "$@"
""")
            os.chmod(wrapper_path, 0o755)
            print(f"✅ Created module wrapper script at {wrapper_path}")
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ RunPod module execution failed.")
            return False

if __name__ == "__main__":
    success = fix_runpod_command()
    if success:
        print("\n✅ RunPod command issues have been fixed.")
        print("You can now use one of these commands to start the handler:")
        runpod_exec = get_runpod_executable()
        if runpod_exec:
            print(f"  - {runpod_exec} --handler-path /app/src/handler.py")
        print(f"  - {os.environ.get('APP_DIR', '/app')}/runpod_wrapper.sh --handler-path {os.environ.get('APP_DIR', '/app')}/src/handler.py")
        print("  - python -m runpod --handler-path src/handler.py")
    else:
        print("\n❌ Failed to fix RunPod command issues.")
        print("Please try reinstalling the RunPod package manually:")
        print("  pip install --force-reinstall runpod==0.10.0")
EOF

# Make the fix script executable
chmod +x ${RUNPOD_FIX_SCRIPT}

# Run the RunPod fix utility
echo "$(date): Running RunPod fix utility"
python ${RUNPOD_FIX_SCRIPT}

# Create an inline handler if needed
HANDLER_PATH="${APP_DIR}/src/handler.py"
if [ ! -f "${HANDLER_PATH}" ]; then
    echo "WARNING: Handler not found at ${HANDLER_PATH}. Creating a minimal handler."
    
    mkdir -p $(dirname ${HANDLER_PATH})
    
    cat > ${HANDLER_PATH} << 'EOF'
#!/usr/bin/env python3
import torch
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def handler(event):
    """
    RunPod handler function.
    
    Args:
        event (dict): The event payload
        
    Returns:
        dict: The response
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Get operation type
        operation = event.get("input", {}).get("operation", "health")
        
        if operation == "health":
            # Health check operation
            logger.info("Processing health check")
            return {
                "success": True,
                "gpu_info": {
                    "available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                },
                "env_info": {
                    "app_dir": os.environ.get("APP_DIR", "/app"),
                    "model_dir": os.environ.get("MODEL_DIR", "/app/models"),
                    "python_version": sys.version
                }
            }
        else:
            # Unknown operation
            logger.warning(f"Unknown operation: {operation}")
            return {
                "success": False,
                "error": f"Operation '{operation}' not supported"
            }
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
EOF
fi

# Start the RunPod handler
echo "$(date): Starting RunPod handler"
cd ${APP_DIR}

# Try multiple ways to start the handler
echo "$(date): Attempting to start handler using multiple methods"

# Define handler path
HANDLER_REL_PATH="src/handler.py"
HANDLER_ABS_PATH="${APP_DIR}/${HANDLER_REL_PATH}"

# Method 1: Try wrapper script
if [ -f "${APP_DIR}/runpod_wrapper.sh" ]; then
    echo "Method 1: Using runpod wrapper script"
    ${APP_DIR}/runpod_wrapper.sh --handler-path ${HANDLER_REL_PATH}
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Handler started successfully using wrapper script"
    else
        echo "Failed to start handler using wrapper script (exit code: $RESULT)"
    fi
else
    echo "Wrapper script not found, trying other methods"
fi

# Method 2: Try direct runpod command
if command -v runpod &> /dev/null; then
    echo "Method 2: Using system runpod command"
    runpod --handler-path ${HANDLER_REL_PATH}
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Handler started successfully using system runpod command"
    else
        echo "Failed to start handler using system runpod command (exit code: $RESULT)"
    fi
else
    echo "System runpod command not found, trying other methods"
fi

# Method 3: Try Python module
echo "Method 3: Using python -m runpod"
python -m runpod --handler-path ${HANDLER_REL_PATH}
RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "Handler started successfully using python -m runpod"
else
    echo "Failed to start handler using python -m runpod (exit code: $RESULT)"
fi

# Fallback: Manual handler execution
echo "Method 4: Directly executing handler script"
echo "from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup\nimport handler\nimport json\n\nwhile True:\n    print('Waiting for requests...')\n    try:\n        job_id = input()\n        event = json.loads(input())\n        print(json.dumps(handler.handler(event)))\n    except Exception as e:\n        print(json.dumps({'error': str(e)}))\n" > ${APP_DIR}/tmp/runner.py
python ${APP_DIR}/tmp/runner.py

# Keep the container running until it's stopped
echo "$(date): All handler methods failed, keeping container alive for debugging"
tail -f /dev/null 