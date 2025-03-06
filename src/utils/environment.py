"""
Environment configuration utilities for Jamba Threat Detection model.
Centralizes environment variables, path configuration, and directory creation.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths and environment variables
DEFAULT_PATHS = {
    "APP_DIR": os.environ.get("APP_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))),
    "MODEL_DIR": os.environ.get("MODEL_DIR", "./models"),
    "LOGS_DIR": os.environ.get("LOGS_DIR", "./logs"),
    "DATA_DIR": os.environ.get("DATA_DIR", "./data"),
}

REQUIRED_ENV_VARS = [
    "RUNPOD_API_KEY",
    "RUNPOD_ENDPOINT_ID"
]

def setup_environment(create_dirs=True):
    """
    Set up the environment for the Jamba Threat Detection model.
    
    Args:
        create_dirs: Whether to create directories if they don't exist
        
    Returns:
        dict: Environment configuration
    """
    # Set up paths
    paths = {}
    for key, default_path in DEFAULT_PATHS.items():
        value = os.environ.get(key, default_path)
        paths[key] = value
        os.environ[key] = value
    
    # Create directories if needed
    if create_dirs:
        for key in ["MODEL_DIR", "LOGS_DIR", "DATA_DIR"]:
            os.makedirs(paths[key], exist_ok=True)
            logger.info(f"Ensured {key} directory exists at {paths[key]}")
    
    # Set up Python path
    app_dir = paths["APP_DIR"]
    src_dir = os.path.join(app_dir, "src")
    
    if app_dir not in sys.path:
        sys.path.append(app_dir)
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    logger.info(f"Environment setup complete. APP_DIR: {app_dir}")
    
    return paths

def check_environment_variables():
    """
    Check if required environment variables are set.
    
    Returns:
        tuple: (bool, list) - Success flag and list of missing variables
    """
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing_vars.append(var)
            logger.warning(f"Environment variable {var} is not set")
    
    return len(missing_vars) == 0, missing_vars

def get_model_path():
    """
    Get the path to the model file.
    
    Returns:
        str: Path to the model file
    """
    model_dir = os.environ.get("MODEL_DIR", "./models")
    return os.path.join(model_dir, "jamba_model.pth")

def list_environment():
    """
    List the current environment configuration.
    
    Returns:
        dict: Current environment configuration
    """
    env_info = {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "python_path": sys.path,
        "environment_variables": {
            key: os.environ.get(key) 
            for key in DEFAULT_PATHS.keys()
        }
    }
    
    # Add required environment variables (masked for security)
    for var in REQUIRED_ENV_VARS:
        val = os.environ.get(var)
        if val:
            # Mask API keys and sensitive information
            env_info["environment_variables"][var] = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "Set"
        else:
            env_info["environment_variables"][var] = "Not set"
    
    return env_info

if __name__ == "__main__":
    # When run directly, set up environment and print configuration
    paths = setup_environment()
    env_status, missing = check_environment_variables()
    
    logger.info(f"Environment status: {'Ready' if env_status else 'Missing variables'}")
    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")
    
    logger.info("Environment configuration:")
    for key, value in list_environment().items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}") 