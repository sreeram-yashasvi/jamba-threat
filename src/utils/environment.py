"""
Environment configuration utilities for Jamba Threat Detection.

This module provides functions for handling environment variables,
path configuration, and directory setup across the application.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path configurations
DEFAULT_APP_DIR = os.environ.get("APP_DIR", "/app")
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_APP_DIR, "models")
DEFAULT_LOGS_DIR = os.path.join(DEFAULT_APP_DIR, "logs")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_APP_DIR, "data")

def setup_environment():
    """
    Set up the environment with default paths and create necessary directories.
    
    Returns:
        dict: A dictionary of configured environment paths
    """
    # Configure paths with environment variable overrides or defaults
    env_config = {
        "APP_DIR": os.environ.get("APP_DIR", DEFAULT_APP_DIR),
        "MODEL_DIR": os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        "LOGS_DIR": os.environ.get("LOGS_DIR", DEFAULT_LOGS_DIR),
        "DATA_DIR": os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
    }
    
    # Create directories if they don't exist
    for dir_name, dir_path in env_config.items():
        if dir_name.endswith("_DIR"):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured {dir_name} directory exists at: {dir_path}")
    
    # Add src directory to Python path if not already there
    src_dir = os.path.join(env_config["APP_DIR"], "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
        logger.info(f"Added {src_dir} to Python path")
    
    # Add app directory to Python path if not already there
    if env_config["APP_DIR"] not in sys.path:
        sys.path.append(env_config["APP_DIR"])
        logger.info(f"Added {env_config['APP_DIR']} to Python path")
    
    return env_config

def check_environment_variables():
    """
    Check for required environment variables.
    
    Returns:
        tuple: (missing_vars, dict of available vars)
    """
    required_vars = {
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY"),
        "RUNPOD_ENDPOINT_ID": os.environ.get("RUNPOD_ENDPOINT_ID")
    }
    
    # Check for missing variables
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("All required environment variables are set")
    
    return missing_vars, required_vars

def load_env_file(env_file=".env"):
    """
    Load environment variables from .env file if it exists.
    
    Args:
        env_file (str): Path to .env file
        
    Returns:
        bool: True if env file was loaded successfully
    """
    app_dir = os.environ.get("APP_DIR", DEFAULT_APP_DIR)
    env_path = os.path.join(app_dir, env_file)
    
    if not os.path.exists(env_path):
        logger.info(f".env file not found at {env_path}")
        return False
    
    logger.info(f"Loading environment variables from {env_path}")
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                os.environ[key] = value
        return True
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        return False

def get_python_info():
    """
    Get information about the Python environment.
    
    Returns:
        dict: Python environment information
    """
    import platform
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_path": sys.path,
        "executable": sys.executable,
        "cwd": os.getcwd()
    }

def print_directory_structure(base_dir=None):
    """
    Print the directory structure for debugging.
    
    Args:
        base_dir (str): Base directory to list
    """
    if base_dir is None:
        base_dir = os.environ.get("APP_DIR", DEFAULT_APP_DIR)
    
    logger.info(f"Directory structure of {base_dir}:")
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                logger.info(f"  📁 {item}/")
            else:
                logger.info(f"  📄 {item}")
                
        src_dir = os.path.join(base_dir, "src")
        if os.path.exists(src_dir):
            logger.info(f"Contents of {src_dir}:")
            for item in os.listdir(src_dir):
                item_path = os.path.join(src_dir, item)
                if os.path.isdir(item_path):
                    logger.info(f"  📁 src/{item}/")
                else:
                    logger.info(f"  📄 src/{item}")
    except Exception as e:
        logger.error(f"Error listing directory structure: {e}")

# Setup logging configuration
def configure_logging(level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level to set
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set level for our own loggers
    logging.getLogger("jamba_threat").setLevel(level)
    
    logger.info(f"Logging configured at level {logging.getLevelName(level)}")

# Run setup if this module is executed directly
if __name__ == "__main__":
    configure_logging()
    env_config = setup_environment()
    missing_vars, _ = check_environment_variables()
    print_directory_structure(env_config["APP_DIR"])
    logger.info(f"Environment setup complete. APP_DIR: {env_config['APP_DIR']}")
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    python_info = get_python_info()
    logger.info(f"Python version: {python_info['python_version']}") 