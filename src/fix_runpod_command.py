#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RunPod Command Fix Utility

This script helps diagnose and fix issues with the RunPod command not being found.
It verifies the installation, finds the correct path to the executable, and provides
instructions for fixing PATH-related issues.
"""

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
        wrapper_path = "/app/runpod_wrapper.sh"
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
            wrapper_path = "/app/runpod_wrapper.sh"
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
        print("  - /app/runpod_wrapper.sh --handler-path /app/src/handler.py")
        print("  - python -m runpod --handler-path /app/src/handler.py")
    else:
        print("\n❌ Failed to fix RunPod command issues.")
        print("Please try reinstalling the RunPod package manually:")
        print("  pip install --force-reinstall runpod==0.10.0") 