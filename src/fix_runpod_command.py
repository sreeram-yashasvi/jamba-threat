#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RunPod Command Fix Utility

This script helps diagnose and fix issues with the RunPod command not being found.
It verifies the installation, finds the correct path to the executable, and provides
instructions for fixing PATH-related issues.
"""

import sys
from jamba.utils.runpod_utils import RunPodInstaller, RunPodError

def main():
    """Main function to fix RunPod command issues."""
    print("RunPod Command Fix Utility")
    print("==========================")
    
    installer = RunPodInstaller()
    
    try:
        # Attempt complete setup
        result = installer.setup()
        
        print("\n✅ RunPod command issues have been fixed.")
        print("You can now use one of these commands to start the handler:")
        
        if result["executable_path"]:
            print(f"  - {result['executable_path']} --handler-path /app/src/handler.py")
        if result["wrapper_path"]:
            print(f"  - {result['wrapper_path']} --handler-path /app/src/handler.py")
        print("  - python -m runpod --handler-path /app/src/handler.py")
        
        return 0
        
    except RunPodError as e:
        print(f"\n❌ {str(e)}")
        print("Please try reinstalling the RunPod package manually:")
        print("  pip install --force-reinstall runpod==0.10.0")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 