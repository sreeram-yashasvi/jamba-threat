import os
import sys
import site
import subprocess
import importlib.util
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum, auto

class RunPodError(Exception):
    """Base exception for RunPod-related errors."""
    pass

class InstallationError(RunPodError):
    """Error during RunPod installation."""
    pass

class ExecutableError(RunPodError):
    """Error finding or accessing RunPod executable."""
    pass

class InstallationState(Enum):
    """Possible states during RunPod installation."""
    NOT_INSTALLED = auto()
    INSTALLING = auto()
    INSTALLED = auto()
    FAILED = auto()

@dataclass
class RunPodPaths:
    """Container for RunPod-related paths."""
    package_path: Optional[str] = None
    executable_path: Optional[str] = None
    symlink_path: Optional[str] = None
    wrapper_path: Optional[str] = None

class PathFinder:
    """Handles finding RunPod-related paths."""
    
    def __init__(self):
        self.paths = RunPodPaths()
    
    def find_package_location(self) -> Optional[str]:
        """Find RunPod package installation location."""
        try:
            spec = importlib.util.find_spec("runpod")
            if spec and spec.origin:
                self.paths.package_path = os.path.dirname(spec.origin)
                return self.paths.package_path
        except ImportError:
            pass
        return None
    
    def find_executable(self) -> Optional[str]:
        """Find RunPod executable in common locations."""
        # Check PATH
        for path in os.environ.get("PATH", "").split(os.pathsep):
            runpod_path = os.path.join(path, "runpod")
            if os.path.isfile(runpod_path) and os.access(runpod_path, os.X_OK):
                self.paths.executable_path = runpod_path
                return runpod_path
        
        # Check site-packages
        for site_pkg in site.getsitepackages():
            possible_paths = [
                os.path.join(site_pkg, "runpod", "bin", "runpod"),
                os.path.join(site_pkg, "bin", "runpod"),
                os.path.join(os.path.dirname(site_pkg), "bin", "runpod")
            ]
            for path in possible_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    self.paths.executable_path = path
                    return path
        
        return None
    
    def find_writable_bin_directory(self) -> Optional[str]:
        """Find a writable bin directory for symlinks."""
        for bin_dir in ["/usr/local/bin", "/usr/bin"]:
            if os.path.isdir(bin_dir) and os.access(bin_dir, os.W_OK):
                return bin_dir
        return None

class RunPodInstaller:
    """Handles RunPod installation and setup."""
    
    def __init__(self, version: str = "0.10.0"):
        self.version = version
        self.state = InstallationState.NOT_INSTALLED
        self.path_finder = PathFinder()
    
    def check_installation(self) -> bool:
        """Check if RunPod is installed."""
        try:
            spec = importlib.util.find_spec("runpod")
            return spec is not None
        except ImportError:
            return False
    
    def install_package(self) -> bool:
        """Install RunPod package."""
        self.state = InstallationState.INSTALLING
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--force-reinstall", f"runpod=={self.version}"
            ])
            self.state = InstallationState.INSTALLED
            return True
        except subprocess.CalledProcessError:
            self.state = InstallationState.FAILED
            raise InstallationError("Failed to install RunPod package")
    
    def create_symlink(self, executable_path: str) -> Optional[str]:
        """Create symlink to RunPod executable."""
        bin_dir = self.path_finder.find_writable_bin_directory()
        if not bin_dir:
            return None
        
        symlink_path = os.path.join(bin_dir, "runpod")
        try:
            if not os.path.exists(symlink_path):
                os.symlink(executable_path, symlink_path)
            return symlink_path
        except OSError:
            return None
    
    def create_wrapper_script(self, executable_path: str, wrapper_path: str = "/app/runpod_wrapper.sh") -> str:
        """Create a wrapper script for RunPod."""
        wrapper_content = f"""#!/bin/bash
# RunPod command wrapper
{executable_path} "$@"
"""
        try:
            with open(wrapper_path, "w") as f:
                f.write(wrapper_content)
            os.chmod(wrapper_path, 0o755)
            return wrapper_path
        except OSError as e:
            raise InstallationError(f"Failed to create wrapper script: {str(e)}")
    
    def verify_installation(self) -> bool:
        """Verify RunPod installation and command availability."""
        try:
            subprocess.check_call(
                ["runpod", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def setup(self) -> Dict[str, str]:
        """
        Complete RunPod setup process.
        
        Returns:
            Dict with paths and status information
        """
        if not self.check_installation():
            self.install_package()
        
        # Find paths
        package_path = self.path_finder.find_package_location()
        if not package_path:
            raise InstallationError("Could not find RunPod package location")
        
        executable_path = self.path_finder.find_executable()
        if not executable_path:
            raise ExecutableError("Could not find RunPod executable")
        
        # Create symlink and wrapper
        symlink_path = self.create_symlink(executable_path)
        wrapper_path = self.create_wrapper_script(executable_path)
        
        return {
            "package_path": package_path,
            "executable_path": executable_path,
            "symlink_path": symlink_path,
            "wrapper_path": wrapper_path,
            "status": "success"
        } 