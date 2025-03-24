from typing import Dict, List, Optional
from dataclasses import dataclass
from packaging import version

@dataclass
class VersionInfo:
    """Information about a model version."""
    version: str
    compatible_versions: List[str]
    features: List[str]
    breaking_changes: List[str]

class VersionManager:
    """Manages model version compatibility."""
    
    def __init__(self):
        self.versions: Dict[str, VersionInfo] = {
            "1.0.0": VersionInfo(
                version="1.0.0",
                compatible_versions=["1.0.0"],
                features=[
                    "Basic threat detection",
                    "Multi-head attention",
                    "Feature extraction layers"
                ],
                breaking_changes=[]
            ),
            "0.9.0": VersionInfo(
                version="0.9.0",
                compatible_versions=["0.9.0", "1.0.0"],
                features=[
                    "Basic threat detection",
                    "Simple neural network"
                ],
                breaking_changes=[
                    "Different model architecture",
                    "No attention mechanism"
                ]
            )
        }
    
    def is_compatible(self, model_version: str, current_version: str) -> bool:
        """Check if model version is compatible with current version."""
        if model_version not in self.versions:
            return False
        
        return current_version in self.versions[model_version].compatible_versions
    
    def get_version_info(self, version_str: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        return self.versions.get(version_str)
    
    def get_latest_compatible_version(self, target_version: str) -> Optional[str]:
        """Get the latest version compatible with the target version."""
        compatible_versions = []
        
        for ver, info in self.versions.items():
            if target_version in info.compatible_versions:
                compatible_versions.append(ver)
        
        if not compatible_versions:
            return None
        
        # Sort versions and return the latest
        return str(max(map(version.parse, compatible_versions)))
    
    def validate_version(self, model_version: str, current_version: str) -> tuple[bool, str]:
        """
        Validate version compatibility and provide detailed message.
        
        Returns:
            tuple: (is_valid, message)
        """
        if not self.is_compatible(model_version, current_version):
            latest_compatible = self.get_latest_compatible_version(current_version)
            message = f"Model version {model_version} is not compatible with current version {current_version}."
            
            if latest_compatible:
                message += f" Latest compatible version is {latest_compatible}."
            
            if model_version in self.versions:
                version_info = self.versions[model_version]
                message += "\nBreaking changes:"
                for change in version_info.breaking_changes:
                    message += f"\n- {change}"
            
            return False, message
        
        return True, f"Version {model_version} is compatible with {current_version}"

# Global version manager instance
version_manager = VersionManager() 