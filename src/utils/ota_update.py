"""OTA (Over-The-Air) model update system for OpenEyes.

Provides:
- Model download and installation
- Version checking
- Rollback capabilities
- Update verification
"""

import os
import hashlib
import json
import time
import tempfile
import shutil
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
import urllib.request
import urllib.error


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    filename: str
    checksum: str
    size_bytes: int
    url: str
    release_date: str
    changelog: str


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    message: str
    previous_version: Optional[str] = None
    new_version: Optional[str] = None


class OTAModelUpdater:
    """OTA model updater for OpenEyes.
    
    Features:
    - Download and install new models
    - Version management
    - Checksum verification
    - Rollback support
    - Progress callbacks
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        manifest_url: Optional[str] = None,
        backup_dir: str = "models/backup",
    ):
        self._models_dir = Path(models_dir)
        self._backup_dir = Path(backup_dir)
        self._manifest_url = manifest_url
        
        self._models_dir.mkdir(exist_ok=True)
        self._backup_dir.mkdir(exist_ok=True)
        
        self._current_versions: Dict[str, str] = {}
        self._update_callbacks: List[Callable[[float, str], None]] = []
        
        self._logger = self._get_logger()
        
        self._load_version_info()
    
    def _get_logger(self):
        try:
            from src.utils.logger import get_logger
            return get_logger(__name__)
        except:
            import logging
            return logging.getLogger(__name__)
    
    def _load_version_info(self) -> None:
        """Load current version information."""
        version_file = self._models_dir / "versions.json"
        
        if version_file.exists():
            try:
                with open(version_file, "r") as f:
                    self._current_versions = json.load(f)
            except:
                pass
    
    def _save_version_info(self) -> None:
        """Save version information."""
        version_file = self._models_dir / "versions.json"
        
        with open(version_file, "w") as f:
            json.dump(self._current_versions, f, indent=2)
    
    def register_update_callback(self, callback: Callable[[float, str], None]) -> None:
        """Register progress callback."""
        self._update_callbacks.append(callback)
    
    def get_current_version(self, model_name: str) -> Optional[str]:
        """Get current version of a model."""
        return self._current_versions.get(model_name)
    
    def check_for_updates(
        self,
        manifest: Optional[Dict[str, List[ModelVersion]]] = None
    ) -> Dict[str, Optional[ModelVersion]]:
        """Check for available updates.
        
        Args:
            manifest: Model manifest (fetched from server if not provided)
            
        Returns:
            Dict of model_name -> latest available version or None
        """
        updates = {}
        
        if manifest is None and self._manifest_url:
            manifest = self._fetch_manifest()
        
        if manifest is None:
            return updates
        
        for model_name, versions in manifest.items():
            if not versions:
                continue
            
            latest = max(versions, key=lambda v: v.version)
            current = self._current_versions.get(model_name)
            
            if current is None or latest.version != current:
                updates[model_name] = latest
            else:
                updates[model_name] = None
        
        return updates
    
    def _fetch_manifest(self) -> Optional[Dict[str, List[ModelVersion]]]:
        """Fetch manifest from server."""
        if not self._manifest_url:
            return None
        
        try:
            with urllib.request.urlopen(self._manifest_url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            manifest = {}
            for model_name, versions in data.get("models", {}).items():
                manifest[model_name] = [
                    ModelVersion(**v) for v in versions
                ]
            
            return manifest
            
        except Exception as e:
            self._logger.warning(f"Failed to fetch manifest: {e}")
            return None
    
    def update_model(
        self,
        model_name: str,
        version: ModelVersion,
        force: bool = False,
    ) -> UpdateResult:
        """Update a specific model.
        
        Args:
            model_name: Name of the model
            version: Version to install
            force: Force update even if same version
            
        Returns:
            UpdateResult with success status
        """
        previous_version = self._current_versions.get(model_name)
        
        if previous_version == version.version and not force:
            return UpdateResult(
                success=True,
                message="Already at latest version",
                previous_version=previous_version,
                new_version=version.version,
            )
        
        try:
            model_path = self._models_dir / version.filename
            
            if model_path.exists():
                backup_path = self._backup_dir / f"{model_name}_{previous_version}_{int(time.time())}"
                shutil.copy2(model_path, backup_path)
                self._logger.info(f"Backed up to {backup_path}")
            
            self._download_model(version, model_path)
            
            self._verify_checksum(model_path, version.checksum)
            
            self._current_versions[model_name] = version.version
            self._save_version_info()
            
            self._logger.info(f"Updated {model_name} to {version.version}")
            
            return UpdateResult(
                success=True,
                message=f"Updated {model_name} to {version.version}",
                previous_version=previous_version,
                new_version=version.version,
            )
            
        except Exception as e:
            self._logger.error(f"Update failed: {e}")
            return UpdateResult(
                success=False,
                message=f"Update failed: {e}",
                previous_version=previous_version,
                new_version=None,
            )
    
    def _download_model(self, version: ModelVersion, output_path: Path) -> None:
        """Download model file."""
        self._logger.info(f"Downloading {version.filename}...")
        
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                progress = min(downloaded / total_size, 1.0)
                
                for callback in self._update_callbacks:
                    try:
                        callback(progress, f"Downloading {version.filename}")
                    except:
                        pass
        
        urllib.request.urlretrieve(
            version.url,
            output_path,
            reporthook=report_progress
        )
        
        for callback in self._update_callbacks:
            try:
                callback(1.0, "Download complete")
            except:
                pass
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> None:
        """Verify file checksum."""
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        actual = sha256.hexdigest()
        
        if actual != expected_checksum:
            raise ValueError(f"Checksum mismatch: expected {expected_checksum}, got {actual}")
        
        self._logger.info("Checksum verified")
    
    def rollback(self, model_name: str) -> UpdateResult:
        """Rollback to previous version.
        
        Args:
            model_name: Name of model to rollback
            
        Returns:
            UpdateResult with rollback status
        """
        current = self._current_versions.get(model_name)
        
        if current is None:
            return UpdateResult(
                success=False,
                message="No version info available",
            )
        
        backup_files = sorted(
            self._backup_dir.glob(f"{model_name}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not backup_files:
            return UpdateResult(
                success=False,
                message="No backup available",
                previous_version=current,
            )
        
        latest_backup = backup_files[0]
        
        model_filename = latest_backup.name.split("_")[1]
        if "_" in model_filename:
            model_filename = f"{model_name}.pt"
        
        model_path = self._models_dir / model_filename
        
        try:
            shutil.copy2(latest_backup, model_path)
            
            version_from_backup = latest_backup.name.split("_")[2]
            self._current_versions[model_name] = version_from_backup
            self._save_version_info()
            
            return UpdateResult(
                success=True,
                message=f"Rolled back to {version_from_backup}",
                previous_version=current,
                new_version=version_from_backup,
            )
            
        except Exception as e:
            return UpdateResult(
                success=False,
                message=f"Rollback failed: {e}",
                previous_version=current,
            )
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available model versions."""
        result = {"available": [], "current": {}}
        
        for model_name, version in self._current_versions.items():
            result["current"][model_name] = version
            result["available"].append(model_name)
        
        return result
    
    def create_local_manifest(
        self,
        models: Dict[str, List[str]],
        output_path: Optional[str] = None
    ) -> Dict[str, List[ModelVersion]]:
        """Create a local manifest from existing models.
        
        Args:
            models: Dict of model_name -> list of filenames
            output_path: Optional path to save manifest
            
        Returns:
            Manifest dict
        """
        manifest = {}
        
        for model_name, filenames in models.items():
            versions = []
            
            for filename in filenames:
                file_path = self._models_dir / filename
                
                if not file_path.exists():
                    continue
                
                stat = file_path.stat()
                
                sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                
                versions.append(ModelVersion(
                    version="1.0.0",
                    filename=filename,
                    checksum=sha256.hexdigest(),
                    size_bytes=stat.st_size,
                    url="",
                    release_date="",
                    changelog="Local model"
                ))
            
            if versions:
                manifest[model_name] = versions
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=2)
        
        return manifest


def create_ota_updater(
    models_dir: str = "models",
    manifest_url: Optional[str] = None,
) -> OTAModelUpdater:
    """Factory function to create OTA updater."""
    return OTAModelUpdater(
        models_dir=models_dir,
        manifest_url=manifest_url,
    )