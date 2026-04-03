"""Fleet model registry - manages model versions across devices."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import hashlib
from pathlib import Path


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    name: str
    version: str
    checksum: str
    file_path: str
    file_size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    deployed_to: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "created_at": self.created_at,
            "deployed_to": self.deployed_to,
            "notes": self.notes,
        }


class ModelRegistry:
    """Registry for managing model versions across the fleet."""

    def __init__(self):
        self._models: Dict[str, Dict[str, ModelVersion]] = {}

    def register_model(
        self,
        name: str,
        version: str,
        file_path: str,
        notes: str = "",
    ) -> ModelVersion:
        """Register a new model version."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        checksum = self._compute_checksum(file_path)
        file_size = path.stat().st_size

        model_version = ModelVersion(
            name=name,
            version=version,
            checksum=checksum,
            file_path=file_path,
            file_size_bytes=file_size,
            notes=notes,
        )

        if name not in self._models:
            self._models[name] = {}

        self._models[name][version] = model_version
        return model_version

    def get_model(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._models.get(name, {}).get(version)

    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        if name not in self._models:
            return None
        versions = self._models[name]
        if not versions:
            return None
        return max(versions.values(), key=lambda v: v.created_at)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return list(self._models.get(name, {}).values())

    def mark_deployed(self, name: str, version: str, device_id: str) -> None:
        """Mark a model as deployed to a device."""
        model = self.get_model(name, version)
        if model and device_id not in model.deployed_to:
            model.deployed_to.append(device_id)

    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a model file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
