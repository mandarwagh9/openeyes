"""Industry templates for OpenEyes.

Pre-configured pipelines for highest-demand industries:
- Warehouse/Logistics: Package detection, damage inspection, pallet counting
- Manufacturing QA: Defect detection, assembly verification, PPE compliance
- Agriculture: Weed detection, crop health monitoring, yield estimation
- Retail: Shelf monitoring, inventory counting, customer analytics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
from pathlib import Path

from src.utils.logger import get_logger


@dataclass
class TemplateConfig:
    """Industry-specific pipeline configuration."""
    name: str
    description: str
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    detection_model: str = "yolo11n"
    depth_model: str = "midas-small"
    depth_enabled: bool = True
    depth_skip_frames: int = 8
    face_enabled: bool = False
    gesture_enabled: bool = False
    pose_enabled: bool = False
    tracking_enabled: bool = True
    tracking_max_age: int = 30
    tracking_min_hits: int = 3
    tracking_iou_threshold: float = 0.3
    world_model_enabled: bool = False
    world_model_type: str = "lewm"
    plan_horizon: int = 10
    plan_samples: int = 100
    safety_enabled: bool = False
    max_velocity: float = 0.5
    min_distance: float = 0.3
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    extra_args: Dict = field(default_factory=dict)
    classes_filter: List[str] = field(default_factory=list)
    custom_settings: Dict = field(default_factory=dict)


TEMPLATES: Dict[str, TemplateConfig] = {
    "warehouse": TemplateConfig(
        name="Warehouse/Logistics",
        description="Package detection, damage inspection, pallet counting, forklift safety",
        detection_model="yolo11n",
        depth_enabled=True,
        depth_skip_frames=8,
        face_enabled=False,
        gesture_enabled=False,
        pose_enabled=False,
        tracking_enabled=True,
        tracking_max_age=60,
        tracking_min_hits=3,
        world_model_enabled=True,
        world_model_type="lewm",
        plan_horizon=15,
        plan_samples=200,
        safety_enabled=True,
        max_velocity=1.0,
        min_distance=0.5,
        confidence_threshold=0.6,
        classes_filter=["person", "forklift", "truck", "pallet", "box", "package"],
        custom_settings={
            "package_inspection": True,
            "pallet_counting": True,
            "forklift_safety": True,
        },
    ),
    "manufacturing-qa": TemplateConfig(
        name="Manufacturing QA",
        description="Defect detection, assembly verification, PPE compliance monitoring",
        detection_model="yolo11n",
        depth_enabled=False,
        face_enabled=True,
        gesture_enabled=True,
        pose_enabled=True,
        tracking_enabled=True,
        tracking_max_age=30,
        tracking_min_hits=3,
        world_model_enabled=False,
        safety_enabled=True,
        max_velocity=0.5,
        min_distance=0.3,
        confidence_threshold=0.7,
        classes_filter=["person", "hard_hat", "safety_vest", "gloves", "product", "defect"],
        custom_settings={
            "defect_detection": True,
            "ppe_compliance": True,
            "assembly_verification": True,
        },
    ),
    "agriculture": TemplateConfig(
        name="Agriculture",
        description="Weed detection, crop health monitoring, yield estimation, outdoor robustness",
        detection_model="yolo11n",
        depth_enabled=True,
        depth_skip_frames=16,
        face_enabled=False,
        gesture_enabled=False,
        pose_enabled=False,
        tracking_enabled=True,
        tracking_max_age=120,
        tracking_min_hits=5,
        world_model_enabled=True,
        world_model_type="lewm",
        plan_horizon=20,
        plan_samples=100,
        safety_enabled=False,
        confidence_threshold=0.4,
        classes_filter=["plant", "weed", "crop", "pest", "fruit", "vegetable", "person"],
        custom_settings={
            "weed_detection": True,
            "crop_health": True,
            "yield_estimation": True,
            "outdoor_robustness": True,
        },
    ),
    "retail": TemplateConfig(
        name="Retail",
        description="Shelf monitoring, inventory counting, customer analytics, privacy-preserving",
        detection_model="yolo11n",
        depth_enabled=True,
        depth_skip_frames=12,
        face_enabled=False,
        gesture_enabled=True,
        pose_enabled=False,
        tracking_enabled=True,
        tracking_max_age=300,
        tracking_min_hits=3,
        world_model_enabled=False,
        safety_enabled=False,
        confidence_threshold=0.5,
        classes_filter=["person", "product", "shelf", "cart", "bottle", "can", "box"],
        custom_settings={
            "shelf_monitoring": True,
            "inventory_counting": True,
            "customer_analytics": True,
            "privacy_preserving": True,
        },
    ),
}


class TemplateManager:
    """Manages industry-specific pipeline templates."""

    def __init__(self):
        self._logger = get_logger(__name__)

    def get_template(self, name: str) -> Optional[TemplateConfig]:
        """Get a template by name."""
        name_lower = name.lower().replace("-", "").replace("_", "")
        for key, template in TEMPLATES.items():
            key_normalized = key.lower().replace("-", "").replace("_", "")
            if key_normalized == name_lower:
                return template
        return None

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(TEMPLATES.keys())

    def get_template_info(self, name: str) -> Optional[Dict]:
        """Get template information as dict."""
        template = self.get_template(name)
        if template:
            return {
                "name": template.name,
                "description": template.description,
                "detection_model": template.detection_model,
                "depth_enabled": template.depth_enabled,
                "tracking_enabled": template.tracking_enabled,
                "world_model_enabled": template.world_model_enabled,
                "safety_enabled": template.safety_enabled,
                "confidence_threshold": template.confidence_threshold,
                "classes_filter": template.classes_filter,
                "custom_settings": template.custom_settings,
            }
        return None

    def save_template(self, name: str, config: TemplateConfig, output_path: str) -> str:
        """Save a template to a YAML file."""
        data = {
            "template": {
                "name": config.name,
                "description": config.description,
            },
            "camera": {
                "width": config.camera_width,
                "height": config.camera_height,
                "fps": config.camera_fps,
            },
            "models": {
                "detection": config.detection_model,
                "depth": config.depth_model,
                "depth_enabled": config.depth_enabled,
                "depth_skip_frames": config.depth_skip_frames,
            },
            "features": {
                "face": config.face_enabled,
                "gesture": config.gesture_enabled,
                "pose": config.pose_enabled,
                "tracking": config.tracking_enabled,
                "world_model": config.world_model_enabled,
                "safety": config.safety_enabled,
            },
            "thresholds": {
                "confidence": config.confidence_threshold,
                "iou": config.iou_threshold,
            },
            "classes_filter": config.classes_filter,
            "custom_settings": config.custom_settings,
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self._logger.info(f"Template saved to: {path}")
        return str(path)

    def load_template(self, path: str) -> Optional[TemplateConfig]:
        """Load a template from a YAML file."""
        path = Path(path)
        if not path.exists():
            self._logger.warning(f"Template file not found: {path}")
            return None

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "camera" not in data:
            self._logger.warning(f"Invalid template file: {path}")
            return None

        camera = data.get("camera", {})
        models = data.get("models", {})
        features = data.get("features", {})
        thresholds = data.get("thresholds", {})

        return TemplateConfig(
            name=data.get("template", {}).get("name", path.stem),
            description=data.get("template", {}).get("description", ""),
            camera_width=camera.get("width", 640),
            camera_height=camera.get("height", 480),
            camera_fps=camera.get("fps", 30),
            detection_model=models.get("detection", "yolo11n"),
            depth_model=models.get("depth", "midas-small"),
            depth_enabled=models.get("depth_enabled", True),
            depth_skip_frames=models.get("depth_skip_frames", 8),
            face_enabled=features.get("face", False),
            gesture_enabled=features.get("gesture", False),
            pose_enabled=features.get("pose", False),
            tracking_enabled=features.get("tracking", True),
            world_model_enabled=features.get("world_model", False),
            safety_enabled=features.get("safety", False),
            confidence_threshold=thresholds.get("confidence", 0.5),
            iou_threshold=thresholds.get("iou", 0.45),
            classes_filter=data.get("classes_filter", []),
            custom_settings=data.get("custom_settings", {}),
        )
