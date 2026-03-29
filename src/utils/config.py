from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger


class Config:
    def __init__(self, config_path: Optional[Path] = None):
        self._config: Dict[str, Any] = {}
        self._config_path = config_path or self._get_default_config_path()
        self._logger = get_logger(__name__)
        self._load()

    def _get_default_config_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "config.yaml"

    def _load(self) -> None:
        load_dotenv()

        if self._config_path.exists():
            with open(self._config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}

        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        env_mappings = {
            "CAMERA_INDEX": ("camera", "source"),
            "MODEL_PATH": ("models", "yolo", "path"),
            "CONFIDENCE_THRESHOLD": ("models", "yolo", "confidence"),
            "OUTPUT_HOST": ("output", "host"),
            "OUTPUT_PORT": ("output", "port"),
            "DEBUG": ("debug",),
        }

        for env_key, config_path in env_mappings.items():
            value = self._get_env_value(env_key)
            if value is not None:
                self._set_nested(config_path, value)

    def _get_env_value(self, key: str) -> Optional[str]:
        import os
        return os.environ.get(key)

    def _set_nested(self, path: tuple, value: str) -> None:
        d = self._config
        for key in path[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        
        final_key = path[-1]
        if final_key in ("source", "port", "width", "height", "fps", "target_fps"):
            d[final_key] = int(value)
        elif final_key in ("confidence", "iou_threshold"):
            d[final_key] = float(value)
        else:
            d[final_key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        d = self._config
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    @property
    def camera_source(self) -> int:
        return self.get("camera", "source", default=0)

    @property
    def camera_width(self) -> int:
        return self.get("camera", "width", default=640)

    @property
    def camera_height(self) -> int:
        return self.get("camera", "height", default=480)

    @property
    def camera_fps(self) -> int:
        return self.get("camera", "fps", default=30)

    @property
    def yolo_path(self) -> str:
        path = self.get("models", "yolo", "path", default="models/yolov8n.pt")
        base_dir = self._config_path.parent
        if not Path(path).is_absolute():
            path = str(base_dir / path)
        if not Path(path).exists():
            self._logger.warning(f"YOLO model path does not exist: {path}")
        return path

    @property
    def yolo_confidence(self) -> float:
        return self.get("models", "yolo", "confidence", default=0.5)

    @property
    def yolo_iou_threshold(self) -> float:
        return self.get("models", "yolo", "iou_threshold", default=0.45)

    @property
    def output_host(self) -> str:
        return self.get("output", "host", default="127.0.0.1")

    @property
    def output_port(self) -> int:
        return self.get("output", "port", default=5000)

    @property
    def target_fps(self) -> int:
        return self.get("performance", "target_fps", default=30)

    @property
    def use_tensorrt(self) -> bool:
        return self.get("performance", "use_tensorrt", default=True)

    @property
    def tensorrt_precision(self) -> str:
        return self.get("performance", "tensorrt", "precision", default="fp16")

    @property
    def tensorrt_dla_enabled(self) -> bool:
        return self.get("performance", "tensorrt", "dla_enabled", default=False)

    @property
    def tensorrt_dla_core(self) -> int:
        return self.get("performance", "tensorrt", "dla_core", default=0)

    @property
    def batch_inference_enabled(self) -> bool:
        return self.get("performance", "batch_inference", "enabled", default=False)

    @property
    def batch_size(self) -> int:
        return self.get("performance", "batch_inference", "batch_size", default=1)

    @property
    def max_batch_size(self) -> int:
        return self.get("performance", "batch_inference", "max_batch_size", default=4)

    @property
    def performance_monitoring_enabled(self) -> bool:
        return self.get("performance", "monitoring", "enabled", default=True)

    @property
    def performance_stats_interval(self) -> int:
        return self.get("performance", "monitoring", "stats_interval_sec", default=5)

    @property
    def log_performance(self) -> bool:
        return self.get("performance", "monitoring", "log_performance", default=True)

    @property
    def tracking_enabled(self) -> bool:
        return self.get("tracking", "enabled", default=True)

    @property
    def tracking_max_age(self) -> int:
        return self.get("tracking", "max_age", default=30)

    @property
    def tracking_min_hits(self) -> int:
        return self.get("tracking", "min_hits", default=3)

    @property
    def tracking_iou_threshold(self) -> float:
        return self.get("tracking", "iou_threshold", default=0.3)

    @property
    def follow_enabled(self) -> bool:
        return self.get("tracking", "follow_enabled", default=False)

    @property
    def debug(self) -> bool:
        return self.get("debug", default=False)


DEFAULT_CONFIG = """\
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

models:
  yolo:
    path: models/yolov8n.pt
    confidence: 0.5
    iou_threshold: 0.45
  depth:
    enabled: true
    path: models/depth_midas.pt

output:
  format: json
  protocol: udp
  host: 127.0.0.1
  port: 5000
  fps: 30

performance:
  target_fps: 30
  max_latency_ms: 50
  use_tensorrt: false

debug: false
"""


def create_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(DEFAULT_CONFIG)
