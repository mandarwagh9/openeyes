import pytest
from pathlib import Path
from src.utils.config import Config, create_default_config


class TestConfig:
    def test_default_values(self):
        config = Config()
        assert config.camera_source == 0
        assert config.camera_width == 640
        assert config.camera_height == 480
        assert config.camera_fps == 30
        assert config.yolo_confidence == 0.5
        assert config.yolo_iou_threshold == 0.45
        assert config.output_host == "127.0.0.1"
        assert config.output_port == 5000
        assert config.target_fps == 30
        assert config.debug is False

    def test_get_with_keys(self):
        config = Config()
        assert config.get("camera", "source") == 0
        assert config.get("models", "yolo", "confidence") == 0.5
        assert config.get("nonexistent", default="default") == "default"

    def test_create_default_config(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        create_default_config(config_file)
        assert config_file.exists()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OUTPUT_PORT", "6000")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
        config = Config()
        assert config.output_port == 6000
        assert config.yolo_confidence == 0.8
