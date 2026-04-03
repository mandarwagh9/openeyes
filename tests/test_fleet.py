import pytest
import time
import json

from src.fleet.protocol import DeviceHeartbeat, ModelDeployment, FleetClient
from src.fleet.model_registry import ModelRegistry, ModelVersion


class TestDeviceHeartbeat:
    def test_creation(self):
        hb = DeviceHeartbeat(
            device_id="robot-01",
            group="warehouse",
            platform="jetson-orin-nano",
            uptime_seconds=3600,
            fps=5.5,
            latency_ms=200.0,
            cpu_percent=45.0,
            gpu_percent=80.0,
            memory_percent=60.0,
            temperature_c=55.0,
        )
        assert hb.device_id == "robot-01"
        assert hb.group == "warehouse"
        assert hb.fps == 5.5

    def test_to_dict(self):
        hb = DeviceHeartbeat(
            device_id="robot-01",
            group="warehouse",
            platform="jetson-orin-nano",
            uptime_seconds=3600,
            fps=5.5,
            latency_ms=200.0,
            cpu_percent=45.0,
            gpu_percent=80.0,
            memory_percent=60.0,
            temperature_c=55.0,
        )
        d = hb.to_dict()
        assert d["device_id"] == "robot-01"
        assert d["fps"] == 5.5
        assert isinstance(d["timestamp"], float)

    def test_from_dict(self):
        data = {
            "device_id": "robot-02",
            "group": "qa",
            "platform": "pi5",
            "uptime_seconds": 7200,
            "fps": 3.0,
            "latency_ms": 300.0,
            "cpu_percent": 30.0,
            "gpu_percent": 0.0,
            "memory_percent": 50.0,
            "temperature_c": 40.0,
            "model_versions": {"detection": "yolo26n-v1.0"},
            "error_count": 0,
            "timestamp": time.time(),
        }
        hb = DeviceHeartbeat.from_dict(data)
        assert hb.device_id == "robot-02"
        assert hb.model_versions == {"detection": "yolo26n-v1.0"}

    def test_json_roundtrip(self):
        hb = DeviceHeartbeat(
            device_id="robot-01",
            group="warehouse",
            platform="jetson-orin-nano",
            uptime_seconds=3600,
            fps=5.5,
            latency_ms=200.0,
            cpu_percent=45.0,
            gpu_percent=80.0,
            memory_percent=60.0,
            temperature_c=55.0,
        )
        json_str = hb.to_json()
        restored = DeviceHeartbeat.from_json(json_str)
        assert restored.device_id == hb.device_id
        assert restored.fps == hb.fps
        assert restored.group == hb.group


class TestModelDeployment:
    def test_creation(self):
        dep = ModelDeployment(
            model_name="yolo26n",
            version="v1.2",
            checksum="abc123",
            download_url="https://example.com/yolo26n.engine",
            target_devices=["robot-01", "robot-02"],
            target_groups=["warehouse"],
        )
        assert dep.model_name == "yolo26n"
        assert dep.version == "v1.2"

    def test_is_targeted_by_device(self):
        dep = ModelDeployment(
            model_name="yolo26n",
            version="v1.2",
            checksum="abc123",
            download_url="https://example.com/yolo26n.engine",
            target_devices=["robot-01"],
        )
        assert dep.is_targeted("robot-01", "warehouse") is True
        assert dep.is_targeted("robot-02", "warehouse") is False

    def test_is_targeted_by_group(self):
        dep = ModelDeployment(
            model_name="yolo26n",
            version="v1.2",
            checksum="abc123",
            download_url="https://example.com/yolo26n.engine",
            target_devices=["*"],
            target_groups=["warehouse"],
        )
        assert dep.is_targeted("robot-01", "warehouse") is True
        assert dep.is_targeted("robot-02", "qa") is False

    def test_is_targeted_wildcard(self):
        dep = ModelDeployment(
            model_name="yolo26n",
            version="v1.2",
            checksum="abc123",
            download_url="https://example.com/yolo26n.engine",
            target_devices=["*"],
        )
        assert dep.is_targeted("any-device", "any-group") is True

    def test_json_roundtrip(self):
        dep = ModelDeployment(
            model_name="yolo26n",
            version="v1.2",
            checksum="abc123",
            download_url="https://example.com/yolo26n.engine",
            target_devices=["*"],
            target_groups=["warehouse"],
            rollback_version="v1.1",
            requires_restart=True,
        )
        json_str = dep.to_json()
        restored = ModelDeployment.from_json(json_str)
        assert restored.model_name == dep.model_name
        assert restored.rollback_version == dep.rollback_version
        assert restored.requires_restart is True


class TestFleetClient:
    def test_creation(self):
        client = FleetClient(device_id="robot-01", server_url="http://fleet.local")
        assert client.device_id == "robot-01"
        assert client.server_url == "http://fleet.local"

    def test_create_heartbeat(self):
        client = FleetClient(device_id="robot-01")
        client.set_group("warehouse")
        client.set_platform("jetson-orin-nano")

        hb = client.create_heartbeat(
            fps=5.5,
            latency_ms=200.0,
            cpu_percent=45.0,
            temperature_c=55.0,
        )
        assert hb.device_id == "robot-01"
        assert hb.group == "warehouse"
        assert hb.platform == "jetson-orin-nano"
        assert hb.fps == 5.5
        assert hb.uptime_seconds >= 0

    def test_should_send_heartbeat(self):
        client = FleetClient(device_id="robot-01")
        client.set_heartbeat_interval(1)

        assert client.should_send_heartbeat() is True
        assert client.should_send_heartbeat() is False

    def test_heartbeat_interval(self):
        client = FleetClient(device_id="robot-01")
        assert client._heartbeat_interval == 30
        client.set_heartbeat_interval(60)
        assert client._heartbeat_interval == 60


class TestModelRegistry:
    def test_register_model(self):
        registry = ModelRegistry()
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"fake model data")
            tmp_path = f.name

        try:
            model = registry.register_model(
                name="yolo26n",
                version="v1.0",
                file_path=tmp_path,
                notes="Initial release",
            )
            assert model.name == "yolo26n"
            assert model.version == "v1.0"
            assert len(model.checksum) == 64
            assert model.file_size_bytes == 15
        finally:
            os.unlink(tmp_path)

    def test_get_model(self):
        registry = ModelRegistry()
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"fake model data")
            tmp_path = f.name

        try:
            registry.register_model("yolo26n", "v1.0", tmp_path)
            model = registry.get_model("yolo26n", "v1.0")
            assert model is not None
            assert model.name == "yolo26n"
        finally:
            os.unlink(tmp_path)

    def test_get_latest_version(self):
        registry = ModelRegistry()
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"fake model data v1")
            tmp_path1 = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"fake model data v2")
            tmp_path2 = f.name

        try:
            registry.register_model("yolo26n", "v1.0", tmp_path1)
            import time
            time.sleep(0.01)
            registry.register_model("yolo26n", "v2.0", tmp_path2)

            latest = registry.get_latest_version("yolo26n")
            assert latest.version == "v2.0"
        finally:
            os.unlink(tmp_path1)
            os.unlink(tmp_path2)

    def test_list_models(self):
        registry = ModelRegistry()
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"model1")
            tmp1 = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"model2")
            tmp2 = f.name

        try:
            registry.register_model("yolo26n", "v1.0", tmp1)
            registry.register_model("da3-small", "v1.0", tmp2)

            models = registry.list_models()
            assert "yolo26n" in models
            assert "da3-small" in models
        finally:
            os.unlink(tmp1)
            os.unlink(tmp2)

    def test_mark_deployed(self):
        registry = ModelRegistry()
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".engine") as f:
            f.write(b"model")
            tmp_path = f.name

        try:
            registry.register_model("yolo26n", "v1.0", tmp_path)
            registry.mark_deployed("yolo26n", "v1.0", "robot-01")
            registry.mark_deployed("yolo26n", "v1.0", "robot-02")

            model = registry.get_model("yolo26n", "v1.0")
            assert "robot-01" in model.deployed_to
            assert "robot-02" in model.deployed_to
        finally:
            os.unlink(tmp_path)
