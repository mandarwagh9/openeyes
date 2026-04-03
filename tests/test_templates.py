import pytest
import tempfile
import os

from src.templates import TemplateConfig, TemplateManager, TEMPLATES


class TestTemplateManager:
    def test_list_templates(self):
        tm = TemplateManager()
        templates = tm.list_templates()
        assert len(templates) == 4
        assert "warehouse" in templates
        assert "manufacturing-qa" in templates
        assert "agriculture" in templates
        assert "retail" in templates

    def test_get_template_warehouse(self):
        tm = TemplateManager()
        template = tm.get_template("warehouse")
        assert template is not None
        assert template.name == "Warehouse/Logistics"
        assert template.depth_enabled is True
        assert template.face_enabled is False
        assert template.world_model_enabled is True
        assert template.safety_enabled is True
        assert template.confidence_threshold == 0.6
        assert len(template.classes_filter) > 0

    def test_get_template_manufacturing(self):
        tm = TemplateManager()
        template = tm.get_template("manufacturing-qa")
        assert template is not None
        assert template.name == "Manufacturing QA"
        assert template.face_enabled is True
        assert template.gesture_enabled is True
        assert template.pose_enabled is True
        assert template.confidence_threshold == 0.7

    def test_get_template_agriculture(self):
        tm = TemplateManager()
        template = tm.get_template("agriculture")
        assert template is not None
        assert template.name == "Agriculture"
        assert template.depth_skip_frames == 16
        assert template.tracking_max_age == 120
        assert template.confidence_threshold == 0.4

    def test_get_template_retail(self):
        tm = TemplateManager()
        template = tm.get_template("retail")
        assert template is not None
        assert template.name == "Retail"
        assert template.tracking_max_age == 300
        assert template.face_enabled is False
        assert template.gesture_enabled is True

    def test_get_template_not_found(self):
        tm = TemplateManager()
        template = tm.get_template("nonexistent")
        assert template is None

    def test_get_template_info(self):
        tm = TemplateManager()
        info = tm.get_template_info("warehouse")
        assert info is not None
        assert "name" in info
        assert "description" in info
        assert info["detection_model"] == "yolo11n"

    def test_save_and_load_template(self):
        tm = TemplateManager()
        template = tm.get_template("warehouse")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            tmp_path = f.name

        try:
            tm.save_template("warehouse", template, tmp_path)
            assert os.path.exists(tmp_path)

            loaded = tm.load_template(tmp_path)
            assert loaded is not None
            assert loaded.name == template.name
            assert loaded.depth_enabled == template.depth_enabled
            assert loaded.confidence_threshold == template.confidence_threshold
            assert loaded.classes_filter == template.classes_filter
        finally:
            os.unlink(tmp_path)

    def test_load_template_not_found(self):
        tm = TemplateManager()
        loaded = tm.load_template("/nonexistent/path.yaml")
        assert loaded is None

    def test_all_templates_have_required_fields(self):
        for name, template in TEMPLATES.items():
            assert template.name, f"Template {name} missing name"
            assert template.description, f"Template {name} missing description"
            assert template.detection_model, f"Template {name} missing detection_model"
            assert isinstance(template.classes_filter, list)
            assert isinstance(template.custom_settings, dict)
