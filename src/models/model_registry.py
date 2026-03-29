from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger


class ModelType(Enum):
    YOLO11 = "yolo11"
    YOLO12 = "yolo12"
    RTMDET = "rtmdet"
    GRASP = "grasp"
    FALL_DETECTION = "fall_detection"


@dataclass
class ModelConfig:
    name: str
    model_type: ModelType
    default_path: str
    input_size: int
    classes: List[str]
    description: str
    recommended_fps: int


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "yolo11n": ModelConfig(
        name="YOLO11n",
        model_type=ModelType.YOLO11,
        default_path="models/yolo11n.onnx",
        input_size=640,
        classes=[
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ],
        description="Ultralytics YOLO11n - fastest YOLO model",
        recommended_fps=30,
    ),
    "yolo11s": ModelConfig(
        name="YOLO11s",
        model_type=ModelType.YOLO11,
        default_path="models/yolo11s.onnx",
        input_size=640,
        classes=[
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ],
        description="Ultralytics YOLO11s - small balanced model",
        recommended_fps=25,
    ),
    "yolo12n": ModelConfig(
        name="YOLO12n",
        model_type=ModelType.YOLO12,
        default_path="models/yolo12n.onnx",
        input_size=640,
        classes=[
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ],
        description="Ultralytics YOLO12n - latest YOLO version",
        recommended_fps=30,
    ),
    "rtmdet_nano": ModelConfig(
        name="RTMDet-nano",
        model_type=ModelType.RTMDET,
        default_path="models/rtmdet_nano.onnx",
        input_size=640,
        classes=[
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ],
        description="Real-Time Detection - efficient alternative to YOLO",
        recommended_fps=28,
    ),
    "grasp_detector": ModelConfig(
        name="GraspDetector",
        model_type=ModelType.GRASP,
        default_path="models/grasp_model.onnx",
        input_size=640,
        classes=["grasp_point"],
        description="Grasping point detection for robot manipulation",
        recommended_fps=15,
    ),
    "fall_detector": ModelConfig(
        name="FallDetector",
        model_type=ModelType.FALL_DETECTION,
        default_path="models/fall_model.onnx",
        input_size=640,
        classes=["standing", "falling", "fallen"],
        description="Human fall detection for safety monitoring",
        recommended_fps=20,
    ),
}


class ModelRegistry:
    """Registry for managing different vision models."""

    _logger = get_logger(__name__)

    @staticmethod
    def get_model(model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        model_name_lower = model_name.lower()
        
        if model_name_lower in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_name_lower]
        
        for key in MODEL_REGISTRY:
            if model_name_lower in key or key in model_name_lower:
                return MODEL_REGISTRY[key]
        
        return None

    @staticmethod
    def list_models() -> List[str]:
        """List all available model names."""
        return list(MODEL_REGISTRY.keys())

    @staticmethod
    def list_models_by_type(model_type: ModelType) -> List[str]:
        """List models filtered by type."""
        return [
            name for name, config in MODEL_REGISTRY.items()
            if config.model_type == model_type
        ]

    @staticmethod
    def get_detection_models() -> List[str]:
        """Get list of general detection models."""
        return ModelRegistry.list_models_by_type(ModelType.YOLO11) + \
               ModelRegistry.list_models_by_type(ModelType.YOLO12) + \
               ModelRegistry.list_models_by_type(ModelType.RTMDET)

    @staticmethod
    def get_specialized_models() -> List[str]:
        """Get list of specialized models."""
        return ModelRegistry.list_models_by_type(ModelType.GRASP) + \
               ModelRegistry.list_models_by_type(ModelType.FALL_DETECTION)

    @staticmethod
    def suggest_model(fps_target: int = 30, specialized: bool = False) -> str:
        """Suggest best model for target FPS."""
        if specialized:
            models = ModelRegistry.get_specialized_models()
        else:
            models = ModelRegistry.get_detection_models()

        for name in models:
            config = MODEL_REGISTRY[name]
            if config.recommended_fps >= fps_target:
                return name

        return "yolo11n"

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information."""
        config = ModelRegistry.get_model(model_name)
        if config:
            return {
                "name": config.name,
                "type": config.model_type.value,
                "path": config.default_path,
                "input_size": config.input_size,
                "num_classes": len(config.classes),
                "description": config.description,
                "recommended_fps": config.recommended_fps,
            }
        return None
