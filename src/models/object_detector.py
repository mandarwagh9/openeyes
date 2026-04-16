from pathlib import Path
from typing import List, Optional

import numpy as np

from src.camera.types import BoundingBox, Detection
from src.exceptions import ModelError
from src.utils.logger import get_logger


class TensorRTDetector:
    """TensorRT-based YOLO detector for GPU acceleration."""

    def __init__(self, onnx_path: str, fp16: bool = True, int8: bool = False, dla: bool = False, logger=None):
        self.onnx_path = onnx_path
        self.fp16 = fp16
        self.int8 = int8
        self.dla = dla
        self.logger = logger or get_logger(__name__)
        self.engine = None
        self.context = None
        self.input_shape = (1, 3, 640, 640)
        self.output_shape = (1, 84, 8400)
        self._cuda_initialized = False

    def _check_tensorrt_available(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            return True
        except Exception:
            return False

    def build_engine(self, force_rebuild: bool = False) -> bool:
        """Build TensorRT engine from ONNX model."""
        if not self._check_tensorrt_available():
            self.logger.warning("TensorRT not available")
            return False

        import tensorrt as trt

        if self.int8:
            engine_path = self.onnx_path.replace('.onnx', '-int8.engine')
        elif self.dla:
            engine_path = self.onnx_path.replace('.onnx', '-dla.engine')
        else:
            engine_path = self.onnx_path.replace('.onnx', '.engine')
        
        engine_file = Path(engine_path)

        if engine_file.exists() and not force_rebuild:
            self.logger.info(f"Using existing TensorRT engine: {engine_path}")
            return self.load_engine(engine_path)

        self.logger.info("Building TensorRT engine from ONNX (this may take 2-5 minutes)...")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    self.logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if self.int8:
            config.set_flag(trt.BuilderFlag.INT8)
            self.logger.info("Using INT8 precision")
        elif self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("Using FP16 precision")

        if self.dla:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_default_device_type(trt.DeviceType.kDLA)
            config.set_dla_core(0)
            self.logger.info("DLA enabled - will use DLA with GPU fallback")

        self.logger.info("Serializing TensorRT engine...")
        self.engine = builder.build_serialized_network(network, config)

        if self.engine is None:
            self.logger.error("Failed to build TensorRT engine")
            return False

        with open(engine_path, 'wb') as f:
            f.write(self.engine)
        self.logger.info(f"TensorRT engine saved to: {engine_path}")

        return self.load_engine(engine_path)

    def load_engine(self, engine_path: str) -> bool:
        """Load pre-built TensorRT engine."""
        if not self._check_cuda_available():
            self.logger.warning("CUDA not available for TensorRT")
            return False

        try:
            import tensorrt as trt
            import pycuda.driver as cuda

            cuda.init()
            self.logger = cuda.Device(0).make_context()

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(engine_path, 'rb') as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self._cuda_initialized = True
            self.logger.info(f"TensorRT engine loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load TensorRT engine: {e}")
            return False

    def infer(self, input_tensor: np.ndarray) -> Optional[np.ndarray]:
        """Run inference using TensorRT."""
        if not self._cuda_initialized or self.context is None:
            return None

        try:
            import pycuda.driver as cuda

            h_input = cuda.mem_alloc(input_tensor.nbytes)
            h_output = cuda.mem_alloc(np.prod(self.output_shape) * 4)

            cuda.memcpy_htod(h_input, input_tensor)

            self.context.execute_v2([int(h_input), int(h_output)])

            output = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, h_output)

            h_input.free()
            h_output.free()

            return output

        except Exception as e:
            self.logger.error(f"TensorRT inference failed: {e}")
            return None

    def preprocess(self, frame: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
        """Preprocess frame for TensorRT inference."""
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        img = padded.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img


class ObjectDetector:
    """Object detector using YOLO with TensorRT optimization."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        int8: bool = False,
        dla: bool = False,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._int8 = int8
        self._dla = dla
        self._model = None
        self._logger = get_logger(__name__)
        self._using_onnx = False
        self._using_tensorrt = False
        self._onnx_session = None
        self._tensorrt_detector = None
        self._prefer_onnx = ".onnx" in model_path
        self._prefer_engine = ".engine" in model_path

    def load(self) -> None:
        """Load the object detection model."""
        import torch

        model_file = Path(self._model_path)

        if not model_file.exists():
            self._logger.warning(
                f"Model file not found: {self._model_path}. "
                "Will attempt to use YOLO11n from ultralytics."
            )

        if self._prefer_engine:
            self._load_tensorrt_engine_model()
        elif self._prefer_onnx:
            if self._try_load_tensorrt():
                return
            if self._try_load_onnx_model():
                return

        if self._prefer_onnx and not torch.cuda.is_available():
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

    def _load_tensorrt_engine_model(self) -> None:
        """Load TensorRT engine file using Ultralytics."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            self._using_tensorrt = True
            self._logger.info(
                f"Object detector loaded (TensorRT Engine): {self._model_path}"
            )
        except Exception as e:
            self._logger.warning(f"Failed to load TensorRT engine: {e}")
            self._load_pytorch_model()

    def _try_load_tensorrt(self) -> bool:
        """Try to load model using TensorRT."""
        try:
            self._logger.info("Attempting to load TensorRT engine...")
            self._tensorrt_detector = TensorRTDetector(
                self._model_path,
                fp16=not self._int8,
                int8=self._int8,
                dla=self._dla,
                logger=self._logger
            )

            if self._tensorrt_detector.build_engine():
                self._using_tensorrt = True
                self._logger.info("Object detector loaded (TensorRT): {}".format(self._model_path))
                return True
            else:
                self._logger.warning("TensorRT engine build failed, will try ONNX Runtime")
                return False
        except Exception as e:
            self._logger.warning(f"TensorRT loading failed: {e}")
            return False

    def _try_load_onnx_model(self) -> bool:
        """Try to load model using ONNX Runtime."""
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2

            providers = []

            if ort.get_available_providers():
                if "TensorrtExecutionProvider" in ort.get_available_providers():
                    providers.append(
                        (
                            "TensorrtExecutionProvider",
                            {
                                "device_id": 0,
                                "trt_engine_cache_enable": True,
                                "trt_fp16_enable": True,
                                "trt_dla_enable": False,
                            },
                        )
                    )
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.append(
                        (
                            "CUDAExecutionProvider",
                            {
                                "device_id": 0,
                                "arena_extend_strategy": "kSameAsRequested",
                            },
                        )
                    )
                providers.append("CPUExecutionProvider")
            else:
                providers.append("CPUExecutionProvider")

            self._onnx_session = ort.InferenceSession(
                self._model_path,
                sess_options,
                providers=providers,
            )

            self._using_onnx = True
            self._logger.info(
                f"Object detector loaded (ONNX): {self._model_path}"
            )
            self._logger.info(f"ONNX Runtime providers: {providers}")
            return True

        except Exception as e:
            self._logger.warning(
                f"Failed to load ONNX model: {e}. Falling back to PyTorch."
            )
            return False

    def _load_onnx_model(self) -> None:
        """Load ONNX model with ONNX Runtime + TensorRT."""
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2

            providers = []

            if ort.get_available_providers():
                if "TensorrtExecutionProvider" in ort.get_available_providers():
                    providers.append(
                        (
                            "TensorrtExecutionProvider",
                            {
                                "device_id": 0,
                                "trt_engine_cache_enable": True,
                                "trt_fp16_enable": True,
                                "trt_dla_enable": False,
                            },
                        )
                    )
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.append(
                        (
                            "CUDAExecutionProvider",
                            {
                                "device_id": 0,
                                "arena_extend_strategy": "kSameAsRequested",
                            },
                        )
                    )
                providers.append("CPUExecutionProvider")
            else:
                providers.append("CPUExecutionProvider")

            self._onnx_session = ort.InferenceSession(
                self._model_path,
                sess_options,
                providers=providers,
            )

            self._using_onnx = True
            self._logger.info(
                f"Object detector loaded (ONNX): {self._model_path}"
            )
            self._logger.info(f"ONNX Runtime providers: {providers}")

        except Exception as e:
            self._logger.warning(
                f"Failed to load ONNX model: {e}. Falling back to PyTorch."
            )
            self._load_pytorch_model()

    def _load_pytorch_model(self) -> None:
        """Load PyTorch model using ultralytics."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            self._logger.info(
                f"Object detector loaded (PyTorch): {self._model_path}"
            )
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in the given frame."""
        if self._model is None and self._onnx_session is None and self._tensorrt_detector is None:
            raise ModelError("Model not loaded. Call load() first.")

        if self._using_onnx:
            return self._detect_onnx(frame)
        elif self._using_tensorrt:
            if self._tensorrt_detector is not None:
                return self._detect_tensorrt(frame)
            else:
                return self._detect_pytorch(frame)
        else:
            return self._detect_pytorch(frame)

    def _detect_tensorrt(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using TensorRT."""
        if self._tensorrt_detector is None:
            return []

        input_tensor = self._tensorrt_detector.preprocess(frame, (640, 640))

        output = self._tensorrt_detector.infer(input_tensor)

        if output is None:
            self._logger.warning("TensorRT inference returned None, falling back to CPU")
            self._using_tensorrt = False
            return self._detect_pytorch(frame)

        detections = self._postprocess_onnx(
            output, frame.shape, (640, 640)
        )

        return detections

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using ONNX Runtime."""
        input_name = self._onnx_session.get_inputs()[0].name
        output_name = self._onnx_session.get_outputs()[0].name

        input_h, input_w = 640, 640

        img = self._preprocess_onnx(frame, (input_h, input_w))

        outputs = self._onnx_session.run(
            [output_name], {input_name: img}
        )

        detections = self._postprocess_onnx(
            outputs[0], frame.shape, (input_h, input_w)
        )

        return detections

    def _preprocess_onnx(
        self, frame: np.ndarray, target_size: tuple
    ) -> np.ndarray:
        """Preprocess frame for ONNX model."""
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        img = padded.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess_onnx(
        self,
        output: np.ndarray,
        original_shape: tuple,
        input_shape: tuple,
    ) -> List[Detection]:
        """Postprocess ONNX output to detections."""
        import cv2

        predictions = output[0]
        orig_h, orig_w = original_shape[:2]
        input_h, input_w = input_shape

        scale = min(input_w / orig_w, input_h / orig_h)
        pad_w = (input_w - int(orig_w * scale)) // 2
        pad_h = (input_h - int(orig_h * scale)) // 2

        detections = []

        for pred in predictions:
            if len(pred) < 6:
                continue

            x1, y1, x2, y2, conf, cls = pred[:6]

            if conf < self._confidence:
                continue

            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            class_id = int(cls)
            class_name = self._get_class_name(class_id)

            detection = Detection(
                class_name=class_name,
                bbox=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                ),
                confidence=float(conf),
            )
            detections.append(detection)

        detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression."""
        if not detections:
            return detections

        boxes = [
            (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence)
            for d in detections
        ]

        import cv2

        bboxes = np.array(
            [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        )
        scores = np.array([b[4] for b in boxes])

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            self._confidence,
            self._iou_threshold,
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from COCO dataset."""
        coco_classes = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        if 0 <= class_id < len(coco_classes):
            return coco_classes[class_id]
        return f"class_{class_id}"

    def _detect_pytorch(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using PyTorch model."""
        results = self._model(
            frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            verbose=False,
            device="0",
            half=True,
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    detection = Detection(
                        class_name=class_name,
                        bbox=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        confidence=conf,
                    )
                    detections.append(detection)

        return detections

    @property
    def name(self) -> str:
        if self._using_tensorrt:
            return "YOLO11n-TensorRT"
        elif self._using_onnx:
            return "YOLO11n-ONNX"
        return "YOLO11n"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._onnx_session is not None or self._tensorrt_detector is not None
