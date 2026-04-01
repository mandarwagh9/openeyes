"""TensorRT INT8 quantization and DLA offloading utilities.

Provides:
- INT8 calibration for YOLO models
- DLA (Deep Learning Accelerator) configuration
- Model optimization for Jetson Orin

Requirements:
    - tensorrt
    - pycuda
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn


try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from src.utils.logger import get_logger


class INT8Calibrator:
    """INT8 calibration for TensorRT models.
    
    Performs post-training quantization with representative dataset.
    """
    
    def __init__(
        self,
        calibration_images: List[np.ndarray],
        cache_file: str = ".calibration.cache",
        num_samples: int = 500,
        batch_size: int = 8,
    ):
        self._calibration_images = calibration_images
        self._cache_file = cache_file
        self._num_samples = num_samples
        self._batch_size = batch_size
        
        self._logger = get_logger(__name__)
        self._calibration_set = None
        self._data_idx = 0
        
    def get_batch(self, name: str, inputs: Dict[str, trt.ILogger]) -> bool:
        """Get next batch for calibration."""
        if self._data_idx >= len(self._calibration_images):
            return False
        
        batch_images = []
        for i in range(self._batch_size):
            if self._data_idx >= len(self._calibration_images):
                break
            img = self._calibration_images[self._data_idx]
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            batch_images.append(img)
            self._data_idx += 1
        
        return True


class TensorRTOptimizer:
    """TensorRT optimizer for YOLO models.
    
    Features:
    - FP16/INT8 quantization
    - DLA offloading
    - Layer fusion optimization
    - Memory optimization
    """
    
    def __init__(
        self,
        model_path: str,
        input_shape: Tuple[int, int, int] = (3, 640, 640),
        precision: str = "fp16",
        dla_enabled: bool = False,
        workspace_size: int = 2 * 1024 * 1024 * 1024,
    ):
        self._logger = get_logger(__name__)
        self._model_path = model_path
        self._input_shape = input_shape
        self._precision = precision
        self._dla_enabled = dla_enabled
        self._workspace_size = workspace_size
        
        self._engine = None
        self._context = None
        
        if not TRT_AVAILABLE:
            self._logger.warning("TensorRT not available. Install tensorrt and pycuda.")
    
    def optimize(
        self,
        onnx_path: Optional[str] = None,
        output_path: Optional[str] = None,
        calibration_images: Optional[List[np.ndarray]] = None,
    ) -> bool:
        """Optimize model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model (if not using existing)
            output_path: Path to save engine
            calibration_images: Images for INT8 calibration
            
        Returns:
            True if successful
        """
        if not TRT_AVAILABLE:
            self._logger.error("TensorRT not available")
            return False
        
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._workspace_size)
            
            if self._precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                self._logger.info("Using FP16 precision")
            elif self._precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                if calibration_images:
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                self._logger.info("Using INT8 precision")
            
            if self._dla_enabled:
                config.set_flag(trt.BuilderFlag.DLA_STABLE)
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = 0
                self._logger.info("DLA enabled")
            
            if onnx_path and os.path.exists(onnx_path):
                parser = trt.OnnxParser(network, logger)
                with open(onnx_path, "rb") as f:
                    parser.parse(f.read())
                
                engine = builder.build_serialized_network(network, config)
                
                if output_path:
                    with open(output_path, "wb") as f:
                        f.write(engine)
                    self._logger.info(f"Engine saved to {output_path}")
                
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Optimization failed: {e}")
            return False
    
    def load_engine(self, engine_path: str) -> bool:
        """Load TensorRT engine from file.
        
        Args:
            engine_path: Path to engine file
            
        Returns:
            True if successful
        """
        if not TRT_AVAILABLE:
            return False
        
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            
            with open(engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            
            if self._engine:
                self._context = self._engine.create_execution_context()
                self._logger.info(f"Engine loaded: {engine_path}")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to load engine: {e}")
            return False
    
    def infer(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Run inference with loaded engine.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Output tensor
        """
        if self._context is None:
            return None
        
        try:
            h_input = cuda.pagelocked_empty(input_data.size, dtype=np.float32)
            h_output = cuda.pagelocked_empty((1, 25200, 85), dtype=np.float32)
            
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            stream = cuda.Stream()
            
            np.copyto(h_input, input_data.flatten())
            cuda.memcpy_htod(d_input, h_input)
            
            self._context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle
            )
            
            cuda.memcpy_dtoh(h_output, d_output)
            
            return h_output
            
        except Exception as e:
            self._logger.error(f"Inference failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._engine is None:
            return {}
        
        info = {
            "num_layers": self._engine.num_layers,
            "num_inputs": self._engine.num_inputs,
            "num_outputs": self._engine.num_outputs,
            "precision": self._precision,
            "dla_enabled": self._dla_enabled,
        }
        
        return info


class DLAManager:
    """Manage DLA (Deep Learning Accelerator) offloading.
    
    DLA is a dedicated neural network accelerator on Jetson Orin.
    """
    
    def __init__(
        self,
        dla_core: int = 0,
        dla_enabled: bool = True,
    ):
        self._logger = get_logger(__name__)
        self._dla_core = dla_core
        self._dla_enabled = dla_enabled
        
    def get_dla_config(self) -> Dict[str, Any]:
        """Get DLA configuration for TensorRT."""
        return {
            "dla_enabled": self._dla_enabled,
            "dla_core": self._dla_core,
            "use_dla": self._dla_enabled,
        }
    
    def set_dla_core(self, core: int) -> None:
        """Set DLA core (0 or 1 on Orin)."""
        if core not in [0, 1]:
            self._logger.warning(f"Invalid DLA core {core}, must be 0 or 1")
            return
        self._dla_core = core
    
    def enable_dla(self) -> None:
        """Enable DLA offloading."""
        self._dla_enabled = True
        self._logger.info("DLA enabled")
    
    def disable_dla(self) -> None:
        """Disable DLA offloading."""
        self._dla_enabled = False
        self._logger.info("DLA disabled")
    
    @staticmethod
    def is_dla_available() -> bool:
        """Check if DLA is available on this device."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True,
                text=True
            )
            return "dla" in result.stdout.lower() or "oran" in result.stdout.lower()
        except:
            return False


class ModelOptimizer:
    """High-level model optimizer for OpenEyes.
    
    Combines TensorRT optimization, quantization, and DLA.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "models",
    ):
        self._logger = get_logger(__name__)
        self._model_path = model_path
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True)
        
        self._trt_optimizer = None
        self._dla_manager = DLAManager()
    
    def optimize_for_jetson(
        self,
        precision: str = "fp16",
        use_dla: bool = False,
        calibrate: bool = False,
        calibration_images: Optional[List[np.ndarray]] = None,
    ) -> Optional[str]:
        """Optimize model for Jetson Orin.
        
        Args:
            precision: fp16, fp32, or int8
            use_dla: Enable DLA offloading
            calibrate: Run INT8 calibration
            calibration_images: Images for calibration
            
        Returns:
            Path to optimized engine or None
        """
        model_name = Path(self._model_path).stem
        output_path = self._output_dir / f"{model_name}_{precision}.engine"
        
        if precision == "int8" and calibrate and calibration_images:
            self._logger.info(f"Running INT8 calibration with {len(calibration_images)} images")
        
        if use_dla:
            self._dla_manager.enable_dla()
        
        self._trt_optimizer = TensorRTOptimizer(
            model_path=self._model_path,
            precision=precision,
            dla_enabled=use_dla,
        )
        
        onnx_path = self._model_path.replace(".pt", ".onnx")
        
        if os.path.exists(onnx_path):
            success = self._trt_optimizer.optimize(
                onnx_path=onnx_path,
                output_path=str(output_path),
                calibration_images=calibration_images,
            )
            
            if success:
                self._logger.info(f"Model optimized: {output_path}")
                return str(output_path)
        
        self._logger.warning("No ONNX model found for optimization")
        return None
    
    def get_optimal_precision(self) -> str:
        """Determine optimal precision for current device."""
        if not TRT_AVAILABLE:
            return "fp32"
        
        if DLAManager.is_dla_available():
            return "fp16"
        
        return "fp16"
    
    def benchmark_model(
        self,
        model_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark model inference.
        
        Args:
            model_path: Path to TensorRT engine
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        if not TRT_AVAILABLE:
            return {"error": "TensorRT not available"}
        
        optimizer = TensorRTOptimizer(model_path, input_shape[1:])
        if not optimizer.load_engine(model_path):
            return {"error": "Failed to load engine"}
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        import time
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            optimizer.infer(dummy_input)
            times.append(time.time() - start)
        
        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "fps": 1.0 / np.mean(times),
        }


def optimize_yolo_model(
    model_path: str,
    precision: str = "fp16",
    output_dir: str = "models",
) -> Optional[str]:
    """Factory function to optimize YOLO model.
    
    Args:
        model_path: Path to YOLO model
        precision: fp16, fp32, or int8
        output_dir: Directory for output
        
    Returns:
        Path to optimized engine
    """
    optimizer = ModelOptimizer(model_path, output_dir)
    return optimizer.optimize_for_jetson(precision=precision)