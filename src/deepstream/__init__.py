# DeepStream module for OpenEyes
# High-performance inference pipeline for Jetson Orin Nano

from .pipeline import (
    DeepStreamPipeline,
    DeepStreamMultiCameraPipeline,
    run_deepstream,
    DetectionResult,
)

__all__ = [
    'DeepStreamPipeline',
    'DeepStreamMultiCameraPipeline', 
    'run_deepstream',
    'DetectionResult',
]
