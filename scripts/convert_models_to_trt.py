#!/usr/bin/env python3
"""Convert models to TensorRT on Jetson.

Run this on Jetson Orin with GPU:
    python scripts/convert_models_to_trt.py
"""

import tensorrt as trt
import numpy as np
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    """Convert ONNX to TensorRT engine."""
    if os.path.exists(engine_path):
        print(f"Already exists: {engine_path}")
        return True
    
    print(f"Converting: {onnx_path}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print(f"Failed to parse {onnx_path}")
            return False
    
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    shape = input_tensor.shape
    
    # Fix dynamic shape
    min_shape = (1, 3, 640, 640)
    opt_shape = (1, 3, 640, 640) 
    max_shape = (1, 3, 640, 640)
    
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    print("Building engine...")
    engine = builder.build_serialized_network(network, config)
    
    if engine:
        # TensorRT 10.x returns IHostMemory directly
        # Use engine.tobytes() or cast to bytes
        if hasattr(engine, 'tobytes'):
            engine_data = engine.tobytes()
        else:
            # For older TensorRT
            engine_data = bytes(engine)
        
        with open(engine_path, 'wb') as f:
            f.write(engine_data)
        print(f"✅ Saved: {engine_path} ({len(engine_data)/1e6:.1f} MB)")
        return True
    
    return False


def main():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Convert face model
    convert_onnx_to_engine(
        f"{models_dir}/yolov8n-face.onnx",
        f"{models_dir}/yolov8n-face.engine"
    )


if __name__ == "__main__":
    main()