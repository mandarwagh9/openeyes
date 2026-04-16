#!/usr/bin/env python3
"""Rebuild TensorRT engine compatible with current DeepStream/TensorRT version."""

import os
import sys
import tensorrt as trt
from pathlib import Path

def build_engine(onnx_path, engine_path, precision='fp16'):
    """Build TensorRT engine."""
    
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX not found: {onnx_path}")
        return False
    
    engine_file = Path(engine_path)
    if engine_file.exists():
        print(f"Removing old engine: {engine_path}")
        engine_file.unlink()
    
    print(f"Building engine from: {onnx_path}")
    print(f"Precision: {precision}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    
    print("Building engine (this takes 2-5 minutes)...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"Engine saved to: {engine_path}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "models", f"{args.model}.onnx")
    engine_path = os.path.join(base_dir, "models", f"{args.model}.engine")
    
    success = build_engine(onnx_path, engine_path, args.precision)
    sys.exit(0 if success else 1)
