#!/usr/bin/env python3
"""Build TensorRT engine from ONNX model for DeepStream."""

import os
import sys
import subprocess

def build_engine(model_name="yolo11n", precision="fp16"):
    """Build TensorRT engine from ONNX."""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "models", f"{model_name}.onnx")
    engine_path = os.path.join(base_dir, "models", f"{model_name}_{precision}.engine")
    
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        return False
    
    print(f"Building TensorRT engine for {model_name}")
    print(f"Input: {onnx_path}")
    print(f"Output: {engine_path}")
    print(f"Precision: {precision}")
    
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--{precision}",
        "--optShapes=input:1x3x640x640",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Engine built successfully!")
        return True
    else:
        print(f"ERROR: {result.stderr}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build TensorRT engine")
    parser.add_argument("--model", default="yolo11n", help="Model name")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"], help="Precision")
    
    args = parser.parse_args()
    
    success = build_engine(args.model, args.precision)
    sys.exit(0 if success else 1)
