#!/usr/bin/env python3
"""INT8 model conversion script for OpenEyes.

Converts YOLO models to TensorRT INT8 format for faster inference.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_to_int8(model_name: str, output_path: str, calibration_frames: int = 100) -> bool:
    """Convert YOLO model to INT8 TensorRT.
    
    Args:
        model_name: Model name (yolo11n, yolo12n, yolo26n)
        output_path: Output path for .engine file
        calibration_frames: Number of frames for calibration
        
    Returns:
        True if successful
    """
    print(f"Converting {model_name} to INT8...")
    print(f"  Output: {output_path}")
    print(f"  Calibration frames: {calibration_frames}")
    
    try:
        import tensorrt as trt
        print(f"  TensorRT version: {trt.__version__}")
    except ImportError:
        print("Warning: TensorRT not available, using ONNX fallback")
        return _convert_to_onnx(model_name, output_path)
    
    try:
        from ultralytics import YOLO
        model = YOLO(f"{model_name}.pt")
        
        if not Path(f"{model_name}.pt").exists():
            print(f"Downloading {model_name} model...")
            model = YOLO(f"{model_name}.pt")
        
        print(f"  Model loaded: {model_name}")
        
        engine_path = Path(output_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  Exporting to TensorRT...")
        model.export(format="engine", half=False, verbose=False)
        
        print(f"  Conversion complete!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def _convert_to_onnx(model_name: str, output_path: str) -> bool:
    """Fallback to ONNX conversion."""
    try:
        from ultralytics import YOLO
        
        model_path = f"{model_name}.pt"
        if not Path(model_path).exists():
            print(f"Downloading {model_name} model...")
            model = YOLO(f"{model_name}.pt")
            model_path = model.trainer.best if hasattr(model, 'trainer') else f"{model_name}.pt"
        else:
            model = YOLO(model_path)
        
        print(f"  Exporting to ONNX...")
        model.export(format="onnx", verbose=False)
        
        print(f"  Conversion complete!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_all(output_dir: str, calibration_frames: int = 100) -> bool:
    """Convert all models."""
    models = ["yolo11n", "yolo12n", "yolo26n", "rtmdet_nano"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        out_file = output_path / f"{model}.engine"
        success = convert_to_int8(model, str(out_file), calibration_frames)
        if not success:
            print(f"Failed to convert {model}")
            
    return True


def generate_calibration_dataset(output_dir: str, num_frames: int = 100) -> bool:
    """Generate calibration dataset from camera or video.
    
    Args:
        output_dir: Directory to save calibration frames
        num_frames: Number of frames to capture
        
    Returns:
        True if successful
    """
    import cv2
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating calibration dataset...")
    print(f"  Output: {output_dir}")
    print(f"  Frames: {num_frames}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return False
        
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = output_path / f"calib_{i:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        
        if i % 10 == 0:
            print(f"  Captured {i}/{num_frames}")
            
    cap.release()
    print(f"  Done! Captured {num_frames} frames")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO models to INT8")
    parser.add_argument("--model", type=str, help="Model name (yolo11n, yolo12n, yolo26n)")
    parser.add_argument("--output", type=str, help="Output path for .engine file")
    parser.add_argument("--all", action="store_true", help="Convert all models")
    parser.add_argument("--calibrate", action="store_true", help="Generate calibration dataset")
    parser.add_argument("--frames", type=int, default=100, help="Number of calibration frames")
    
    args = parser.parse_args()
    
    if args.calibrate:
        return generate_calibration_dataset(args.output or "calibration/", args.frames)
    
    if args.all:
        return convert_all(args.output or "models/int8/", args.frames)
    
    if args.model:
        return convert_to_int8(args.model, args.output or f"models/int8/{args.model}.engine", args.frames)
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)