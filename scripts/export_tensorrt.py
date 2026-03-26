#!/usr/bin/env python3
"""Export YOLO model to TensorRT engine for Jetson."""

import argparse
from pathlib import Path


def export_to_tensorrt(model_path: str, fp16: bool = True, imgsz: int = 640):
    """Export YOLO model to TensorRT engine."""
    from ultralytics import YOLO

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found: {model_path}")
        return False

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    ext = model_file.suffix
    if ext == ".pt":
        print("PyTorch model detected")
    elif ext == ".onnx":
        print("ONNX model detected")
    else:
        print(f"Unknown model type: {ext}")

    print(f"\nExporting to TensorRT engine...")
    print(f"  FP16: {fp16}")
    print(f"  Image size: {imgsz}")
    print(f"  This may take 2-5 minutes on first run...\n")

    try:
        exported_path = model.export(
            format="engine",
            half=fp16,
            imgsz=imgsz,
            dynamic=False,
            batch=1,
            verbose=True
        )

        print(f"\n✓ Export complete!")
        print(f"  Engine saved to: {exported_path}")
        print(f"\nTo use this engine, update config.yaml:")
        print(f"  models:")
        print(f"    yolo:")
        print(f"      path: {exported_path}")

        return True

    except Exception as e:
        print(f"Error during export: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to TensorRT engine")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="Path to YOLO model (.pt or .onnx)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision (default: True)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )

    args = parser.parse_args()

    export_to_tensorrt(args.model, args.fp16, args.img_size)


if __name__ == "__main__":
    main()