#!/usr/bin/env python3
"""Export YOLO model to TensorRT engine with SOTA optimizations for Jetson Orin Nano.

Usage:
    python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt
    python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt --int8 --calib-dir /path/to/images

Optimizations:
    --best: Exhaustive tactic search (5-15% faster, +5min build time)
    --useCudaGraph: CUDA graph capture (0.5ms -> 0.02ms CPU enqueue)
    --int8: INT8 quantization with calibration (1.6-1.9x speedup)
    --workspace: Builder workspace size (default 2GB)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def export_yolo_to_onnx(model_path: str, imgsz: int = 640) -> str:
    """Export YOLO model to ONNX format."""
    from ultralytics import YOLO

    print(f"[1/3] Exporting {model_path} to ONNX...")
    model = YOLO(model_path)

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        batch=1,
        simplify=True,
        opset=17,
        verbose=False,
    )

    print(f"  ONNX saved to: {onnx_path}")
    return str(onnx_path)


def quantize_int8(onnx_path: str, calib_dir: str, output_path: str, num_images: int = 32):
    """Quantize ONNX model to INT8 using NVIDIA modelopt."""
    print(f"[2/3] INT8 quantization with {num_images} calibration images...")

    try:
        import modelopt
        print(f"  Using modelopt {modelopt.__version__}")
    except ImportError:
        print("  Installing nvidia-modelopt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--extra-index-url", "https://pypi.nvidia.com",
            "nvidia-modelopt[onnx]"
        ])

    cmd = [
        sys.executable, "-m", "modelopt.onnx.quantization",
        "--onnx_path", onnx_path,
        "--quantize_mode", "int8",
        "--output_path", output_path,
        "--calibration_data_dir", calib_dir,
        "--num_calibration_batches", str(num_images),
    ]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"  INT8 ONNX saved to: {output_path}")


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 int8: bool = False, workspace_gb: int = 2, imgsz: int = 640):
    """Build TensorRT engine with SOTA optimizations."""
    print(f"[3/3] Building TensorRT engine...")

    shape = f"1x3x{imgsz}x{imgsz}"
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=images:{shape}",
        f"--optShapes=images:{shape}",
        f"--maxShapes=images:{shape}",
        f"--workspace={workspace_gb * 1024}",
        "--best",
        "--useCudaGraph",
        "--noDataTransfers",
        "--useSpinWait",
        "--warmUp=200",
        "--duration=5",
        "--avgRuns=100",
    ]

    if int8:
        cmd.append("--stronglyTyped")
    elif fp16:
        cmd.append("--fp16")

    print(f"  Running: trtexec --best --useCudaGraph {'--fp16' if fp16 else '--int8'}")
    print(f"  This may take 3-8 minutes with --best flag...\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n  WARNING: trtexec returned non-zero. Trying without --best...")
        cmd_no_best = [c for c in cmd if c != "--best"]
        subprocess.run(cmd_no_best, capture_output=False, text=True)

    print(f"\n  Engine saved to: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to optimized TensorRT engine")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt",
                        help="Path to YOLO model (.pt or .onnx)")
    parser.add_argument("--int8", action="store_true",
                        help="Enable INT8 quantization (requires --calib-dir)")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory with calibration images for INT8")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--workspace", type=int, default=2, help="Builder workspace (GB)")
    parser.add_argument("--output", type=str, default=None, help="Output engine path")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    output_dir = model_path.parent
    if args.output:
        engine_path = args.output
    else:
        suffix = "_int8" if args.int8 else "_fp16"
        engine_path = str(output_dir / f"{model_path.stem}{suffix}.engine")

    if model_path.suffix == ".pt":
        onnx_path = export_yolo_to_onnx(str(model_path), args.img_size)
    else:
        onnx_path = str(model_path)

    int8_onnx = None
    if args.int8:
        if not args.calib_dir:
            print("Error: --calib-dir required for INT8 quantization")
            sys.exit(1)
        if not os.path.isdir(args.calib_dir):
            print(f"Error: Calibration directory not found: {args.calib_dir}")
            sys.exit(1)
        int8_onnx = str(output_dir / f"{model_path.stem}_int8.onnx")
        quantize_int8(onnx_path, args.calib_dir, int8_onnx)

    build_engine(
        int8_onnx or onnx_path,
        engine_path,
        fp16=not args.int8,
        int8=args.int8,
        workspace_gb=args.workspace,
        imgsz=args.img_size,
    )

    print(f"\n{'='*50}")
    print(f"Engine: {engine_path}")
    print(f"Update config.yaml:")
    print(f"  models:")
    print(f"    yolo:")
    print(f"      path: {engine_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
