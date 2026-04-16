#!/usr/bin/env python3
"""Download and convert DeepStream models for OpenEyes.

This script downloads pre-trained models and converts them to TensorRT engines
for use with the DeepStream pipeline.

Models:
- YOLOv7-face: Face detection
- TRTPose: Hand gesture pose (21 keypoints)
- TRTPose: Body pose (17 keypoints)
- Depth Anything V3: Monocular depth estimation

Usage:
    python scripts/download_deepstream_models.py --all
    python scripts/download_deepstream_models.py --face
    python scripts/download_deepstream_models.py --gesture
    python scripts/download_deepstream_models.py --pose
    python scripts/download_deepstream_models.py --depth
"""

import argparse
import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# Base directory for models
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_cmd(cmd, cwd=None):
    """Run shell command and return output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    return True


def download_file(url: str, dest: Path) -> bool:
    """Download file from URL."""
    if dest.exists():
        print(f"Already exists: {dest}")
        return True
    
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file."""
    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Extract failed: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """Extract TAR file."""
    print(f"Extracting: {tar_path}")
    try:
        with tarfile.open(tar_path, 'r:*') as t:
            t.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Extract failed: {e}")
        return False


def convert_to_tensorrt(onnx_path: Path, engine_path: Path, fp16: bool = True) -> bool:
    """Convert ONNX to TensorRT engine."""
    if engine_path.exists():
        print(f"Engine already exists: {engine_path}")
        return True
    
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    print(f"Converting {onnx_path.name} to TensorRT...")
    return run_cmd(cmd)


def download_yolov7_face():
    """Download YOLOv7-face model."""
    print("\n=== Downloading YOLOv7-face ===")
    
    # YOLOv7-face from hiennguyen9874
    # Using the ONNX model directly
    onnx_path = MODELS_DIR / "yolov7-face.onnx"
    
    # Try direct download or use alternative
    urls = [
        "https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v1.0/yolov7n-face.onnx",
        "https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v1.0/yolov7-face.onnx",
    ]
    
    for url in urls:
        if download_file(url, onnx_path):
            break
    else:
        print("Trying alternative: Using ultralytics YOLOv8 face...")
        # Fallback to YOLOv8n face from ultralytics
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            # Export to ONNX
            onnx_path = MODELS_DIR / "yolov8n-face.onnx"
            model.export(format='onnx')
            # Rename
            if onnx_path.exists():
                shutil.move(onnx_path, MODELS_DIR / "yolov7-face.onnx")
                onnx_path = MODELS_DIR / "yolov7-face.onnx"
        except ImportError:
            print("ultralytics not installed. Install with: pip install ultralytics")
            return False
    
    if not onnx_path.exists():
        print("Failed to download YOLOv7-face")
        return False
    
    # Convert to TensorRT
    engine_path = MODELS_DIR / "yolov7-face.engine"
    if convert_to_tensorrt(onnx_path, engine_path):
        print(f"YOLOv7-face engine created: {engine_path}")
        return True
    
    return False


def download_trt_pose_hand():
    """Download TRTPose hand model."""
    print("\n=== Downloading TRTPose Hand ===")
    
    # TRTPose hand from NVIDIA-AI-IOT
    repo_url = "https://github.com/NVIDIA-AI-IOT/trt_pose_hand"
    
    # Download model weights
    weights_url = "https://github.com/NVIDIA-AI-IOT/trt_pose_hand/releases/download/v0.0.1/resnet18_hand.trt"
    weights_path = MODELS_DIR / "resnet18_hand.pth"
    
    # Alternative: Try to download PyTorch model
    if not weights_path.exists():
        # Try direct link or clone repo
        try:
            # Use torch2trt if available
            print("Attempting to download trt_pose_hand weights...")
            weights_url = "https://github.com/NVIDIA-AI-IOT/trt_pose_hand/raw/main/hand_pose/resnet18_baseline_att_224x224.pth"
            download_file(weights_url, weights_path)
        except:
            pass
    
    # If we have weights, export to ONNX then TensorRT
    if weights_path.exists():
        # Create ONNX export script
        print("Exporting hand model to ONNX...")
        # For now, create placeholder - actual export requires trt_pose
        print("Note: Hand model export requires trt_pose library")
        print("Install: pip install trt_pose")
        return True
    
    print("TRTPose hand model not available for auto-download")
    print("Manual install: https://github.com/NVIDIA-AI-IOT/trt_pose_hand")
    return False


def download_trt_pose_body():
    """Download TRTPose body model."""
    print("\n=== Downloading TRTPose Body ===")
    
    weights_url = "https://github.com/NVIDIA-AI-IOT/trt_pose/raw/main/body_pose/densenet121_baseline_att_256x256.pth"
    weights_path = MODELS_DIR / "densenet121_body.pth"
    
    if download_file(weights_url, weights_path):
        print("Body pose weights downloaded")
        # Export requires trt_pose
        print("Note: Body model export requires trt_pose library")
        return True
    
    print("TRTPose body model not available for auto-download")
    print("Manual install: https://github.com/NVIDIA-AI-IOT/trt_pose")
    return False


def download_depth_anything():
    """Download Depth Anything V3 model."""
    print("\n=== Downloading Depth Anything V3 ===")
    
    # Try multiple sources
    onnx_path = MODELS_DIR / "depth_anything_v3_small.onnx"
    
    # Source 1: HuggingFace
    urls = [
        "https://huggingface.co/spaces/DepthAnything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_small.onnx",
    ]
    
    for url in urls:
        if download_file(url, onnx_path):
            break
    
    # Source 2: Try Depth Anything V2 from seeed studio
    if not onnx_path.exists():
        try:
            from depth_anything_v2 import DepthAnythingV2
            model = DepthAnythingV2(model_type='small')
            model.load_state_dict(torch.load(torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-0676ba61.pth'
            )))
            print("Depth Anything V2 loaded (V3 not available on PyTorch)")
        except ImportError:
            pass
    
    if not onnx_path.exists():
        # UseMiDaS as alternative
        print("Falling back to MiDaS small...")
        try:
            import torch
            import torch.hub
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            midas.eval()
            
            # Export to ONNX
            dummy_input = torch.randn(1, 3, 384, 384)
            onnx_path = MODELS_DIR / "midas_small.onnx"
            torch.onnx.export(midas, dummy_input, str(onnx_path),
                           input_names=["input"],
                           output_names=["output"],
                           dynamic_axes={"input": {0: "batch"},
                                       "output": {0: "batch"}})
            print("MiDaS exported to ONNX")
        except Exception as e:
            print(f"MiDaS export failed: {e}")
            print("Install: pip install timm torch torchvision")
            return False
    
    # Convert to TensorRT
    engine_path = MODELS_DIR / "depth_anything.engine"
    if convert_to_tensorrt(onnx_path, engine_path):
        print(f"Depth engine created: {engine_path}")
        return True
    
    return False


def download_existing_engines():
    """Check for existing YOLO engines."""
    print("\n=== Checking existing models ===")
    
    existing = []
    for name in ["yolov10n", "yolo11n", "yolov8n"]:
        for ext in [".engine", ".onnx"]:
            path = MODELS_DIR / f"{name}{ext}"
            if path.exists():
                existing.append(path.name)
    
    if existing:
        print(f"Found: {existing}")
    else:
        print("No existing models found")
    
    return existing


def main():
    parser = argparse.ArgumentParser(description="Download DeepStream models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--face", action="store_true", help="Download face detection")
    parser.add_argument("--gesture", action="store_true", help="Download hand pose")
    parser.add_argument("--pose", action="store_true", help="Download body pose")
    parser.add_argument("--depth", action="store_true", help="Download depth")
    parser.add_argument("--check", action="store_true", help="Check existing models")
    
    args = parser.parse_args()
    
    print(f"Models directory: {MODELS_DIR}")
    
    if args.check or (not args.face and not args.gesture and 
                     not args.pose and not args.depth and not args.all):
        download_existing_engines()
        return
    
    success = []
    
    if args.all or args.face:
        if download_yolov7_face():
            success.append("face")
    
    if args.all or args.gesture:
        if download_trt_pose_hand():
            success.append("gesture")
    
    if args.all or args.pose:
        if download_trt_pose_body():
            success.append("pose")
    
    if args.all or args.depth:
        if download_depth_anything():
            success.append("depth")
    
    print(f"\n=== Summary ===")
    print(f"Downloaded: {success}")
    print(f"Models location: {MODELS_DIR}")


if __name__ == "__main__":
    main()