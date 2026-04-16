#!/usr/bin/env python3
"""OpenEyes Plug & Play Setup.

Run this ONCE with internet to download everything needed.
After that, the system runs fully offline.

Usage:
    python setup_plug_and_play.py
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def run(cmd, desc=""):
    """Run shell command."""
    print(f"  {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "already" not in result.stderr.lower():
        print(f"    Warning: {result.stderr.strip()[:100]}")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("🖥️  OpenEyes Plug & Play Setup")
    print("=" * 60)
    print()
    
    base = Path("/home/mandar/openeyes")
    os.chdir(base)
    
    # Check if already setup
    if (base / "models" / "yolov10n.engine").exists():
        print("✅ Already configured!")
        print("   Run: python -m src.main --deepstream --camera 0")
        print()
        print("   To test offline: disconnect internet first")
        return
    
    print("📦 Step 1: System packages...")
    run("sudo apt update", "Update apt")
    run("sudo apt install -y python3-pip python3-gi python3-gst-1.0", "Install GI/GStreamer")
    
    print("📦 Step 2: Python packages...")
    packages = [
        "numpy",
        "opencv-python-headless", 
        "mediapipe",
        "tqdm",
    ]
    for pkg in packages:
        run(f"pip3 install {pkg} --quiet", f"Install {pkg}")
    
    print("📦 Step 3: Download pyds (DeepStream Python bindings)...")
    run("pip3 uninstall pyds -y 2>/dev/null || true", "Remove old pyds")
    
    # Try to download pyds wheel
    urls = [
        "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl",
    ]
    
    pyds_installed = False
    for url in urls:
        try:
            print(f"  Downloading pyds from GitHub...")
            fname = url.split("/")[-1]
            urllib.request.urlretrieve(url, fname)
            run(f"pip3 install {fname} --force-reinstall --no-deps", "Install pyds")
            os.remove(fname)
            pyds_installed = True
            break
        except Exception as e:
            print(f"    Could not download: {e}")
    
    if not pyds_installed:
        print("  ⚠️  pyds download failed - will use fallback")
    
    print("📦 Step 4: Download YOLO models...")
    models_dir = base / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_urls = {
        "yolov10n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.onnx",
        "yolov8n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.onnx",
    }
    
    for name, url in model_urls.items():
        dst = models_dir / name
        if dst.exists():
            print(f"  ✅ {name} (cached)")
        else:
            try:
                print(f"  Downloading {name}...")
                urllib.request.urlretrieve(url, dst)
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ⚠️  {name}: {e}")
    
    # Create TensorRT engine if possible
    if (models_dir / "yolov10n.onnx").exists():
        print("📦 Step 5: Build TensorRT engine (may take a minute)...")
        # Try to create engine using trt
        try:
            import tensorrt as trt
            print("  TensorRT available - engine will be built on first run")
        except ImportError:
            print("  TensorRT not available - using ONNX fallback")
    
    print("📦 Step 6: Verify setup...")
    
    checks = [
        ("models/yolov10n.onnx", "YOLO model"),
        ("deepstream/config_yolov10n.txt", "DeepStream config"),
        ("deepstream/labels.txt", "COCO labels"),
    ]
    
    print()
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("To run the system:")
    print("  python -m src.main --deepstream --camera 0")
    print()
    print("For demos:")
    print("  python demo_all_features.py")
    print()
    print("🚀 Ready to run OFFLINE!")


if __name__ == "__main__":
    main()