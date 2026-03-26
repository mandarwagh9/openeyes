#!/usr/bin/env python3
"""Install and setup trt_pose for TensorRT-accelerated pose estimation."""

import subprocess
import os
import sys


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch not installed")
        return False


def check_tensorrt():
    """Check TensorRT availability."""
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("TensorRT not installed")
        return False


def clone_and_install_trt_pose():
    """Clone and install trt_pose."""
    print("\n=== Installing trt_pose ===")
    
    # Check if already installed
    try:
        import trt_pose
        print("trt_pose already installed")
        return True
    except ImportError:
        pass
    
    # Install dependencies
    print("Installing dependencies...")
    deps = [
        "pycuda",
        "torchvision"
    ]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep, "--user"], check=False)
    
    # Clone repo
    print("Cloning trt_pose repository...")
    home = os.path.expanduser("~")
    trt_pose_dir = os.path.join(home, "trt_pose")
    
    if os.path.exists(trt_pose_dir):
        print("trt_pose directory already exists")
    else:
        subprocess.run([
            "git", "clone", "https://github.com/NVIDIA-AI-IOT/trt_pose",
            trt_pose_dir
        ], check=True)
    
    # Install trt_pose
    print("Installing trt_pose...")
    os.chdir(trt_pose_dir)
    subprocess.run([sys.executable, "setup.py", "develop"], check=False)
    
    return True


def test_trt_pose():
    """Test trt_pose."""
    try:
        import trt_pose
        print("trt_pose imported successfully")
        return True
    except Exception as e:
        print(f"trt_pose import failed: {e}")
        return False


def download_models():
    """Download trt_pose models."""
    import urllib.request
    import os
    
    print("\n=== Downloading trt_pose models ===")
    
    models_dir = os.path.join(os.path.expanduser("~"), ".trt_pose", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download human pose model
    model_url = "https://github.com/NVIDIA-AI-IOT/trt_pose/raw/master/tasks/human_pose/resnet18_baseline_att_224x224_A.pth"
    model_path = os.path.join(models_dir, "resnet18_baseline_att_224x224_A.pth")
    
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    else:
        print("Model already exists")
    
    return True


def main():
    print("=== trt_pose Setup ===")
    
    if not check_cuda():
        print("CUDA not available!")
        return False
    
    if not check_tensorrt():
        print("TensorRT not available!")
        return False
    
    clone_and_install_trt_pose()
    test_trt_pose()
    download_models()
    
    print("\n=== Setup Complete ===")
    print("To use trt_pose, update your pose estimator to use TensorRT-accelerated inference")
    
    return True


if __name__ == "__main__":
    main()