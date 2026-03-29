#!/usr/bin/env python3
"""Jetson optimization helper for OpenEyes."""

import os
import sys
import subprocess
import json
from pathlib import Path


def is_jetson() -> bool:
    """Check if running on a Jetson device."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return "jetson" in model or "tegra" in model
    except Exception:
        return False


def get_jetson_info() -> dict:
    """Get Jetson system information."""
    info = {
        "is_jetson": False,
        "model": "Unknown",
        "jetpack": "Unknown",
        "cuda": "Unknown",
        "tensorrt": "Unknown",
        "power_mode": "Unknown",
    }

    try:
        with open("/proc/device-tree/model", "r") as f:
            info["model"] = f.read().strip()
            info["is_jetson"] = "jetson" in info["model"].lower()
    except Exception:
        pass

    if info["is_jetson"]:
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5
            )
            info["power_mode"] = result.stdout.strip() or "Unknown"
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line:
                        info["cuda"] = line.strip()
                        break
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["dpkg", "-l", "tensorrt"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "tensorrt" in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            info["tensorrt"] = parts[2]
                        break
        except Exception:
            pass

    return info


def check_optimization() -> dict:
    """Check current optimization status."""
    status = {
        "jetson_clocks": False,
        "power_mode_max": False,
        "performance_governor": False,
    }

    try:
        result = subprocess.run(
            ["jetson_clocks", "--show"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            status["jetson_clocks"] = True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5
        )
        if "0" in result.stdout or "MAX" in result.stdout:
            status["power_mode_max"] = True
    except Exception:
        pass

    try:
        for cpu in range(4):
            governor_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            if Path(governor_path).exists():
                with open(governor_path) as f:
                    if "performance" in f.read():
                        status["performance_governor"] = True
                        break
    except Exception:
        pass

    return status


def print_system_info() -> None:
    """Print detailed system information."""
    info = get_jetson_info()
    print("=" * 50)
    print("Jetson System Information")
    print("=" * 50)
    print(f"  Device: {info['model']}")
    print(f"  JetPack: {info['jetpack']}")
    print(f"  CUDA: {info['cuda']}")
    print(f"  TensorRT: {info['tensorrt']}")
    print(f"  Power Mode: {info['power_mode']}")

    if info["is_jetson"]:
        status = check_optimization()
        print("\n[Optimization Status]")
        print(f"  jetson_clocks: {'OK' if status['jetson_clocks'] else 'NOT RUNNING'}")
        print(f"  Power Mode: {'MAX' if status['power_mode_max'] else 'NOT MAX'}")
        print(f"  CPU Governor: {'performance' if status['performance_governor'] else 'NOT OPTIMIZED'}")

        if not all(status.values()):
            print("\n[Recommendation]")
            print("  Run: sudo python3 scripts/jetson_helper.py --optimize")
    print("=" * 50)


def run_optimization() -> bool:
    """Run Jetson optimization."""
    if os.geteuid() != 0:
        print("Error: Must run as root for optimization")
        print("  sudo python3 scripts/jetson_helper.py --optimize")
        return False

    print("Running Jetson optimization...")

    try:
        subprocess.run(["nvpmodel", "-m", "0"], check=True)
        print("  [OK] Power mode set to MAX (15W)")
    except Exception as e:
        print(f"  [SKIP] nvpmodel: {e}")

    try:
        subprocess.run(["jetson_clocks"], check=True)
        print("  [OK] jetson_clocks enabled")
    except Exception as e:
        print(f"  [SKIP] jetson_clocks: {e}")

    for cpu in range(4):
        try:
            governor_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            with open(governor_path, "w") as f:
                f.write("performance")
            print(f"  [OK] CPU{cpu} governor set to performance")
        except Exception:
            pass

    print("\nOptimization complete!")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Jetson optimization helper")
    parser.add_argument("--info", action="store_true", help="Show system info")
    parser.add_argument(
        "--optimize", action="store_true", help="Run optimization (requires sudo)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check optimization status"
    )

    args = parser.parse_args()

    if args.info:
        print_system_info()
    elif args.optimize:
        run_optimization()
    elif args.check:
        info = get_jetson_info()
        status = check_optimization()
        print(json.dumps({**info, **status}, indent=2))
    else:
        print_system_info()


if __name__ == "__main__":
    main()
