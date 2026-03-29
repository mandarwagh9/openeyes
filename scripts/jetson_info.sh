#!/bin/bash
# Jetson System Information Script for OpenEyes
# Run with: bash scripts/jetson_info.sh

echo "========================================"
echo "Jetson System Information"
echo "========================================"

# Model
echo ""
echo "[System]"
MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
echo "  Model: $MODEL"

# JetPack version
echo ""
echo "[JetPack]"
jetpack_version=$(dpkg -l | grep -i jetpack | head -1 | awk '{print $3}' || echo "Unknown")
echo "  JetPack: $jetpack_version"

# CUDA version
echo ""
echo "[CUDA]"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" | sed 's/^/  /'
else
    echo "  CUDA: Not found"
fi

# TensorRT version
echo ""
echo "[TensorRT]"
if [ -f /usr/lib/aarch64-linux-gnu/libnvinfer.so ]; then
    tensorrt_version=$(dpkg -l | grep -i tensorrt | head -1 | awk '{print $3}' || echo "Unknown")
    echo "  TensorRT: $tensorrt_version"
else
    echo "  TensorRT: Not found"
fi

# Current power mode
echo ""
echo "[Power]"
nvpmodel -q 2>/dev/null || echo "  Mode: Unknown"
jetson_clocks --show 2>/dev/null | grep -E "CPU|MGPU|EMC" | head -3 | sed 's/^/  /' || echo "  Clocks: Unknown"

# Memory
echo ""
echo "[Memory]"
total_mem=$(free -h | awk '/^Mem:/{print $2}')
used_mem=$(free -h | awk '/^Mem:/{print $3}')
echo "  Total: $total_mem"
echo "  Used: $used_mem"

# Disk
echo ""
echo "[Disk]"
df -h / | awk 'NR==2{print "  Used: " $3 " / " $2 " (" $5 ")"}'

# Temperature
echo ""
echo "[Temperature]"
for zone in /sys/class/thermal/thermal_zone*; do
    if [ -f "$zone/temp" ]; then
        temp=$(cat "$zone/temp" 2>/dev/null | head -1)
        type=$(cat "$zone/type" 2>/dev/null | head -1)
        if [ -n "$temp" ]; then
            echo "  $type: $((temp/1000))C"
        fi
    fi
done

# OpenEyes recommended settings
echo ""
echo "[OpenEyes Recommendations]"
if [ "$MODEL" == *"Orin"* ]; then
    echo "  Performance Mode: MAX (15W)"
    echo "  Target FPS: 20-30 (all models), 30+ (minimal)"
else
    echo "  Performance Mode: MAX"
    echo "  Target FPS: 15-20 (all models), 25+ (minimal)"
fi

echo ""
echo "Run 'sudo bash scripts/jetson_perf.sh' to optimize performance"
echo "========================================"
