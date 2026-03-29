#!/bin/bash
# Jetson Performance Optimization Script for OpenEyes
# Run with: sudo bash scripts/jetson_perf.sh

set -e

echo "========================================"
echo "Jetson Performance Optimization"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash $0"
    exit 1
fi

# Detect Jetson model
MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
echo "Detected: $MODEL"

# Set MAX power mode (15W)
echo ""
echo "[1/4] Setting MAX power mode..."
nvpmodel -m 0 2>/dev/null || echo "  (nvpmodel not available, skipping)"
jetson_clocks 2>/dev/null || echo "  (jetson_clocks not available, skipping)"
echo "  Done"

# Enable max performance
echo ""
echo "[2/4] Configuring CPU governor..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done
echo "  Done"

# Set GPU to max frequency
echo ""
echo "[3/4] Configuring GPU..."
echo 1 > /sys/devices/57000000.gpu/power_dpm_force_performance_level 2>/dev/null || true
echo "  Done"

# Optimize memory
echo ""
echo "[4/4] Memory optimization..."
# Reduce swappiness
echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
# Enable zram
modprobe zram num_devices=1 2>/dev/null || true
echo "  Done"

echo ""
echo "========================================"
echo "Optimization complete!"
echo ""
echo "Current settings:"
nvpmodel -q 2>/dev/null || echo "  Power mode: MAX (15W)"
jetson_clocks --show 2>/dev/null | head -5 || echo "  (clocks info not available)"
echo ""
echo "Recommended for OpenEyes:"
echo "  python src/main.py --no-face --no-gesture --no-pose --no-depth"
echo "========================================"
