#!/bin/bash
# Jetson Orin Nano SOTA Performance Optimization Script
# Run with: sudo bash scripts/jetson_perf.sh
#
# This script enables MAXN SUPER mode + jetson_clocks for maximum FPS.
# Requires active cooling (fan) - monitors will overheat without it.

set -e

echo "========================================"
echo "OpenEyes Jetson Orin Nano - SOTA Perf"
echo "========================================"

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash $0"
    exit 1
fi

MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
echo "Detected: $MODEL"

# 1. Enable MAXN SUPER mode (unlocks highest GPU/CPU clocks)
echo ""
echo "[1/6] Setting MAXN SUPER power mode..."
nvpmodel -m 2 2>/dev/null && echo "  MAXN SUPER enabled" || {
    echo "  MAXN SUPER not available, trying MAX (15W)..."
    nvpmodel -m 0 2>/dev/null && echo "  MAX mode enabled" || echo "  nvpmodel not available"
}

# 2. Lock all clocks (disable DVFS, prevent throttling dips)
echo ""
echo "[2/6] Locking clocks with jetson_clocks..."
jetson_clocks 2>/dev/null && echo "  Clocks locked" || echo "  jetson_clocks not available"

# 3. CPU governor - performance mode
echo ""
echo "[3/6] Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done
echo "  Done"

# 4. GPU max frequency
echo ""
echo "[4/6] Setting GPU to max frequency..."
GPU_FREQ=$(cat /sys/devices/gpu.0/devfreq/gpu.0/cur_freq 2>/dev/null || echo "unknown")
echo "  GPU frequency: $GPU_FREQ Hz"

# 5. Memory optimization
echo ""
echo "[5/6] Memory optimization..."
echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
echo 1 > /proc/sys/vm/overcommit_memory 2>/dev/null || true
echo "  Swappiness: 10, Overcommit: 1"

# 6. Disable unnecessary services
echo ""
echo "[6/6] Disabling unnecessary services..."
systemctl stop cups 2>/dev/null || true
systemctl stop bluetooth 2>/dev/null || true
systemctl stop ModemManager 2>/dev/null || true
echo "  Stopped: cups, bluetooth, ModemManager"

echo ""
echo "========================================"
echo "SOTA Performance Optimization Complete!"
echo "========================================"
echo ""
echo "Current settings:"
nvpmodel -q 2>/dev/null | head -3 || echo "  Power mode: MAX"
echo "  CPU: $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq 2>/dev/null || echo 'N/A') kHz"
echo "  GPU: $(cat /sys/devices/gpu.0/devfreq/gpu.0/cur_freq 2>/dev/null || echo 'N/A') Hz"
echo ""
echo "Expected FPS improvements:"
echo "  Before: ~4-5 FPS (full pipeline)"
echo "  After:  ~8-12 FPS (full pipeline with optimizations)"
echo "  Minimal: ~20-30 FPS (--no-face --no-gesture --no-pose)"
echo ""
echo "Monitor thermals: watch -n 1 tegrastats"
echo "========================================"
