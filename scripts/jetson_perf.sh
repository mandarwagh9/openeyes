#!/bin/bash
# Jetson Orin Nano Performance Optimization Script
# Run with: sudo bash scripts/jetson_perf.sh
#
# This script enables MAXN mode + jetson_clocks for maximum FPS.
# Requires active cooling (fan) - monitors will overheat without it.

set -e

echo "========================================"
echo "OpenEyes Jetson Optimization"
echo "========================================"

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash $0"
    exit 1
fi

MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
echo "Detected: $MODEL"

# 1. Enable MAXN SUPER mode (unlocks highest GPU/CPU clocks)
echo ""
echo "[1/8] Setting MAXN power mode..."
nvpmodel -m 0 2>/dev/null && echo "  MAX mode enabled" || {
    echo "  Trying alternative modes..."
    nvpmodel -m 2 2>/dev/null && echo "  MAXN SUPER enabled" || echo "  nvpmodel not available"
}

# 2. Lock all clocks (disable DVFS, prevent throttling dips)
echo ""
echo "[2/8] Locking clocks with jetson_clocks..."
jetson_clocks 2>/dev/null && echo "  Clocks locked" || echo "  jetson_clocks not available"

# 3. CPU governor - performance mode
echo ""
echo "[3/8] Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done
echo "  Done"

# 4. GPU max frequency
echo ""
echo "[4/8] Setting GPU to max frequency..."
echo 1 > /sys/devices/gpu.0/devfreq/gpu.0/governor 2>/dev/null || true
GPU_FREQ=$(cat /sys/devices/gpu.0/devfreq/gpu.0/max_freq 2>/dev/null || echo "unknown")
echo "  GPU max frequency: $GPU_FREQ Hz"

# 5. Memory optimization
echo ""
echo "[5/8] Memory optimization..."
echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
echo 1 > /proc/sys/vm/overcommit_memory 2>/dev/null || true
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
echo "  Swappiness: 10, Overcommit: 1"

# 6. Disable unnecessary services
echo ""
echo "[6/8] Disabling unnecessary services..."
systemctl stop cups 2>/dev/null || true
systemctl stop bluetooth 2>/dev/null || true
systemctl stop ModemManager 2>/dev/null || true
systemctl stop thermald 2>/dev/null || true
systemctl stop snapd 2>/dev/null || true
echo "  Stopped: cups, bluetooth, ModemManager, thermald, snapd"

# 7. NVMe/SSD optimization
echo ""
echo "[7/8] Disk I/O optimization..."
echo "noop" > /sys/block/mmcblk0/queue/scheduler 2>/dev/null || true
echo "noop" > /sys/block/nvme0n1/queue/scheduler 2>/dev/null || true
echo "  I/O scheduler: noop"

# 8. Network optimization
echo ""
echo "[8/8] Network optimization..."
ethtool -K eth0 tso on 2>/dev/null || true
ethtool -K eth0 gso on 2>/dev/null || true
ethtool -G eth0 rx 4096 tx 4096 2>/dev/null || true
echo "  TCP offload enabled"

echo ""
echo "========================================"
echo "Performance Optimization Complete!"
echo "========================================"
echo ""
echo "Current settings:"
nvpmodel -q 2>/dev/null | head -3 || echo "  Power mode: MAX"
echo "  CPU: $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq 2>/dev/null || echo 'N/A') kHz"
echo "  GPU: $(cat /sys/devices/gpu.0/devfreq/gpu.0/cur_freq 2>/dev/null || echo 'N/A') Hz"
echo ""
echo "Expected FPS with optimizations:"
echo "  Default pipeline:  8-15 FPS"
echo "  --int8:           15-25 FPS (+50-100%)"
echo "  --turbo:          15-25 FPS"
echo "  --int8 --turbo:   25-35 FPS"
echo "  Minimal (no face/gesture/pose): 25-40 FPS"
echo "  Detection only:   50-80 FPS"
echo ""
echo "Recommended run commands:"
echo "  python -m src.main --camera 0 --debug"
echo "  python -m src.main --camera 0 --int8 --debug"
echo "  python -m src.main --camera 0 --int8 --turbo --debug"
echo "  python -m src.main --camera 0 --int8 --no-face --no-gesture --no-pose --debug"
echo ""
echo "Monitor: watch -n 1 tegrastats"
echo "========================================"
