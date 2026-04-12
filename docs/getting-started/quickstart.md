# QUICKSTART.md - Quick Start Guide for OpenEyes

> **Version**: v2.5.0  
> **Estimated Time**: 5 minutes

---

## Prerequisites

| Requirement | Check |
|:------------|:------|
| NVIDIA Jetson Orin Nano (or Pi 5, Intel NPU, Hailo, Qualcomm) | ☐ |
| CSI Camera (IMX219) or USB Webcam | ☐ |
| Ubuntu 22.04 installed | ☐ |
| Internet connection | ☐ |

---

## Step 1: Clone the Project

```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
```

---

## Step 2: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## Step 3: Enable Maximum Performance (Recommended)

For best performance on Jetson Orin Nano:

```bash
# One-command optimization (recommended)
sudo bash scripts/jetson_perf.sh

# Or manually:
sudo nvpmodel -m 2  # MAXN SUPER
sudo jetson_clocks
```

---

## Step 4: Run OpenEyes

### Basic Vision Pipeline

```bash
python -m src.main --camera 0 --debug
```

### With Person Following

```bash
python -m src.main --camera 0 --follow --debug
```

### With World Model (Predictive Tracking)

```bash
python -m src.main --camera 0 --world-model lewm --follow --debug
```

### Turbo Mode (Maximum FPS)

```bash
python -m src.main --camera 0 --world-model lewm --follow --turbo --debug
```

### Industry Template

```bash
# Warehouse/Logistics
python -m src.main --camera 0 --template warehouse --debug

# Manufacturing QA
python -m src.main --camera 0 --template manufacturing-qa --debug

# Agriculture
python -m src.main --camera 0 --template agriculture --debug

# Retail
python -m src.main --camera 0 --template retail --debug
```

---

## Step 5: Verify Performance

```bash
# Run benchmarks
python -m benchmarks.run_benchmarks --all --report
```

Expected FPS on Jetson Orin Nano:

| Configuration | FPS |
|:--------------|:----|
| Full pipeline (default) | 4-6 |
| Full pipeline + turbo | 8-12 |
| Minimal (--no-face --no-gesture --no-pose) | 15-20 |
| World model planning | 100-200 Hz |

---

## Next Steps

- **[CLI Reference](../reference/cli.md)** - Complete CLI reference
- **[Optimization Guide](../guides/optimization.md)** - Performance optimization guide
- **[User Guide](user-guide.md)** - Full user guide
- **[World Models](../concepts/world-models.md)** - World models documentation
- **[Technical Spec](../concepts/technical-spec.md)** - Technical specification

---

## Troubleshooting

### No Camera Detected
```bash
# Check camera
ls /dev/video*

# For CSI camera on Jetson:
ls /dev/video0
```

### Low FPS
```bash
# Enable MAXN SUPER mode
sudo bash scripts/jetson_perf.sh

# Use turbo mode
python -m src.main --camera 0 --turbo

# Disable unused models
python -m src.main --camera 0 --no-face --no-gesture --no-pose
```

### Out of Memory
```bash
# GStreamer pipeline now captures at 1280x720 (fixed in v1.5.0)
# If still OOM, reduce resolution in config.yaml
```

### Depth Anything V3 Requires HuggingFace Token
```bash
# DA3 is gated - use MiDaS by default
python -m src.main --camera 0 --depth-model midas-small

# Or login to HuggingFace for DA3
huggingface-cli login
python -m src.main --camera 0 --depth-model da3-small
```
