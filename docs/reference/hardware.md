# Hardware

Hardware specifications and setup for OpenEyes.

---

## Computing Platform: NVIDIA Jetson Orin Nano

| Specification | 4GB Model | 8GB Model |
|:--------------|:----------|:----------|
| **GPU** | 512 CUDA cores, 16 Tensor cores | 1024 CUDA cores, 32 Tensor cores |
| **CPU** | 6-core Arm Cortex-A78AE | 6-core Arm Cortex-A78AE |
| **Memory** | 4GB LPDDR5 | 8GB LPDDR5 |
| **Storage** | 16GB eMMC 5.1 | 32GB eMMC 5.1 |
| **Power** | 7-15W | 7-15W |
| **AI Performance** | 20 TOPS | 40 TOPS |

**Recommended**: 8GB model for better performance with multiple models.

---

## Camera

### Recommended Cameras

| Camera | Resolution | FOV | Interface | Cost |
|:-------|:-----------|:----|:----------|:-----|
| Logitech C920 | 1080p @ 30fps | 78° | USB 2.0 | $50 |
| Logitech C270 | 720p @ 30fps | 60° | USB 2.0 | $25 |
| Razer Kiyo | 1080p @ 30fps | 90° | USB 2.0 | $70 |
| IMX219 (CSI) | 1080p @ 30fps | - | CSI | $20 |

### Camera Requirements

| Requirement | Specification |
|:------------|:------------|
| **Interface** | USB 2.0 or higher |
| **Resolution** | 640x480 minimum, 1280x720 recommended |
| **Frame Rate** | 30 fps minimum |
| **FOV** | 60-90 degrees |
| **Driver** | UVC (USB Video Class) compliant |

---

## Power Supply

| Specification | Value |
|:--------------|:------|
| **Voltage** | 5V DC |
| **Current** | 3A minimum, 4A recommended |
| **Connector** | Barrel jack 5.5mm x 2.1mm |
| **Alternative** | Micro-USB (Jetson includes PMIC) |

### Power Options

| Option | Pros | Cons |
|:-------|:-----|:-----|
| Barrel Jack (5.5mm) | Stable power | Requires adapter |
| USB-C PD | Common cable | Requires 65W+ supply |
| Micro-USB | Easy to find cable | Slower boot |

---

## Power Budget

| Component | Idle | Active (Typical) | Peak |
|:----------|:-----|:-----------------|:-----|
| Jetson Orin Nano | 5W | 10W | 15W |
| USB Camera | 0.5W | 1W | 2W |
| USB Hub (if used) | 0W | 2.5W | 5W |
| **Total** | **5.5W** | **13W** | **22W** |

---

## Thermal

| Metric | Value |
|:-------|:------|
| **Operating Temp** | 0°C to 40°C |
| **Thermal Throttle** | Starts at 85°C |
| **Max Sustained** | 70°C with heatsink |

### Cooling Options

| Solution | Cost | Effectiveness |
|:---------|:-----|:--------------|
| Passive (heatsink only) | $10 | Basic |
| 40mm Fan | $15 | Good |
| Active Cooler (NVIDIA) | $25 | Excellent |

---

## Storage

| Device | Speed | Capacity | Cost |
|:-------|:------|:---------|:-----|
| microSD Card | UHS-I | 32-256GB | $15-40 |
| NVMe SSD (M.2) | PCIe Gen3 | 256GB-1TB | $30-80 |

---

## Network

| Method | Speed | Use Case |
|:-------|:------|:---------|
| Ethernet (Gigabit) | 1 Gbps | Fixed installation |
| WiFi (WiFi 6) | 1.2 Gbps | Mobile robot |

---

## Bill of Materials

| Item | Quantity | Unit Cost | Total |
|:-----|:---------|:----------|:------|
| Jetson Orin Nano 8GB | 1 | $200 | $200 |
| USB Webcam 1080p | 1 | $50 | $50 |
| 64GB microSD | 1 | $20 | $20 |
| 5V/4A Power Supply | 1 | $15 | $15 |
| **Total** | | | **$285** |

---

## Future Hardware Upgrades

| Upgrade | Description | Benefit |
|:--------|:------------|:--------|
| RealSense D435i | Stereo depth camera | Better depth estimation |
| OAK-D Lite | Spatial AI camera | Built-in AI processing |
| Lidar | RPLIDAR A1M8 | Redundant obstacle detection |
| Coral TPU | Edge TPU | Additional AI acceleration |