# HARDWARE.md - Hardware Specifications for PROJECT0

> **Project**: PROJECT0 - Robot Vision System  
> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## 1. Hardware Overview

### 1.1 System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROJECT0 Hardware                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────────┐         ┌──────────────┐                          │
│    │   USB        │         │   Power      │                          │
│    │  Camera      │         │   Supply     │                          │
│    │  (Webcam)    │         │  5V/4A DC    │                          │
│    └──────┬───────┘         └──────┬───────┘                          │
│           │                        │                                   │
│           │                        │                                   │
│           │            ┌────────────┴────────────┐                      │
│           │            │   NVIDIA Jetson        │                      │
│           │            │   Orin Nano            │                      │
│           │            │                        │                      │
│           └───────────▶│  • GPU (Ampere)       │                      │
│                        │  • CPU (Arm Cortex)   │                      │
│                        │  • 4-8GB LPDDR5       │                      │
│                        │  • 5-15W Power        │                      │
│                        └───────────┬────────────┘                      │
│                                    │                                   │
│                                    │ Ethernet / WiFi                    │
│                                    │                                   │
│                                    ▼                                   │
│                        ┌─────────────────────┐                         │
│                        │   Network Output    │                         │
│                        │   (UDP to robot)    │                         │
│                        └─────────────────────┘                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Primary Components

### 2.1 Computing Platform: NVIDIA Jetson Orin Nano

| Specification | 4GB Model | 8GB Model |
|:--------------|:----------|:----------|
| **GPU** | 512 CUDA cores, 16 Tensor cores | 1024 CUDA cores, 32 Tensor cores |
| **CPU** | 6-core Arm Cortex-A78AE | 6-core Arm Cortex-A78AE |
| **Memory** | 4GB LPDDR5 | 8GB LPDDR5 |
| **Storage** | 16GB eMMC 5.1 | 32GB eMMC 5.1 |
| **Power** | 7-15W | 7-15W |
| **AI Performance** | 20 TOPS | 40 TOPS |

**Recommended**: 8GB model for better performance with multiple models.

#### Jetson Orin Nano Pinout

```
┌─────────────────────────────────────────────────────────────┐
│                    Jetson Orin Nano                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 40-pin Header                        │   │
│  │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐   │   │
│  │  │1 │2 │3 │4 │5 │6 │7 │8 │9 │10│11│12│13│14│15│16│   │   │
│  │  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤   │   │
│  │  │17│18│19│20│21│22│23│24│25│26│27│28│29│30│31│32│   │   │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   │   │
│  │                                                      │   │
│  │  GPIO: 3.3V power, I2C, SPI, UART, PWM            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────┐  ┌────────────┐  ┌────────────────┐    │
│  │   USB 3.0      │  │  Ethernet  │  │   HDMI         │    │
│  │   (Type-A)     │  │  (1 Gbps)  │  │   (DisplayPort)│    │
│  └────────────────┘  └────────────┘  └────────────────┘    │
│                                                              │
│  ┌────────────────┐  ┌────────────┐  ┌────────────────┐    │
│  │   USB 2.0      │  │  Micro-USB │  │   M.2 Key M   │    │
│  │   (Type-A)     │  │  (Power)   │  │   (NVMe)      │    │
│  └────────────────┘  └────────────┘  └────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Camera

#### Recommended Cameras

| Camera | Resolution | FOV | Interface | Cost |
|:-------|:-----------|:----|:----------|:-----|
| Logitech C920 | 1080p @ 30fps | 78° | USB 2.0 | $50 |
| Logitech C270 | 720p @ 30fps | 60° | USB 2.0 | $25 |
| Razer Kiyo | 1080p @ 30fps | 90° | USB 2.0 | $70 |

#### Camera Requirements

| Requirement | Specification |
|:------------|:------------|
| **Interface** | USB 2.0 or higher |
| **Resolution** | 640x480 minimum, 1280x720 recommended |
| **Frame Rate** | 30 fps minimum |
| **FOV** | 60-90 degrees |
| **Driver** | UVC (USB Video Class) compliant |

#### USB Connection Diagram

```
┌─────────────┐        USB-A        ┌─────────────┐
│   Webcam    │─────────────────────│   Jetson    │
│             │        USB          │   Orin Nano │
│  ┌───────┐  │                     │             │
│  │ Lens  │  │                     │  ┌───────┐  │
│  │  ⬤   │  │                     │  │ USB   │  │
│  └───────┘  │                     │  │ Host  │  │
│             │                     │  └───────┘  │
└─────────────┘                     └─────────────┘
```

### 2.3 Power Supply

| Specification | Value |
|:--------------|:------|
| **Voltage** | 5V DC |
| **Current** | 3A minimum, 4A recommended |
| **Connector** | Barrel jack 5.5mm x 2.1mm |
| **Alternative** | Micro-USB (Jetson includes PMIC) |

#### Power Options

| Option | Pros | Cons |
|:-------|:-----|:-----|
| Barrel Jack (5.5mm) | Stable power | Requires adapter |
| USB-C PD | Common cable | Requires 65W+ supply |
| Micro-USB | Easy to find cable | Slower boot |

---

## 3. Power Requirements

### 3.1 Power Budget

| Component | Idle | Active (Typical) | Peak |
|:----------|:-----|:-----------------|:-----|
| Jetson Orin Nano | 5W | 10W | 15W |
| USB Camera | 0.5W | 1W | 2W |
| USB Hub (if used) | 0W | 2.5W | 5W |
| **Total** | **5.5W** | **13W** | **22W** |

### 3.2 Thermal Considerations

| Metric | Value |
|:-------|:------|
| **Operating Temp** | 0°C to 40°C |
| **Thermal Throttle** | Starts at 85°C |
| **Max Sustained** | 70°C with heatsink |

#### Cooling Options

| Solution | Cost | Effectiveness |
|:---------|:-----|:--------------|
| Passive (heatsink only) | $10 | Basic |
| 40mm Fan | $15 | Good |
| Active Cooler (NVIDIA) | $25 | Excellent |
| Water Cooling (custom) | $50+ | Best |

---

## 4. Physical Setup

### 4.1 Minimal Setup

```
┌─────────────────────────────────────────────────────────────┐
│                    Minimal Setup Diagram                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌──────────────────────────────────────────────┐         │
│    │           Jetson Orin Nano Dev Kit           │         │
│    │  ┌─────────────────────────────────────┐    │         │
│    │  │                                     │    │         │
│    │  │           [Camera]                  │    │         │
│    │  │            (USB)                     │    │         │
│    │  │              │                      │    │         │
│    │  └──────────────┴──────────────────────┘    │         │
│    │                                            │         │
│    │  ┌─────────────────────────────────────┐    │         │
│    │  │  [=======]  [=======]  [=======]   │    │         │
│    │  │   Power    Ethernet   USB Camera   │    │         │
│    │  └─────────────────────────────────────┘    │         │
│    └──────────────────────────────────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Robot Mounting

For humanoid robot integration:

| Mounting Point | Recommendation |
|:---------------|:---------------|
| **Head** | Camera mounted on 1/4" tripod mount |
| **Mount** | 3D printed mount bracket |
| **Cable** | USB extension cable (3m max without hub) |

---

## 5. Storage

### 5.1 Boot Device Options

| Device | Speed | Capacity | Cost |
|:-------|:------|:---------|:-----|
| microSD Card | UHS-I | 32-256GB | $15-40 |
| NVMe SSD (M.2) | PCIe Gen3 | 256GB-1TB | $30-80 |

### 5.2 Recommended Storage

| Use Case | Recommendation |
|:---------|:---------------|
| Development | 64GB microSD |
| Production | 128GB+ NVMe |

---

## 6. Network

### 6.1 Connection Options

| Method | Speed | Use Case |
|:-------|:------|:---------|
| Ethernet (Gigabit) | 1 Gbps | Fixed installation |
| WiFi (WiFi 6) | 1.2 Gbps | Mobile robot |
| USB Tethering | 480 Mbps | Temporary |

### 6.2 Network Topology

```
PROJECT0                    Robot Controller
    │                              │
    │  UDP:5000                   │
    │◄─────────────────────────────│
    │  (JSON output)               │
    │                              │
    │                              │
    │  Optional: SSH               │
    │◄─────────────────────────────│
```

---

## 7. Assembly Instructions

### 7.1 Step-by-Step

1. **Prepare Storage**
   - Flash JetPack to microSD or NVMe
   - See [INSTALL.md](../INSTALL.md)

2. **Connect Camera**
   - Plug USB webcam into Jetson USB port
   - Verify with: `ls /dev/video*`

3. **Connect Power**
   - Connect 5V/4A barrel jack
   - OR connect USB-C PD charger

4. **Network Connection**
   - Connect Ethernet OR configure WiFi

5. **First Boot**
   - Power on Jetson
   - Complete Ubuntu setup
   - Configure wireless (if needed)

---

## 8. Troubleshooting

### 8.1 Common Hardware Issues

| Issue | Cause | Solution |
|:------|:------|:---------|
| Camera not detected | USB not powered | Use powered USB hub |
| Jetson won't boot | Bad SD card | Re-flash with Etcher |
| Power issues | Insufficient current | Use 4A+ supply |
| Thermal throttling | Poor cooling | Add fan/heatsink |

### 8.2 Power LED Indicators

| LED State | Meaning |
|:----------|:-------|
| Solid Green | Power on, booting |
| Blinking Green | Boot in progress |
| Solid Red | Power failure |
| No LED | No power |

---

## 9. Future Hardware Upgrades

### 9.1 Potential Enhancements

| Upgrade | Description | Benefit |
|:--------|:------------|:--------|
| RealSense D435i | Stereo depth camera | Better depth estimation |
| OAK-D Lite | Spatial AI camera | Built-in AI processing |
| Lidar | RPLIDAR A1M8 | Redundant obstacle detection |
| Coral TPU | Edge TPU | Additional AI acceleration |

### 9.2 Multi-Camera Setup

For advanced robot vision:

```
┌─────────────────────────────────────────────┐
│          Multi-Camera Configuration          │
├─────────────────────────────────────────────┤
│                                              │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│   │  Left  │  │ Center  │  │  Right  │    │
│   │ Camera │  │ Camera  │  │ Camera  │    │
│   └───┬─────┘  └────┬────┘  └───┬─────┘    │
│       │             │            │          │
│       └─────────────┼────────────┘          │
│                     │                        │
│                     ▼                        │
│              ┌─────────────┐                │
│              │ USB Hub     │                │
│              │ (Powered)   │                │
│              └──────┬──────┘                │
│                     │                        │
│                     ▼                        │
│              ┌─────────────┐                │
│              │ Jetson      │                │
│              │ Orin Nano   │                │
│              └─────────────┘                │
│                                              │
└─────────────────────────────────────────────┘
```

---

## Appendix: Bill of Materials

| Item | Quantity | Unit Cost | Total |
|:-----|:---------|:----------|:------|
| Jetson Orin Nano 8GB | 1 | $200 | $200 |
| USB Webcam 1080p | 1 | $50 | $50 |
| 64GB microSD | 1 | $20 | $20 |
| 5V/4A Power Supply | 1 | $15 | $15 |
| **Total** | | | **$285** |
