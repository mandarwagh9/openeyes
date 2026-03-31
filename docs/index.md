---
title: Home
---

<style>
/* Hero Banner */
.md-content > article > h1:first-of-type + p,
.md-content > article > p:first-of-type {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
  padding: 2rem;
  border-radius: 16px;
  text-align: center;
  margin: 2rem 0 !important;
  color: #e2e8f0 !important;
  position: relative;
  overflow: hidden;
}

.md-content > article > h1:first-of-type + p::before,
.md-content > article > p:first-of-type::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at 50% 0%, rgba(124, 58, 237, 0.4) 0%, transparent 60%);
}

/* Make first h1 special */
.md-content > article > h1:first-of-type {
  display: none !important;
}

/* Logo */
.logo-img {
  display: block;
  width: 120px;
  margin: 0 auto 1.5rem;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

/* Buttons */
.md-button {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  border-radius: 10px;
  font-weight: 600;
  text-decoration: none;
  margin: 0.25rem;
  transition: all 0.3s ease;
}

.md-button--primary {
  background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
  color: white !important;
  box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
}

.md-button--primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
}

/* Feature Cards Grid */
.md-content .grid-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.md-content .grid-cards > div {
  background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
  border-radius: 16px;
  padding: 1.25rem;
  border: 1px solid #334155;
  transition: all 0.3s ease;
}

.md-content .grid-cards > div:hover {
  transform: translateY(-4px);
  border-color: #7c3aed;
  box-shadow: 0 8px 25px rgba(124, 58, 237, 0.2);
}

.md-content .grid-cards h3 {
  color: #a78bfa !important;
  font-size: 0.95rem !important;
  margin-bottom: 0.5rem !important;
}

.md-content .grid-cards p {
  color: #94a3b8 !important;
  font-size: 0.8rem !important;
  margin: 0 !important;
}

/* Tables */
.md-content table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border-radius: 12px;
  overflow: hidden;
  margin: 1.5rem 0;
}

.md-content th {
  background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
  color: white !important;
  padding: 0.875rem 1rem !important;
  font-weight: 600;
}

.md-content td {
  padding: 0.75rem 1rem !important;
  border-bottom: 1px solid #334155;
  color: #cbd5e1 !important;
}

.md-content tr:hover td {
  background: #1e293b !important;
}

/* Badge */
.badge {
  background: rgba(124, 58, 237, 0.2);
  color: #a78bfa;
  padding: 0.2rem 0.6rem;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 0.8rem;
}

/* Hardware Grid */
.hardware-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.hardware-grid > div {
  background: #1e293b;
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
  border: 1px solid #334155;
}

.hardware-grid h3 {
  font-size: 0.85rem !important;
  color: #06b6d4 !important;
  margin-bottom: 0.25rem !important;
}

.hardware-grid p {
  font-size: 0.75rem !important;
  color: #94a3b8 !important;
  margin: 0 !important;
}

/* Section Titles */
.md-content h2 {
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: #f1f5f9 !important;
  margin: 2.5rem 0 1.5rem !important;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid #7c3aed;
  text-align: center;
}

/* Code Blocks */
.md-content pre {
  background: #1e293b !important;
  border-radius: 12px !important;
  border: 1px solid #334155 !important;
  margin: 1rem 0 !important;
}

/* Lists */
.md-content ul {
  list-style: none;
  padding-left: 0;
}

.md-content ul li {
  padding: 0.25rem 0;
  color: #cbd5e1 !important;
}

/* Center content */
.center-text {
  text-align: center;
}

.center-text a {
  color: #06b6d4 !important;
}
</style>

<img src="assets/images/logo.svg" class="logo-img" alt="OpenEyes Logo">

<div style="text-align: center;">

## 🤖 We Give Robots Vision

A humanoid robot needs to see the world like a human does. Not just pixels — but **understanding**. **Distance**. **Intent**. **Action**.

OpenEyes runs entirely on **NVIDIA Jetson** — no cloud, no lag, no dependencies.

[🚀 Quick Start](getting-started/quickstart.md){ .md-button .md-button--primary }
[⭐ Star on GitHub](https://github.com/mandarwagh9/openeyes){ .md-button }

</div>

---

## ✨ Features

<div class="grid-cards">

- **🔍 Object Detection** — Real-time detection of 80+ object classes

- **📏 Depth Estimation** — Measure distance to everything in the scene

- **👤 Face Detection** — Who's in the room? Identify and track faces

- **👋 Gesture Recognition** — Understand hand signals — stop, wave, point

- **🦴 Pose Estimation** — Detect body positions and movements

- **🎯 Object Tracking** — Follow specific objects across frames

- **🚶 Person Following** — Autonomous person tracking and following

- **🗺️ Visual SLAM** — Build maps and navigate autonomously

</div>

---

## ⚡ Performance

| Configuration | FPS | Use Case |
|:--------------|:----|:---------|
| All models enabled | <span class="badge">10-15</span> | Full capability |
| Minimal | <span class="badge">25-30</span> | Speed critical |
| Optimized INT8 | <span class="badge">30-40</span> | Production |

---

## 💻 Hardware

<div class="hardware-grid">

- **🟢 Jetson Orin Nano** — 4GB or 8GB variants

- **📷 Camera** — CSI (IMX219) or USB Webcam

- **🖥️ OS** — Ubuntu 22.04 + JetPack 5.1+

- **⚡ Power** — 7-15W consumption

</div>

---

## 🛠️ Quick Start

```bash
# Clone and install
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt

# Run with debug window
python src/main.py --debug
```

```bash
# Jetson Optimization
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 📦 Supported Models

| Model | Type | Size | Purpose |
|:------|:-----|:-----|:--------|
| YOLO11n | Object Detection | 5.4MB | Real-time detection |
| MiDaS v2.1 | Depth Estimation | 350MB | Monocular depth |
| MediaPipe | Face/Gesture/Pose | ~20MB | Multi-modal ML |
| SmolVLA | VLA | ~450M params | Vision-Language-Action |
| OpenVLA | VLA | 7B params | State-of-the-art VLA |

---

## 📅 The Journey

| Version | Milestone |
|:--------|:----------|
| v0.6.x | Real VLA models (SmolVLA, OpenVLA, Octo) |
| v0.5.x | Visual odometry, SLAM, Nav2 |
| v0.4.x | VLA, event camera |
| v0.3.x | Model selection |
| v0.2.x | Tracking, performance, ROS2 |
| v0.1.x | Core vision (detection, depth, face, gesture, pose) |

---

<div class="center-text">

## 🤝 Contribute

OpenEyes is built by people like you. Developers, researchers, hobbyists, dreamers.

See [Contributing Guide](development/contributing.md) to join us.

---

> 🤖 The future of robotics is open. Let's build it together.
>
> **Apache 2.0 License**

</div>