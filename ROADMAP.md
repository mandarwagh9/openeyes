# ROADMAP.md - Project Roadmap for OpenEyes

> **Version**: v2.0.0 (Hardware-Agnostic Edge Vision Framework)
> **Last Updated**: 2026-04-02

---

## Overview

OpenEyes is an open-source, hardware-agnostic robot vision framework designed for edge AI on NVIDIA Jetson, Raspberry Pi, Hailo, Intel NPU, and Qualcomm platforms. It bridges the full train → optimize → deploy → operate loop, enabling small teams to go from zero to production robot vision in 5 minutes.

**Position**: "The Fastest Path from Zero to Robot Vision on Edge"

---

## Version History

| Version | Status | Date | Description |
|:--------|:-------|:-----|:------------|
| v2.5.0 | Planned | 2026-Q4 | World Models & Predictive Intelligence |
| v2.0.0 | Planned | 2026-Q3 | Hardware-Agnostic Edge Vision Framework |
| v1.5.0 | Planned | 2026-Q2 | YOLO26 + Depth Anything V3 + Fleet Management |
| v1.0.0 | Current | 2026-04-01 | Safety & Reliability + Diffusion Policy |
| v0.8.0 | Released | 2026-04-01 | VLA Integration + Action Chunking + TensorRT |
| v0.7.0 | Released | 2026-04-01 | Multi-Modal Sensing + LIDAR + Sensor Fusion |
| v0.6.0 | Released | 2026-03-30 | Navigation + Obstacle Avoidance |
| v0.5.0 | Released | 2026-03-30 | SLAM + Nav2 + VLA Integration |
| v0.4.4 | Released | 2026-03-30 | Person Following + Gesture Owner |
| v0.4.0 | Released | 2026-03-28 | VLA + Event Camera |
| v0.3.0 | Released | 2026-03-27 | Model Selection |
| v0.2.x | Released | 2026-03-26 | Tracking + ROS2 |
| v0.1.0 | Released | 2026-03-28 | Command Subscription + Full ROS2 |

---

## Phase 1: Foundation (v0.1.0 - v1.0.0) ✅ COMPLETE

All phases of the original 18-month industry standard roadmap have been implemented.

### v0.7.0 - Multi-Modal Sensing ✅ COMPLETE
- [x] Isaac ROS VSLAM integration
- [x] LIDAR processing and obstacle detection
- [x] Sensor fusion (camera + depth + LIDAR)
- [x] Multi-camera support
- [x] RealSense D455 integration

### v0.8.0 - VLA & Performance ✅ COMPLETE
- [x] Action chunking for 10-30 Hz control
- [x] LoRA fine-tuning for VLA customization
- [x] TensorRT INT8 quantization with calibration
- [x] DLA offloading

### v1.0.0 - Safety & Reliability ✅ COMPLETE
- [x] Health monitor for 24/7 operation
- [x] Safety controller with E-STOP
- [x] OTA updates with rollback
- [x] Diffusion Policy integration

---

## Phase 2: Hardware Abstraction & Model Updates (v1.5.0)

**Target**: Q2 2026 | **Theme**: "Run Anywhere, Detect Everything"

### YOLO26 Integration (P0)

The latest Ultralytics YOLO26 (Jan 2026) brings NMS-free end-to-end predictions with 43% faster CPU inference vs YOLO11.

- [ ] Add YOLO26n model support (~40 FPS on Orin Nano, 40.9% mAP)
- [ ] Add YOLO26s model support (~25 FPS on Orin Nano, 48.6% mAP)
- [ ] INT8 QAT export pipeline for YOLO26
- [ ] Benchmark suite comparing YOLO11 vs YOLO26 on all supported hardware
- [ ] Backward compatibility: `--model yolo26n` flag

### Depth Anything V3 Integration (P0)

Depth Anything V3 (ByteDance, ICLR 2026 Oral) is the new SOTA with single plain transformer architecture, outperforming DA2 by 35.7% in camera pose accuracy.

- [ ] Replace MiDaS with Depth Anything V3 as default depth estimator
- [ ] TensorRT export for DA3 (TensorRT ROS2 nodes exist from RWTH Aachen)
- [ ] Multi-view depth support (DA3 capability)
- [ ] Depth-ray representation for improved geometric accuracy
- [ ] CLI arg: `--depth-model da3-small` (edge-optimized variant)

### Hardware Abstraction Layer (P0)

The #1 developer pain point is fragmented tooling. HAL provides a single API targeting TensorRT, OpenVINO, TVM, and Hailo DFC.

- [ ] `src/backends/` directory with backend abstraction
- [ ] TensorRT backend (existing, refactor into backend interface)
- [ ] OpenVINO backend for Intel CPU/NPU deployment
- [ ] TVM backend for hardware-agnostic auto-tuning
- [ ] Hailo DFC backend for Raspberry Pi AI HAT+ 2
- [ ] Auto-detection: `--backend auto` selects best available backend
- [ ] Unified model export: `openeyes export --backend openvino --model yolo26n`
- [ ] Backend benchmarking CLI: `openeyes benchmark --all-backends`

### Multi-Platform Support (P0)

Expand beyond Jetson to support the full edge AI hardware ecosystem.

- [ ] Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H, 40 TOPS, $150 total)
- [ ] Intel Core Ultra + OpenVINO (48 TOPS NPU)
- [ ] Qualcomm RB5/RB6 (15-30 TOPS Hexagon NPU)
- [ ] Hailo-8 standalone accelerator (26 TOPS, 3.5W)
- [ ] Platform detection: `openeyes platform-info` shows detected hardware
- [ ] Hardware-specific optimization profiles

### SAM 3 Integration (P1)

SAM 3 (Meta, Mar 2026) introduces concept-aware segmentation with 4M unique concept labels - a new paradigm for promptable object segmentation.

- [ ] SAM 3 integration for concept-aware segmentation
- [ ] EdgeSAM variant for edge deployment (40x speedup, ~11ms)
- [ ] Text-prompted segmentation: `--segment "red boxes"`
- [ ] Video tracking with SAM 3's built-in tracker

### Fleet Management Foundation (P1)

No open-source solution exists for managing model versioning, OTA updates, and performance telemetry across heterogeneous edge vision devices.

- [ ] Device registration and heartbeat protocol
- [ ] Model version registry with signed artifacts
- [ ] Performance telemetry collection (FPS, latency, errors)
- [ ] Fleet dashboard (web-based, lightweight)
- [ ] Group-based model deployment (deploy to all "warehouse-robots")
- [ ] CLI: `openeyes fleet list`, `openeyes fleet deploy --group warehouse`

---

## Phase 3: Production & Scale (v2.0.0)

**Target**: Q3 2026 | **Theme**: "From Prototype to Production"

### Unified Inference Pipeline (P0)

A single pipeline abstraction that handles detection, depth, segmentation, and tracking with automatic backend selection and model orchestration.

- [ ] `Pipeline` class: declarative pipeline definition
- [ ] Automatic model scheduling (run detection every frame, depth every 3rd)
- [ ] Zero-copy GPU memory management
- [ ] Pipeline configuration via YAML
- [ ] Pipeline visualization: `openeyes pipeline visualize config.yaml`
- [ ] Hot-reload pipeline configuration without restart

### Edge-Cloud Split Inference (P0)

Practical approach for models too large for edge (VLA 7B params → 1-2 FPS on Orin Nano).

- [ ] Edge-cloud split inference protocol
- [ ] Lightweight edge model for fast filtering
- [ ] Cloud fallback for complex scenes
- [ ] Adaptive routing: edge handles 90%, cloud handles 10% edge cases
- [ ] Latency budget management (edge < 50ms, cloud < 500ms)
- [ ] Offline mode: graceful degradation when cloud unavailable

### Real-World Benchmarking Suite (P0)

Comprehensive benchmarks across all supported hardware and models.

- [ ] FPS benchmarks per model per hardware
- [ ] Power consumption measurements
- [ ] Accuracy benchmarks (mAP, depth error, tracking IDF1)
- [ ] Thermal throttling detection and reporting
- [ ] Benchmark report generation: `openeyes benchmark --report`
- [ ] CI integration: run benchmarks on supported hardware nightly

### Production Deployment Toolkit (P0)

Everything needed to go from prototype to production deployment.

- [ ] Docker images for all supported platforms
- [ ] Systemd service templates
- [ ] Auto-start on boot with health monitoring
- [ ] Log rotation and structured logging (JSON)
- [ ] Prometheus metrics endpoint
- [ ] Grafana dashboard templates
- [ ] Deployment scripts: `openeyes deploy --target jetson-orin-nano`

### Advanced Tracking (P1)

Upgrade from ByteTrack to latest tracking algorithms with occlusion handling.

- [ ] ByteTrack (current, keep as default)
- [ ] BoT-SORT with ReID for crowded scenes
- [ ] OC-SORT for non-linear motion handling
- [ ] SAM 3 Tracker for concept-aware tracking
- [ ] BoostTrack for improved occlusion handling
- [ ] Auto-switching: select tracker based on scene complexity

### Industry Templates (P1)

Pre-configured pipelines for highest-demand industries.

- [ ] **Warehouse/Logistics**: Package detection, damage inspection, pallet counting
- [ ] **Manufacturing QA**: Defect detection, assembly verification, PPE compliance
- [ ] **Agriculture**: Weed detection, crop health monitoring, yield estimation
- [ ] **Retail**: Shelf monitoring, inventory counting, customer analytics
- [ ] Template CLI: `openeyes init --template warehouse`

### VLA Edge Support (P2)

Quantized VLA inference for edge devices with realistic performance expectations.

- [ ] OpenVLA INT4 quantization (~3.5GB, ~2 FPS on Orin Nano 8GB)
- [ ] GR00T N1.6 Jetson deployment guide
- [ ] Edge-cloud VLA with RoboECC pattern
- [ ] Instruction-tuned VLA for domain-specific tasks
- [ ] VLA performance monitoring and fallback to rule-based

### EU AI Act Compliance (P2)

Full enforcement from August 2026 - critical for European deployments.

- [ ] On-device data anonymization (face blurring, license plate masking)
- [ ] Model provenance tracking
- [ ] Dataset lineage documentation
- [ ] Audit trail for all decisions
- [ ] Human-in-the-loop escalation
- [ ] Compliance report generation: `openeyes compliance report`

---

## Phase 4: World Models & Predictive Intelligence (v2.5.0)

**Target**: Q4 2026 - Q1 2027 | **Theme**: "From Reactive to Predictive"

### LeWorldModel Integration (P0)

15M parameter latent-space world model for real-time planning at 100-200 Hz on Jetson Orin Nano.

- [ ] `src/world_model/` module with abstract `WorldModel` interface
- [ ] LeWorldModel implementation (DINOv2 encoder + transition model)
- [ ] CEM planner for goal-conditioned planning in latent space
- [ ] Predictive tracking: handle 5-10 frame occlusions
- [ ] Predictive safety evaluation: test actions before execution
- [ ] CLI args: `--world-model`, `--plan-horizon`, `--plan-samples`, `--safety-predict`
- [ ] ROS2 topic: `/vision/predictions` for predicted trajectories
- [ ] Tests: latency <10ms, memory <100MB, power <5W

### V-JEPA 2 Perception Enhancement (P1)

80M parameter video JEPA for temporal feature extraction at 10-20 FPS.

- [ ] TensorRT 3D-RoPE custom plugin (2-3 week effort)
- [ ] `src/models/vjepa2_extractor.py` for feature extraction
- [ ] Feature fusion with YOLO26 detection head
- [ ] Temporal consistency for SAM 3 segmentation
- [ ] ONNX fallback if TensorRT plugin delayed
- [ ] CLI args: `--vjepa-frames`, `--vjepa-precision`, `--feature-fusion`

### Edge-Cloud Split Architecture (P0)

Adaptive routing between edge (reactive) and cloud (deliberative) world models.

- [ ] `src/pipeline/edge_cloud_router.py` for routing logic
- [ ] Cloud API (REST/gRPC) for V-JEPA ViT-L / Kairos 4B inference
- [ ] Adaptive complexity detector: decide when to offload
- [ ] Fallback handling: graceful degradation when cloud unavailable
- [ ] Telemetry: track routing statistics, cloud latency
- [ ] CLI args: `--edge-cloud`, `--cloud-url`, `--complexity-threshold`

### Advanced World Model Features (P2)

- [ ] Synthetic data generation for rare edge cases
- [ ] Multi-step manipulation planning
- [ ] Physical interaction prediction (push, drop, collision)
- [ ] V-JEPA 2.1 dense features for detection/segmentation
- [ ] Kairos 3.0 integration (when Jetson Thor available)
- [ ] Hierarchical planning (short + long horizon)

---

## Feature Priority Matrix

| Feature | Priority | Version | Effort | Impact |
|:--------|:---------|:--------|:-------|:-------|
| YOLO26n/s integration | P0 | v1.5.0 | 2 weeks | High |
| Depth Anything V3 | P0 | v1.5.0 | 2 weeks | High |
| Hardware Abstraction Layer | P0 | v1.5.0 | 4 weeks | Critical |
| Multi-platform support | P0 | v1.5.0 | 4 weeks | Critical |
| Fleet management | P1 | v1.5.0 | 3 weeks | High |
| SAM 3 integration | P1 | v1.5.0 | 2 weeks | Medium |
| Unified inference pipeline | P0 | v2.0.0 | 4 weeks | Critical |
| Edge-cloud split inference | P0 | v2.0.0 | 3 weeks | High |
| Benchmarking suite | P0 | v2.0.0 | 2 weeks | High |
| Production deployment toolkit | P0 | v2.0.0 | 3 weeks | Critical |
| Advanced tracking | P1 | v2.0.0 | 2 weeks | Medium |
| Industry templates | P1 | v2.0.0 | 3 weeks | High |
| VLA edge support | P2 | v2.0.0 | 4 weeks | Medium |
| EU AI Act compliance | P2 | v2.0.0 | 2 weeks | Medium |
| LeWorldModel integration | P0 | v2.5.0 | 3 weeks | Critical |
| V-JEPA 2 perception | P1 | v2.5.0 | 6 weeks | High |
| Edge-cloud split (world models) | P0 | v2.5.0 | 4 weeks | High |
| Synthetic data generation | P2 | v2.5.0 | 4 weeks | Medium |

---

## Target Industries

| Industry | Market Size | Growth | Key Features Needed |
|:---------|:-----------|:-------|:-------------------|
| Warehouse/Logistics | $29.98B | 18.7% CAGR | Package detection, damage inspection, fleet management |
| Manufacturing QA | Largest CV segment | 13% CAGR | Defect detection, assembly verification, PPE monitoring |
| Agriculture | $18.5B → $74.2B | 16.4% CAGR | Weed detection, crop monitoring, outdoor robustness |
| Retail | $8.58B by 2032 | 24.3% CAGR | Shelf monitoring, inventory counting, privacy-preserving |
| Energy/Utilities | Steady | Growing | Infrastructure inspection, predictive maintenance |

---

## Hardware Support Matrix

| Platform | TOPS | Power | Price | Status | Backend |
|:---------|:-----|:------|:------|:-------|:--------|
| Jetson Orin Nano | 40 | 5-15W | $199-249 | Current | TensorRT |
| Jetson Orin NX | 100 | 10-25W | $399-499 | Planned v1.5 | TensorRT |
| Jetson T4000 | 1,200 | 40-70W | TBD | Planned v2.0 | TensorRT |
| Pi 5 + AI HAT+ 2 | 40 | ~12W | ~$150 | Planned v1.5 | Hailo DFC |
| Intel Core Ultra | 48 | 15-45W | $300-600 | Planned v1.5 | OpenVINO |
| Qualcomm RB5/RB6 | 15-30 | 5-15W | $600-800 | Planned v2.0 | QNN |
| Hailo-8 | 26 | 3.5W | $150-200 | Planned v1.5 | Hailo DFC |

---

## Competitive Positioning

| Dimension | OpenEyes | Isaac ROS | LeRobot | yolo_ros |
|:----------|:---------|:----------|:--------|:---------|
| Hardware support | Multi-vendor | NVIDIA only | Any (training) | NVIDIA only |
| Full pipeline | Yes | Yes | No (learning only) | No (detection only) |
| Fleet management | Yes | No | No | No |
| Setup time | 5 minutes | Hours-days | Days | 30 minutes |
| Open source | Apache 2.0 | Apache 2.0 | Apache 2.0 | GPL-3.0 |
| Edge-optimized | Primary focus | Secondary | No | Yes |
| Production-ready | Yes | Yes | No | Partial |

---

## Contributing to Roadmap

Want to suggest features? Please [open an issue](https://github.com/mandarwagh9/openeyes/issues) with:
- Feature description
- Use case
- Priority suggestion

---

## Notes

- Timeline is approximate and may change based on resources and feedback
- Priorities may shift based on user requirements and market changes
- Community contributions can accelerate development
- All new modules follow ROS2 standards for interoperability
- Hardware abstraction layer is the critical path for v1.5.0
