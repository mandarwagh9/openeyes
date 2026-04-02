# World Models for OpenEyes

> **Version**: v1.0.0
> **Date**: 2026-04-02
> **Status**: Planning Complete, Phase 1 Ready for Implementation

---

## 1. What Are World Models?

World models are neural networks that learn **internal representations of environmental dynamics**. They function as "internal simulators" that predict how the world will evolve given current observations and actions.

**Key distinction from traditional vision**:
- **Traditional vision (YOLO, depth, tracking)**: Discriminative — classifies what's in the current frame
- **World models**: Generative and predictive — imagines what the next frame(s) will look like given actions

**Core concept**: Instead of learning a direct policy mapping observations to actions, a world model first learns a compressed model of the environment's dynamics, then plans within that learned model. This is analogous to how humans imagine consequences before acting.

---

## 2. Why World Models for OpenEyes?

### Current Pipeline Limitations

```
Frame → Detect → Track → Output
```

| Limitation | Impact |
|-----------|--------|
| No future prediction | Cannot anticipate object movement |
| No "what if" reasoning | Cannot evaluate actions before executing |
| Loses objects during occlusion | Tracking fails when objects are hidden |
| No safety evaluation | Actions executed without foresight |
| Reactive only | No proactive behavior |

### World Model-Enhanced Pipeline

```
Frame → Detect → Track → [World Model: Predict N steps] → Plan → Output
```

| New Capability | Benefit |
|---------------|---------|
| **Predictive tracking** | Handle 5-10 frame occlusions |
| **Counterfactual reasoning** | "What if I turn left vs right?" |
| **Safety evaluation** | Test actions in imagined scenarios |
| **Persistent world state** | Remember objects outside camera view |
| **Physical intuition** | Understand gravity, collision, inertia |
| **Synthetic data generation** | Create rare edge cases for training |

---

## 3. World Model Landscape (2026)

### Models Evaluated for OpenEyes

| Model | Params | Type | Edge Ready | Open Source | Relevance |
|-------|--------|------|-----------|-------------|-----------|
| **LeWorldModel** | 15M | Latent-space JEPA | ✅ 100-200 Hz | Yes | **Primary** |
| **V-JEPA 2 ViT-B** | 80M | Video JEPA | ✅ 10-20 FPS | Yes | **Secondary** |
| **V-JEPA 2 ViT-L** | 300M | Video JEPA | ⚠️ 3-6 FPS | Yes | Cloud |
| **V-JEPA 2.1 ViT-B** | 80M | Dense features | ✅ 10-20 FPS | Yes | **Perception** |
| **DINO-WM** | 15M | DINO + transition | ✅ 100+ FPS | Yes | **Primary** |
| **Kairos 3.0-4B** | 4B | Generative world model | ❌ Needs Thor | Yes | Future |
| **NVIDIA Cosmos** | 4B-14B | Diffusion/Autoregressive | ❌ Server only | Partial | Cloud |
| **Genie 3** | Closed | Interactive 3D | ❌ Not available | No | Not relevant |
| **DreamerV3** | ~50M | Latent dynamics | ✅ Possible | Yes | Alternative |

### Architecture Comparison

| Architecture | Approach | Pros | Cons |
|-------------|----------|------|------|
| **JEPA** (V-JEPA 2) | Predict embeddings, not pixels | Efficient, semantically meaningful | Custom ops (3D-RoPE), no TensorRT support |
| **Latent-space** (LeWM, Dreamer) | Predict in compressed latent space | Very fast, low memory | Less interpretable |
| **Diffusion** (Cosmos) | Generate future frames via diffusion | High fidelity, multi-modal | Very slow (minutes per action) |
| **Autoregressive** (Genie 3, Kairos) | Next-token prediction for video | Leverages transformer scaling | Error accumulation, heavy |

---

## 4. Recommended Architecture

### Edge Stack (Jetson Orin Nano)

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenEyes v2.0+ Pipeline                    │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Camera   │→│ Detect   │→│ Track    │→│ World Model│  │
│  │ (30 FPS) │  │ (YOLO26) │  │(ByteTrack)│ │(LeWM 15M)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────┬──────┘  │
│                                                   │          │
│  ┌──────────────────────────────────────────┐     │          │
│  │           Prediction & Planning           │     │          │
│  │  ┌────────────┐  ┌────────────┐  ┌─────┐ │     │          │
│  │  │ Trajectory │  │ Collision  │  │Goal │ │     │          │
│  │  │ Prediction │  │ Evaluation │  │Plan │ │     │          │
│  │  └────────────┘  └────────────┘  └─────┘ │     │          │
│  └──────────────────────────────────────────┘     │          │
│                                                   ↓          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Output Layer                         │   │
│  │  /vision/detections  /vision/predictions  /vision/cmd│   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Performance: 100-200 Hz planning, <10ms latency, <100MB    │
│  Memory, 3-5W power                                         │
└─────────────────────────────────────────────────────────────┘
```

### Edge-Cloud Split (Phase 3)

```
Edge (Orin Nano)                     Cloud (A100/V100)
┌─────────────────────┐             ┌─────────────────────┐
│ LeWM 15M            │             │ V-JEPA 2 ViT-L 300M │
│ 100 Hz reactive     │────HTTPS───→│ or Kairos 4B        │
│ <10ms latency       │←────────────│ 500ms latency       │
│                     │             │ Deliberative plan   │
│ V-JEPA ViT-B 80M    │             │                     │
│ 15 Hz perception    │             │                     │
└─────────────────────┘             └─────────────────────┘
```

---

## 5. Phase 1: LeWorldModel Integration

### 5.1 What is LeWorldModel?

LeWorldModel (LeWM) is a 15M parameter latent-space world model based on the JEPA architecture. It uses:
- **Frozen DINOv2 encoder**: Extracts patch-level features from frames
- **Transition model**: Small ViT that predicts next latent state given action
- **CEM planning**: Cross-Entropy Method for optimization in latent space

**Key paper**: arXiv:2603.19312 (Mila/NYU/LeCun, Mar 2026)

### 5.2 Performance on Jetson Orin Nano

| Metric | Value |
|--------|-------|
| Model size | 15M parameters |
| Memory (FP16) | <100MB total |
| Encoding latency | 1-2ms |
| Prediction latency | 0.5ms |
| Planning (100 samples) | 3-5ms |
| Total loop | 5-10ms |
| Control rate | 100-200 Hz |
| Power consumption | 3-5W |

### 5.3 Implementation Plan

```
src/world_model/
├── __init__.py
├── base.py              # WorldModel abstract interface
├── lewm.py              # LeWorldModel implementation
├── planner.py           # CEM planner
├── safety_evaluator.py  # Predictive safety checks
└── types.py             # Data types (Prediction, Plan, etc.)
```

### 5.4 Integration with Existing Modules

| Module | Integration | Change |
|--------|-------------|--------|
| `src/utils/tracker.py` | Predict positions during occlusion | Add `predict_next_position()` method |
| `src/core/frame_processor.py` | Add prediction stage | Insert after tracking step |
| `src/utils/safety_controller.py` | Evaluate actions before execution | Add `evaluate_action_safety()` method |
| `src/ros2/vision_node.py` | Publish predictions | Add `/vision/predictions` topic |

### 5.5 New CLI Flags

```bash
--world-model          str     lewm        World model (lewm/vjepa2/none)
--plan-horizon         int     10          Planning horizon (steps)
--plan-samples         int     100         CEM sample count
--prediction-fps       int     30          Prediction update rate
--occlusion-frames     int     5           Max frames to predict through occlusion
--safety-predict       flag    False       Enable predictive safety evaluation
```

### 5.6 Example Usage

```bash
# Basic world model with predictive tracking
python -m src.main --camera 0 --world-model lewm --follow

# With safety evaluation
python -m src.main --camera 0 --world-model lewm --safety-predict --min-distance 0.5

# Custom planning parameters
python -m src.main --camera 0 --world-model lewm --plan-horizon 20 --plan-samples 200

# With ROS2 publishing predictions
python -m src.main --camera 0 --world-model lewm --ros2
```

### 5.7 ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/predictions` | JSON | Predicted object positions N steps ahead |
| `/vision/plan` | JSON | Current action plan from world model |
| `/vision/safety` | JSON | Safety evaluation results |

**Message format** (`/vision/predictions`):
```json
{
  "type": "predictions",
  "timestamp": 1712000000.0,
  "horizon": 10,
  "predictions": [
    {
      "track_id": 1,
      "class": "person",
      "positions": [
        {"step": 1, "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}},
        {"step": 2, "bbox": {"x1": 110, "y1": 55, "x2": 210, "y2": 305}},
        ...
      ],
      "confidence": 0.92
    }
  ]
}
```

---

## 6. Phase 2: V-JEPA 2 Perception Enhancement

### 6.1 What is V-JEPA 2?

V-JEPA 2 (Video Joint Embedding Predictive Architecture) is Meta FAIR's self-supervised video model that predicts future video embeddings rather than pixels. This makes it computationally efficient compared to pixel-generation approaches.

**Key features**:
- 3D-RoPE positional embeddings for spatiotemporal understanding
- Predicts masked tubelet embeddings (2×16×16 patches)
- Open-source (Apache 2.0), HuggingFace models available
- V-JEPA 2.1 (Mar 2026) adds dense features for detection/segmentation

### 6.2 Performance on Jetson Orin Nano

| Variant | Params | FPS (TensorRT FP16) | Memory | Power |
|---------|--------|---------------------|--------|-------|
| ViT-B (80M) | 80M | 10-20 FPS (16 frames) | ~710MB | 6-9W |
| ViT-L (300M) | 300M | 3-6 FPS (16 frames) | ~1.45GB | 10-14W |
| ViT-g (1B) | 1B | 1-2 FPS | ~4GB | 13-15W |

**Recommended**: ViT-B for edge, ViT-L for cloud.

### 6.3 TensorRT Export Challenge

V-JEPA 2 uses **3D-RoPE** (3D Rotary Position Embeddings) which is not natively supported by TensorRT. The export path requires:

1. Export to ONNX with custom op for 3D-RoPE
2. Implement TensorRT plugin for 3D-RoPE
3. Build TensorRT engine with FP16/INT8 precision
4. Validate accuracy against PyTorch reference

**Estimated effort**: 2-3 weeks for TensorRT plugin development.

### 6.4 Integration with Detection

V-JEPA 2.1's dense features are specifically designed for dense prediction tasks:

```
V-JEPA 2 features [N_patches, 768]
        ↓
    [Feature Fusion]
        ↓
YOLO26 features [N_patches, 256]
        ↓
    [Detection Head]
        ↓
  Object detections
```

Expected improvement: +3-5% mAP on cluttered scenes, better small object detection.

---

## 7. Phase 3: Edge-Cloud Split

### 7.1 Architecture

| Layer | Model | Frequency | Latency | Purpose |
|-------|-------|-----------|---------|---------|
| Edge | LeWM 15M | 100 Hz | <10ms | Reactive control |
| Edge | V-JEPA ViT-B | 15 Hz | ~50ms | Perception enhancement |
| Cloud | V-JEPA ViT-L | 2 Hz | ~500ms | Deliberative planning |
| Cloud | Kairos 4B | 1 Hz | ~1s | Complex scene understanding |

### 7.2 Adaptive Routing

The edge world model decides when to consult the cloud:
- **Low complexity** (familiar scene, high confidence): Edge only
- **Medium complexity** (novel objects, moderate confidence): Edge + async cloud verification
- **High complexity** (unseen environment, low confidence): Sync cloud consultation

### 7.3 Fallback Handling

If cloud is unavailable:
1. Edge continues with local world model
2. Warning logged, telemetry recorded
3. Planning horizon reduced to maintain safety
4. Reconnection attempted with exponential backoff

---

## 8. Benchmarking

### 8.1 Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Planning latency | <10ms | `time.perf_counter()` |
| Prediction accuracy (occlusion) | >80% IoU | Held-out sequences |
| Detection improvement | +5% mAP | COCO/robot dataset |
| Power consumption | <10W additional | `jetson-stats` |
| Memory usage | <1GB additional | `nvidia-smi` |
| Control rate | 100+ Hz | Timing benchmarks |

### 8.2 Benchmark Commands

```bash
# World model latency benchmark
openeyes benchmark --world-model lewm --iterations 1000

# Prediction accuracy benchmark
openeyes benchmark --prediction-accuracy --dataset robot-logs

# Detection improvement benchmark
openeyes benchmark --detection --with-world-model --without-world-model

# Power consumption benchmark
openeyes benchmark --power --world-model lewm --duration 300
```

---

## 9. Data Requirements

### 9.1 Training Data for Fine-tuning

| Model | Data Needed | Source |
|-------|------------|--------|
| LeWM transition model | 10-50 hours of robot interaction | Self-collected |
| V-JEPA 2 action-conditioned | 62 hours (DROID-scale) | Open X-Embodiment |
| V-JEPA 2.1 dense features | Pre-trained, no fine-tuning needed | HuggingFace |

### 9.2 Data Collection

Use LeRobot dataset format for collecting training data:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="openeyes/warehouse-robot",
    fps=30,
    video=True,
    image_writer_processes=10,
)
```

---

## 10. Limitations and Known Issues

### 10.1 Current Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| No TensorRT support for V-JEPA 2 | Requires ONNX fallback (slower) | Custom plugin development (Phase 2) |
| LeWM trained on simple environments | May not generalize to complex scenes | Fine-tune on domain-specific data |
| Short planning horizon (10-20 steps) | Limited long-horizon planning | Hierarchical planning (Phase 4) |
| Image-only goals (no language) | Cannot accept text goals | V-JEPA 2.1 + language model (Phase 4) |
| Correlation ≠ causation | Novel interventions may fail | Combine with physics-based models |

### 10.2 What World Models DON'T Do

- **Replace object detection**: Still need YOLO for classification
- **Replace depth estimation**: Still need DA3 for metric depth
- **Perfect simulation**: Approximations, not perfect physics
- **Eliminate sim2real gap**: Shift it from hand-tuned to data-driven

---

## 11. References

### Papers
- **V-JEPA 2**: "Video Joint Embedding Predictive Architecture 2" (Meta FAIR, 2025)
- **V-JEPA 2.1**: "Dense Predictive Features for Video Understanding" (Meta FAIR, Mar 2026)
- **DINO-WM**: "World Models on Pre-trained Features for Zero-shot Planning" (ICML 2025)
- **LeWorldModel**: arXiv:2603.19312 (Mila/NYU/LeCun, Mar 2026)
- **DreamerV3**: "Mastering Diverse Control through Latent Imagination" (Nature 2025)
- **Kairos 3.0**: "Generative World Model for Embodied Intelligence" (ACE Robotics, Mar 2026)

### Code Repos
- **V-JEPA 2**: https://github.com/facebookresearch/vjepa2 (3.5k stars)
- **DINO-WM**: https://github.com/gaoyuezhou/dino_wm
- **LeRobot**: https://github.com/huggingface/lerobot (22.9k stars)
- **Kairos**: https://github.com/kairos-agi/kairos-sensenova
- **NVIDIA Cosmos**: https://github.com/NVIDIA/Cosmos

### HuggingFace Models
- `facebook/vjepa2-vitl-fpc64-256` (300M)
- `facebook/vjepa2-vith-fpc64-256` (600M)
- `facebook/vjepa2-vitg-fpc64-256` (1B)
- `facebook/vjepa2-vitg-fpc64-384` (1B, higher res)

---

## 12. FAQ

**Q: Can world models replace our current detection/tracking pipeline?**
A: No. World models complement reactive vision. You still need YOLO for detection, DA3 for depth, and ByteTrack for tracking. World models add prediction and planning on top.

**Q: Will this slow down our current pipeline?**
A: LeWM adds <10ms latency (100+ Hz). V-JEPA 2 ViT-B adds ~50ms per inference (10-20 FPS). The total pipeline stays above 6-10 FPS with all features enabled.

**Q: Can we run this without a Jetson?**
A: LeWM can run on any GPU with PyTorch. V-JEPA 2 requires more compute. For CPU-only systems, use ONNX Runtime with reduced frame counts.

**Q: How much training data do we need?**
A: LeWM's transition model can be fine-tuned with 10-50 hours of robot interaction data. V-JEPA 2 action-conditioned needs ~62 hours of DROID-scale data.

**Q: Is this production-ready?**
A: Phase 1 (LeWM) is designed for production on Jetson Orin Nano. Phase 2 (V-JEPA 2) requires TensorRT plugin development. Phase 3 (edge-cloud) needs cloud infrastructure.
