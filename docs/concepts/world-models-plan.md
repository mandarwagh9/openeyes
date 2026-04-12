# World Models Integration Plan

> **Version**: v1.0.0
> **Date**: 2026-04-02
> **Status**: Approved for Implementation

---

## Executive Summary

This plan outlines the integration of world models into OpenEyes, adding **predictive capabilities** (trajectory prediction, counterfactual reasoning, safety evaluation) to the existing reactive vision pipeline.

**Key Principle**: World models **augment**, not replace, the current detection/tracking/depth pipeline.

**Target Hardware**: Jetson Orin Nano (8GB RAM, 40 TOPS)
**Power Budget**: <10W additional for world model stack
**Control Rate**: 100+ Hz for planning, 10-20 Hz for perception enhancement

---

## Why World Models?

### Current Limitation (Reactive Pipeline)
```
Frame → Detect → Track → Output
```
- No future prediction
- No "what if" reasoning
- Loses objects during occlusion
- No safety evaluation before action

### World Model-Enhanced Pipeline (Predictive)
```
Frame → Detect → Track → [World Model: Predict N steps] → Plan → Output
```
- Predicts object trajectories during occlusion
- Evaluates actions in imagined scenarios
- Maintains persistent world state
- Enables model-predictive control

---

## Phase 1: LeWorldModel Integration (Q2 2026)

**Timeline**: 3 weeks | **Effort**: 2 developers | **Risk**: Low

### 1.1 Objectives

- Integrate LeWorldModel (15M params) for latent-space planning
- Achieve 100+ Hz control rate on Jetson Orin Nano
- Augment existing tracking with trajectory prediction
- Add predictive safety evaluation

### 1.2 Deliverables

| Deliverable | Description | Timeline |
|------------|-------------|----------|
| `src/world_model/` module | Core world model abstraction | Week 1 |
| `src/world_model/lewm.py` | LeWorldModel implementation | Week 1-2 |
| Predictive tracking | Occlusion handling via prediction | Week 2 |
| Safety evaluation | Predict collision scenarios | Week 2-3 |
| CLI flags | `--world-model`, `--plan-horizon` | Week 3 |
| Tests | Unit + integration tests | Week 3 |
| Documentation | Technical docs + examples | Week 3 |

### 1.3 Technical Approach

```python
# src/world_model/base.py
class WorldModel(ABC):
    """Abstract world model interface."""
    
    @abstractmethod
    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode frame to latent state."""
        ...
    
    @abstractmethod
    def predict(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next latent state given action."""
        ...
    
    @abstractmethod
    def plan(self, current: np.ndarray, goal: np.ndarray, 
             horizon: int = 10) -> list[np.ndarray]:
        """Plan action sequence to reach goal."""
        ...
```

```python
# src/world_model/lewm.py
class LeWorldModel(WorldModel):
    """LeWorldModel: 15M param latent-space world model.
    
    Architecture:
    - Encoder: Frozen DINOv2 ViT-S (21M params, frozen)
    - Transition: Decoder-only ViT (15M params)
    - Planning: CEM in latent space
    
    Performance on Jetson Orin Nano:
    - Encoding: ~1-2ms
    - Prediction: ~0.5ms
    - Planning (100 samples): ~3-5ms
    - Total loop: ~5-10ms (100-200 Hz)
    - Memory: <100MB total
    - Power: 3-5W
    """
    
    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        self.encoder = self._load_dinov2_encoder()
        self.transition = self._load_transition_model()
        self.device = device
        
    def plan(self, current, goal, horizon=10):
        """CEM planning in latent space."""
        # 1. Sample action sequences
        # 2. Rollout with transition model
        # 3. Score against goal embedding
        # 4. Return best action sequence
        ...
```

### 1.4 Integration Points

| Existing Module | Integration | Benefit |
|----------------|-------------|---------|
| `src/utils/tracker.py` | Predict object positions during occlusion | Handle 5-10 frame occlusions |
| `src/core/frame_processor.py` | Add prediction stage after tracking | Predictive obstacle avoidance |
| `src/utils/safety_controller.py` | Evaluate actions before execution | Prevent collisions proactively |
| `src/ros2/vision_node.py` | Publish predicted trajectories | `/vision/predictions` topic |

### 1.5 New CLI Flags

```bash
--world-model          str     lewm        World model to use (lewm/vjepa2/none)
--plan-horizon         int     10          Planning horizon (steps)
--plan-samples         int     100         CEM sample count
--prediction-fps       int     30          Prediction update rate
--occlusion-frames     int     5           Max frames to predict through occlusion
--safety-predict       flag    False       Enable predictive safety evaluation
```

### 1.6 Testing Strategy

```python
# tests/test_world_model.py
class TestLeWorldModel:
    def test_encode_produces_correct_shape(self):
        ...
    
    def test_predict_next_state(self):
        ...
    
    def test_plan_reaches_goal(self):
        ...
    
    def test_occlusion_handling(self):
        ...
    
    def test_latency_under_10ms(self):
        ...
    
    def test_memory_under_100mb(self):
        ...
```

### 1.7 Dependencies

```txt
# New dependencies for Phase 1
torch>=2.0.0
timm>=1.0.0  # For DINOv2
```

---

## Phase 2: V-JEPA 2 Perception Enhancement (Q3 2026)

**Timeline**: 6 weeks | **Effort**: 2-3 developers | **Risk**: Medium

### 2.1 Objectives

- Integrate V-JEPA 2 ViT-B (80M) as perception feature extractor
- Achieve 10-20 FPS on Jetson Orin Nano
- Augment detection/segmentation with temporal features
- Implement TensorRT export (custom 3D-RoPE plugin)

### 2.2 Deliverables

| Deliverable | Description | Timeline |
|------------|-------------|----------|
| TensorRT 3D-RoPE plugin | Custom plugin for V-JEPA 2 | Week 1-3 |
| `src/models/vjepa2_extractor.py` | V-JEPA 2 feature extractor | Week 2-3 |
| Detection head integration | Feed V-JEPA features to detector | Week 3-4 |
| Segmentation head integration | Feed V-JEPA features to SAM | Week 4-5 |
| Benchmark suite | Compare with/without V-JEPA | Week 5-6 |
| Tests | Unit + integration + benchmark | Week 6 |

### 2.3 Technical Approach

**TensorRT Export Path**:
1. Export V-JEPA 2 to ONNX (handle 3D-RoPE as custom op)
2. Implement TensorRT plugin for 3D-RoPE
3. Build TensorRT engine with FP16/INT8 precision
4. Benchmark and validate accuracy

```python
# src/models/vjepa2_extractor.py
class VJEPA2FeatureExtractor:
    """V-JEPA 2 ViT-B feature extractor for perception enhancement.
    
    Architecture:
    - ViT-B encoder (80M params)
    - 3D-RoPE positional embeddings
    - Tubelet patchify: 2x16x16
    - Output: [1, N_patches, 768] per clip
    
    Performance on Jetson Orin Nano (TensorRT FP16):
    - 7 frames: 20-30 FPS
    - 16 frames: 10-20 FPS
    - 32 frames: 3-5 FPS
    - Memory: ~710MB total
    - Power: 6-9W
    """
    
    def __init__(self, variant: str = "vitb", num_frames: int = 16):
        self.variant = variant
        self.num_frames = num_frames
        self.engine = self._load_tensorrt_engine()
        
    def extract(self, frames: list[np.ndarray]) -> np.ndarray:
        """Extract spatiotemporal features from frame sequence."""
        ...
```

### 2.4 Integration Points

| Existing Module | Integration | Benefit |
|----------------|-------------|---------|
| `src/models/object_detector.py` | Fuse V-JEPA features with YOLO features | Better detection in clutter |
| `src/models/depth_estimator.py` | Use V-JEPA 3D-aware features | Improved depth estimation |
| `src/models/sam3_segmenter.py` | Temporal consistency in segmentation | Reduced flickering |

### 2.5 New CLI Flags

```bash
--vjepa-frames         int     16          Number of frames for V-JEPA 2
--vjepa-precision      str     fp16        Precision (fp16/int8)
--feature-fusion       flag    False       Enable V-JEPA feature fusion
```

---

## Phase 3: Edge-Cloud Split Architecture (Q4 2026)

**Timeline**: 4 weeks | **Effort**: 2 developers | **Risk**: Low

### 3.1 Objectives

- Implement edge-cloud split inference
- Edge: LeWM (100 Hz) + V-JEPA ViT-B (15 Hz)
- Cloud: V-JEPA ViT-L or Kairos 4B for deliberative planning
- Adaptive routing based on scene complexity

### 3.2 Deliverables

| Deliverable | Description | Timeline |
|------------|-------------|----------|
| `src/pipeline/edge_cloud_router.py` | Routing logic | Week 1 |
| Cloud API | REST/gRPC endpoint for cloud inference | Week 1-2 |
| Adaptive complexity detector | Decide when to offload | Week 2-3 |
| Fallback handling | Graceful degradation | Week 3 |
| Telemetry | Track routing statistics | Week 3-4 |
| Tests | Integration + failure mode tests | Week 4 |

### 3.3 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Edge (Orin Nano)                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ LeWM 15M     │───│ Complexity   │───│ Local Result   │  │
│  │ 100 Hz       │    │ Detector     │    │ (< 10ms)      │  │
│  └──────────────┘    │              │    └───────────────┘  │
│                      │ If complex   │                       │
│                      │ or novel     │                       │
│                      └──────┬───────┘                       │
│                             │                                │
├─────────────────────────────┼────────────────────────────────┤
│                             │ HTTPS/gRPC                     │
│                         ┌───┴────┐                           │
│                         │ Cloud   │                           │
│                         │ V-JEPA  │                           │
│                         │ ViT-L   │                           │
│                         └───┬────┘                           │
│                             │                                │
│                         ┌───┴────┐                           │
│                         │ Cloud   │                           │
│                         │ Result  │                           │
│                         │ (< 500ms)│                          │
│                         └────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 New CLI Flags

```bash
--edge-cloud           flag    False       Enable edge-cloud split
--cloud-url            str     ""          Cloud inference endpoint
--cloud-api-key        str     ""          API key for cloud
--complexity-threshold float  0.7          Threshold for offloading
--max-cloud-latency    int     500         Max cloud latency (ms)
--edge-only-fallback   flag    True        Use edge only if cloud fails
```

---

## Phase 4: Advanced World Model Features (2027)

**Timeline**: Ongoing | **Effort**: 1-2 developers | **Risk**: Medium

### 4.1 Planned Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Synthetic data generation | Generate rare edge cases for training | P1 |
| Multi-step manipulation planning | Plan complex manipulation sequences | P1 |
| Physical interaction prediction | Predict object dynamics (push, drop, etc.) | P2 |
| Hierarchical planning | Multi-scale planning (short + long horizon) | P2 |
| V-JEPA 2.1 dense features | Use latest V-JEPA 2.1 for detection/segmentation | P1 |
| Kairos 3.0 integration | When Jetson Thor becomes available | P2 |

---

## Resource Requirements

### Development Team

| Role | Time Commitment | Duration |
|------|----------------|----------|
| ML Engineer (world models) | Full-time | 13 weeks |
| Systems Engineer (TensorRT/edge) | Full-time | 9 weeks |
| ROS2 Integration Engineer | Part-time (50%) | 13 weeks |

### Hardware

| Item | Purpose | Cost |
|------|---------|------|
| Jetson Orin Nano dev kit | Primary development | $349 |
| Jetson Orin NX dev kit | Testing larger models | $499 |
| Cloud GPU (A100/V100) | Cloud inference endpoint | ~$500/month |

### Compute

| Task | Compute | Duration |
|------|---------|----------|
| TensorRT plugin development | Local (Orin Nano) | 3 weeks |
| V-JEPA 2 fine-tuning | Cloud GPU (1x A100) | 1-2 weeks |
| Benchmarking | Orin Nano + Orin NX | 1 week |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TensorRT 3D-RoPE plugin complexity | High | High | Budget 3 weeks, have ONNX fallback |
| LeWM accuracy insufficient for planning | Medium | Medium | Fallback to reactive tracking |
| V-JEPA 2 accuracy drop with quantization | Medium | Medium | Use INT8 QAT, not PTQ |
| Cloud latency exceeds budget | Low | Medium | Edge-only fallback, adaptive routing |
| Memory pressure on Orin Nano | Medium | High | Strict memory budgeting, profiling |
| Power budget exceeded | Low | Medium | Thermal monitoring, adaptive FPS |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Planning latency | <10ms | Benchmark suite |
| Prediction accuracy (occlusion) | >80% IoU | Test with held-out sequences |
| Detection improvement | +5% mAP | COCO/robot dataset benchmark |
| Power consumption | <10W additional | Jetson power monitoring |
| Memory usage | <1GB additional | `nvidia-smi` + `psutil` |
| Control rate | 100+ Hz for planning | Timing benchmarks |
| Test coverage | 80%+ for world_model module | pytest-cov |

---

## Timeline Summary

```
Q2 2026 (Apr-Jun)
├── Week 1-2: LeWM module + DINOv2 encoder
├── Week 3-4: Predictive tracking + safety
├── Week 5-6: CLI flags + ROS2 integration
└── Week 7-8: Testing + documentation

Q3 2026 (Jul-Sep)
├── Week 1-3: TensorRT 3D-RoPE plugin
├── Week 4-5: V-JEPA 2 feature extractor
├── Week 6-7: Detection/segmentation integration
└── Week 8-9: Benchmarking + optimization

Q4 2026 (Oct-Dec)
├── Week 1-2: Edge-cloud router
├── Week 3-4: Cloud API + adaptive routing
├── Week 5-6: Telemetry + fallback handling
└── Week 7-8: Integration testing + docs

2027
├── Synthetic data generation
├── Multi-step manipulation planning
├── V-JEPA 2.1 dense features
└── Kairos 3.0 (Jetson Thor)
```

---

## Next Steps (Immediate)

1. **Week 1**: Clone LeWorldModel/DINO-WM repos, set up development environment
2. **Week 1**: Create `src/world_model/` module structure
3. **Week 2**: Implement LeWM encoder + transition model
4. **Week 2**: Benchmark on Jetson Orin Nano
5. **Week 3**: Integrate with tracking module
6. **Week 3**: Add CLI flags and ROS2 topics
7. **Week 4**: Write tests and documentation

---

## References

- **V-JEPA 2**: https://github.com/facebookresearch/vjepa2
- **DINO-WM**: https://github.com/gaoyuezhou/dino_wm
- **LeRobot**: https://github.com/huggingface/lerobot
- **Kairos 3.0**: https://github.com/kairos-agi/kairos-sensenova
- **NVIDIA Cosmos**: https://github.com/NVIDIA/Cosmos
- **LeWorldModel paper**: arXiv:2603.19312
