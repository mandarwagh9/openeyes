# AGENTS.md - Developer Guidelines for PROJECT0

> This file contains guidelines for AI agents working on the PROJECT0 codebase.

---

## Project Overview

**PROJECT0** is a vision system for humanoid robots - providing AI-powered computer vision capabilities (object detection, depth estimation, face recognition, gesture recognition) running on NVIDIA Jetson Orin Nano.

---

## Version

- **Current Version**: v0.0.1
- **License**: Apache 2.0

---

## Build Commands

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Vision System

```bash
# Run main vision system
python src/main.py

# Run with specific camera
python src/main.py --camera 0

# Run with debug output
python src/main.py --debug
```

### Testing

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_camera.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Linting & Type Checking

```bash
# Run linting
python -m pylint src/

# Run type checking
python -m mypy src/

# Format code
python -m black src/
python -m isort src/
```

### Model Management

```bash
# Download AI models
python scripts/download_models.py

# Convert models to TensorRT
python scripts/convert_to_tensorrt.py --model yolov8n
```

---

## Code Style Guidelines

### Language
- **Primary**: Python 3.10+
- **Type Hints**: Required for all functions

### Import Organization

Order imports as follows (use `isort`):

```python
# 1. Standard library
import os
import sys
from typing import List, Dict, Optional

# 2. Third-party libraries
import cv2
import numpy as np
from ultralytics import YOLO

# 3. Internal modules
from src.camera import CameraHandler
from src.models import ObjectDetector

# 4. Local files
from .constants import *
```

### Naming Conventions

| Element | Convention | Example |
|:--------|:-----------|:--------|
| Files | snake_case | `camera_handler.py` |
| Classes | PascalCase | `CameraHandler` |
| Functions | camelCase | `processFrame()` |
| Constants | UPPER_SNAKE | `MAX_WIDTH = 640` |
| Variables | snake_case | `frame_buffer` |
| Type Aliases | PascalCase | `DetectionResult = Dict[str, Any]` |

### TypeScript-like Type Annotations Required

```python
# Good
def detect_objects(frame: np.ndarray, conf: float = 0.5) -> List[DetectionResult]:
    """Detect objects in the given frame.
    
    Args:
        frame: Input image as numpy array (H, W, 3)
        conf: Confidence threshold for detections
        
    Returns:
        List of detection results with bbox, label, and confidence
    """
    ...

# Bad
def detect_objects(frame, conf=0.5):
    ...
```

### Error Handling

```python
class VisionError(Exception):
    """Base exception for vision-related errors."""
    pass

class CameraError(VisionError):
    """Raised when camera initialization or read fails."""
    pass

class ModelError(VisionError):
    """Raised when AI model fails to load or run."""
    pass

# Usage
try:
    detector = ObjectDetector()
except ModelError as e:
    logger.error(f"Failed to load model: {e}")
    raise
```

### Docstrings

Use Google-style docstrings:

```python
def process_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Process a single frame through the vision pipeline.
    
    Args:
        frame: Input frame as BGR numpy array
        
    Returns:
        Dictionary containing:
            - objects: List of detected objects
            - depth: Depth map (if enabled)
            - faces: List of detected faces
            
    Raises:
        CameraError: If frame capture fails
        
    Example:
        >>> frame = camera.read()
        >>> result = process_frame(frame)
        >>> print(f"Detected {len(result['objects'])} objects")
    """
    ...
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_frame(frame: np.ndarray) -> Dict[str, Any]:
    logger.debug(f"Processing frame shape: {frame.shape}")
    # ... processing code
    logger.info(f"Detected {len(detections)} objects")
```

---

## Architecture

### Data Flow

```
Camera → Preprocess → AI Inference → Postprocess → Output
    ↓            ↓            ↓           ↓
  OpenCV    Resize/Norm   TensorRT    JSON/UDP
```

### Module Structure

```
src/
├── camera/
│   ├── __init__.py
│   ├── camera_handler.py    # Camera abstraction
│   └── types.py             # Camera-related types
├── models/
│   ├── __init__.py
│   ├── object_detector.py   # YOLOv8 wrapper
│   ├── depth_estimator.py  # MiDaS wrapper
│   └── face_recognizer.py  # Face detection
├── inference/
│   ├── __init__.py
│   ├── engine.py            # TensorRT wrapper
│   └── optimizer.py         # Model optimization
├── output/
│   ├── __init__.py
│   ├── json_output.py       # JSON output handler
│   └── udp_output.py        # UDP streaming
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging setup
│   └── config.py            # Configuration
└── main.py                  # Entry point
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_object_detector.py
import pytest
import numpy as np
from src.models.object_detector import ObjectDetector

class TestObjectDetector:
    """Tests for ObjectDetector class."""
    
    @pytest.fixture
    def detector(self):
        return ObjectDetector(model_path="models/yolov8n.pt")
    
    def test_detector_initializes(self, detector):
        assert detector is not None
        
    def test_detect_returns_list(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_models/

# With verbose output
pytest -v tests/

# Stop on first failure
pytest -x tests/
```

---

## Git Workflow

### Branch Naming

```
main              - Stable release
develop           - Integration branch
docs/<feature>    - Documentation
feat/<feature>   - New features
fix/<issue>      - Bug fixes
refactor/<area>  - Code improvements
```

### Commit Messages

```
<type>: <short description>

<detailed description>

Closes #<issue>
```

**Types:**
- `docs:` - Documentation changes
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code restructuring
- `test:` - Testing
- `chore:` - Maintenance

**Examples:**

```
docs: Add API documentation for ObjectDetector

Added detailed docstrings and type hints to the ObjectDetector
class following Google-style documentation format.

Closes #12
```

```
feat: Add depth estimation module

Implemented MiDaS depth estimation with TensorRT optimization.
Supports single-camera depth perception for obstacle avoidance.

Closes #8
```

---

## Performance Guidelines

### Optimization Targets

| Metric | Target | Priority |
|:-------|:-------|:---------|
| FPS | 20-30 | High |
| Latency | <50ms | High |
| Memory | <2GB | Medium |
| Model Size | <50MB | Medium |

### Profiling

```bash
# Profile code
python -m cProfile -o output.prof src/main.py

# Analyze with snakeviz
snakeviz output.prof
```

---

## Configuration

### Environment Variables

```bash
# .env file
CAMERA_INDEX=0
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
OUTPUT_FORMAT=json
DEBUG=false
```

### Runtime Flags

```python
# Using argparse
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--model', type=str, default='yolov8n')
parser.add_argument('--debug', action='store_true')
```

---

## Common Tasks

### Adding a New AI Model

1. Create model class in `src/models/`
2. Implement `load()` and `predict()` methods
3. Add tests in `tests/test_models/`
4. Update documentation

### Modifying Output Format

1. Edit `src/output/json_output.py` or `src/output/udp_output.py`
2. Update `API_SPEC.md`
3. Add tests for new format

### Debugging

```python
# Enable debug logging
python src/main.py --debug

# Or set environment variable
export DEBUG=true
python src/main.py
```

---

## Resources

- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) - Full technical spec
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
