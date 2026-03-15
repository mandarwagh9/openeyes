# AGENTS.md - Developer Guidelines for PROJECT0

> AI agent guidelines for PROJECT0 - vision system for humanoid robots.

---

## Quick Commands

```bash
# Install & Run
pip install -r requirements.txt
python src/main.py --camera 0 --debug

# Testing
pytest tests/                          # All tests
pytest tests/test_camera.py -v        # Single file (verbose)
pytest tests/ -x                      # Stop on first failure
pytest tests/ --cov=src --cov-report=html

# Linting & Formatting
pylint src/
mypy src/
black src/
isort src/
```

---

## Code Style

### Language
- Python 3.10+
- Type hints required on ALL functions

### Imports (use isort)
```python
# 1. Standard library
import os
from typing import List, Optional

# 2. Third-party
import cv2
import numpy as np

# 3. Internal modules
from src.camera import CameraHandler
from src.models import ObjectDetector
```

### Naming Conventions

| Element | Convention | Example |
|:--------|:-----------|:--------|
| Files | snake_case | `camera_handler.py` |
| Classes | PascalCase | `CameraHandler` |
| Functions | snake_case | `process_frame()` |
| Variables | snake_case | `frame_buffer` |
| Constants | UPPER_SNAKE | `MAX_WIDTH = 640` |

### Type Annotations Required
```python
def detect_objects(frame: np.ndarray, conf: float = 0.5) -> List[Detection]:
    """Detect objects in the given frame."""
    ...
```

### Error Handling
Use custom exceptions from `src/exceptions.py`:
```python
from src.exceptions import CameraError, ModelError, VisionError

try:
    detector.load()
except ModelError as e:
    logger.error(f"Failed to load model: {e}")
    raise
```

### Docstrings
Google-style, required on all public functions:
```python
def process_frame(frame: np.ndarray) -> VisionResult:
    """Process a single frame through the vision pipeline.

    Args:
        frame: Input frame as BGR numpy array

    Returns:
        VisionResult containing detections, depth, faces, gestures, pose

    Raises:
        CameraError: If frame capture fails
    """
```

### Logging
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info(f"Detected {len(detections)} objects")
```

---

## Architecture

```
src/
├── camera/           # CameraHandler, types
├── models/           # ObjectDetector, depth_estimator, etc.
├── output/           # json_formatter, udp_sender
├── utils/            # config, logger
└── main.py           # Entry point
```

### Data Flow
Camera → ObjectDetector → JSON Formatter → UDP Sender

---

## Testing

### Test Structure
```python
import pytest
from src.models.object_detector import ObjectDetector

class TestObjectDetector:
    @pytest.fixture
    def detector(self):
        return ObjectDetector(model_path="models/yolov8n.pt")

    def test_detect_returns_list(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)
```

---

## Git Workflow

### Branch Naming
- `feat/<feature>` - New features
- `fix/<issue>` - Bug fixes
- `docs/<feature>` - Documentation

### Commit Messages
```
<type>: <short description>

Closes #<issue>
```
Types: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

---

## Common Tasks

### Add New AI Model
1. Create class in `src/models/`
2. Implement `load()` and `detect()` methods
3. Add type hints and docstrings
4. Add tests in `tests/`

### Modify Output
1. Edit `src/output/json_formatter.py`
2. Ensure output matches API_SPEC.md schema

---

## Performance Targets

| Metric | Target |
|:-------|:-------|
| FPS | 20-30 |
| Latency | <50ms |
| Memory | <2GB |

---

## Resources

- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md)
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
