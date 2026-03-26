# CONTRIBUTING.md - Contributing to OpenEyes

> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## Welcome!

Thank you for your interest in contributing to OpenEyes! This document will help you get started.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what's best for the community

---

## How to Contribute

### 1. Report Bugs

Found a bug? Please [open an issue](https://github.com/mandarwagh9/openeyes/issues) with:

- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots/logs if applicable

### 2. Suggest Features

Have an idea? [Open a feature request](https://github.com/mandarwagh9/openeyes/issues) with:

- Clear description of the feature
- Use cases
- Any implementation ideas

### 3. Pull Requests

Want to contribute code? Follow these steps:

#### Step 1: Fork the Repository

```
1. Go to https://github.com/mandarwagh9/openeyes
2. Click "Fork" button
3. Clone your fork
```

#### Step 2: Create a Branch

```bash
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/bug-description
```

#### Step 3: Make Changes

```bash
# Make your changes
# Follow our coding standards (see AGENTS.md)
```

#### Step 4: Test Your Changes

```bash
# Run tests
pytest tests/

# Run linting
pylint src/
mypy src/
```

#### Step 5: Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

#### Step 6: Push to GitHub

```bash
git push origin feature/your-feature-name
```

#### Step 7: Open a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill in the PR template
4. Submit

---

## Development Setup

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10+ |
| Git | Latest |
| CUDA | 11.8+ (for GPU development) |

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Requirements Files

| File | Purpose |
|:-----|:--------|
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Development dependencies |
| `requirements-test.txt` | Testing dependencies |

---

## Coding Standards

See [AGENTS.md](AGENTS.md) for detailed coding standards:

### Key Points

- **Type Hints**: Required for all functions
- **Docstrings**: Use Google-style format
- **Naming**: Follow snake_case for variables, PascalCase for classes
- **Imports**: Use isort for organization

### Code Style

```python
def process_frame(frame: np.ndarray, config: Config) -> Result:
    """Process a single frame through the vision pipeline.
    
    Args:
        frame: Input frame as BGR numpy array
        config: Configuration object
        
    Returns:
        Processing result with detections
        
    Raises:
        CameraError: If frame capture fails
    """
    # Implementation
    pass
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_camera.py

# With coverage
pytest --cov=src --cov-report=html
```

### Writing Tests

```python
# tests/test_object_detector.py
import pytest
import numpy as np
from src.models import ObjectDetector

class TestObjectDetector:
    @pytest.fixture
    def detector(self):
        return ObjectDetector("models/yolov8n.pt")
    
    def test_detect_returns_list(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)
```

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|:-----|:------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `style` | Code style (formatting) |
| `refactor` | Code refactoring |
| `test` | Testing |
| `chore` | Maintenance |

### Examples

```
feat: add depth estimation module

Implemented MiDaS depth estimation with TensorRT optimization.
Supports single-camera depth perception for obstacle avoidance.

Closes #8
```

```
fix: resolve camera disconnect handling

Added auto-reconnect with exponential backoff.
Now handles USB camera unplugged gracefully.

Closes #15
```

---

## Pull Request Guidelines

### PR Requirements

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Commits are logical and well-described

### PR Review Process

1. Automated checks run
2. At least one maintainer review required
3. Address feedback promptly
4. Squash commits before merge

---

## Recognition

Contributors will be recognized in:

- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- GitHub profile

---

## Questions?

| Channel | Link |
|:--------|:-----|
| GitHub Issues | https://github.com/mandarwagh9/openeyes/issues |
| Discussions | https://github.com/mandarwagh9/openeyes/discussions |

---

## Thank You!

Your contributions make OpenEyes better for everyone.
