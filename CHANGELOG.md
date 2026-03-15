# CHANGELOG.md - Version History for PROJECT0

> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.0.1] - 2026-03-15

### Added

- **Documentation**
  - README.md with project overview
  - AGENTS.md developer guidelines
  - TECHNICAL_SPEC.md technical specifications
  - ARCHITECTURE.md system architecture
  - HARDWARE.md hardware specifications
  - API_SPEC.md API documentation
  - QUICKSTART.md quick start guide
  - INSTALL.md detailed installation
  - USER_GUIDE.md user guide
  - TROUBLESHOOTING.md common issues
  - CONTRIBUTING.md contribution guidelines
  - ROADMAP.md project roadmap
  - CHANGELOG.md version history

- **Project Structure**
  - Directory structure for src/, models/, docs/
  - requirements.txt with dependencies
  - LICENSE (Apache 2.0)

- **Source Code**
  - config.yaml with default configuration
  - camera/ module with CameraHandler
  - models/ module with ObjectDetector (YOLOv8)
  - output/ module with JSON formatter and UDP sender
  - utils/ module with config loader and logger
  - main.py entry point

### Changed

- Initial repository setup
- Project named "PROJECT0"
- License set to Apache 2.0

### Known Issues

- No tests implemented yet

---

## [Unreleased]

### Planned for v0.0.2

- [x] YOLOv8n object detection model integration
- [x] Camera handler implementation
- [x] Basic output (JSON/UDP)
- [ ] Working object detection demo
- [ ] Unit tests

### Planned for v0.0.3

- [ ] MiDaS depth estimation integration
- [ ] Depth visualization
- [ ] Distance calculation utilities

---

## Version Format

Given a version number `MAJOR.MINOR.PATCH`:

- **MAJOR** - Incompatible API changes
- **MINOR** - New functionality (backwards compatible)
- **PATCH** - Bug fixes

---

## Upgrade Guide

### From v0.0.1 to v0.0.2

1. Update requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Download models:
   ```bash
   python scripts/download_models.py
   ```

3. Run vision system:
   ```bash
   python src/main.py
   ```

---

## Release Cycle

| Version | Type | Target |
|:--------|:-----|:-------|
| v0.0.1 | Initial | March 2026 |
| v0.0.2 | Minor | April 2026 |
| v0.0.3 | Minor | April 2026 |
| v1.0.0 | Major | June 2026 |

---

## Acknowledgments

This CHANGELOG format is based on [Keep a Changelog](https://keepachangelog.com).
