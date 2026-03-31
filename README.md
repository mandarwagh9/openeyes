# OpenEyes Documentation

Documentation website for OpenEyes - open-source vision system for humanoid robots.

## Quick Start

```bash
# Install dependencies
pip install mkdocs-material

# Serve locally
mkdocs serve

# Build for production
mkdocs build
```

## Deployment

This site is configured for deployment to Vercel. Connect the repository to Vercel and it will automatically build.

## Structure

```
docs/
├── index.md              # Landing page
├── getting-started/      # Quickstart, installation, config
├── user-guide/           # Commands, ROS2
├── development/          # Architecture, API, contributing
├── reference/            # Technical specs, hardware
└── troubleshooting/      # Common issues, FAQ
```

## Customization

- Edit `mkdocs.yml` to change theme, navigation, colors
- Add custom CSS in `assets/css/extra.css`
- Edit templates in `overrides/` (if needed)

## License

Apache 2.0 - Same as main project.