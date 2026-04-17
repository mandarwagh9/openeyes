# OpenEyes Agent Notes

## Repo shape
- This repo has two deliverables: Python runtime in `src/` and website output at root (`index.html`, `docs.html`, `hardware.html`, `demos.html`, `community.html`).
- Website source-of-truth is now Astro in `web/` (components/pages/styles), then built to root HTML.
- Treat executable code as source of truth. Many prose docs are aspirational or stale.

## Source of truth files
- CLI flags/modes: `src/cli/argparse.py`
- Runtime entry flow: `src/main.py`
- Core processing loop: `src/core/vision_system.py` and `src/core/frame_processor.py`
- DeepStream path (`--deepstream`): `src/deepstream/pipeline.py`
- Config and env overrides: `src/utils/config.py` + `config.yaml`

## Commands that are reliable
```bash
pip install -r requirements.txt
python -m src.main --debug
python -m src.main --deepstream --camera 0
python -m src.main --api --api-port 8000 --api-host 127.0.0.1
python -m src.main --video input.mp4 --output output.mp4
sudo bash scripts/jetson_perf.sh
python -m benchmarks.run_benchmarks --all
```

## Testing
- Full suite: `pytest tests -q`
- Focused run: `pytest tests/test_config.py -q`
- Current baseline (verified): `pytest tests -q` has 1 failure:
  - `tests/test_deepstream_pipeline.py::TestDeepStreamPipeline::test_initialization_defaults`
  - Test expects default width `640`; `DeepStreamPipeline` default is `1280`.

## Runtime quirks that save time
- Prefer `python -m src.main` for local runs; docs and launch files mix invocation styles.
- DeepStream currently forces `yolov10n` config in `DeepStreamPipeline._get_config_path()` even if another model is requested.
- `src/main.py` injects ROS2 Python paths and sets `DISPLAY=:0` when unset.
- Keep the ROS2 init delay in mind: `VisionSystem._init_ros2()` intentionally sleeps `0.5s` after startup.
- `Config` env overrides are limited to `CAMERA_INDEX`, `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, `OUTPUT_HOST`, `OUTPUT_PORT`, `DEBUG`.
- `.gitignore` excludes model artifacts (`*.onnx`, `*.engine`, `models/*.pt`), so model binaries will not appear in git status unless unignored.
- `launch/openeyes.launch.py` hardcodes `python3 src/main.py ...` values and does not wire declared launch args into the command.
- `scripts/deploy-docs.sh` is destructive (`git checkout -B gh-pages` + `git push --force`); do not run casually.

## Website editing gotchas
- Edit `web/` files, not root HTML output.
- Build website with `npm run build` (or Dockerized Node if Node is not installed locally), then promote `dist/` output to root files.
- Preferred links use clean routes (`/docs`, `/hardware`, `/demos`, `/community`) that redirect to `.html` pages.
- Use `npm run publish:root` to build + sync root pages in one command.
- `demos` page uses local GIFs in `demo/`.

## Known docs drift
- Version strings are inconsistent across code/docs (`README.md` v3.0.1, `src/main.py` and API v2.6.0, `src/__init__.py` v0.0.1).
- Some documented commands are not implemented in current CLI (for example, `openeyes fleet ...` subcommands and `scripts/optimize_jetson.py`).
- When unsure, validate against `src/cli/argparse.py` and runnable tests, not markdown docs.
- There is no checked-in CI workflow, pre-commit config, or formatter config at repo root.
