"""OpenEyes REST API."""

import time
from typing import Optional

from fastapi import FastAPI

from src.api.routes import health, metrics, models, control

_app: Optional[FastAPI] = None
_start_time: float = time.time()


def create_api() -> FastAPI:
    global _app, _start_time
    _start_time = time.time()

    _app = FastAPI(
        title="OpenEyes Vision API",
        description="REST API for OpenEyes robot vision system",
        version="2.6.0",
    )

    _app.include_router(health.router, prefix="/health", tags=["health"])
    _app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
    _app.include_router(models.router, prefix="/models", tags=["models"])
    _app.include_router(control.router, prefix="/control", tags=["control"])

    return _app


def get_app() -> Optional[FastAPI]:
    return _app


def get_uptime() -> float:
    return time.time() - _start_time