"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    uptime: float
    fps: float = 0.0


@router.get("", response_model=HealthResponse)
async def health_check():
    from src.api import get_uptime
    from src.core.vision_system import VisionSystem

    fps = 0.0
    try:
        if VisionSystem._instance:
            perf = VisionSystem._instance._perf_monitor
            if perf:
                fps = perf.get_stats().fps
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        uptime=get_uptime(),
        fps=fps,
    )


@router.get("/live")
async def liveness():
    return {"status": "alive"}


@router.get("/ready")
async def readiness():
    from src.core.vision_system import VisionSystem
    ready = VisionSystem._instance is not None
    return {"ready": ready}