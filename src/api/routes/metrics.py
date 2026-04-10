"""Prometheus metrics endpoint."""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("", response_class=PlainTextResponse)
async def get_metrics():
    from src.utils.prometheus_exporter import get_exporter

    exporter = get_exporter()
    return PlainTextResponse(
        content=exporter.generate(),
        media_type=exporter.content_type(),
    )