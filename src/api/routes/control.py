"""Control endpoints for runtime configuration."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_control_state = {
    "following": False,
    "template": None,
    "turbo": False,
}


class ControlUpdate(BaseModel):
    following: bool = None
    template: str = None
    turbo: bool = None


class ControlResponse(BaseModel):
    status: str
    following: bool = False
    template: str = None
    turbo: bool = False


@router.get("")
async def get_control():
    return ControlResponse(
        status="ok",
        following=_control_state["following"],
        template=_control_state["template"],
        turbo=_control_state["turbo"],
    )


@router.post("")
async def update_control(req: ControlUpdate):
    global _control_state

    if req.following is not None:
        _control_state["following"] = req.following
    if req.template is not None:
        _control_state["template"] = req.template
    if req.turbo is not None:
        _control_state["turbo"] = req.turbo

    return ControlResponse(
        status="updated",
        following=_control_state["following"],
        template=_control_state["template"],
        turbo=_control_state["turbo"],
    )


@router.post("/stop")
async def stop_system():
    global _control_state
    _control_state["following"] = False
    return {"status": "stopped"}


@router.post("/start")
async def start_system():
    return {"status": "started"}