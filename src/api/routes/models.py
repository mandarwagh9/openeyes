"""Model registry endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ModelRegisterRequest(BaseModel):
    name: str
    path: str
    version: str = "v1.0"
    notes: str = ""


class ModelInfo(BaseModel):
    name: str
    version: str
    path: str
    size_bytes: int = 0


@router.get("")
async def list_models():
    from src.fleet.model_registry import ModelRegistry

    registry = ModelRegistry()
    model_names = registry.list_models()

    models = []
    for name in model_names:
        versions = registry.list_versions(name)
        for v in versions:
            models.append(ModelInfo(
                name=v.name,
                version=v.version,
                path=v.file_path,
                size_bytes=v.file_size_bytes,
            ))

    return {"models": models}


@router.post("")
async def register_model(req: ModelRegisterRequest):
    from src.fleet.model_registry import ModelRegistry

    try:
        registry = ModelRegistry()
        model = registry.register_model(
            name=req.name,
            version=req.version,
            file_path=req.path,
            notes=req.notes,
        )
        return {"status": "registered", "checksum": model.checksum}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}/{version}")
async def delete_model(name: str, version: str):
    return {"status": "not_implemented", "message": "Model deletion not yet implemented"}