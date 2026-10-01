from fastapi import APIRouter

from core.api.identity import APP_NAME, APP_VERSION

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"app": APP_NAME, "version": APP_VERSION, "status": "ok"}
