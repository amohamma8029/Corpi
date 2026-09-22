from collections.abc import Iterable

from fastapi import FastAPI

from core.api.auth import AuthMiddleware
from core.api.identity import APP_VERSION
from core.api.routes import router

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
)


def create_app(token: str, allowed_origins: Iterable[str] | None = None) -> FastAPI:
    """Build the core FastAPI app.

    `token` is the per-launch bearer credential. `allowed_origins` defaults to the
    Tauri 2 webview origins plus the Vite dev server; the server entry point can
    override it via CORPI_CORE_ALLOWED_ORIGINS.
    """
    origins = (
        frozenset(allowed_origins)
        if allowed_origins is not None
        else frozenset(DEFAULT_ALLOWED_ORIGINS)
    )
    app = FastAPI(
        title="Corpi Core",
        version=APP_VERSION,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(AuthMiddleware, token=token, allowed_origins=origins)
    app.include_router(router)
    return app
