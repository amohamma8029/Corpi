from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from core.api.auth import AuthMiddleware
from core.api.identity import APP_VERSION
from core.api.jobs import router as jobs_router
from core.api.routes import router
from core.persistence.store import PersistenceStore


class LifecycleService(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    # macOS and Linux webviews use the tauri scheme; Windows uses http(s).
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
)


def create_app(
    token: str,
    allowed_origins: Iterable[str] | None = None,
    *,
    store: PersistenceStore | None = None,
    runner: LifecycleService | None = None,
    sweeper: LifecycleService | None = None,
) -> FastAPI:
    """Build the core FastAPI app.

    `token` is the per-launch bearer credential. `allowed_origins` defaults to the
    Tauri 2 webview origins plus the Vite dev server; the server entry point can
    override it via CORPI_CORE_ALLOWED_ORIGINS. When a `store` is provided the
    job-status routes are mounted; `runner` and `sweeper` are started and
    stopped with the app's lifespan.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        started = [service for service in (runner, sweeper) if service is not None]
        for service in started:
            await service.start()
        try:
            yield
        finally:
            for service in reversed(started):
                await service.stop()

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
        lifespan=lifespan,
    )
    app.state.store = store
    app.add_middleware(AuthMiddleware, token=token, allowed_origins=origins)
    app.include_router(router)
    # Mounted unconditionally so the API contract (and its generated types)
    # include the job routes; handlers answer 503 when no store is configured.
    app.include_router(jobs_router)
    return app
