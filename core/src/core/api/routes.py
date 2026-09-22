import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.api.identity import APP_NAME, APP_VERSION

router = APIRouter()

STREAM_STEPS = 5
STREAM_TICK_SECONDS = 0.1


@router.get("/health")
async def health() -> dict[str, str]:
    return {"app": APP_NAME, "version": APP_VERSION, "status": "ok"}


@router.get("/demo/stream")
async def demo_stream() -> StreamingResponse:
    """Transport proof for authenticated streaming: SSE-formatted chunks over plain HTTP.

    Read with fetch + Authorization header, never browser EventSource (which cannot
    send the per-launch token). Later import-progress and answer-token streams reuse
    this shape.
    """
    return StreamingResponse(
        _events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


async def _events() -> AsyncIterator[str]:
    for step in range(1, STREAM_STEPS + 1):
        await asyncio.sleep(STREAM_TICK_SECONDS)
        yield f"data: {json.dumps({'step': step, 'total': STREAM_STEPS})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"
