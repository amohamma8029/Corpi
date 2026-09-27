"""Job status endpoints (issue #9, R7): state and per-step progress.

Read-only views over the durable records the runner maintains. Every route
sits behind the existing bearer-token, Origin, and Host middleware; write
endpoints are deliberately absent until a milestone needs them (adding one
requires extending the CORS preflight's allowed methods first).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from core.persistence.models import JobStatus
from core.persistence.store import PersistenceStore

router = APIRouter()


def _store(request: Request) -> PersistenceStore:
    store = getattr(request.app.state, "store", None)
    if store is None:  # pragma: no cover - the server always configures a store
        raise HTTPException(status_code=503, detail="job store is unavailable")
    return store


def _status_payload(status: JobStatus, kind: str | None = None) -> dict:
    payload: dict = {
        "job_id": status.job_id,
        "status": status.status,
        "current_step_key": status.current_step_key,
        "attempt_count": status.attempt_count,
        "failure_reason": status.failure_reason,
        "updated_at": status.updated_at,
    }
    if kind is not None:
        payload["kind"] = kind
    return payload


@router.get("/jobs")
async def list_jobs(request: Request) -> list[dict]:
    store = _store(request)
    return [_status_payload(status, kind) for status, kind in store.list_job_overviews()]


@router.get("/jobs/{job_id}")
async def job_detail(job_id: str, request: Request) -> dict:
    store = _store(request)
    status = store.load_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job does not exist")
    job = store.load_job_by_id(job_id)
    steps = [
        {
            "step_key": step.step_key,
            "committed_at": step.committed_at,
            "checkpoint_format_version": step.checkpoint_format_version,
            "checkpoint": step.checkpoint,
        }
        for step in store.load_job_steps(job_id)
    ]
    attempts = [
        {
            "step_key": attempt.step_key,
            "attempt_number": attempt.attempt_number,
            "started_at": attempt.started_at,
            "finished_at": attempt.finished_at,
            "failure_reason": attempt.failure_reason,
        }
        for attempt in store.load_job_attempts(job_id)
    ]
    return {
        "job": {**_status_payload(status), "kind": job.kind},
        "steps": steps,
        "attempts": attempts,
    }
