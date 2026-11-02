"""Managed, supported local-model setup behind the existing authenticated boundary."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.persistence.errors import ConflictingWriteError, PersistenceError
from core.search.controls import PreparationControl
from core.search.handoff import ALTERNATIVE_PROFILE_ID, DEFAULT_PROFILE_ID, SUPPORTED_PROFILE_IDS
from core.search.install import ModelSetupService
from core.search.switching import NotebookModelService

router = APIRouter()


class ModelSetupPayload(BaseModel):
    profile_id: str
    manifest_id: str
    model: str
    language_scope: str
    is_default: bool
    download_bytes: int
    state: str
    generation: int
    received_bytes: int
    failure_code: str | None
    can_cancel: bool
    can_retry: bool


class ModelSetupRequest(BaseModel):
    request_key: str = Field(min_length=1, max_length=200)
    generation: int = Field(ge=0)


class ModelGenerationRequest(BaseModel):
    generation: int = Field(ge=0)


class NotebookModelPayload(BaseModel):
    notebook_id: str
    active_profile_id: str
    pending_profile_id: str | None
    generation: int
    ready_sources: int
    rebuilt_sources: int
    failed_sources: int = Field(default=0, ge=0)
    cancelled_sources: int = Field(default=0, ge=0)


class NotebookModelChoice(ModelSetupRequest):
    profile_id: str


class SourceSearchProgress(BaseModel):
    source_id: str
    status: str
    step: str | None
    waiting_reason: str | None
    failure_code: str | None
    completed_chunks: int
    total_chunks: int
    job_id: str | None = None
    can_cancel: bool = False
    can_resume: bool = False


class PreparationAction(BaseModel):
    job_id: str


class ResumePreparation(PreparationAction):
    request_key: str = Field(min_length=1, max_length=200)


def _preparation_control(request: Request) -> PreparationControl:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(503, "search store is unavailable")
    runner = getattr(request.app.state, "runner", None)
    return PreparationControl(store, notify=runner.notify if runner else lambda: None)


@router.post("/sources/{source_id}/search/cancel", response_model=PreparationAction)
def cancel_search_preparation(
    source_id: str, payload: PreparationAction, request: Request
) -> PreparationAction:
    try:
        return PreparationAction(
            job_id=_preparation_control(request).cancel(source_id, payload.job_id)
        )
    except ConflictingWriteError as error:
        raise HTTPException(409, str(error)) from None
    except PersistenceError:
        raise HTTPException(404, "source does not exist") from None


@router.post("/sources/{source_id}/search/resume", response_model=PreparationAction)
def resume_search_preparation(
    source_id: str, payload: ResumePreparation, request: Request
) -> PreparationAction:
    try:
        return PreparationAction(
            job_id=_preparation_control(request).resume(
                source_id, payload.job_id, payload.request_key
            )
        )
    except ConflictingWriteError as error:
        raise HTTPException(409, str(error)) from None
    except PersistenceError:
        raise HTTPException(404, "source does not exist") from None


@router.get("/sources/{source_id}/search-progress", response_model=SourceSearchProgress)
def source_search_progress(source_id: str, request: Request) -> SourceSearchProgress:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(503, "search store is unavailable")
    try:
        store.load_source(source_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(404, "source does not exist") from error
    with store.database.connect() as connection:
        row = connection.execute(
            """SELECT j.id,s.status,s.current_step_key,s.failure_code,h.waiting_reason,
                      h.state AS handoff_state
               FROM jobs j JOIN job_status s ON s.job_id=j.id
               JOIN search_handoffs h ON h.job_id=j.id
               WHERE j.source_id=? AND h.purpose='import'
               ORDER BY j.created_at DESC,j.rowid DESC LIMIT 1""",
            (source_id,),
        ).fetchone()
        if row is None:
            return SourceSearchProgress(
                source_id=source_id,
                status="waiting_for_extraction",
                step=None,
                waiting_reason=None,
                failure_code=None,
                completed_chunks=0,
                total_chunks=0,
            )
        plan = connection.execute(
            "SELECT output_index_version_id FROM job_steps"
            " WHERE job_id=? AND step_key='plan-search'",
            (row["id"],),
        ).fetchone()
        total = 0
        completed = 0
        if plan is not None and plan[0] is not None:
            total = connection.execute(
                "SELECT count(*) FROM retrieval_chunks WHERE index_version_id=?", (plan[0],)
            ).fetchone()[0]
            for batch in connection.execute(
                "SELECT chunk_ids_json FROM search_batches WHERE index_version_id=? AND complete=1",
                (plan[0],),
            ):
                import json

                completed += len(json.loads(batch[0]))
        return SourceSearchProgress(
            source_id=source_id,
            status="cancelled" if row["handoff_state"] == "cancelled" else row["status"],
            step=row["current_step_key"],
            waiting_reason=row["waiting_reason"],
            failure_code=row["failure_code"],
            completed_chunks=completed,
            total_chunks=total,
            job_id=row["id"],
            can_cancel=row["handoff_state"] != "cancelled"
            and row["status"] in {"queued", "running"},
            can_resume=row["status"] in {"failed", "cancelled"}
            and PreparationControl.can_resume(connection, source_id, row["id"]),
        )


def _service(request: Request) -> ModelSetupService:
    service = getattr(request.app.state, "model_setup", None)
    if service is None:
        raise HTTPException(503, "model setup is unavailable")
    return service


def _notebook_service(request: Request) -> NotebookModelService:
    service = getattr(request.app.state, "notebook_models", None)
    if service is None:
        raise HTTPException(503, "notebook model service is unavailable")
    return service


def notebook_payload(service: NotebookModelService, notebook_id: str) -> NotebookModelPayload:
    from core.search.profiles import profiles

    row = service.state(notebook_id)
    identities = {profiles()[key].identity: key for key in SUPPORTED_PROFILE_IDS}
    return NotebookModelPayload(
        notebook_id=notebook_id,
        active_profile_id=identities[row["active_manifest_id"]],
        pending_profile_id=identities.get(row["pending_manifest_id"]),
        generation=row["selection_generation"],
        ready_sources=row["ready_sources"],
        rebuilt_sources=row["rebuilt_sources"],
        failed_sources=row["failed_sources"],
        cancelled_sources=row["cancelled_sources"],
    )


@router.get("/notebooks/{notebook_id}/model", response_model=NotebookModelPayload)
def get_notebook_model(notebook_id: str, request: Request) -> NotebookModelPayload:
    try:
        return notebook_payload(_notebook_service(request), notebook_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(404, "notebook does not exist") from error


@router.post("/notebooks/{notebook_id}/model", response_model=NotebookModelPayload)
def choose_notebook_model(
    notebook_id: str, payload: NotebookModelChoice, request: Request
) -> NotebookModelPayload:
    service = _notebook_service(request)
    try:
        service.choose(notebook_id, payload.profile_id, payload.generation, payload.request_key)
    except ConflictingWriteError as error:
        raise HTTPException(409, str(error)) from error
    except (PersistenceError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    return notebook_payload(service, notebook_id)


@router.post("/notebooks/{notebook_id}/model/cancel", response_model=NotebookModelPayload)
def cancel_notebook_model(
    notebook_id: str, payload: ModelGenerationRequest, request: Request
) -> NotebookModelPayload:
    service = _notebook_service(request)
    try:
        service.cancel(notebook_id, payload.generation)
    except ConflictingWriteError as error:
        raise HTTPException(409, str(error)) from error
    except (PersistenceError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    return notebook_payload(service, notebook_id)


def model_payload(service: ModelSetupService, profile_id: str) -> ModelSetupPayload:
    profile = service.profile(profile_id)
    row = service.state(profile_id)
    return ModelSetupPayload(
        profile_id=profile.id,
        manifest_id=profile.identity,
        model=profile.model,
        language_scope="English, French and Arabic; same-language search"
        if profile.id == DEFAULT_PROFILE_ID
        else "Multilingual; wider language/source checks pending"
        if profile.id == ALTERNATIVE_PROFILE_ID
        else "English",
        is_default=profile.id == DEFAULT_PROFILE_ID,
        download_bytes=profile.download_bytes,
        state=row["state"],
        generation=row["generation"],
        received_bytes=row["received_bytes"],
        failure_code=row["failure_code"],
        can_cancel=row["state"] == "downloading",
        can_retry=row["state"] in ("missing", "failed", "cancelled"),
    )


@router.get("/models", response_model=list[ModelSetupPayload])
def list_models(request: Request) -> list[ModelSetupPayload]:
    service = _service(request)
    return [model_payload(service, profile) for profile in SUPPORTED_PROFILE_IDS]


@router.post("/models/{profile_id}/setup", response_model=ModelSetupPayload)
def setup_model(profile_id: str, payload: ModelSetupRequest, request: Request) -> ModelSetupPayload:
    service = _service(request)
    try:
        service.begin(profile_id, payload.request_key, payload.generation)
    except ConflictingWriteError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return model_payload(service, profile_id)


@router.post("/models/{profile_id}/setup/cancel", response_model=ModelSetupPayload)
def cancel_model_setup(
    profile_id: str, payload: ModelGenerationRequest, request: Request
) -> ModelSetupPayload:
    service = _service(request)
    try:
        service.cancel(profile_id, payload.generation)
    except ConflictingWriteError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return model_payload(service, profile_id)
