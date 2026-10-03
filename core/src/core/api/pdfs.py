"""Authenticated notebook PDF acquisition, history, and verified-byte routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import NoReturn
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from core.ingest.service import PdfImportError, PdfIngestService
from core.persistence.errors import PersistenceError
from core.persistence.models import PdfAcquisition
from core.persistence.store import PersistenceStore

router = APIRouter()


class AcquisitionCreate(BaseModel):
    retry_key: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=255)
    intent: str = "add"
    target_source_id: str | None = None


class AcquisitionPayload(BaseModel):
    id: str
    retry_key: str
    notebook_id: str
    intent: str
    target_source_id: str | None
    filename: str
    state: str
    generation: int
    received_size_bytes: int
    expected_size_bytes: int | None
    input_sha256: str | None
    planned_managed_file_id: str | None
    outcome_kind: str | None
    source_id: str | None
    import_id: str | None
    failure_code: str | None
    updated_at: str
    can_retry: bool
    can_cancel: bool
    requires_upload: bool


class ImportPayload(BaseModel):
    id: str
    source_id: str
    source_version_id: str
    filename: str
    state: str
    failure_code: str | None
    size_bytes: int
    sha256: str
    created_at: str
    trashed: bool
    is_current: bool
    is_pending: bool


class StoragePayload(BaseModel):
    original_file_count: int
    original_bytes: int
    bytes_pending_removal: int
    retained_staging_bytes: int
    max_upload_bytes: int
    rendered_page_pixel_limit: int


class TrashPayload(BaseModel):
    import_id: str | None = None
    source_id: str
    purge_deadline_at: str


class ProcessingActionPayload(BaseModel):
    job_id: str
    status: str


def _store(request: Request) -> PersistenceStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="PDF store is unavailable")
    return store


def _service(request: Request) -> PdfIngestService:
    service = getattr(request.app.state, "pdf_ingest", None)
    if service is None:
        raise HTTPException(status_code=503, detail="PDF import service is unavailable")
    return service


def _acquisition_payload(item: PdfAcquisition) -> AcquisitionPayload:
    terminal_failure = item.failure_code in (
        "integrity_failure",
        "integrity_conflict",
        "source_deleted",
    )
    return AcquisitionPayload(
        id=item.id,
        retry_key=item.retry_key,
        notebook_id=item.notebook_id,
        intent=item.intent,
        target_source_id=item.target_source_id,
        filename=item.display_filename,
        state=item.state,
        generation=item.finalization_generation,
        received_size_bytes=item.received_size_bytes,
        expected_size_bytes=item.input_size_bytes,
        input_sha256=item.input_sha256,
        planned_managed_file_id=item.planned_managed_file_id,
        outcome_kind=item.outcome_kind,
        source_id=item.outcome_source_id,
        import_id=item.outcome_import_id,
        failure_code=item.failure_code,
        updated_at=item.updated_at,
        can_retry=(
            item.input_complete_at is not None
            and item.state in ("staged", "finalizing", "waiting_cleanup", "failed")
            and not terminal_failure
        ),
        can_cancel=item.state not in ("completed", "cancelled"),
        requires_upload=(
            item.input_complete_at is None
            and item.state in ("awaiting_upload", "failed")
            and not terminal_failure
        ),
    )


def _raise_pdf_error(error: PdfImportError) -> NoReturn:
    raise HTTPException(
        status_code=error.status,
        detail={"code": error.code, "message": error.detail},
    ) from error


@router.post(
    "/notebooks/{notebook_id}/pdf-acquisitions",
    response_model=AcquisitionPayload,
    status_code=201,
)
def create_acquisition(
    notebook_id: str, payload: AcquisitionCreate, request: Request
) -> AcquisitionPayload:
    try:
        result = _service(request).create_acquisition(
            payload.retry_key,
            notebook_id,
            payload.intent,
            payload.filename,
            payload.target_source_id,
        )
    except PdfImportError as error:
        _raise_pdf_error(error)
    return _acquisition_payload(result)


@router.get("/notebooks/{notebook_id}/pdf-acquisitions", response_model=list[AcquisitionPayload])
def list_unfinished_acquisitions(notebook_id: str, request: Request) -> list[AcquisitionPayload]:
    store = _store(request)
    try:
        store.load_notebook(notebook_id)
        return [
            _acquisition_payload(item)
            for item in store.list_unfinished_pdf_acquisitions(notebook_id)
        ]
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=404, detail="notebook does not exist") from error


@router.get("/pdf-acquisitions/{acquisition_id}", response_model=AcquisitionPayload)
def get_acquisition(acquisition_id: str, request: Request) -> AcquisitionPayload:
    try:
        item = _store(request).load_pdf_acquisition(acquisition_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=404, detail="PDF acquisition does not exist") from error
    return _acquisition_payload(item)


@router.put(
    "/pdf-acquisitions/{acquisition_id}/content",
    response_model=AcquisitionPayload,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
async def upload_pdf(acquisition_id: str, request: Request) -> AcquisitionPayload:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    if content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_media_type",
                "message": "Send PDF bytes as application/pdf.",
            },
        )
    try:
        item = await _service(request).receive_and_finalize(acquisition_id, request)
    except PdfImportError as error:
        _raise_pdf_error(error)
    return _acquisition_payload(item)


@router.post("/pdf-acquisitions/{acquisition_id}/retry", response_model=AcquisitionPayload)
def retry_pdf(acquisition_id: str, request: Request) -> AcquisitionPayload:
    try:
        item = _service(request).retry(acquisition_id)
    except PdfImportError as error:
        _raise_pdf_error(error)
    return _acquisition_payload(item)


@router.post("/pdf-acquisitions/{acquisition_id}/cancel", response_model=AcquisitionPayload)
def cancel_pdf(acquisition_id: str, request: Request) -> AcquisitionPayload:
    try:
        item = _service(request).cancel(acquisition_id)
    except PdfImportError as error:
        _raise_pdf_error(error)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(
            status_code=409, detail="PDF acquisition cannot be cancelled"
        ) from error
    return _acquisition_payload(item)


@router.get("/sources/{source_id}/imports", response_model=list[ImportPayload])
def list_imports(source_id: str, request: Request) -> list[ImportPayload]:
    store = _store(request)
    try:
        store.load_source(source_id)
        state = store.load_source_pdf_state(source_id)
        imports = store.list_pdf_imports(source_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=404, detail="source does not exist") from error
    result: list[ImportPayload] = []
    for item, trashed in imports:
        try:
            managed = store.load_import_managed_file(item.id)
        except PersistenceError:
            continue
        result.append(
            ImportPayload(
                id=item.id,
                source_id=item.source_id,
                source_version_id=item.source_version_id,
                filename=item.display_filename,
                state=item.state,
                failure_code=item.failure_code,
                size_bytes=managed.size_bytes,
                sha256=managed.sha256,
                created_at=item.created_at,
                trashed=trashed,
                is_current=state is not None and state.current_import_id == item.id,
                is_pending=state is not None and state.pending_import_id == item.id,
            )
        )
    return result


@router.delete(
    "/sources/{source_id}/imports/{import_id}",
    response_model=TrashPayload,
)
def trash_import(source_id: str, import_id: str, request: Request) -> TrashPayload:
    store = _store(request)
    try:
        item = store.load_pdf_import(import_id)
        if item.source_id != source_id:
            raise HTTPException(status_code=404, detail="PDF import does not exist")
        deadline = (
            (datetime.now(UTC) + timedelta(days=30))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        trashed = store.trash_pdf_import(import_id, deadline)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return TrashPayload(
        import_id=trashed.import_id,
        source_id=trashed.source_id,
        purge_deadline_at=trashed.purge_deadline_at,
    )


@router.post(
    "/sources/{source_id}/imports/{import_id}/retry",
    response_model=ProcessingActionPayload,
)
def retry_import_processing(
    source_id: str, import_id: str, request: Request
) -> ProcessingActionPayload:
    store = _store(request)
    try:
        item = store.load_pdf_import(import_id)
        if item.source_id != source_id:
            raise HTTPException(status_code=404, detail="PDF import does not exist")
        job_id = store.retry_pdf_import_processing(import_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    service = getattr(request.app.state, "pdf_ingest", None)
    if service is not None and service.notify_jobs is not None:
        service.notify_jobs()
    return ProcessingActionPayload(job_id=job_id, status="queued")


@router.post(
    "/sources/{source_id}/imports/{import_id}/cancel",
    response_model=ProcessingActionPayload,
)
def cancel_import_processing(
    source_id: str, import_id: str, request: Request
) -> ProcessingActionPayload:
    store = _store(request)
    try:
        item = store.load_pdf_import(import_id)
        if item.source_id != source_id:
            raise HTTPException(status_code=404, detail="PDF import does not exist")
        job_id = store.cancel_pdf_import_processing(import_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return ProcessingActionPayload(job_id=job_id, status="cancelled")


@router.delete("/sources/{source_id}", response_model=TrashPayload)
def trash_source(source_id: str, request: Request) -> TrashPayload:
    deadline = (
        (datetime.now(UTC) + timedelta(days=30))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    try:
        trashed = _store(request).trash_source(source_id, deadline)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=404, detail="source does not exist") from error
    return TrashPayload(source_id=trashed.source_id, purge_deadline_at=trashed.purge_deadline_at)


@router.get(
    "/sources/{source_id}/imports/{import_id}/pdf",
    responses={
        200: {
            "description": "The verified saved PDF bytes.",
            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
def open_import(source_id: str, import_id: str, request: Request) -> Response:
    try:
        item = _store(request).load_pdf_import(import_id)
        if item.source_id != source_id:
            raise HTTPException(status_code=404, detail="PDF import does not exist")
        content, filename = _service(request).read_verified_import(import_id)
    except PdfImportError as error:
        _raise_pdf_error(error)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=404, detail="PDF import does not exist") from error
    safe_name = quote(filename.replace("\r", "").replace("\n", ""), safe="")
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f"inline; filename*=UTF-8''{safe_name}",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/storage/originals", response_model=StoragePayload)
def storage_report(request: Request) -> StoragePayload:
    service = _service(request)
    report = service.storage_report()
    return StoragePayload(**report)
