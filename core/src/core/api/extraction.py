"""Extraction report and partial-source review endpoints (M1.5).

Read views over the durable extraction records plus the two review write endpoints
(Confirm/Decline and Review Again). Every route sits behind the existing bearer-token,
Origin, and Host middleware, whose CORS preflight already allows POST. Handlers are
synchronous on purpose: FastAPI runs them in its worker thread pool. Failure codes
(`no_usable_text`, `page_limit_exceeded`) surface explicitly instead of a generic
processing-failed label.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.persistence.errors import ConflictingWriteError, PersistenceError
from core.persistence.models import ExtractionReport, ExtractionSummary
from core.persistence.store import PersistenceStore

router = APIRouter()


class ExtractionPagePayload(BaseModel):
    page_number: int
    classification: str
    text_drawn: bool
    extracted_chars: int
    unmapped_ratio: float
    image_area_ratio: float


class ExtractionReviewPayload(BaseModel):
    state: str
    generation: int
    decided_at: str | None


class ExtractionReportPayload(BaseModel):
    source_id: str
    extraction_id: str | None
    outcome: str | None
    failure_code: str | None
    failure_detail: str | None
    parser_name: str | None
    parser_version: str | None
    normalization_version: str | None
    passage_count: int
    pages: list[ExtractionPagePayload]
    review: ExtractionReviewPayload | None


class ReviewDecisionRequest(BaseModel):
    decision: str
    generation: int


def _store(request: Request) -> PersistenceStore:
    store = getattr(request.app.state, "store", None)
    if store is None:  # pragma: no cover - the server always configures a store
        raise HTTPException(status_code=503, detail="extraction store is unavailable")
    return store


def _report_payload(
    source_id: str,
    report: ExtractionSummary | None,
    failure_reason: str | None,
    failure_code: str | None,
) -> ExtractionReportPayload:
    if report is None or report.outcome is None:
        return ExtractionReportPayload(
            source_id=source_id,
            extraction_id=None if report is None else report.extraction.id,
            outcome=None,
            failure_code=failure_code,
            failure_detail=failure_reason,
            parser_name=None,
            parser_version=None,
            normalization_version=None,
            passage_count=0,
            pages=[],
            review=None,
        )
    return ExtractionReportPayload(
        source_id=source_id,
        extraction_id=report.extraction.id,
        outcome=report.outcome.outcome,
        failure_code=None,
        failure_detail=None,
        parser_name=report.extraction.parser_name,
        parser_version=report.extraction.parser_version,
        normalization_version=report.extraction.normalization_version,
        passage_count=report.passage_count,
        pages=[
            ExtractionPagePayload(
                page_number=page.page_number,
                classification=page.classification,
                text_drawn=page.text_drawn,
                extracted_chars=page.extracted_chars,
                unmapped_ratio=page.unmapped_ratio,
                image_area_ratio=page.image_area_ratio,
            )
            for page in report.pages
        ],
        review=None
        if report.review is None
        else ExtractionReviewPayload(
            state=report.review.state,
            generation=report.review.generation,
            decided_at=report.review.decided_at,
        ),
    )


def _review_payload(report: ExtractionReport) -> ExtractionReviewPayload:
    review = report.review
    if review is None:
        raise HTTPException(status_code=409, detail="extraction has no review state")
    return ExtractionReviewPayload(
        state=review.state, generation=review.generation, decided_at=review.decided_at
    )


def _require_source(store: PersistenceStore, source_id: str) -> None:
    try:
        store.load_source(source_id)
    except PersistenceError as error:
        raise HTTPException(status_code=404, detail="source does not exist") from error


@router.get("/sources/{source_id}/extraction", response_model=ExtractionReportPayload)
def get_extraction(source_id: str, request: Request) -> ExtractionReportPayload:
    store = _store(request)
    _require_source(store, source_id)
    report = store.load_extraction_summary_for_source(source_id)
    status = store.load_latest_extraction_job_status(source_id)
    return _report_payload(
        source_id,
        report,
        None if status is None else status.failure_reason,
        None if status is None else status.failure_code,
    )


@router.post("/extractions/{extraction_id}/review", response_model=ExtractionReviewPayload)
def decide_review(
    extraction_id: str, payload: ReviewDecisionRequest, request: Request
) -> ExtractionReviewPayload:
    """Confirm or Decline the exact extraction the user reviewed (currency rechecked)."""
    store = _store(request)
    if payload.decision not in ("confirmed", "declined"):
        raise HTTPException(status_code=400, detail="decision must be 'confirmed' or 'declined'")
    try:
        report = store.load_extraction_report(extraction_id)
    except PersistenceError as error:
        raise HTTPException(status_code=404, detail="extraction does not exist") from error
    if report.outcome is None:
        raise HTTPException(status_code=404, detail="extraction does not exist")
    try:
        store.record_review_decision(report.extraction.id, payload.decision, payload.generation)
    except ConflictingWriteError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    notify = getattr(getattr(request.app.state, "runner", None), "notify", None)
    if notify is not None:
        notify()
    return _review_payload(store.load_extraction_report(report.extraction.id))


@router.post("/extractions/{extraction_id}/review/reopen", response_model=ExtractionReviewPayload)
def reopen_review(extraction_id: str, request: Request) -> ExtractionReviewPayload:
    """Review Again: reopen the decision and supersede older open dialogs."""
    store = _store(request)
    try:
        report = store.load_extraction_report(extraction_id)
    except PersistenceError as error:
        raise HTTPException(status_code=404, detail="extraction does not exist") from error
    if report.outcome is None:
        raise HTTPException(status_code=404, detail="extraction does not exist")
    try:
        store.reopen_review(report.extraction.id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _review_payload(store.load_extraction_report(report.extraction.id))
