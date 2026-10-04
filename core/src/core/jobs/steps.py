"""Registered job step sets, keyed by job kind.

M1.5 registers `pdf-extract` here. The step parses in a supervisable child process (a stuck
parser is stopped and reaped before any retry), keeps every database write in the guarded
commit, and reuses a saved extraction across jobs when the E2 identity matches. The runner
refuses jobs whose kind has no registered step set.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from importlib.metadata import version as _package_version
from typing import Any

from core.extract.classify import (
    IMAGE_AREA_LIMIT,
    MAX_EXTRACTION_PAGES,
    SHORT_PAGE_CHARS,
    UNMAPPED_RATIO_LIMIT,
    classify_page,
    decide_document_outcome,
)
from core.extract.locations import (
    LOCATION_FORMAT_VERSION,
    classify_block_kind,
    layout_bbox_to_display,
    location_json,
)
from core.extract.normalize import normalize_text
from core.extract.supervisor import (
    ExtractionCancelled,
    ExtractionChildError,
    ExtractionRun,
    ExtractionTimeout,
    run_extraction,
)
from core.extract.types import PageExtraction
from core.jobs.runner import (
    STEP_POLICIES,
    JobStepCancelled,
    PermanentStepFailure,
    StepCommitContext,
    StepContext,
    StepDefinition,
    StepOutcome,
)
from core.persistence.models import (
    ExtractionOutcome,
    ExtractionPage,
    ExtractionReview,
    ExtractionVersion,
    JSONValue,
    Passage,
    VisualLocation,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore

STEP_SETS: dict[str, tuple[StepDefinition, ...]] = {}

PDF_EXTRACT_STEP_KEY = "extract-text"
PDF_EXTRACT_POLICY_CLASS = "extract"
CHECKPOINT_FORMAT_VERSION = 1
NORMALIZATION_VERSION = "faithfully-normalized-text-v1"
SETTINGS_FORMAT_VERSION = 1
EXTRACTION_SETTINGS: JSONValue = {
    "rules_version": "corpi-rules-1",
    "thresholds": {
        "short_page_chars": SHORT_PAGE_CHARS,
        "unmapped_ratio_limit": UNMAPPED_RATIO_LIMIT,
        "image_area_limit": IMAGE_AREA_LIMIT,
        "max_pages": MAX_EXTRACTION_PAGES,
    },
}

# The child gets a slightly smaller budget than the recorded policy so a timeout
# stops and reaps the parser before the runner abandons the attempt.
STOP_MARGIN_S = 2.0
CHILD_TIMEOUT_S = STEP_POLICIES[PDF_EXTRACT_POLICY_CLASS].timeout_seconds - STOP_MARGIN_S

ExtractRunner = Callable[..., ExtractionRun]


def _parser_version() -> str:
    return _package_version("pdfminer.six")


def _page_records(
    extraction_id: str,
    source_id: str,
    source_version_id: str,
    pages: list[PageExtraction],
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    page_rows: list[dict[str, Any]] = []
    passage_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    classifications: list[str] = []
    for page in pages:
        classification = classify_page(page.metrics)
        classifications.append(classification)
        page_rows.append(
            {
                "id": new_id(),
                "source_id": source_id,
                "source_version_id": source_version_id,
                "extraction_version_id": extraction_id,
                "page_number": page.page_number,
                "text_drawn": page.metrics.text_drawn,
                "extracted_chars": page.metrics.extracted_chars,
                "unmapped_ratio": page.metrics.unmapped_ratio,
                "image_area_ratio": page.metrics.image_area_ratio,
                "classification": classification,
                "created_at": created_at,
            }
        )
        for block in page.blocks:
            text = normalize_text(block.text)
            if not text:
                continue
            bbox = layout_bbox_to_display(
                block.bbox,
                cropbox=page.frame.cropbox,
                mediabox=page.frame.mediabox,
                rotation=page.frame.rotation,
            )
            passage_rows.append(
                {
                    "id": new_id(),
                    "source_id": source_id,
                    "source_version_id": source_version_id,
                    "extraction_version_id": extraction_id,
                    "ordinal": len(passage_rows),
                    "text": text,
                    "location_format_version": LOCATION_FORMAT_VERSION,
                    "location": location_json(
                        page_number=page.page_number,
                        kind=classify_block_kind(block.text),
                        bbox=bbox,
                        reading_order=block.reading_order,
                    ),
                    "created_at": created_at,
                }
            )
        for ordinal, box in enumerate(page.image_boxes):
            bbox = layout_bbox_to_display(
                box,
                cropbox=page.frame.cropbox,
                mediabox=page.frame.mediabox,
                rotation=page.frame.rotation,
            )
            visual_rows.append(
                {
                    "id": new_id(),
                    "source_id": source_id,
                    "source_version_id": source_version_id,
                    "extraction_version_id": extraction_id,
                    "page_number": page.page_number,
                    "ordinal": ordinal,
                    "kind": "image",
                    "bbox": list(bbox),
                    "created_at": created_at,
                }
            )
    return page_rows, passage_rows, visual_rows, classifications


def _payload_from_report(report: Any) -> dict[str, Any]:
    """Serialize a saved extraction report for idempotent re-commit (cross-job reuse)."""
    extraction = report.extraction
    outcome = report.outcome
    review = report.review
    return {
        "extraction": {
            "id": extraction.id,
            "source_id": extraction.source_id,
            "source_version_id": extraction.source_version_id,
            "parser_name": extraction.parser_name,
            "parser_version": extraction.parser_version,
            "normalization_version": extraction.normalization_version,
            "settings_format_version": extraction.settings_format_version,
            "settings": extraction.settings,
            "created_at": extraction.created_at,
        },
        "passages": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "source_version_id": item.source_version_id,
                "extraction_version_id": item.extraction_version_id,
                "ordinal": item.ordinal,
                "text": item.text,
                "location_format_version": item.location_format_version,
                "location": item.location,
                "created_at": item.created_at,
            }
            for item in report.passages
        ],
        "pages": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "source_version_id": item.source_version_id,
                "extraction_version_id": item.extraction_version_id,
                "page_number": item.page_number,
                "text_drawn": item.text_drawn,
                "extracted_chars": item.extracted_chars,
                "unmapped_ratio": item.unmapped_ratio,
                "image_area_ratio": item.image_area_ratio,
                "classification": item.classification,
                "created_at": item.created_at,
            }
            for item in report.pages
        ],
        "visuals": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "source_version_id": item.source_version_id,
                "extraction_version_id": item.extraction_version_id,
                "page_number": item.page_number,
                "ordinal": item.ordinal,
                "kind": item.kind,
                "bbox": item.bbox,
                "created_at": item.created_at,
            }
            for item in report.visuals
        ],
        "outcome": None
        if outcome is None
        else {
            "extraction_version_id": outcome.extraction_version_id,
            "source_id": outcome.source_id,
            "source_version_id": outcome.source_version_id,
            "outcome": outcome.outcome,
            "created_at": outcome.created_at,
        },
        "review": None
        if review is None
        else {
            "extraction_version_id": review.extraction_version_id,
            "source_id": review.source_id,
            "source_version_id": review.source_version_id,
            "state": review.state,
            "generation": review.generation,
            "decided_at": review.decided_at,
            "updated_at": review.updated_at,
        },
    }


def _records_from_payload(
    payload: dict[str, Any],
) -> tuple[
    ExtractionVersion,
    list[Passage],
    list[ExtractionPage],
    list[VisualLocation],
    ExtractionOutcome | None,
    ExtractionReview | None,
]:
    raw_extraction = payload["extraction"]
    extraction = ExtractionVersion(
        raw_extraction["id"],
        raw_extraction["source_id"],
        raw_extraction["source_version_id"],
        raw_extraction["parser_name"],
        raw_extraction["parser_version"],
        raw_extraction["normalization_version"],
        raw_extraction["settings_format_version"],
        raw_extraction["settings"],
        raw_extraction["created_at"],
    )
    passages = [
        Passage(
            item["id"],
            item["source_id"],
            item["source_version_id"],
            item["extraction_version_id"],
            item["ordinal"],
            item["text"],
            item["location_format_version"],
            item["location"],
            item["created_at"],
        )
        for item in payload["passages"]
    ]
    pages = [
        ExtractionPage(
            item["id"],
            item["source_id"],
            item["source_version_id"],
            item["extraction_version_id"],
            item["page_number"],
            item["text_drawn"],
            item["extracted_chars"],
            item["unmapped_ratio"],
            item["image_area_ratio"],
            item["classification"],
            item["created_at"],
        )
        for item in payload["pages"]
    ]
    visuals = [
        VisualLocation(
            item["id"],
            item["source_id"],
            item["source_version_id"],
            item["extraction_version_id"],
            item["page_number"],
            item["ordinal"],
            item["kind"],
            item["bbox"],
            item["created_at"],
        )
        for item in payload["visuals"]
    ]
    raw_outcome = payload["outcome"]
    outcome = (
        None
        if raw_outcome is None
        else ExtractionOutcome(
            raw_outcome["extraction_version_id"],
            raw_outcome["source_id"],
            raw_outcome["source_version_id"],
            raw_outcome["outcome"],
            raw_outcome["created_at"],
        )
    )
    raw_review = payload["review"]
    review = (
        None
        if raw_review is None
        else ExtractionReview(
            raw_review["extraction_version_id"],
            raw_review["source_id"],
            raw_review["source_version_id"],
            raw_review["state"],
            raw_review["generation"],
            raw_review["decided_at"],
            raw_review["updated_at"],
        )
    )
    return extraction, passages, pages, visuals, outcome, review


def build_pdf_extract_steps(
    store: PersistenceStore,
    *,
    extract: ExtractRunner = run_extraction,
) -> tuple[StepDefinition, ...]:
    """Build the `pdf-extract` step set for one store (run/commit closures hold it)."""

    def inputs(context: StepContext) -> JSONValue:
        return {
            "step": "pdf-extract",
            "parser_name": "pdfminer.six",
            "parser_version": _parser_version(),
            "normalization_version": NORMALIZATION_VERSION,
            "settings_format_version": SETTINGS_FORMAT_VERSION,
            "settings": EXTRACTION_SETTINGS,
        }

    def run(context: StepContext) -> StepOutcome:
        job = store.load_job_by_id(context.job_id)
        if job.source_version_id is None:
            raise PermanentStepFailure("no_usable_text: job has no source version to extract")
        existing = store.lookup_reusable_extraction(
            job.source_id,
            job.source_version_id,
            "pdfminer.six",
            _parser_version(),
            NORMALIZATION_VERSION,
            SETTINGS_FORMAT_VERSION,
            EXTRACTION_SETTINGS,
        )
        if existing is not None:
            return StepOutcome(CHECKPOINT_FORMAT_VERSION, _payload_from_report(existing))
        source_version = store.load_source_version(job.source_version_id)
        if source_version.managed_file_id is None:
            raise PermanentStepFailure(
                "source version has no managed original to extract text from"
            )
        managed = store.load_managed_file(source_version.managed_file_id)
        pdf_path = store.database.config.data_dir / managed.relative_path

        cancel = threading.Event()

        def _watch_stop() -> None:
            while not cancel.is_set():
                if context.should_stop():
                    cancel.set()
                    return
                cancel.wait(0.1)

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()
        try:
            result = extract(pdf_path, timeout_s=CHILD_TIMEOUT_S, cancel=cancel)
        except ExtractionCancelled as error:
            if context.is_cancelled():
                raise JobStepCancelled("extraction cancelled") from error
            raise TimeoutError(f"extraction stopped at its deadline: {error}") from error
        except ExtractionTimeout as error:
            raise TimeoutError(f"extraction child stopped: {error}") from error
        except ExtractionChildError as error:
            if error.code is not None:
                raise PermanentStepFailure(str(error), code=error.code) from error
            raise
        finally:
            cancel.set()
            watcher.join(timeout=1.0)

        created_at = utc_timestamp()
        extraction_id = new_id()
        page_rows, passage_rows, visual_rows, classifications = _page_records(
            extraction_id, job.source_id, job.source_version_id, list(result.pages), created_at
        )
        decision = decide_document_outcome(classifications)
        if decision.outcome == "failed":
            raise PermanentStepFailure(
                f"{decision.failure_code}: the document has no usable text to extract",
                code=decision.failure_code,
            )
        raw_review = (
            None
            if decision.outcome != "partial"
            else {
                "extraction_version_id": extraction_id,
                "source_id": job.source_id,
                "source_version_id": job.source_version_id,
                "state": "pending",
                "generation": 1,
                "decided_at": None,
                "updated_at": created_at,
            }
        )
        payload = {
            "extraction": {
                "id": extraction_id,
                "source_id": job.source_id,
                "source_version_id": job.source_version_id,
                "parser_name": "pdfminer.six",
                "parser_version": result.parser_version,
                "normalization_version": NORMALIZATION_VERSION,
                "settings_format_version": SETTINGS_FORMAT_VERSION,
                "settings": EXTRACTION_SETTINGS,
                "created_at": created_at,
            },
            "passages": passage_rows,
            "pages": page_rows,
            "visuals": visual_rows,
            "outcome": {
                "extraction_version_id": extraction_id,
                "source_id": job.source_id,
                "source_version_id": job.source_version_id,
                "outcome": decision.outcome,
                "created_at": created_at,
            },
            "review": raw_review,
            "page_count": len(result.pages),
        }
        return StepOutcome(CHECKPOINT_FORMAT_VERSION, payload)

    def commit(store_arg: PersistenceStore, commit: StepCommitContext) -> None:
        payload = commit.outcome.checkpoint
        assert isinstance(payload, dict)
        extraction, passages, pages, visuals, outcome, review = _records_from_payload(payload)
        store_arg.commit_extraction_step(
            commit.job_id,
            commit.step_key,
            extraction,
            passages,
            commit.outcome.checkpoint_format_version,
            commit.outcome.checkpoint,
            guard=commit.guard,
            inputs_sha256=commit.inputs_sha256,
            pages=pages,
            visuals=visuals,
            outcome=outcome,
            review=review,
        )

    return (
        StepDefinition(
            step_key=PDF_EXTRACT_STEP_KEY,
            run=run,
            policy_class=PDF_EXTRACT_POLICY_CLASS,
            commit=commit,
            inputs=inputs,
        ),
    )


def register_pdf_extract_steps(store: PersistenceStore) -> dict[str, tuple[StepDefinition, ...]]:
    """Register the M1.5 step sets into the module registry the server hands the runner."""
    STEP_SETS.clear()
    STEP_SETS["pdf-extract"] = build_pdf_extract_steps(store)
    return STEP_SETS
