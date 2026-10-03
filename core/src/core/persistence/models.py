"""Typed values persisted by Corpi's local store."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

_UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str) -> str:
    if not _UUID4.fullmatch(value):
        raise ValueError("identifier must be a canonical lowercase UUIDv4")
    return value


def validate_sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError("checksum must be 64 lowercase hexadecimal characters")
    return value


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except (AttributeError, ValueError) as error:
        raise ValueError("timestamp must be RFC 3339 UTC with six fractional digits") from error
    canonical = parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if value != canonical:
        raise ValueError("timestamp must be RFC 3339 UTC with six fractional digits")
    return value


def _validate_json_value(value: JSONValue) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("value is not valid JSON data")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _validate_json_value(item)
        return
    raise ValueError("value is not valid JSON data")


def encode_json(value: JSONValue) -> str:
    try:
        _validate_json_value(value)
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not valid JSON data") from error


def decode_json(value: str) -> JSONValue:
    try:
        decoded = json.loads(value)
        _validate_json_value(decoded)
        return decoded
    except (TypeError, ValueError) as error:
        raise ValueError("stored JSON is invalid") from error


@dataclass(frozen=True, slots=True)
class Notebook:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    notebook_id: str
    display_name: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedFile:
    id: str
    relative_path: str
    size_bytes: int
    sha256: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SourceVersion:
    id: str
    source_id: str
    managed_file_id: str | None
    content_sha256: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ExtractionVersion:
    id: str
    source_id: str
    source_version_id: str
    parser_name: str
    parser_version: str
    normalization_version: str
    settings_format_version: int
    settings: JSONValue
    created_at: str


@dataclass(frozen=True, slots=True)
class Passage:
    id: str
    source_id: str
    source_version_id: str
    extraction_version_id: str
    ordinal: int
    text: str
    location_format_version: int
    location: JSONValue
    created_at: str


@dataclass(frozen=True, slots=True)
class ExtractionPage:
    id: str
    source_id: str
    source_version_id: str
    extraction_version_id: str
    page_number: int
    text_drawn: bool
    extracted_chars: int
    unmapped_ratio: float
    image_area_ratio: float
    classification: str
    created_at: str


@dataclass(frozen=True, slots=True)
class VisualLocation:
    id: str
    source_id: str
    source_version_id: str
    extraction_version_id: str
    page_number: int
    ordinal: int
    kind: str
    bbox: JSONValue
    created_at: str


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    extraction_version_id: str
    source_id: str
    source_version_id: str
    outcome: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ExtractionReview:
    extraction_version_id: str
    source_id: str
    source_version_id: str
    state: str
    generation: int
    decided_at: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    extraction: ExtractionVersion
    passages: tuple[Passage, ...]
    pages: tuple[ExtractionPage, ...]
    visuals: tuple[VisualLocation, ...]
    outcome: ExtractionOutcome | None
    review: ExtractionReview | None


@dataclass(frozen=True, slots=True)
class IndexVersion:
    id: str
    source_id: str
    source_version_id: str
    extraction_version_id: str
    index_kind: str
    chunker_version: str
    settings_format_version: int
    settings: JSONValue
    embedding_model: str | None
    embedding_revision: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class RetrievalChunk:
    id: str
    index_version_id: str
    source_id: str
    source_version_id: str
    extraction_version_id: str
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class ChunkPassage:
    retrieval_chunk_id: str
    passage_id: str
    index_version_id: str
    source_id: str
    source_version_id: str
    extraction_version_id: str
    ordinal: int
    start_offset: int | None
    end_offset: int | None


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    logical_work_key: str
    kind: str
    source_id: str
    source_version_id: str | None
    extraction_version_id: str | None
    index_version_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class JobStep:
    job_id: str
    step_key: str
    checkpoint_format_version: int
    checkpoint: JSONValue
    inputs_sha256: str | None
    output_source_version_id: str | None
    output_extraction_version_id: str | None
    output_index_version_id: str | None
    committed_at: str


@dataclass(frozen=True, slots=True)
class JobStatus:
    job_id: str
    status: str
    current_step_key: str | None
    attempt_count: int
    owner_instance_id: str | None
    claimed_at: str | None
    heartbeat_at: str | None
    failure_reason: str | None
    superseded_by_job_id: str | None
    policy: JSONValue
    updated_at: str


@dataclass(frozen=True, slots=True)
class JobStepAttempt:
    job_id: str
    step_key: str
    attempt_number: int
    started_at: str
    finished_at: str | None
    failure_reason: str | None
    outcome: str | None


@dataclass(frozen=True, slots=True)
class AttemptGuard:
    """Proof a commit or publication belongs to the current attempt.

    Every output-saving call and every publication verifies this in the same
    transaction as the write, so a cancelled job, a superseded attempt, an
    expired attempt, or another instance's attempt can never record evidence.
    """

    owner_instance_id: str
    step_key: str
    expected_attempt_number: int


@dataclass(frozen=True, slots=True)
class SourceReadiness:
    source_id: str
    source_version_id: str
    index_version_ids: tuple[str, ...]
    published_by_job_id: str | None
    published_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedFileRemoval:
    managed_file_id: str
    relative_path: str
    requested_at: str
    removed_at: str | None
    attempts: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class SourceTrash:
    source_id: str
    trashed_at: str
    purge_deadline_at: str
    files_removed_at: str | None
    snapshots_removed_at: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class PdfAcquisition:
    id: str
    retry_key: str
    notebook_id: str
    intent: str
    target_source_id: str | None
    expected_current_import_id: str | None
    display_filename: str
    state: str
    finalization_generation: int
    owner_instance_id: str
    staging_relative_path: str
    input_sha256: str | None
    input_size_bytes: int | None
    received_size_bytes: int
    input_complete_at: str | None
    last_write_at: str
    planned_reuse_managed_file_id: str | None
    planned_managed_file_id: str | None
    planned_relative_path: str | None
    planned_sha256: str | None
    planned_size_bytes: int | None
    placement_state: str
    outcome_kind: str | None
    outcome_source_id: str | None
    outcome_import_id: str | None
    failure_code: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class PdfImport:
    id: str
    source_id: str
    source_version_id: str
    acquisition_id: str | None
    display_filename: str
    state: str
    failure_code: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SourcePdfState:
    source_id: str
    current_import_id: str | None
    pending_import_id: str | None
    revision: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class PdfImportTrash:
    import_id: str
    source_id: str
    trashed_at: str
    purge_deadline_at: str
    file_removed_at: str | None
    snapshots_removed_at: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class ExtractionGraph:
    extraction: ExtractionVersion
    passages: tuple[Passage, ...]


@dataclass(frozen=True, slots=True)
class IndexGraph:
    index: IndexVersion
    chunks: tuple[RetrievalChunk, ...]
    mappings: tuple[ChunkPassage, ...]


@dataclass(frozen=True, slots=True)
class PersistenceSnapshot:
    notebooks: tuple[Notebook, ...]
    sources: tuple[Source, ...]
    managed_files: tuple[ManagedFile, ...]
    source_versions: tuple[SourceVersion, ...]
    extractions: tuple[ExtractionGraph, ...]
    indexes: tuple[IndexGraph, ...]
    jobs: tuple[Job, ...]
    job_steps: tuple[JobStep, ...]
