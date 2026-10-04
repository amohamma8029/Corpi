"""Typed, transactional save/load operations for local Corpi records."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import PurePosixPath, PureWindowsPath

from core.persistence.database import Database
from core.persistence.errors import (
    ConflictingWriteError,
    PersistenceError,
    translate_sqlite_error,
)
from core.persistence.models import (
    AttemptGuard,
    ChunkPassage,
    ExtractionGraph,
    ExtractionOutcome,
    ExtractionPage,
    ExtractionReport,
    ExtractionReview,
    ExtractionVersion,
    IndexGraph,
    IndexVersion,
    Job,
    JobStatus,
    JobStep,
    JobStepAttempt,
    JSONValue,
    ManagedFile,
    ManagedFileRemoval,
    Notebook,
    Passage,
    PdfAcquisition,
    PdfImport,
    PdfImportTrash,
    PersistenceSnapshot,
    RetrievalChunk,
    Source,
    SourcePdfState,
    SourceReadiness,
    SourceTrash,
    SourceVersion,
    VisualLocation,
    decode_json,
    encode_json,
    new_id,
    utc_timestamp,
    validate_id,
    validate_sha256,
    validate_utc_timestamp,
)


class PersistenceStore:
    """Application-owned store; callers cannot access a long-lived connection."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_notebook(self, notebook_id: str, title: str) -> Notebook:
        validate_id(notebook_id)
        title = _required_text(title, "notebook title")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _fetchone(connection, "SELECT * FROM notebooks WHERE id=?", (notebook_id,))
            if row is None:
                connection.execute(
                    "INSERT INTO notebooks(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                    (notebook_id, title, timestamp, timestamp),
                )
                row = _fetchone(connection, "SELECT * FROM notebooks WHERE id=?", (notebook_id,))
            elif row["title"] != title:
                raise ConflictingWriteError("notebook identity already has different content")
            return _notebook(_require_row(row))

    def rename_notebook(self, notebook_id: str, title: str) -> Notebook:
        validate_id(notebook_id)
        title = _required_text(title, "notebook title")
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE notebooks SET title=?, updated_at=? WHERE id=?",
                (title, utc_timestamp(), notebook_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("notebook does not exist")
            row = _fetchone(connection, "SELECT * FROM notebooks WHERE id=?", (notebook_id,))
            return _notebook(_require_row(row))

    def create_source(self, source_id: str, notebook_id: str, display_name: str) -> Source:
        validate_id(source_id)
        validate_id(notebook_id)
        display_name = _required_text(display_name, "source display name")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _fetchone(connection, "SELECT * FROM sources WHERE id=?", (source_id,))
            if row is None:
                connection.execute(
                    """INSERT INTO sources(id,notebook_id,display_name,created_at,updated_at)
                       VALUES(?,?,?,?,?)""",
                    (source_id, notebook_id, display_name, timestamp, timestamp),
                )
                row = _fetchone(connection, "SELECT * FROM sources WHERE id=?", (source_id,))
            elif row["notebook_id"] != notebook_id or row["display_name"] != display_name:
                raise ConflictingWriteError("source identity already has different content")
            return _source(_require_row(row))

    def rename_source(self, source_id: str, display_name: str) -> Source:
        validate_id(source_id)
        display_name = _required_text(display_name, "source display name")
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE sources SET display_name=?, updated_at=? WHERE id=?",
                (display_name, utc_timestamp(), source_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("source does not exist")
            row = _fetchone(connection, "SELECT * FROM sources WHERE id=?", (source_id,))
            return _source(_require_row(row))

    def list_notebooks(self) -> tuple[Notebook, ...]:
        with self.database.connect() as connection:
            return tuple(
                _notebook(row)
                for row in connection.execute("SELECT * FROM notebooks ORDER BY updated_at DESC,id")
            )

    def list_notebook_sources(
        self, notebook_id: str
    ) -> tuple[tuple[Source, SourcePdfState | None], ...]:
        validate_id(notebook_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT source.*, state.source_id AS state_source_id,
                          state.current_import_id,state.pending_import_id,
                          state.revision,state.updated_at AS state_updated_at
                   FROM sources AS source
                   LEFT JOIN source_pdf_state AS state ON state.source_id=source.id
                   WHERE source.notebook_id=?
                     AND NOT EXISTS (SELECT 1 FROM source_trash WHERE source_id=source.id)
                   ORDER BY source.updated_at DESC,source.id""",
                (notebook_id,),
            ).fetchall()
            result: list[tuple[Source, SourcePdfState | None]] = []
            for row in rows:
                state = None
                if row["state_source_id"] is not None:
                    state = SourcePdfState(
                        row["state_source_id"],
                        row["current_import_id"],
                        row["pending_import_id"],
                        int(row["revision"]),
                        row["state_updated_at"],
                    )
                result.append((_source(row), state))
            return tuple(result)

    def create_pdf_acquisition(
        self,
        retry_key: str,
        notebook_id: str,
        intent: str,
        display_filename: str,
        owner_instance_id: str,
        target_source_id: str | None = None,
    ) -> PdfAcquisition:
        retry_key = _required_text(retry_key, "retry key")
        display_filename = _required_text(display_filename, "PDF filename")
        owner_instance_id = _required_text(owner_instance_id, "owner instance id")
        validate_id(notebook_id)
        if intent not in ("add", "replace"):
            raise ValueError("PDF acquisition intent must be add or replace")
        if target_source_id is not None:
            validate_id(target_source_id)
        if (intent == "replace") != (target_source_id is not None):
            raise ValueError("replace acquisitions require exactly one target source")
        acquisition_id = new_id()
        timestamp = utc_timestamp()
        staging_path = f"files/staging/{acquisition_id}.part"
        with self._transaction() as connection:
            existing = _fetchone(
                connection, "SELECT * FROM pdf_acquisitions WHERE retry_key=?", (retry_key,)
            )
            if existing is not None:
                expected = (notebook_id, intent, target_source_id, display_filename)
                actual = tuple(
                    existing[field]
                    for field in ("notebook_id", "intent", "target_source_id", "display_filename")
                )
                if actual != expected:
                    raise ConflictingWriteError(
                        "retry key already identifies a different PDF action"
                    )
                return _pdf_acquisition(existing)
            if target_source_id is not None:
                target = _fetchone(
                    connection,
                    """SELECT source.notebook_id,state.current_import_id
                       FROM sources AS source
                       LEFT JOIN source_pdf_state AS state ON state.source_id=source.id
                       WHERE source.id=? AND NOT EXISTS (
                           SELECT 1 FROM source_trash WHERE source_id=source.id
                       )""",
                    (target_source_id,),
                )
                if target is None or target["notebook_id"] != notebook_id:
                    raise PersistenceError("replacement source does not exist in this notebook")
                expected_current = target["current_import_id"]
                if expected_current is None:
                    raise PersistenceError("source has no saved PDF to replace")
            else:
                if (
                    _fetchone(connection, "SELECT 1 FROM notebooks WHERE id=?", (notebook_id,))
                    is None
                ):
                    raise PersistenceError("notebook does not exist")
                expected_current = None
            connection.execute(
                """INSERT INTO pdf_acquisitions(
                       id,retry_key,notebook_id,intent,target_source_id,expected_current_import_id,
                       display_filename,state,finalization_generation,owner_instance_id,
                       staging_relative_path,last_write_at,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,'awaiting_upload',1,?,?,?,?,?)""",
                (
                    acquisition_id,
                    retry_key,
                    notebook_id,
                    intent,
                    target_source_id,
                    expected_current,
                    display_filename,
                    owner_instance_id,
                    staging_path,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def load_pdf_acquisition(self, acquisition_id: str) -> PdfAcquisition:
        validate_id(acquisition_id)
        with self.database.connect() as connection:
            row = _fetchone(
                connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
            )
            if row is None:
                raise PersistenceError("PDF acquisition does not exist")
            return _pdf_acquisition(row)

    def list_recoverable_pdf_acquisitions(self) -> tuple[PdfAcquisition, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM pdf_acquisitions
                   WHERE state IN ('staged','finalizing','waiting_cleanup')
                   ORDER BY created_at,id"""
            ).fetchall()
            return tuple(_pdf_acquisition(row) for row in rows)

    def list_unfinished_pdf_acquisitions(self, notebook_id: str) -> tuple[PdfAcquisition, ...]:
        """Discover saved attempts without exposing another notebook's work or replay receipts."""
        validate_id(notebook_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM pdf_acquisitions
                   WHERE notebook_id=? AND state NOT IN ('completed','cancelled')
                   ORDER BY created_at,id""",
                (notebook_id,),
            ).fetchall()
            return tuple(_pdf_acquisition(row) for row in rows)

    def begin_pdf_upload(self, acquisition_id: str, owner_instance_id: str) -> PdfAcquisition:
        validate_id(acquisition_id)
        owner_instance_id = _required_text(owner_instance_id, "owner instance id")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            if row["state"] not in ("awaiting_upload", "failed"):
                raise ConflictingWriteError("PDF acquisition cannot receive another upload")
            if row["input_complete_at"] is not None:
                raise ConflictingWriteError(
                    "complete staged input must be retried without re-upload"
                )
            connection.execute(
                """UPDATE pdf_acquisitions
                   SET state='receiving',finalization_generation=finalization_generation+1,
                       owner_instance_id=?,failure_code=NULL,last_write_at=?,updated_at=?
                   WHERE id=?""",
                (owner_instance_id, timestamp, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def record_pdf_upload(
        self,
        acquisition_id: str,
        owner_instance_id: str,
        generation: int,
        checksum: str,
        size_bytes: int,
    ) -> PdfAcquisition:
        validate_id(acquisition_id)
        validate_sha256(checksum)
        _integer(size_bytes, "PDF size")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "receiving")
            if row["input_sha256"] is not None and (
                row["input_sha256"] != checksum or row["input_size_bytes"] != size_bytes
            ):
                raise ConflictingWriteError("retry key already has different accepted PDF bytes")
            connection.execute(
                """UPDATE pdf_acquisitions SET state='staged',input_sha256=?,input_size_bytes=?,
                          input_complete_at=?,last_write_at=?,updated_at=? WHERE id=?""",
                (checksum, size_bytes, timestamp, timestamp, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def note_pdf_upload_progress(
        self, acquisition_id: str, owner_instance_id: str, generation: int, received_size_bytes: int
    ) -> None:
        validate_id(acquisition_id)
        _integer(received_size_bytes, "received PDF size")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "receiving")
            if received_size_bytes < int(row["received_size_bytes"]):
                raise ConflictingWriteError("PDF transfer progress cannot move backwards")
            connection.execute(
                """UPDATE pdf_acquisitions SET received_size_bytes=?,last_write_at=?,updated_at=?
                   WHERE id=?""",
                (received_size_bytes, timestamp, timestamp, acquisition_id),
            )

    def clear_pdf_reuse_hold(
        self, acquisition_id: str, owner_instance_id: str, generation: int
    ) -> None:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "finalizing")
            connection.execute(
                """UPDATE pdf_acquisitions
                   SET planned_reuse_managed_file_id=NULL,updated_at=? WHERE id=?""",
                (timestamp, acquisition_id),
            )

    def claim_pdf_finalization(
        self, acquisition_id: str, owner_instance_id: str, *, recovery: bool = False
    ) -> PdfAcquisition:
        validate_id(acquisition_id)
        owner_instance_id = _required_text(owner_instance_id, "owner instance id")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            if row["input_complete_at"] is None:
                raise ConflictingWriteError("PDF acquisition has no complete staged input")
            if row["state"] not in ("staged", "finalizing", "waiting_cleanup"):
                raise ConflictingWriteError("PDF acquisition is not finalizable")
            generation = int(row["finalization_generation"])
            if recovery or row["owner_instance_id"] != owner_instance_id:
                generation += 1
            connection.execute(
                """UPDATE pdf_acquisitions SET state='finalizing',finalization_generation=?,
                          owner_instance_id=?,failure_code=NULL,updated_at=? WHERE id=?""",
                (generation, owner_instance_id, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def resume_failed_pdf_acquisition(
        self, acquisition_id: str, owner_instance_id: str
    ) -> PdfAcquisition:
        validate_id(acquisition_id)
        owner_instance_id = _required_text(owner_instance_id, "owner instance id")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            if row["state"] != "failed" or row["input_complete_at"] is None:
                raise ConflictingWriteError(
                    "failed PDF acquisition has no retryable complete input"
                )
            if row["failure_code"] in ("integrity_failure", "integrity_conflict", "source_deleted"):
                raise ConflictingWriteError("this PDF failure is terminal and cannot be retried")
            connection.execute(
                """UPDATE pdf_acquisitions SET state='staged',
                          finalization_generation=finalization_generation+1,
                          owner_instance_id=?,failure_code=NULL,updated_at=? WHERE id=?""",
                (owner_instance_id, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def expire_pdf_finalization(
        self, acquisition_id: str, owner_instance_id: str, generation: int
    ) -> PdfAcquisition:
        """Invalidate a timed-out attempt; its already-running thread cannot commit later."""
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            if row["state"] == "completed":
                return _pdf_acquisition(row)
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "finalizing")
            connection.execute(
                """UPDATE pdf_acquisitions SET state='staged',failure_code='finalization_timeout',
                          finalization_generation=finalization_generation+1,
                          updated_at=? WHERE id=?""",
                (timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def set_pdf_reuse_hold(
        self, acquisition_id: str, owner_instance_id: str, generation: int, managed_file_id: str
    ) -> PdfAcquisition:
        validate_id(acquisition_id)
        validate_id(managed_file_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "finalizing")
            connection.execute(
                """UPDATE pdf_acquisitions SET planned_reuse_managed_file_id=?,
                          last_write_at=?,updated_at=? WHERE id=?""",
                (managed_file_id, timestamp, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def set_pdf_placement_plan(
        self,
        acquisition_id: str,
        owner_instance_id: str,
        generation: int,
        managed_file_id: str,
        relative_path: str,
        checksum: str,
        size_bytes: int,
    ) -> PdfAcquisition:
        validate_id(acquisition_id)
        validate_id(managed_file_id)
        validate_sha256(checksum)
        _validate_relative_path(relative_path)
        _integer(size_bytes, "planned PDF size")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "finalizing")
            planned = row["planned_managed_file_id"]
            if planned is not None and (
                planned != managed_file_id
                or row["planned_relative_path"] != relative_path
                or row["planned_sha256"] != checksum
                or row["planned_size_bytes"] != size_bytes
            ):
                raise ConflictingWriteError(
                    "PDF acquisition already has a different placement plan"
                )
            connection.execute(
                """UPDATE pdf_acquisitions SET planned_managed_file_id=?,planned_relative_path=?,
                          planned_sha256=?,planned_size_bytes=?,placement_state='planned',
                          last_write_at=?,updated_at=? WHERE id=?""",
                (
                    managed_file_id,
                    relative_path,
                    checksum,
                    size_bytes,
                    timestamp,
                    timestamp,
                    acquisition_id,
                ),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def mark_pdf_placement_complete(
        self, acquisition_id: str, owner_instance_id: str, generation: int
    ) -> PdfAcquisition:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "finalizing")
            if row["planned_managed_file_id"] is None:
                raise ConflictingWriteError("PDF placement has no durable plan")
            connection.execute(
                "UPDATE pdf_acquisitions SET placement_state='placed',updated_at=? WHERE id=?",
                (timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def find_active_notebook_pdf(
        self, notebook_id: str, checksum: str, size_bytes: int
    ) -> tuple[Source, PdfImport, ManagedFile] | None:
        validate_id(notebook_id)
        validate_sha256(checksum)
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                """SELECT source.id AS source_id,source.notebook_id,source.display_name,
                          source.created_at AS source_created_at,
                          source.updated_at AS source_updated_at,
                          pdf_import.id AS import_id,pdf_import.source_version_id,
                          pdf_import.acquisition_id,pdf_import.display_filename,
                          pdf_import.state AS import_state,
                          pdf_import.failure_code,pdf_import.created_at AS import_created_at,
                          pdf_import.updated_at AS import_updated_at,
                          managed_files.id AS managed_file_id,managed_files.relative_path,
                          managed_files.size_bytes,managed_files.sha256 AS managed_sha256,
                          managed_files.created_at AS managed_created_at
                   FROM sources AS source
                   JOIN source_pdf_state AS state ON state.source_id=source.id
                   JOIN pdf_imports AS pdf_import
                     ON pdf_import.id IN (state.current_import_id,state.pending_import_id)
                    AND pdf_import.source_id=source.id
                   JOIN source_versions AS version
                     ON version.id=pdf_import.source_version_id AND version.source_id=source.id
                   JOIN managed_files ON managed_files.id=version.managed_file_id
                   WHERE source.notebook_id=? AND version.content_sha256=?
                     AND managed_files.sha256=?
                     AND managed_files.size_bytes=?
                     AND NOT EXISTS (SELECT 1 FROM source_trash WHERE source_id=source.id)
                     AND NOT EXISTS (SELECT 1 FROM pdf_import_trash WHERE import_id=pdf_import.id)
                   ORDER BY source.created_at,source.id LIMIT 1""",
                (notebook_id, checksum, checksum, size_bytes),
            )
            if row is None:
                return None
            return (
                Source(
                    row["source_id"],
                    row["notebook_id"],
                    row["display_name"],
                    row["source_created_at"],
                    row["source_updated_at"],
                ),
                _pdf_import_from_join(row),
                _managed_file_from_join(row),
            )

    def find_registry_managed_file(self, checksum: str, size_bytes: int) -> ManagedFile | None:
        validate_sha256(checksum)
        _integer(size_bytes, "PDF size")
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                """SELECT managed_files.* FROM original_registry
                   JOIN managed_files ON managed_files.id=original_registry.managed_file_id
                   WHERE original_registry.sha256=? AND original_registry.size_bytes=?""",
                (checksum, size_bytes),
            )
            return None if row is None else _managed_file(row)

    def canonical_pdf_has_eligible_owner(self, checksum: str, size_bytes: int, now: str) -> bool:
        validate_sha256(checksum)
        validate_utc_timestamp(now)
        with self.database.connect() as connection:
            return (
                _fetchone(
                    connection,
                    """SELECT 1 FROM original_registry AS registry
                       WHERE registry.sha256=? AND registry.size_bytes=?
                         AND EXISTS (
                             SELECT 1 FROM source_versions AS version
                             JOIN sources AS source ON source.id=version.source_id
                             LEFT JOIN source_trash AS source_trash
                               ON source_trash.source_id=source.id
                             WHERE version.managed_file_id=registry.managed_file_id
                               AND (source_trash.source_id IS NULL
                                    OR source_trash.purge_deadline_at>?)
                               AND (source_trash.source_id IS NOT NULL OR NOT EXISTS (
                                   SELECT 1 FROM pdf_imports AS item
                                   JOIN pdf_import_trash AS item_trash
                                     ON item_trash.import_id=item.id
                                   WHERE item.source_version_id=version.id
                                     AND item_trash.purge_deadline_at<=?
                               ))
                         ) LIMIT 1""",
                    (checksum, size_bytes, now, now),
                )
                is not None
            )

    def managed_file_is_removing(self, managed_file_id: str) -> bool:
        validate_id(managed_file_id)
        with self.database.connect() as connection:
            return (
                _fetchone(
                    connection,
                    "SELECT 1 FROM managed_file_removals WHERE managed_file_id=?",
                    (managed_file_id,),
                )
                is not None
            )

    def has_removal_marked_pdf_fingerprint(self, checksum: str, size_bytes: int) -> bool:
        validate_sha256(checksum)
        _integer(size_bytes, "PDF size")
        with self.database.connect() as connection:
            return (
                _fetchone(
                    connection,
                    """SELECT 1 FROM managed_files AS file
                       JOIN managed_file_removals AS removal ON removal.managed_file_id=file.id
                       WHERE file.sha256=? AND file.size_bytes=? LIMIT 1""",
                    (checksum, size_bytes),
                )
                is not None
            )

    def has_live_pdf_reuse_hold(self, managed_file_id: str) -> bool:
        validate_id(managed_file_id)
        with self.database.connect() as connection:
            return (
                _fetchone(
                    connection,
                    """SELECT 1 FROM pdf_acquisitions
                   WHERE planned_reuse_managed_file_id=?
                     AND state IN ('staged','finalizing','waiting_cleanup') LIMIT 1""",
                    (managed_file_id,),
                )
                is not None
            )

    def list_legacy_pdf_candidates(
        self, checksum: str, size_bytes: int, now: str
    ) -> tuple[ManagedFile, ...]:
        validate_sha256(checksum)
        validate_utc_timestamp(now)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT managed_files.* FROM managed_files
                   JOIN source_versions ON source_versions.managed_file_id=managed_files.id
                   JOIN sources ON sources.id=source_versions.source_id
                   LEFT JOIN source_trash ON source_trash.source_id=sources.id
                   WHERE managed_files.sha256=? AND managed_files.size_bytes=?
                     AND NOT EXISTS (SELECT 1 FROM managed_file_removals
                                     WHERE managed_file_id=managed_files.id)
                      AND (source_trash.source_id IS NULL OR source_trash.purge_deadline_at>?)
                      AND (source_trash.source_id IS NULL OR source_trash.files_removed_at IS NULL)
                      AND (source_trash.source_id IS NOT NULL OR NOT EXISTS (
                          SELECT 1 FROM pdf_imports AS item
                          JOIN pdf_import_trash AS item_trash ON item_trash.import_id=item.id
                          WHERE item.source_version_id=source_versions.id
                            AND item_trash.purge_deadline_at<=?
                      ))
                   ORDER BY managed_files.created_at,managed_files.id""",
                (checksum, size_bytes, now, now),
            ).fetchall()
            return tuple(_managed_file(row) for row in rows)

    def load_pdf_import(self, import_id: str) -> PdfImport:
        validate_id(import_id)
        with self.database.connect() as connection:
            row = _fetchone(connection, "SELECT * FROM pdf_imports WHERE id=?", (import_id,))
            if row is None:
                raise PersistenceError("PDF import does not exist")
            return _pdf_import(row)

    def trash_pdf_import(self, import_id: str, purge_deadline_at: str) -> PdfImportTrash:
        validate_id(import_id)
        validate_utc_timestamp(purge_deadline_at)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            existing = _fetchone(
                connection, "SELECT * FROM pdf_import_trash WHERE import_id=?", (import_id,)
            )
            if existing is not None:
                return _pdf_import_trash(existing)
            item = _require_row(
                _fetchone(connection, "SELECT * FROM pdf_imports WHERE id=?", (import_id,))
            )
            state = _fetchone(
                connection, "SELECT * FROM source_pdf_state WHERE source_id=?", (item["source_id"],)
            )
            if state is not None and import_id in (
                state["current_import_id"],
                state["pending_import_id"],
            ):
                raise ConflictingWriteError(
                    "current or pending PDF import cannot enter history trash"
                )
            active_job = _fetchone(
                connection,
                """SELECT 1 FROM jobs JOIN job_status ON job_status.job_id=jobs.id
                   WHERE jobs.source_version_id=? AND job_status.status IN ('queued','running')
                   LIMIT 1""",
                (item["source_version_id"],),
            )
            if active_job is not None:
                raise ConflictingWriteError(
                    "PDF import is still required by an active processing job"
                )
            connection.execute(
                """INSERT INTO pdf_import_trash(
                       import_id,source_id,trashed_at,purge_deadline_at,file_removed_at,updated_at
                   ) VALUES(?,?,?,?,NULL,?)
                   ON CONFLICT(import_id) DO UPDATE SET
                       purge_deadline_at=excluded.purge_deadline_at,updated_at=excluded.updated_at""",
                (import_id, item["source_id"], timestamp, purge_deadline_at, timestamp),
            )
            return _pdf_import_trash(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_import_trash WHERE import_id=?", (import_id,)
                    )
                )
            )

    def list_expired_pdf_import_trash(self, now: str) -> tuple[PdfImportTrash, ...]:
        validate_utc_timestamp(now)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM pdf_import_trash WHERE purge_deadline_at<=?
                   ORDER BY purge_deadline_at,import_id""",
                (now,),
            ).fetchall()
            return tuple(_pdf_import_trash(row) for row in rows)

    def mark_pdf_import_file_cleanup_complete(self, import_id: str) -> None:
        validate_id(import_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE pdf_import_trash SET file_removed_at=?,updated_at=?
                   WHERE import_id=?""",
                (timestamp, timestamp, import_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("PDF import is not in trash")

    def mark_pdf_import_snapshots_removed(self, import_id: str) -> None:
        validate_id(import_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE pdf_import_trash SET snapshots_removed_at=?,updated_at=?
                   WHERE import_id=?""",
                (timestamp, timestamp, import_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("PDF import is not in trash")

    def reserve_pdf_import_file_removal(
        self, managed_file_id: str, relative_path: str, import_id: str
    ) -> ManagedFileRemoval | None:
        validate_id(managed_file_id)
        validate_id(import_id)
        relative_path = _required_text(relative_path, "relative path")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            target = _fetchone(
                connection,
                """SELECT source_id,source_version_id FROM pdf_imports WHERE id=?""",
                (import_id,),
            )
            if target is None:
                raise PersistenceError("PDF import does not exist")
            expired = _fetchone(
                connection,
                """SELECT 1 FROM pdf_import_trash
                   WHERE import_id=? AND purge_deadline_at<=?""",
                (import_id, timestamp),
            )
            if expired is None:
                raise ConflictingWriteError("PDF import is not eligible for physical removal")
            state = _fetchone(
                connection,
                "SELECT * FROM source_pdf_state WHERE source_id=?",
                (target["source_id"],),
            )
            if state is not None and import_id in (
                state["current_import_id"],
                state["pending_import_id"],
            ):
                raise ConflictingWriteError("current or pending PDF import cannot be removed")
            active = _fetchone(
                connection,
                """SELECT 1 FROM jobs JOIN job_status ON job_status.job_id=jobs.id
                   WHERE jobs.source_version_id=? AND job_status.status IN ('queued','running')
                   LIMIT 1""",
                (target["source_version_id"],),
            )
            if active is not None:
                raise ConflictingWriteError("PDF import is required by active processing")
            file_row = _fetchone(
                connection,
                """SELECT 1 FROM source_versions
                   WHERE managed_file_id=? AND NOT (
                       id=? AND source_id=?
                   ) LIMIT 1""",
                (managed_file_id, target["source_version_id"], target["source_id"]),
            )
            if file_row is not None:
                return None
            hold = _fetchone(
                connection,
                """SELECT 1 FROM pdf_acquisitions
                   WHERE planned_reuse_managed_file_id=?
                     AND state IN ('staged','finalizing','waiting_cleanup') LIMIT 1""",
                (managed_file_id,),
            )
            if hold is not None:
                return None
            connection.execute(
                """INSERT INTO managed_file_removals(
                       managed_file_id,relative_path,requested_at,removed_at,attempts,last_error
                   ) VALUES(?,?,?,NULL,0,NULL) ON CONFLICT(managed_file_id) DO NOTHING""",
                (managed_file_id, relative_path, timestamp),
            )
            return _managed_file_removal(
                _require_row(
                    _fetchone(
                        connection,
                        "SELECT * FROM managed_file_removals WHERE managed_file_id=?",
                        (managed_file_id,),
                    )
                )
            )

    def remove_trashed_pdf_import_records(self, import_id: str) -> None:
        validate_id(import_id)
        with self._transaction() as connection:
            trash = _fetchone(
                connection, "SELECT * FROM pdf_import_trash WHERE import_id=?", (import_id,)
            )
            if trash is None:
                raise PersistenceError("PDF import is not in trash")
            if trash["purge_deadline_at"] > utc_timestamp():
                raise ConflictingWriteError("PDF import trash retention has not expired")
            if trash["file_removed_at"] is None or trash["snapshots_removed_at"] is None:
                raise PersistenceError("PDF import purge phases are incomplete")
            item = _require_row(
                _fetchone(connection, "SELECT * FROM pdf_imports WHERE id=?", (import_id,))
            )
            state = _fetchone(
                connection, "SELECT * FROM source_pdf_state WHERE source_id=?", (item["source_id"],)
            )
            if state is not None and import_id in (
                state["current_import_id"],
                state["pending_import_id"],
            ):
                raise ConflictingWriteError("current or pending PDF import cannot be purged")
            active_job = _fetchone(
                connection,
                """SELECT 1 FROM jobs JOIN job_status ON job_status.job_id=jobs.id
                   WHERE jobs.source_version_id=? AND job_status.status IN ('queued','running')
                   LIMIT 1""",
                (item["source_version_id"],),
            )
            if active_job is not None:
                raise ConflictingWriteError(
                    "PDF import became required by an active processing job"
                )
            version = _fetchone(
                connection,
                "SELECT managed_file_id FROM source_versions WHERE id=?",
                (item["source_version_id"],),
            )
            managed_file_id = None if version is None else version["managed_file_id"]
            removal = None
            if managed_file_id is not None:
                removal = _fetchone(
                    connection,
                    "SELECT removed_at FROM managed_file_removals WHERE managed_file_id=?",
                    (managed_file_id,),
                )
                if removal is not None and removal["removed_at"] is None:
                    raise PersistenceError("PDF original removal is still pending")
            jobs = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM jobs WHERE source_version_id=?",
                    (item["source_version_id"],),
                ).fetchall()
            ]
            for job_id in jobs:
                connection.execute("DELETE FROM job_steps WHERE job_id=?", (job_id,))
                connection.execute("DELETE FROM job_step_attempts WHERE job_id=?", (job_id,))
                connection.execute("DELETE FROM job_status WHERE job_id=?", (job_id,))
                connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            connection.execute(
                "DELETE FROM chunk_passages WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM retrieval_chunks WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM index_versions WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM passages WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM extraction_reviews WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM extraction_outcomes WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM extraction_visuals WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM extraction_pages WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute(
                "DELETE FROM extraction_versions WHERE source_version_id=?",
                (item["source_version_id"],),
            )
            connection.execute("DELETE FROM pdf_import_trash WHERE import_id=?", (import_id,))
            connection.execute("DELETE FROM pdf_imports WHERE id=?", (import_id,))
            connection.execute(
                "DELETE FROM source_versions WHERE id=?", (item["source_version_id"],)
            )
            if (
                managed_file_id is not None
                and removal is not None
                and removal["removed_at"] is not None
            ):
                connection.execute(
                    "DELETE FROM managed_file_removals WHERE managed_file_id=?",
                    (managed_file_id,),
                )
                connection.execute(
                    """DELETE FROM managed_files WHERE id=?
                       AND NOT EXISTS (SELECT 1 FROM source_versions WHERE managed_file_id=?)""",
                    (managed_file_id, managed_file_id),
                )

    def find_notebook_historical_pdf(
        self, notebook_id: str, checksum: str, size_bytes: int, now: str
    ) -> tuple[Source, PdfImport, bool] | None:
        validate_id(notebook_id)
        validate_sha256(checksum)
        validate_utc_timestamp(now)
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                """SELECT source.id AS source_id,source.notebook_id,source.display_name,
                          source.created_at AS source_created_at,
                          source.updated_at AS source_updated_at,
                          item.id AS import_id,item.source_version_id,item.acquisition_id,
                          item.display_filename,item.state AS import_state,item.failure_code,
                          item.created_at AS import_created_at,item.updated_at AS import_updated_at,
                          CASE WHEN trash.import_id IS NULL THEN 0 ELSE 1 END AS import_trashed
                   FROM sources AS source
                   JOIN pdf_imports AS item ON item.source_id=source.id
                   JOIN source_versions AS version
                     ON version.id=item.source_version_id AND version.source_id=item.source_id
                   JOIN managed_files AS file ON file.id=version.managed_file_id
                   LEFT JOIN pdf_import_trash AS trash ON trash.import_id=item.id
                   LEFT JOIN source_pdf_state AS active ON active.source_id=source.id
                   WHERE source.notebook_id=? AND file.sha256=? AND file.size_bytes=?
                      AND item.id IS NOT active.current_import_id
                      AND item.id IS NOT active.pending_import_id
                    ORDER BY item.created_at DESC,item.id LIMIT 1""",
                (notebook_id, checksum, size_bytes),
            )
            if row is None:
                return None
            return (
                Source(
                    row["source_id"],
                    row["notebook_id"],
                    row["display_name"],
                    row["source_created_at"],
                    row["source_updated_at"],
                ),
                PdfImport(
                    row["import_id"],
                    row["source_id"],
                    row["source_version_id"],
                    row["acquisition_id"],
                    row["display_filename"],
                    row["import_state"],
                    row["failure_code"],
                    row["import_created_at"],
                    row["import_updated_at"],
                ),
                bool(row["import_trashed"]),
            )

    def load_import_managed_file(self, import_id: str) -> ManagedFile:
        validate_id(import_id)
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                """SELECT file.* FROM pdf_imports AS item
                   JOIN source_versions AS version ON version.id=item.source_version_id
                   JOIN managed_files AS file ON file.id=version.managed_file_id
                   WHERE item.id=?""",
                (import_id,),
            )
            if row is None:
                raise PersistenceError("PDF import has no managed original")
            return _managed_file(row)

    def list_pdf_imports(self, source_id: str) -> tuple[tuple[PdfImport, bool], ...]:
        validate_id(source_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT pdf_import.*,
                          CASE WHEN trash.import_id IS NULL THEN 0 ELSE 1 END AS trashed
                   FROM pdf_imports AS pdf_import
                   LEFT JOIN pdf_import_trash AS trash ON trash.import_id=pdf_import.id
                   WHERE pdf_import.source_id=?
                   ORDER BY pdf_import.created_at DESC,pdf_import.id""",
                (source_id,),
            ).fetchall()
            return tuple((_pdf_import(row), bool(row["trashed"])) for row in rows)

    def retry_pdf_import_processing(self, import_id: str) -> str:
        validate_id(import_id)
        timestamp = utc_timestamp()
        job_id = new_id()
        retry_key = f"pdf-extract-retry:{import_id}:{new_id()}"
        with self._transaction() as connection:
            item = _require_row(
                _fetchone(connection, "SELECT * FROM pdf_imports WHERE id=?", (import_id,))
            )
            if (
                _fetchone(
                    connection, "SELECT 1 FROM pdf_import_trash WHERE import_id=?", (import_id,)
                )
                is not None
            ):
                raise ConflictingWriteError("PDF import in trash cannot be processed")
            active = _fetchone(
                connection,
                """SELECT 1 FROM jobs JOIN job_status ON job_status.job_id=jobs.id
                   WHERE jobs.source_version_id=? AND job_status.status IN ('queued','running')
                   LIMIT 1""",
                (item["source_version_id"],),
            )
            if active is not None:
                raise ConflictingWriteError("PDF import already has processing work queued")
            state = _fetchone(
                connection, "SELECT * FROM source_pdf_state WHERE source_id=?", (item["source_id"],)
            )
            if state is None:
                raise PersistenceError("PDF source state is unavailable")
            if state["current_import_id"] != import_id:
                connection.execute(
                    """UPDATE source_pdf_state SET pending_import_id=?,revision=revision+1,
                              updated_at=? WHERE source_id=?""",
                    (import_id, timestamp, item["source_id"]),
                )
            connection.execute(
                "UPDATE pdf_imports SET state='pending',failure_code=NULL,updated_at=? WHERE id=?",
                (timestamp, import_id),
            )
            connection.execute(
                """INSERT INTO jobs(
                       id,logical_work_key,kind,source_id,source_version_id,
                       extraction_version_id,index_version_id,created_at,updated_at
                   ) VALUES(?,?,'pdf-extract',?,?,NULL,NULL,?,?)""",
                (
                    job_id,
                    retry_key,
                    item["source_id"],
                    item["source_version_id"],
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT INTO job_status(job_id,status,updated_at) VALUES(?,'queued',?)",
                (job_id, timestamp),
            )
            return job_id

    def cancel_pdf_import_processing(self, import_id: str) -> str:
        validate_id(import_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            item = _require_row(
                _fetchone(connection, "SELECT * FROM pdf_imports WHERE id=?", (import_id,))
            )
            if (
                _fetchone(
                    connection, "SELECT 1 FROM pdf_import_trash WHERE import_id=?", (import_id,)
                )
                is not None
            ):
                raise ConflictingWriteError("PDF import is in trash")
            job = _fetchone(
                connection,
                """SELECT jobs.id FROM jobs JOIN job_status ON job_status.job_id=jobs.id
                   WHERE jobs.source_version_id=? AND job_status.status IN ('queued','running')
                   ORDER BY jobs.created_at DESC,jobs.id DESC LIMIT 1""",
                (item["source_version_id"],),
            )
            if job is None:
                raise ConflictingWriteError("PDF import has no active processing job")
            connection.execute(
                """UPDATE job_status SET status='cancelled',
                          failure_reason='user cancelled PDF processing',updated_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (timestamp, job["id"]),
            )
            connection.execute(
                """DELETE FROM source_readiness_indexes WHERE source_id IN (
                       SELECT source_id FROM source_readiness WHERE published_by_job_id=?
                   )""",
                (job["id"],),
            )
            connection.execute(
                "DELETE FROM source_readiness WHERE published_by_job_id=?", (job["id"],)
            )
            _mark_pdf_import_failed(
                connection, item["source_version_id"], "processing_cancelled", timestamp
            )
            return str(job["id"])

    def load_source_pdf_state(self, source_id: str) -> SourcePdfState | None:
        validate_id(source_id)
        with self.database.connect() as connection:
            row = _fetchone(
                connection, "SELECT * FROM source_pdf_state WHERE source_id=?", (source_id,)
            )
            return None if row is None else _source_pdf_state(row)

    def finish_pdf_acquisition(
        self,
        acquisition_id: str,
        owner_instance_id: str,
        generation: int,
        outcome_kind: str,
        *,
        managed_file: ManagedFile | None = None,
        existing_source_id: str | None = None,
        existing_import_id: str | None = None,
        new_source_id: str | None = None,
        source_version_id: str | None = None,
        import_id: str | None = None,
        pending_replacement: bool = False,
    ) -> PdfAcquisition:
        """Commit a PDF import outcome, references, and queued job atomically."""
        validate_id(acquisition_id)
        if outcome_kind not in ("created", "duplicate", "replaced", "unchanged"):
            raise ValueError("unknown PDF acquisition outcome")
        for identifier in (
            existing_source_id,
            existing_import_id,
            new_source_id,
            source_version_id,
            import_id,
        ):
            if identifier is not None:
                validate_id(identifier)
        if managed_file is not None:
            _validate_managed_file(managed_file)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            acquisition = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(
                acquisition, owner_instance_id, generation, "finalizing"
            )
            if acquisition["input_complete_at"] is None:
                raise ConflictingWriteError("PDF acquisition input is incomplete")
            if outcome_kind in ("duplicate", "unchanged"):
                if existing_source_id is None or existing_import_id is None:
                    raise ValueError(
                        "duplicate/no-op outcomes require an existing source and import"
                    )
                source_id = existing_source_id
                result_import_id = existing_import_id
                existing = _fetchone(
                    connection,
                    """SELECT source.notebook_id,item.source_id,version.managed_file_id,
                              file.sha256,file.size_bytes,state.current_import_id,
                              state.pending_import_id
                       FROM sources AS source
                       JOIN pdf_imports AS item
                         ON item.id=? AND item.source_id=source.id
                       JOIN source_versions AS version
                         ON version.id=item.source_version_id AND version.source_id=item.source_id
                       JOIN managed_files AS file ON file.id=version.managed_file_id
                       JOIN source_pdf_state AS state ON state.source_id=source.id
                       WHERE source.id=?
                         AND NOT EXISTS (SELECT 1 FROM source_trash WHERE source_id=source.id)
                         AND NOT EXISTS (SELECT 1 FROM pdf_import_trash WHERE import_id=item.id)
                         AND NOT EXISTS (SELECT 1 FROM managed_file_removals
                                         WHERE managed_file_id=file.id)""",
                    (existing_import_id, existing_source_id),
                )
                if (
                    existing is None
                    or existing["notebook_id"] != acquisition["notebook_id"]
                    or existing_import_id
                    not in (existing["current_import_id"], existing["pending_import_id"])
                    or existing["sha256"] != acquisition["input_sha256"]
                    or existing["size_bytes"] != acquisition["input_size_bytes"]
                ):
                    raise ConflictingWriteError("duplicate PDF source changed before finalization")
                if (
                    outcome_kind == "unchanged"
                    and existing["current_import_id"] != existing_import_id
                ):
                    raise ConflictingWriteError("unchanged PDF is no longer the current import")
            else:
                if source_version_id is None or import_id is None:
                    raise ValueError(
                        "new PDF outcomes require source-version and import identities"
                    )
                source_id = acquisition["target_source_id"] or new_source_id
                if source_id is None:
                    raise ValueError("Add outcome requires a new source identity")
                validate_id(source_id)
                if acquisition["target_source_id"] is None:
                    if (
                        _fetchone(
                            connection,
                            "SELECT 1 FROM notebooks WHERE id=?",
                            (acquisition["notebook_id"],),
                        )
                        is None
                    ):
                        raise ConflictingWriteError("PDF acquisition notebook was removed")
                    connection.execute(
                        """INSERT INTO sources(id,notebook_id,display_name,created_at,updated_at)
                           VALUES(?,?,?,?,?)""",
                        (
                            source_id,
                            acquisition["notebook_id"],
                            acquisition["display_filename"],
                            timestamp,
                            timestamp,
                        ),
                    )
                else:
                    target_state = _fetchone(
                        connection,
                        """SELECT state.*,source.notebook_id FROM source_pdf_state AS state
                           JOIN sources AS source ON source.id=state.source_id
                           WHERE state.source_id=?
                             AND NOT EXISTS (
                                 SELECT 1 FROM source_trash WHERE source_id=source.id
                             )""",
                        (source_id,),
                    )
                    current_import = (
                        None if target_state is None else target_state["current_import_id"]
                    )
                    if (
                        target_state is None
                        or target_state["notebook_id"] != acquisition["notebook_id"]
                    ):
                        raise ConflictingWriteError("replacement source was deleted or moved")
                    if current_import != acquisition["expected_current_import_id"]:
                        raise ConflictingWriteError(
                            "replacement target changed during PDF acquisition"
                        )
                    superseded_import_id = target_state["pending_import_id"]
                    if superseded_import_id is not None:
                        superseded = _fetchone(
                            connection,
                            "SELECT source_version_id FROM pdf_imports WHERE id=?",
                            (superseded_import_id,),
                        )
                        if superseded is not None:
                            connection.execute(
                                """UPDATE job_status SET status='cancelled',
                                          failure_reason='superseded by newer PDF replacement',
                                          updated_at=?
                                   WHERE status IN ('queued','running') AND job_id IN (
                                       SELECT id FROM jobs WHERE source_version_id=?
                                   )""",
                                (timestamp, superseded["source_version_id"]),
                            )
                            connection.execute(
                                """UPDATE pdf_imports SET state='failed',
                                          failure_code='replacement_superseded',updated_at=?
                                   WHERE id=? AND state='pending'""",
                                (timestamp, superseded_import_id),
                            )
                    stale_acquisitions = connection.execute(
                        """SELECT * FROM pdf_acquisitions WHERE target_source_id=? AND id<>?
                           AND state IN ('awaiting_upload','receiving','staged','finalizing',
                                         'waiting_cleanup')""",
                        (source_id, acquisition_id),
                    ).fetchall()
                    for stale in stale_acquisitions:
                        connection.execute(
                            """INSERT INTO acquisition_file_cleanup(
                                   id,acquisition_id,relative_path,file_kind,expected_sha256,
                                   expected_size_bytes,created_at
                               ) VALUES(?,?,?,'staging',?,?,?)
                               ON CONFLICT(acquisition_id,relative_path) DO NOTHING""",
                            (
                                new_id(),
                                stale["id"],
                                stale["staging_relative_path"],
                                stale["input_sha256"],
                                stale["input_size_bytes"],
                                timestamp,
                            ),
                        )
                        if stale["planned_relative_path"] is not None:
                            connection.execute(
                                """INSERT INTO acquisition_file_cleanup(
                                       id,acquisition_id,relative_path,file_kind,expected_sha256,
                                       expected_size_bytes,created_at
                                   ) VALUES(?,?,?,'planned_original',?,?,?)
                                   ON CONFLICT(acquisition_id,relative_path) DO NOTHING""",
                                (
                                    new_id(),
                                    stale["id"],
                                    stale["planned_relative_path"],
                                    stale["planned_sha256"],
                                    stale["planned_size_bytes"],
                                    timestamp,
                                ),
                            )
                        connection.execute(
                            """UPDATE pdf_acquisitions SET state='cancelled',
                                      failure_code='replacement_superseded',
                                      finalization_generation=finalization_generation+1,
                                      planned_reuse_managed_file_id=NULL,updated_at=? WHERE id=?""",
                            (timestamp, stale["id"]),
                        )
                file_id = (
                    managed_file.id
                    if managed_file is not None
                    else (
                        acquisition["planned_reuse_managed_file_id"]
                        or acquisition["planned_managed_file_id"]
                    )
                )
                if file_id is None:
                    raise ConflictingWriteError("PDF acquisition has no verified managed file")
                if acquisition["input_sha256"] != acquisition["planned_sha256"] and (
                    acquisition["planned_managed_file_id"] is not None
                ):
                    raise ConflictingWriteError("planned PDF fingerprint changed")
                if managed_file is not None:
                    if managed_file.id != file_id:
                        raise ConflictingWriteError(
                            "managed file does not match the acquisition plan"
                        )
                    _save_managed_file(connection, managed_file)
                elif acquisition["planned_reuse_managed_file_id"] != file_id:
                    raise ConflictingWriteError("existing managed file has no recorded reuse hold")
                checksum = acquisition["input_sha256"]
                size_bytes = acquisition["input_size_bytes"]
                assert checksum is not None and size_bytes is not None
                file_row = _fetchone(
                    connection, "SELECT * FROM managed_files WHERE id=?", (file_id,)
                )
                if (
                    file_row is None
                    or file_row["sha256"] != checksum
                    or file_row["size_bytes"] != size_bytes
                ):
                    raise ConflictingWriteError("verified managed file identity changed")
                registry = _fetchone(
                    connection,
                    "SELECT managed_file_id FROM original_registry WHERE sha256=? AND size_bytes=?",
                    (checksum, size_bytes),
                )
                if registry is not None and registry["managed_file_id"] != file_id:
                    raise ConflictingWriteError("another managed original is already canonical")
                if registry is None:
                    connection.execute(
                        """INSERT INTO original_registry(
                               sha256,size_bytes,managed_file_id,created_at
                           )
                           VALUES(?,?,?,?)""",
                        (checksum, size_bytes, file_id, timestamp),
                    )
                source_version = SourceVersion(
                    source_version_id, source_id, file_id, checksum, timestamp
                )
                _validate_source_version(source_version)
                _save_source_version(connection, source_version)
                connection.execute(
                    """INSERT INTO pdf_imports(
                           id,source_id,source_version_id,acquisition_id,display_filename,state,
                           created_at,updated_at
                       ) VALUES(?,?,?,?,?,'pending',?,?)""",
                    (
                        import_id,
                        source_id,
                        source_version_id,
                        acquisition_id,
                        acquisition["display_filename"],
                        timestamp,
                        timestamp,
                    ),
                )
                current_state = _fetchone(
                    connection, "SELECT * FROM source_pdf_state WHERE source_id=?", (source_id,)
                )
                if acquisition["target_source_id"] is None:
                    connection.execute(
                        """INSERT INTO source_pdf_state(
                               source_id,current_import_id,pending_import_id,revision,updated_at
                           ) VALUES(?,?,NULL,1,?)""",
                        (source_id, import_id, timestamp),
                    )
                elif pending_replacement:
                    if current_state is None:
                        connection.execute(
                            """INSERT INTO source_pdf_state(
                                   source_id,current_import_id,pending_import_id,revision,updated_at
                               ) VALUES(?,NULL,?,1,?)""",
                            (source_id, import_id, timestamp),
                        )
                    else:
                        connection.execute(
                            """UPDATE source_pdf_state SET pending_import_id=?,revision=revision+1,
                                      updated_at=? WHERE source_id=?""",
                            (import_id, timestamp, source_id),
                        )
                else:
                    raise ConflictingWriteError("replacement must remain pending until publication")
                self._insert_queued_pdf_job(connection, source_id, source_version_id, timestamp)
                result_import_id = import_id
            connection.execute(
                """UPDATE pdf_acquisitions SET state='completed',outcome_kind=?,outcome_source_id=?,
                          outcome_import_id=?,planned_reuse_managed_file_id=NULL,failure_code=NULL,
                          updated_at=? WHERE id=?""",
                (outcome_kind, source_id, result_import_id, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    @staticmethod
    def _insert_queued_pdf_job(
        connection: sqlite3.Connection, source_id: str, source_version_id: str, timestamp: str
    ) -> None:
        logical_key = f"pdf-extract:{source_version_id}"
        proposed_id = new_id()
        connection.execute(
            """INSERT INTO jobs(
                   id,logical_work_key,kind,source_id,source_version_id,
                   extraction_version_id,index_version_id,created_at,updated_at
               ) VALUES(?,?,'pdf-extract',?,?,NULL,NULL,?,?)
               ON CONFLICT(logical_work_key) DO NOTHING""",
            (proposed_id, logical_key, source_id, source_version_id, timestamp, timestamp),
        )
        job = _fetchone(connection, "SELECT * FROM jobs WHERE logical_work_key=?", (logical_key,))
        if (
            job is None
            or job["source_id"] != source_id
            or job["source_version_id"] != source_version_id
        ):
            raise ConflictingWriteError("PDF processing key already identifies different work")
        connection.execute(
            """INSERT INTO job_status(job_id,status,updated_at) VALUES(?,'queued',?)
               ON CONFLICT(job_id) DO NOTHING""",
            (job["id"], timestamp),
        )

    def mark_pdf_acquisition_waiting_cleanup(
        self, acquisition_id: str, owner_instance_id: str, generation: int
    ) -> PdfAcquisition:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            _verify_pdf_acquisition_attempt(row, owner_instance_id, generation, "finalizing")
            connection.execute(
                """UPDATE pdf_acquisitions SET state='waiting_cleanup',
                          planned_reuse_managed_file_id=NULL,failure_code='file_cleanup_pending',
                          updated_at=? WHERE id=?""",
                (timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def fail_pdf_acquisition(
        self,
        acquisition_id: str,
        owner_instance_id: str,
        generation: int,
        failure_code: str,
        *,
        cancelled: bool = False,
    ) -> PdfAcquisition:
        validate_id(acquisition_id)
        failure_code = _required_text(failure_code, "failure code")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _require_row(
                _fetchone(
                    connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                )
            )
            if (
                row["owner_instance_id"] != owner_instance_id
                or int(row["finalization_generation"]) != generation
            ):
                raise ConflictingWriteError("PDF acquisition attempt was superseded")
            if row["state"] in ("completed", "cancelled"):
                raise ConflictingWriteError("PDF acquisition is already terminal")
            state = "cancelled" if cancelled else "failed"
            connection.execute(
                """UPDATE pdf_acquisitions SET state=?,failure_code=?,
                          finalization_generation=finalization_generation+1,
                          planned_reuse_managed_file_id=NULL,updated_at=? WHERE id=?""",
                (state, failure_code, timestamp, acquisition_id),
            )
            return _pdf_acquisition(
                _require_row(
                    _fetchone(
                        connection, "SELECT * FROM pdf_acquisitions WHERE id=?", (acquisition_id,)
                    )
                )
            )

    def record_acquisition_file_cleanup(
        self,
        acquisition_id: str,
        relative_path: str,
        file_kind: str,
        checksum: str | None = None,
        size_bytes: int | None = None,
    ) -> str:
        validate_id(acquisition_id)
        _validate_relative_path(relative_path)
        if file_kind not in ("staging", "planned_original"):
            raise ValueError("unknown acquisition cleanup file kind")
        if checksum is not None:
            validate_sha256(checksum)
        if size_bytes is not None:
            _integer(size_bytes, "cleanup file size")
        cleanup_id = new_id()
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            existing = _fetchone(
                connection,
                """SELECT id FROM acquisition_file_cleanup
                   WHERE acquisition_id=? AND relative_path=?""",
                (acquisition_id, relative_path),
            )
            if existing is not None:
                return str(existing["id"])
            connection.execute(
                """INSERT INTO acquisition_file_cleanup(
                       id,acquisition_id,relative_path,file_kind,expected_sha256,
                       expected_size_bytes,created_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    cleanup_id,
                    acquisition_id,
                    relative_path,
                    file_kind,
                    checksum,
                    size_bytes,
                    timestamp,
                ),
            )
            return cleanup_id

    def list_pending_acquisition_file_cleanup(
        self,
    ) -> tuple[tuple[str, str, str, str, str | None, int | None], ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT id,acquisition_id,relative_path,file_kind,
                          expected_sha256,expected_size_bytes
                   FROM acquisition_file_cleanup
                   WHERE removed_at IS NULL ORDER BY created_at,id"""
            ).fetchall()
            return tuple(
                (
                    row["id"],
                    row["acquisition_id"],
                    row["relative_path"],
                    row["file_kind"],
                    row["expected_sha256"],
                    row["expected_size_bytes"],
                )
                for row in rows
            )

    def mark_acquisition_file_removed(self, cleanup_id: str) -> None:
        validate_id(cleanup_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE acquisition_file_cleanup SET removed_at=?,last_error=NULL WHERE id=?",
                (timestamp, cleanup_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("acquisition cleanup record does not exist")

    def record_acquisition_file_cleanup_failure(self, cleanup_id: str, error: str) -> None:
        validate_id(cleanup_id)
        error = _required_text(error, "cleanup error")
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE acquisition_file_cleanup SET attempts=attempts+1,last_error=?
                   WHERE id=? AND removed_at IS NULL""",
                (error, cleanup_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("acquisition cleanup record is not pending")

    def save_managed_file(self, managed_file: ManagedFile) -> ManagedFile:
        _validate_managed_file(managed_file)
        with self._transaction() as connection:
            _save_managed_file(connection, managed_file)
            return _managed_file(
                _require_row(
                    _fetchone(
                        connection,
                        "SELECT * FROM managed_files WHERE id=?",
                        (managed_file.id,),
                    )
                )
            )

    def create_source_version(
        self,
        source_id: str,
        content_sha256: str,
        managed_file_id: str | None = None,
        version_id: str | None = None,
    ) -> SourceVersion:
        source_version = SourceVersion(
            id=version_id or new_id(),
            source_id=source_id,
            managed_file_id=managed_file_id,
            content_sha256=content_sha256,
            created_at=utc_timestamp(),
        )
        _validate_source_version(source_version)
        with self._transaction() as connection:
            _save_source_version(connection, source_version)
            return _source_version(
                _require_row(
                    _fetchone(
                        connection,
                        "SELECT * FROM source_versions WHERE id=?",
                        (source_version.id,),
                    )
                )
            )

    def save_extraction(
        self,
        extraction: ExtractionVersion,
        passages: Sequence[Passage],
    ) -> ExtractionGraph:
        with self._transaction() as connection:
            _save_extraction(connection, extraction, passages)
            return _load_extraction(connection, extraction.id)

    def save_extraction_result(
        self,
        extraction: ExtractionVersion,
        passages: Sequence[Passage],
        pages: Sequence[ExtractionPage],
        visuals: Sequence[VisualLocation],
        outcome: ExtractionOutcome,
        review: ExtractionReview | None = None,
    ) -> ExtractionReport:
        """Commit one extraction result as a single guarded transaction.

        The extraction version, its passages, per-page quality records, visual metadata, the
        recorded outcome, and (exactly for partial outcomes) its review state commit together
        with save_extraction's existing nonempty-passage and retry-conflict discipline. A retry
        of the same extraction must carry the same records.
        """
        with self._transaction() as connection:
            _save_extraction_result(
                connection, extraction, passages, pages, visuals, outcome, review
            )
            return _load_extraction_report(connection, extraction.id)

    def load_extraction_report(self, extraction_version_id: str) -> ExtractionReport:
        validate_id(extraction_version_id)
        with self.database.connect() as connection:
            return _load_extraction_report(connection, extraction_version_id)

    def lookup_reusable_extraction(
        self,
        source_id: str,
        source_version_id: str,
        parser_name: str,
        parser_version: str,
        normalization_version: str,
        settings_format_version: int,
        settings: JSONValue,
    ) -> ExtractionReport | None:
        """Find a saved extraction a new job can reuse (cross-job reuse identity).

        Identity covers source version, verified content (the version pins it), parser and
        normalization versions, and the effective rule/settings payload. The lookup is scoped
        to one source, so notebooks sharing PDF bytes never combine evidence ownership.
        """
        validate_id(source_id)
        validate_id(source_version_id)
        _required_text(parser_name, "parser name")
        _required_text(parser_version, "parser version")
        _required_text(normalization_version, "normalization version")
        _integer(settings_format_version, "settings format version", minimum=1)
        settings_json = encode_json(settings)
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT id FROM extraction_versions
                   WHERE source_id=? AND source_version_id=?
                     AND parser_name=? AND parser_version=?
                     AND normalization_version=? AND settings_format_version=?
                     AND settings_json=?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (
                    source_id,
                    source_version_id,
                    parser_name,
                    parser_version,
                    normalization_version,
                    settings_format_version,
                    settings_json,
                ),
            ).fetchone()
            if row is None:
                return None
            return _load_extraction_report(connection, row["id"])

    def record_review_decision(
        self,
        extraction_version_id: str,
        decision: str,
        generation: int,
    ) -> ExtractionReview:
        """Record Confirm/Decline idempotently against the reviewed generation.

        Confirm rechecks currency first — a superseded source version, a trashed source, or a
        trashed candidate import cannot be confirmed (the readiness-publication recheck shape).
        Repeating the same decision at the same generation is a no-op returning the recorded
        state; a stale generation or a conflicting decision is rejected.
        """
        validate_id(extraction_version_id)
        if decision not in ("confirmed", "declined"):
            raise ValueError("review decision must be 'confirmed' or 'declined'")
        _integer(generation, "review generation", minimum=1)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _fetchone(
                connection,
                "SELECT * FROM extraction_reviews WHERE extraction_version_id=?",
                (extraction_version_id,),
            )
            if row is None:
                raise PersistenceError("extraction has no review state")
            review = _extraction_review(row)
            if generation < review.generation:
                raise ConflictingWriteError(
                    "review decision is stale; a newer review generation exists"
                )
            if generation > review.generation:
                raise ValueError("review generation does not exist")
            if review.state == decision:
                return review
            if review.state != "pending":
                raise ConflictingWriteError("review was already decided differently")
            if decision == "confirmed":
                _verify_review_currency(connection, review)
            connection.execute(
                """UPDATE extraction_reviews
                   SET state=?, decided_at=?, updated_at=?
                   WHERE extraction_version_id=?""",
                (decision, timestamp, timestamp, extraction_version_id),
            )
            updated = _fetchone(
                connection,
                "SELECT * FROM extraction_reviews WHERE extraction_version_id=?",
                (extraction_version_id,),
            )
            assert updated is not None
            return _extraction_review(updated)

    def reopen_review(self, extraction_version_id: str) -> ExtractionReview:
        """Review Again: reopen the decision and supersede older open dialogs.

        Bumping the generation invalidates any open dialog from an earlier generation, so only
        the newest one can confirm.
        """
        validate_id(extraction_version_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            row = _fetchone(
                connection,
                "SELECT * FROM extraction_reviews WHERE extraction_version_id=?",
                (extraction_version_id,),
            )
            if row is None:
                raise PersistenceError("extraction has no review state")
            connection.execute(
                """UPDATE extraction_reviews
                   SET state='pending', generation=generation+1, decided_at=NULL, updated_at=?
                   WHERE extraction_version_id=?""",
                (timestamp, extraction_version_id),
            )
            updated = _fetchone(
                connection,
                "SELECT * FROM extraction_reviews WHERE extraction_version_id=?",
                (extraction_version_id,),
            )
            assert updated is not None
            return _extraction_review(updated)

    def save_index(
        self,
        index: IndexVersion,
        chunks: Sequence[RetrievalChunk],
        mappings: Sequence[ChunkPassage],
    ) -> IndexGraph:
        with self._transaction() as connection:
            _save_index(connection, index, chunks, mappings)
            return _load_index(connection, index.id)

    def create_job(
        self,
        logical_work_key: str,
        kind: str,
        source_id: str,
        source_version_id: str | None = None,
        extraction_version_id: str | None = None,
        index_version_id: str | None = None,
    ) -> Job:
        logical_work_key = _required_text(logical_work_key, "logical work key")
        kind = _required_text(kind, "job kind")
        _validate_job_targets(source_id, source_version_id, extraction_version_id, index_version_id)
        timestamp = utc_timestamp()
        proposed_id = new_id()
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO jobs(
                       id,logical_work_key,kind,source_id,source_version_id,
                       extraction_version_id,index_version_id,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(logical_work_key) DO NOTHING""",
                (
                    proposed_id,
                    logical_work_key,
                    kind,
                    source_id,
                    source_version_id,
                    extraction_version_id,
                    index_version_id,
                    timestamp,
                    timestamp,
                ),
            )
            row = _fetchone(
                connection,
                "SELECT * FROM jobs WHERE logical_work_key=?",
                (logical_work_key,),
            )
            if row is None:
                raise PersistenceError("job could not be saved")
            expected = (kind, source_id, source_version_id, extraction_version_id, index_version_id)
            actual = tuple(
                row[field]
                for field in (
                    "kind",
                    "source_id",
                    "source_version_id",
                    "extraction_version_id",
                    "index_version_id",
                )
            )
            if actual != expected:
                raise ConflictingWriteError("logical work key already identifies different work")
            return _job(row)

    def commit_source_version_step(
        self,
        job_id: str,
        step_key: str,
        source_version: SourceVersion,
        checkpoint_format_version: int,
        checkpoint: JSONValue,
        *,
        guard: AttemptGuard,
        managed_file: ManagedFile | None = None,
        inputs_sha256: str | None = None,
    ) -> JobStep:
        _validate_source_version(source_version)
        if managed_file is not None:
            _validate_managed_file(managed_file)
            if source_version.managed_file_id != managed_file.id:
                raise ValueError("source version and managed-file identities do not match")
        with self._transaction() as connection:
            _verify_attempt_current(connection, job_id, step_key, guard)
            if managed_file is not None:
                _save_managed_file(connection, managed_file)
            _save_source_version(connection, source_version)
            _save_job_step(
                connection,
                job_id,
                step_key,
                checkpoint_format_version,
                checkpoint,
                inputs_sha256,
                output_source_version_id=source_version.id,
            )
            return _load_job_step(connection, job_id, step_key)

    def commit_extraction_step(
        self,
        job_id: str,
        step_key: str,
        extraction: ExtractionVersion,
        passages: Sequence[Passage],
        checkpoint_format_version: int,
        checkpoint: JSONValue,
        *,
        guard: AttemptGuard,
        inputs_sha256: str | None = None,
        pages: Sequence[ExtractionPage] = (),
        visuals: Sequence[VisualLocation] = (),
        outcome: ExtractionOutcome | None = None,
        review: ExtractionReview | None = None,
    ) -> JobStep:
        """Commit the extraction step output in one guarded transaction.

        With an outcome record the full M1.5 result — passages, per-page quality records,
        visual metadata, and the review state — commits together with the step completion.
        """
        with self._transaction() as connection:
            _verify_attempt_current(connection, job_id, step_key, guard)
            if outcome is not None:
                _save_extraction_result(
                    connection, extraction, passages, pages, visuals, outcome, review
                )
            else:
                _save_extraction(connection, extraction, passages)
            _save_job_step(
                connection,
                job_id,
                step_key,
                checkpoint_format_version,
                checkpoint,
                inputs_sha256,
                output_extraction_version_id=extraction.id,
            )
            return _load_job_step(connection, job_id, step_key)

    def commit_index_step(
        self,
        job_id: str,
        step_key: str,
        index: IndexVersion,
        chunks: Sequence[RetrievalChunk],
        mappings: Sequence[ChunkPassage],
        checkpoint_format_version: int,
        checkpoint: JSONValue,
        *,
        guard: AttemptGuard,
        inputs_sha256: str | None = None,
    ) -> JobStep:
        with self._transaction() as connection:
            _verify_attempt_current(connection, job_id, step_key, guard)
            _save_index(connection, index, chunks, mappings)
            _save_job_step(
                connection,
                job_id,
                step_key,
                checkpoint_format_version,
                checkpoint,
                inputs_sha256,
                output_index_version_id=index.id,
            )
            return _load_job_step(connection, job_id, step_key)

    # --- Runner and lifecycle records (subproblem 1.3) ---

    def commit_step(
        self,
        job_id: str,
        step_key: str,
        checkpoint_format_version: int,
        checkpoint: JSONValue,
        *,
        guard: AttemptGuard,
        inputs_sha256: str | None = None,
    ) -> JobStep:
        """Commit a step completion that has no database version output.

        The transaction verifies the attempt guard first, so a cancellation, a
        superseded attempt, or another instance can never record evidence.
        """
        step_key = _required_text(step_key, "step key")
        if checkpoint_format_version <= 0:
            raise ValueError("checkpoint format version must be positive")
        with self._transaction() as connection:
            _verify_attempt_current(connection, job_id, step_key, guard)
            _save_job_step(
                connection,
                job_id,
                step_key,
                checkpoint_format_version,
                checkpoint,
                inputs_sha256,
            )
            return _load_job_step(connection, job_id, step_key)

    def load_source_version(self, source_version_id: str) -> SourceVersion:
        validate_id(source_version_id)
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                "SELECT * FROM source_versions WHERE id=?",
                (source_version_id,),
            )
            if row is None:
                raise PersistenceError("source version does not exist")
            return _source_version(row)

    def load_managed_file(self, managed_file_id: str) -> ManagedFile:
        validate_id(managed_file_id)
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                "SELECT * FROM managed_files WHERE id=?",
                (managed_file_id,),
            )
            if row is None:
                raise PersistenceError("managed file does not exist")
            return _managed_file(row)

    def load_extraction_report_for_source(self, source_id: str) -> ExtractionReport | None:
        """The newest extraction on the source's current version, if one is committed."""
        validate_id(source_id)
        with self.database.connect() as connection:
            current_row = connection.execute(
                """SELECT id FROM source_versions WHERE source_id=?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (source_id,),
            ).fetchone()
            if current_row is None:
                return None
            row = connection.execute(
                """SELECT id FROM extraction_versions
                   WHERE source_id=? AND source_version_id=?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (source_id, current_row[0]),
            ).fetchone()
            if row is None:
                return None
            return _load_extraction_report(connection, row["id"])

    def load_latest_extraction_job_status(self, source_id: str) -> JobStatus | None:
        """The newest `pdf-extract` job status for a source (carries failure codes)."""
        validate_id(source_id)
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT job_status.* FROM job_status
                   JOIN jobs ON jobs.id = job_status.job_id
                   WHERE jobs.source_id=? AND jobs.kind='pdf-extract'
                   ORDER BY jobs.created_at DESC, jobs.rowid DESC LIMIT 1""",
                (source_id,),
            ).fetchone()
            return None if row is None else _job_status(row)

    def load_job_by_id(self, job_id: str) -> Job:
        validate_id(job_id)
        with self.database.connect() as connection:
            row = _fetchone(connection, "SELECT * FROM jobs WHERE id=?", (job_id,))
            if row is None:
                raise PersistenceError("job does not exist")
            return _job(_require_row(row))

    def load_job_steps(self, job_id: str) -> tuple[JobStep, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_steps WHERE job_id=? ORDER BY committed_at, step_key",
                (job_id,),
            ).fetchall()
            return tuple(_job_step(row) for row in rows)

    def list_claimable_jobs(self) -> tuple[JobStatus, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM job_status
                   WHERE status='queued' ORDER BY updated_at, job_id"""
            ).fetchall()
            return tuple(_job_status(row) for row in rows)

    def list_job_overviews(self) -> tuple[tuple[JobStatus, str], ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT status.*, job.kind AS job_kind
                   FROM job_status AS status
                   JOIN jobs AS job ON job.id = status.job_id
                   ORDER BY status.updated_at DESC, status.job_id"""
            ).fetchall()
            return tuple((_job_status(row), row["job_kind"]) for row in rows)

    def load_job_attempts(self, job_id: str) -> tuple[JobStepAttempt, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM job_step_attempts
                   WHERE job_id=? ORDER BY step_key, attempt_number""",
                (job_id,),
            ).fetchall()
            return tuple(_job_step_attempt(row) for row in rows)

    def list_source_managed_files(self, source_id: str) -> tuple[ManagedFile, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT file.*
                   FROM managed_files AS file
                   JOIN source_versions AS version ON version.managed_file_id = file.id
                   WHERE version.source_id=?
                   ORDER BY file.created_at, file.id""",
                (source_id,),
            ).fetchall()
            return tuple(_managed_file(row) for row in rows)

    def cancel_job(self, job_id: str, failure_reason: str = "cancelled") -> JobStatus:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_status SET status='cancelled', failure_reason=?, updated_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (failure_reason, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("job is not cancellable in its current status")
            job = _fetchone(connection, "SELECT source_version_id FROM jobs WHERE id=?", (job_id,))
            if job is not None and job["source_version_id"] is not None:
                _mark_pdf_import_failed(
                    connection, job["source_version_id"], "processing_cancelled", timestamp
                )
            # Hide exactly what this job published; a newer job's readiness
            # remains untouched.
            connection.execute(
                """DELETE FROM source_readiness_indexes
                   WHERE source_id IN (
                       SELECT source_id FROM source_readiness WHERE published_by_job_id=?
                   )""",
                (job_id,),
            )
            connection.execute(
                "DELETE FROM source_readiness WHERE published_by_job_id=?", (job_id,)
            )
            row = _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
            return _job_status(_require_row(row))

    def annotate_queued_job(self, job_id: str, reason: str | None) -> None:
        """Set or clear a visible blocking reason on a still-queued job."""
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_status SET failure_reason=?, updated_at=?
                   WHERE job_id=? AND status='queued'""",
                (reason, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("job is not queued")

    def release_running_jobs(self) -> tuple[str, ...]:
        """Return every running job to queued when a core starts.

        Safe without a heartbeat-age cutoff because the exclusive
        data-directory lock proves no other core is alive: a `running` job at
        startup belongs to a previous instance, however recently it
        heartbeated. An immediate restart therefore recovers promptly instead
        of leaving the job stuck for a heartbeat window.
        """
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT job_id FROM job_status WHERE status='running'"
            ).fetchall()
            connection.execute(
                """UPDATE job_status SET status='queued', owner_instance_id=NULL, updated_at=?
                   WHERE status='running'""",
                (timestamp,),
            )
            return tuple(row[0] for row in rows)

    def create_job_status(self, job_id: str) -> JobStatus:
        validate_id(job_id)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO job_status(
                       job_id,status,current_step_key,attempt_count,
                       owner_instance_id,claimed_at,heartbeat_at,failure_reason,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO NOTHING""",
                (job_id, "queued", None, 0, None, None, None, None, timestamp),
            )
            row = _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
            return _job_status(_require_row(row))

    def load_job_status(self, job_id: str) -> JobStatus | None:
        with self.database.connect() as connection:
            row = _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
            return None if row is None else _job_status(row)

    def claim_job(
        self,
        job_id: str,
        owner_instance_id: str,
        *,
        policy: JSONValue | None = None,
    ) -> JobStatus:
        owner_instance_id = _required_text(owner_instance_id, "owner instance id")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            current = _job_status(
                _require_row(
                    _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
                )
            )
            if current.status not in ("queued", "running"):
                raise ConflictingWriteError("job is not claimable in its current status")
            if current.status == "running" and current.owner_instance_id not in (
                None,
                owner_instance_id,
            ):
                raise ConflictingWriteError("job is already claimed by another owner")
            connection.execute(
                """UPDATE job_status
                   SET status='running', owner_instance_id=?,
                       claimed_at=COALESCE(claimed_at, ?), heartbeat_at=?, updated_at=?,
                       policy_json=COALESCE(policy_json, ?)
                   WHERE job_id=?""",
                (
                    owner_instance_id,
                    timestamp,
                    timestamp,
                    timestamp,
                    encode_json(policy) if policy is not None else None,
                    job_id,
                ),
            )
            row = _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
            return _job_status(_require_row(row))

    def heartbeat_job(self, job_id: str, owner_instance_id: str) -> None:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_status SET heartbeat_at=?, updated_at=?
                   WHERE job_id=? AND owner_instance_id=? AND status='running'""",
                (timestamp, timestamp, job_id, owner_instance_id),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("job heartbeat rejected: ownership or status mismatch")

    def update_job_progress(self, job_id: str, current_step_key: str) -> JobStatus:
        current_step_key = _required_text(current_step_key, "current step key")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE job_status SET current_step_key=?, updated_at=? WHERE job_id=?",
                (current_step_key, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("job status record does not exist")
            row = _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
            return _job_status(_require_row(row))

    def finish_job(
        self,
        job_id: str,
        status: str,
        failure_reason: str | None = None,
        *,
        failure_code: str | None = None,
    ) -> JobStatus:
        if status not in ("completed", "failed", "cancelled"):
            raise ValueError("job can only finish as completed, failed, or cancelled")
        if failure_code is not None:
            failure_code = _required_text(failure_code, "failure code")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_status SET status=?, failure_reason=?, failure_code=?, updated_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (status, failure_reason, failure_code, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("job is not active; finish rejected")
            if status in ("failed", "cancelled"):
                job = _fetchone(
                    connection, "SELECT source_version_id FROM jobs WHERE id=?", (job_id,)
                )
                if job is not None and job["source_version_id"] is not None:
                    _mark_pdf_import_failed(
                        connection,
                        job["source_version_id"],
                        failure_code or "processing_failed",
                        timestamp,
                    )
            row = _fetchone(connection, "SELECT * FROM job_status WHERE job_id=?", (job_id,))
            return _job_status(_require_row(row))

    def start_step_attempt(self, job_id: str, step_key: str) -> JobStepAttempt:
        step_key = _required_text(step_key, "step key")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            next_number = int(
                connection.execute(
                    """SELECT COALESCE(MAX(attempt_number), 0) + 1
                       FROM job_step_attempts WHERE job_id=? AND step_key=?""",
                    (job_id, step_key),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO job_step_attempts(
                       job_id,step_key,attempt_number,started_at,finished_at,failure_reason
                   ) VALUES(?,?,?,?,?,?)""",
                (job_id, step_key, next_number, timestamp, None, None),
            )
            cursor = connection.execute(
                """UPDATE job_status
                   SET current_step_key=?, attempt_count=attempt_count+1, updated_at=?
                   WHERE job_id=?""",
                (step_key, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("job status record does not exist")
            row = _fetchone(
                connection,
                "SELECT * FROM job_step_attempts"
                " WHERE job_id=? AND step_key=? AND attempt_number=?",
                (job_id, step_key, next_number),
            )
            return _job_step_attempt(_require_row(row))

    def finish_step_attempt(
        self,
        job_id: str,
        step_key: str,
        attempt_number: int,
        failure_reason: str | None = None,
        outcome: str | None = None,
    ) -> JobStepAttempt:
        if outcome is None:
            outcome = "ok" if failure_reason is None else "failed"
        if outcome not in ("ok", "failed", "unknown"):
            raise ValueError("attempt outcome must be ok, failed, or unknown")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_step_attempts
                   SET finished_at=?, failure_reason=?, outcome=?
                   WHERE job_id=? AND step_key=? AND attempt_number=? AND finished_at IS NULL""",
                (timestamp, failure_reason, outcome, job_id, step_key, attempt_number),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("step attempt is unknown or already finished")
            row = _fetchone(
                connection,
                "SELECT * FROM job_step_attempts"
                " WHERE job_id=? AND step_key=? AND attempt_number=?",
                (job_id, step_key, attempt_number),
            )
            return _job_step_attempt(_require_row(row))

    def record_unknown_paid_outcome(
        self,
        job_id: str,
        step_key: str,
        attempt_number: int,
        attempt_reason: str,
        job_reason: str,
    ) -> JobStepAttempt:
        """Record an unknown paid-call outcome and fail the job in ONE transaction.

        The single write closes the crash window between marking the attempt
        and marking the job; even if a crash splits them elsewhere, the
        attempt's `outcome` stays 'unknown' and recovery never replays it.
        """
        attempt_reason = _required_text(attempt_reason, "attempt reason")
        job_reason = _required_text(job_reason, "job reason")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_step_attempts
                   SET finished_at=?, failure_reason=?, outcome='unknown'
                   WHERE job_id=? AND step_key=? AND attempt_number=? AND finished_at IS NULL""",
                (timestamp, attempt_reason, job_id, step_key, attempt_number),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("step attempt is unknown or already finished")
            job_cursor = connection.execute(
                """UPDATE job_status SET status='failed', failure_reason=?, updated_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (job_reason, timestamp, job_id),
            )
            if job_cursor.rowcount != 1:
                raise ConflictingWriteError("job is not active; unknown outcome not recorded")
            row = _fetchone(
                connection,
                "SELECT * FROM job_step_attempts"
                " WHERE job_id=? AND step_key=? AND attempt_number=?",
                (job_id, step_key, attempt_number),
            )
            return _job_step_attempt(_require_row(row))

    def load_step_attempts(self, job_id: str, step_key: str) -> tuple[JobStepAttempt, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM job_step_attempts
                   WHERE job_id=? AND step_key=? ORDER BY attempt_number""",
                (job_id, step_key),
            ).fetchall()
            return tuple(_job_step_attempt(row) for row in rows)

    def publish_readiness(
        self,
        job_id: str,
        source_id: str,
        source_version_id: str,
        index_version_ids: Sequence[str],
        *,
        guard: AttemptGuard,
    ) -> SourceReadiness:
        """Publish readiness only after verifying the publishing job in-transaction.

        The publishing instance, job, and attempt are verified (the job must be
        running, owned by the guard, and the guarded attempt current and
        unresolved); the job must belong to this source, target the source's
        current version, and have committed every required index as one of its
        own step outputs. An older job cannot replace a newer job's readiness,
        a trashed source can never publish, an extraction whose decision is
        missing, pending, or declined never publishes, and cancelling the job
        later hides exactly the readiness it published.
        """
        validate_id(job_id)
        validate_id(source_id)
        validate_id(source_version_id)
        cleaned_indexes = tuple(dict.fromkeys(validate_id(item) for item in index_version_ids))
        if not cleaned_indexes:
            raise ValueError("readiness requires at least one finished index")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            _verify_attempt_current(connection, job_id, guard.step_key, guard)
            status_row = _fetchone(
                connection,
                "SELECT superseded_by_job_id FROM job_status WHERE job_id=?",
                (job_id,),
            )
            if status_row is not None and status_row["superseded_by_job_id"] is not None:
                raise ConflictingWriteError(
                    "job was superseded by newer work; publication rejected"
                )
            job_row = _fetchone(
                connection,
                "SELECT source_id, source_version_id, created_at FROM jobs WHERE id=?",
                (job_id,),
            )
            if job_row is None or job_row["source_id"] != source_id:
                raise ConflictingWriteError(
                    "job does not belong to this source; publication rejected"
                )
            current_row = connection.execute(
                """SELECT id FROM source_versions WHERE source_id=?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (source_id,),
            ).fetchone()
            if (
                current_row is None
                or current_row[0] != source_version_id
                or job_row["source_version_id"] != source_version_id
            ):
                raise ConflictingWriteError(
                    "job does not target the current source version; publication rejected"
                )
            trash_row = _fetchone(
                connection, "SELECT 1 FROM source_trash WHERE source_id=?", (source_id,)
            )
            if trash_row is not None:
                raise ConflictingWriteError("source is deleted; publication rejected")
            _verify_extraction_decision(connection, cleaned_indexes)
            committed_outputs = {
                row[0]
                for row in connection.execute(
                    """SELECT output_index_version_id FROM job_steps
                       WHERE job_id=? AND output_index_version_id IS NOT NULL""",
                    (job_id,),
                ).fetchall()
            }
            missing = [item for item in cleaned_indexes if item not in committed_outputs]
            if missing:
                raise ConflictingWriteError(
                    "published indexes were not committed by this job; publication rejected"
                )
            current_readiness = _fetchone(
                connection,
                "SELECT published_by_job_id FROM source_readiness WHERE source_id=?",
                (source_id,),
            )
            if current_readiness is not None and current_readiness["published_by_job_id"] not in (
                None,
                job_id,
            ):
                current_publisher = _fetchone(
                    connection,
                    "SELECT created_at FROM jobs WHERE id=?",
                    (current_readiness["published_by_job_id"],),
                )
                if (
                    current_publisher is not None
                    and current_publisher["created_at"] > job_row["created_at"]
                ):
                    raise ConflictingWriteError(
                        "a newer job's readiness cannot be replaced by an older job"
                    )
            connection.execute(
                """INSERT INTO source_readiness(
                       source_id,source_version_id,published_by_job_id,published_at,updated_at
                   ) VALUES(?,?,?,?,?)
                   ON CONFLICT(source_id) DO UPDATE SET
                       source_version_id=excluded.source_version_id,
                       published_by_job_id=excluded.published_by_job_id,
                       published_at=excluded.published_at,
                       updated_at=excluded.updated_at""",
                (source_id, source_version_id, job_id, timestamp, timestamp),
            )
            connection.execute(
                "DELETE FROM source_readiness_indexes WHERE source_id=?", (source_id,)
            )
            for index_version_id in cleaned_indexes:
                connection.execute(
                    "INSERT INTO source_readiness_indexes(source_id,index_version_id) VALUES(?,?)",
                    (source_id, index_version_id),
                )
            pdf_import = _fetchone(
                connection,
                "SELECT id FROM pdf_imports WHERE source_id=? AND source_version_id=?",
                (source_id, source_version_id),
            )
            if pdf_import is not None:
                import_id = pdf_import["id"]
                state = _fetchone(
                    connection, "SELECT * FROM source_pdf_state WHERE source_id=?", (source_id,)
                )
                if state is None or import_id not in (
                    state["current_import_id"],
                    state["pending_import_id"],
                ):
                    raise ConflictingWriteError(
                        "PDF version is no longer the source's current candidate"
                    )
                if (
                    state["pending_import_id"] is not None
                    and state["pending_import_id"] != import_id
                ):
                    raise ConflictingWriteError("a newer PDF replacement is pending")
                connection.execute(
                    """UPDATE pdf_imports SET state='ready',failure_code=NULL,updated_at=?
                       WHERE id=?""",
                    (timestamp, import_id),
                )
                connection.execute(
                    """UPDATE source_pdf_state SET current_import_id=?,pending_import_id=NULL,
                              revision=revision+1,updated_at=? WHERE source_id=?""",
                    (import_id, timestamp, source_id),
                )
            # Mark every older or same-instant peer superseded. The mark is
            # durable and independent of the readiness record, so cancelling
            # this (or any newer) result never restores an older job's right
            # to publish. Ties on created_at are resolved by first publisher.
            connection.execute(
                """UPDATE job_status
                   SET superseded_by_job_id=?, updated_at=?
                   WHERE job_id<>? AND superseded_by_job_id IS NULL
                     AND job_id IN (
                         SELECT id FROM jobs WHERE source_id=? AND created_at <= ?
                     )""",
                (job_id, timestamp, job_id, source_id, job_row["created_at"]),
            )
            return _load_readiness(connection, source_id)

    def load_readiness(self, source_id: str) -> SourceReadiness | None:
        with self.database.connect() as connection:
            row = _fetchone(
                connection, "SELECT 1 FROM source_readiness WHERE source_id=?", (source_id,)
            )
            return None if row is None else _load_readiness(connection, source_id)

    def remove_readiness(self, source_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM source_readiness_indexes WHERE source_id=?", (source_id,)
            )
            connection.execute("DELETE FROM source_readiness WHERE source_id=?", (source_id,))

    def trash_source(self, source_id: str, purge_deadline_at: str) -> SourceTrash:
        """One transaction: hide readiness, cancel active jobs, record the trash deadline."""
        validate_id(source_id)
        validate_utc_timestamp(purge_deadline_at)
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            acquisitions = connection.execute(
                """SELECT * FROM pdf_acquisitions WHERE target_source_id=?
                   AND state IN ('awaiting_upload','receiving','staged','finalizing',
                                 'waiting_cleanup')""",
                (source_id,),
            ).fetchall()
            for acquisition in acquisitions:
                connection.execute(
                    """INSERT INTO acquisition_file_cleanup(
                               id,acquisition_id,relative_path,file_kind,expected_sha256,
                               expected_size_bytes,created_at
                           ) VALUES(?,?,?,'staging',?,?,?)
                           ON CONFLICT(acquisition_id,relative_path) DO NOTHING""",
                    (
                        new_id(),
                        acquisition["id"],
                        acquisition["staging_relative_path"],
                        acquisition["input_sha256"],
                        acquisition["input_size_bytes"],
                        timestamp,
                    ),
                )
                if acquisition["planned_relative_path"] is not None:
                    connection.execute(
                        """INSERT INTO acquisition_file_cleanup(
                               id,acquisition_id,relative_path,file_kind,expected_sha256,
                               expected_size_bytes,created_at
                           ) VALUES(?,?,?,'planned_original',?,?,?)
                           ON CONFLICT(acquisition_id,relative_path) DO NOTHING""",
                        (
                            new_id(),
                            acquisition["id"],
                            acquisition["planned_relative_path"],
                            acquisition["planned_sha256"],
                            acquisition["planned_size_bytes"],
                            timestamp,
                        ),
                    )
            connection.execute(
                "DELETE FROM source_readiness_indexes WHERE source_id=?", (source_id,)
            )
            connection.execute("DELETE FROM source_readiness WHERE source_id=?", (source_id,))
            connection.execute(
                """UPDATE job_status SET status='cancelled', failure_reason='source deleted',
                       updated_at=?
                   WHERE status IN ('queued','running')
                     AND job_id IN (SELECT id FROM jobs WHERE source_id=?)""",
                (timestamp, source_id),
            )
            connection.execute(
                """UPDATE pdf_acquisitions
                   SET state='cancelled',failure_code='source_deleted',
                       finalization_generation=finalization_generation+1,
                       planned_reuse_managed_file_id=NULL,updated_at=?
                   WHERE target_source_id=?
                     AND state IN ('awaiting_upload','receiving','staged','finalizing',
                                   'waiting_cleanup')""",
                (timestamp, source_id),
            )
            connection.execute(
                """INSERT INTO source_trash(
                       source_id,trashed_at,purge_deadline_at,files_removed_at,
                       snapshots_removed_at,updated_at
                   ) VALUES(?,?,?,?,?,?)
                   ON CONFLICT(source_id) DO UPDATE SET
                       purge_deadline_at=excluded.purge_deadline_at,
                       updated_at=excluded.updated_at""",
                (source_id, timestamp, purge_deadline_at, None, None, timestamp),
            )
            row = _fetchone(
                connection, "SELECT * FROM source_trash WHERE source_id=?", (source_id,)
            )
            return _source_trash(_require_row(row))

    def load_trash(self, source_id: str) -> SourceTrash | None:
        with self.database.connect() as connection:
            row = _fetchone(
                connection, "SELECT * FROM source_trash WHERE source_id=?", (source_id,)
            )
            return None if row is None else _source_trash(row)

    def list_expired_trash(self, now: str) -> tuple[SourceTrash, ...]:
        validate_utc_timestamp(now)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM source_trash
                   WHERE purge_deadline_at <= ? ORDER BY purge_deadline_at""",
                (now,),
            ).fetchall()
            return tuple(_source_trash(row) for row in rows)

    def mark_trash_files_removed(self, source_id: str) -> SourceTrash:
        return self._mark_trash_phase(source_id, "files_removed_at")

    def mark_trash_snapshots_removed(self, source_id: str) -> SourceTrash:
        return self._mark_trash_phase(source_id, "snapshots_removed_at")

    def _mark_trash_phase(self, source_id: str, column: str) -> SourceTrash:
        if column not in ("files_removed_at", "snapshots_removed_at"):
            raise ValueError("unknown trash phase column")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                f"UPDATE source_trash SET {column}=?, updated_at=? WHERE source_id=?",
                (timestamp, timestamp, source_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("source is not in trash")
            row = _fetchone(
                connection, "SELECT * FROM source_trash WHERE source_id=?", (source_id,)
            )
            return _source_trash(_require_row(row))

    def remove_trashed_records(self, source_id: str) -> None:
        """Atomic final purge phase: records, source row, and trash tracker.

        File and snapshot removal happen outside the database (tracked by
        ``files_removed_at``/``snapshots_removed_at``); this transaction removes
        the remaining records in foreign-key order and the tracker together with
        the source row, so a tracker row always implies the source still exists.
        """
        validate_id(source_id)
        with self._transaction() as connection:
            tracker = _fetchone(
                connection, "SELECT 1 FROM source_trash WHERE source_id=?", (source_id,)
            )
            if tracker is None:
                raise PersistenceError("source is not in trash")
            pending_cleanup = _fetchone(
                connection,
                """SELECT 1 FROM acquisition_file_cleanup AS cleanup
                   JOIN pdf_acquisitions AS acquisition ON acquisition.id=cleanup.acquisition_id
                   WHERE cleanup.removed_at IS NULL
                     AND (acquisition.target_source_id=? OR acquisition.outcome_source_id=?)
                   LIMIT 1""",
                (source_id, source_id),
            )
            if pending_cleanup is not None:
                raise PersistenceError("PDF acquisition cleanup is still pending")
            managed_file_ids = [
                row[0]
                for row in connection.execute(
                    """SELECT DISTINCT managed_file_id FROM source_versions
                       WHERE source_id=? AND managed_file_id IS NOT NULL""",
                    (source_id,),
                ).fetchall()
            ]
            connection.execute(
                """DELETE FROM acquisition_file_cleanup
                   WHERE acquisition_id IN (
                       SELECT id FROM pdf_acquisitions
                       WHERE target_source_id=? OR outcome_source_id=?
                   )""",
                (source_id, source_id),
            )
            connection.execute("DELETE FROM pdf_import_trash WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM source_pdf_state WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM pdf_imports WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM chunk_passages WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM retrieval_chunks WHERE source_id=?", (source_id,))
            connection.execute(
                "DELETE FROM source_readiness_indexes WHERE source_id=?", (source_id,)
            )
            connection.execute("DELETE FROM source_readiness WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM passages WHERE source_id=?", (source_id,))
            connection.execute(
                """DELETE FROM job_steps
                   WHERE job_id IN (SELECT id FROM jobs WHERE source_id=?)""",
                (source_id,),
            )
            connection.execute(
                """DELETE FROM job_step_attempts
                   WHERE job_id IN (SELECT id FROM jobs WHERE source_id=?)""",
                (source_id,),
            )
            connection.execute(
                """DELETE FROM job_status
                   WHERE job_id IN (SELECT id FROM jobs WHERE source_id=?)""",
                (source_id,),
            )
            connection.execute("DELETE FROM jobs WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM index_versions WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM extraction_reviews WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM extraction_outcomes WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM extraction_visuals WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM extraction_pages WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM extraction_versions WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM source_versions WHERE source_id=?", (source_id,))
            connection.execute(
                """UPDATE pdf_acquisitions SET target_source_id=NULL,outcome_source_id=NULL,
                          outcome_import_id=NULL,expected_current_import_id=NULL
                   WHERE target_source_id=? OR outcome_source_id=?""",
                (source_id, source_id),
            )
            for managed_file_id in managed_file_ids:
                removal = _fetchone(
                    connection,
                    "SELECT * FROM managed_file_removals WHERE managed_file_id=?",
                    (managed_file_id,),
                )
                if removal is not None and removal["removed_at"] is None:
                    # A pending removal keeps both the marker and the row so
                    # the next sweep retries the file and new references stay
                    # blocked in the meantime.
                    continue
                connection.execute(
                    "DELETE FROM managed_file_removals WHERE managed_file_id=?",
                    (managed_file_id,),
                )
                connection.execute(
                    """DELETE FROM managed_files
                       WHERE id=?
                         AND NOT EXISTS (
                             SELECT 1 FROM source_versions
                             WHERE managed_file_id = managed_files.id
                         )""",
                    (managed_file_id,),
                )
            connection.execute("DELETE FROM source_trash WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM sources WHERE id=?", (source_id,))

    def reserve_managed_file_removal(
        self, managed_file_id: str, relative_path: str, source_id: str
    ) -> ManagedFileRemoval | None:
        """Atomically check surviving references and reserve the removal.

        Returns None when another source still references the file and creates
        nothing; otherwise the marker exists in the same transaction, so no
        reference can be inserted between the check and the reservation (the
        marker's trigger rejects later references). A reference committed
        before this transaction is seen by the check.
        """
        relative_path = _required_text(relative_path, "relative path")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            shared = connection.execute(
                """SELECT 1 FROM source_versions
                   WHERE managed_file_id=? AND source_id<>? LIMIT 1""",
                (managed_file_id, source_id),
            ).fetchone()
            if shared is not None:
                return None
            held = connection.execute(
                """SELECT 1 FROM pdf_acquisitions
                   WHERE planned_reuse_managed_file_id=?
                     AND state IN ('staged','finalizing','waiting_cleanup') LIMIT 1""",
                (managed_file_id,),
            ).fetchone()
            if held is not None:
                return None
            connection.execute(
                """INSERT INTO managed_file_removals(
                       managed_file_id,relative_path,requested_at,removed_at,attempts,last_error
                   ) VALUES(?,?,?,?,?,?) ON CONFLICT(managed_file_id) DO NOTHING""",
                (managed_file_id, relative_path, timestamp, None, 0, None),
            )
            row = _fetchone(
                connection,
                "SELECT * FROM managed_file_removals WHERE managed_file_id=?",
                (managed_file_id,),
            )
            return _managed_file_removal(_require_row(row))

    def mark_managed_file_removed(self, managed_file_id: str) -> ManagedFileRemoval:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE managed_file_removals
                   SET removed_at=?, last_error=NULL
                   WHERE managed_file_id=?""",
                (timestamp, managed_file_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("managed file is not marked for removal")
            connection.execute(
                "DELETE FROM original_registry WHERE managed_file_id=?", (managed_file_id,)
            )
            connection.execute(
                """UPDATE pdf_import_trash SET file_removed_at=?,updated_at=?
                   WHERE import_id IN (
                       SELECT item.id FROM pdf_imports AS item
                       JOIN source_versions AS version ON version.id=item.source_version_id
                       WHERE version.managed_file_id=?
                   )""",
                (timestamp, timestamp, managed_file_id),
            )
            row = _fetchone(
                connection,
                "SELECT * FROM managed_file_removals WHERE managed_file_id=?",
                (managed_file_id,),
            )
            return _managed_file_removal(_require_row(row))

    def record_managed_file_removal_failure(
        self, managed_file_id: str, error: str
    ) -> ManagedFileRemoval:
        error = _required_text(error, "removal error summary")
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE managed_file_removals
                   SET attempts=attempts+1, last_error=?
                   WHERE managed_file_id=? AND removed_at IS NULL""",
                (error, managed_file_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("managed file is not pending removal")
            row = _fetchone(
                connection,
                "SELECT * FROM managed_file_removals WHERE managed_file_id=?",
                (managed_file_id,),
            )
            return _managed_file_removal(_require_row(row))

    def load_managed_file_removal(self, managed_file_id: str) -> ManagedFileRemoval | None:
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                "SELECT * FROM managed_file_removals WHERE managed_file_id=?",
                (managed_file_id,),
            )
            return None if row is None else _managed_file_removal(row)

    def load_notebook(self, notebook_id: str) -> Notebook:
        validate_id(notebook_id)
        with self.database.connect() as connection:
            row = _fetchone(connection, "SELECT * FROM notebooks WHERE id=?", (notebook_id,))
            if row is None:
                raise PersistenceError("notebook does not exist")
            return _notebook(_require_row(row))

    def load_source(self, source_id: str) -> Source:
        validate_id(source_id)
        with self.database.connect() as connection:
            row = _fetchone(connection, "SELECT * FROM sources WHERE id=?", (source_id,))
            if row is None:
                raise PersistenceError("source does not exist")
            return _source(_require_row(row))

    def load_extraction(self, extraction_version_id: str) -> ExtractionGraph:
        validate_id(extraction_version_id)
        with self.database.connect() as connection:
            return _load_extraction(connection, extraction_version_id)

    def load_index(self, index_version_id: str) -> IndexGraph:
        validate_id(index_version_id)
        with self.database.connect() as connection:
            return _load_index(connection, index_version_id)

    def load_job(self, logical_work_key: str) -> Job:
        with self.database.connect() as connection:
            row = _fetchone(
                connection,
                "SELECT * FROM jobs WHERE logical_work_key=?",
                (logical_work_key,),
            )
            if row is None:
                raise PersistenceError("job does not exist")
            return _job(row)

    def load_job_step(self, job_id: str, step_key: str) -> JobStep:
        validate_id(job_id)
        with self.database.connect() as connection:
            return _load_job_step(connection, job_id, step_key)

    def load_snapshot(self) -> PersistenceSnapshot:
        """Load every 1.2 record through one read transaction for restart evidence."""
        with self._transaction(write=False) as connection:
            notebooks = tuple(
                _notebook(row) for row in connection.execute("SELECT * FROM notebooks ORDER BY id")
            )
            sources = tuple(
                _source(row) for row in connection.execute("SELECT * FROM sources ORDER BY id")
            )
            managed_files = tuple(
                _managed_file(row)
                for row in connection.execute("SELECT * FROM managed_files ORDER BY id")
            )
            source_versions = tuple(
                _source_version(row)
                for row in connection.execute("SELECT * FROM source_versions ORDER BY id")
            )
            extractions = tuple(
                _load_extraction(connection, row["id"])
                for row in connection.execute("SELECT id FROM extraction_versions ORDER BY id")
            )
            indexes = tuple(
                _load_index(connection, row["id"])
                for row in connection.execute("SELECT id FROM index_versions ORDER BY id")
            )
            jobs = tuple(_job(row) for row in connection.execute("SELECT * FROM jobs ORDER BY id"))
            job_steps = tuple(
                _job_step(row)
                for row in connection.execute("SELECT * FROM job_steps ORDER BY job_id,step_key")
            )
            return PersistenceSnapshot(
                notebooks,
                sources,
                managed_files,
                source_versions,
                extractions,
                indexes,
                jobs,
                job_steps,
            )

    @contextmanager
    def _transaction(self, write: bool = True) -> Iterator[sqlite3.Connection]:
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                try:
                    yield connection
                    connection.execute("COMMIT")
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
        except sqlite3.Error as error:
            translated = translate_sqlite_error(error)
            raise translated from error


def _fetchone(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...] = (),
) -> sqlite3.Row | None:
    return connection.execute(query, parameters).fetchone()


def _verify_pdf_acquisition_attempt(
    row: sqlite3.Row,
    owner_instance_id: str,
    generation: int,
    expected_state: str,
) -> None:
    if row["owner_instance_id"] != owner_instance_id:
        raise ConflictingWriteError("PDF acquisition belongs to another core instance")
    if int(row["finalization_generation"]) != generation:
        raise ConflictingWriteError("PDF acquisition attempt was superseded")
    if row["state"] != expected_state:
        raise ConflictingWriteError("PDF acquisition is not in the expected state")


def _mark_pdf_import_failed(
    connection: sqlite3.Connection,
    source_version_id: str,
    failure_code: str,
    timestamp: str,
) -> None:
    row = _fetchone(
        connection,
        "SELECT id,source_id FROM pdf_imports WHERE source_version_id=?",
        (source_version_id,),
    )
    if row is None:
        return
    import_id = row["id"]
    source_id = row["source_id"]
    connection.execute(
        "UPDATE pdf_imports SET state='failed',failure_code=?,updated_at=? WHERE id=?",
        (failure_code, timestamp, import_id),
    )
    connection.execute(
        """UPDATE source_pdf_state SET pending_import_id=NULL,revision=revision+1,updated_at=?
           WHERE source_id=? AND pending_import_id=?""",
        (timestamp, source_id, import_id),
    )


def _validate_relative_path(path: str) -> None:
    pure_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        not path
        or "\\" in path
        or pure_path.is_absolute()
        or windows_path.drive
        or ".." in pure_path.parts
        or not pure_path.parts
        or pure_path.as_posix() != path
    ):
        raise ValueError("path must be a normalized app-data-relative path")


def _require_row(row: sqlite3.Row | None) -> sqlite3.Row:
    if row is None:
        raise PersistenceError("expected database record was not found")
    return row


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


def _integer(value: int, label: str, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer greater than or equal to {minimum}")


def _validate_managed_file(managed_file: ManagedFile) -> None:
    validate_id(managed_file.id)
    validate_sha256(managed_file.sha256)
    path = managed_file.relative_path
    pure_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        not path
        or "\\" in path
        or pure_path.is_absolute()
        or windows_path.drive
        or ".." in pure_path.parts
        or not pure_path.parts
        or pure_path.as_posix() != path
    ):
        raise ValueError("managed-file reference must be a normalized app-data-relative path")
    _integer(managed_file.size_bytes, "managed-file size")
    validate_utc_timestamp(managed_file.created_at)


def _validate_source_version(source_version: SourceVersion) -> None:
    validate_id(source_version.id)
    validate_id(source_version.source_id)
    if source_version.managed_file_id is not None:
        validate_id(source_version.managed_file_id)
    validate_sha256(source_version.content_sha256)
    validate_utc_timestamp(source_version.created_at)


def _validate_extraction(extraction: ExtractionVersion, passages: Sequence[Passage]) -> None:
    validate_id(extraction.id)
    validate_id(extraction.source_id)
    validate_id(extraction.source_version_id)
    _required_text(extraction.parser_name, "parser name")
    _required_text(extraction.parser_version, "parser version")
    _required_text(extraction.normalization_version, "normalization version")
    _integer(extraction.settings_format_version, "settings format version", minimum=1)
    encode_json(extraction.settings)
    validate_utc_timestamp(extraction.created_at)
    if len({passage.id for passage in passages}) != len(passages):
        raise ValueError("passage identities must be unique within an extraction")
    if len({passage.ordinal for passage in passages}) != len(passages):
        raise ValueError("passage positions must be unique within an extraction")
    for passage in passages:
        validate_id(passage.id)
        if (
            passage.source_id != extraction.source_id
            or passage.source_version_id != extraction.source_version_id
            or passage.extraction_version_id != extraction.id
        ):
            raise ValueError("passage ownership does not match its extraction")
        _integer(passage.ordinal, "passage position")
        _integer(passage.location_format_version, "passage location format version", minimum=1)
        if not passage.text:
            raise ValueError("passage position and text must be valid")
        encode_json(passage.location)
        validate_utc_timestamp(passage.created_at)


def _validate_index(
    index: IndexVersion,
    chunks: Sequence[RetrievalChunk],
    mappings: Sequence[ChunkPassage],
    connection: sqlite3.Connection,
) -> None:
    for identifier in (
        index.id,
        index.source_id,
        index.source_version_id,
        index.extraction_version_id,
    ):
        validate_id(identifier)
    _required_text(index.index_kind, "index kind")
    _required_text(index.chunker_version, "chunker version")
    _integer(index.settings_format_version, "settings format version", minimum=1)
    if (index.embedding_model is None) != (index.embedding_revision is None):
        raise ValueError("embedding model and revision must be supplied together")
    encode_json(index.settings)
    validate_utc_timestamp(index.created_at)
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("retrieval chunk identities must be unique within an index")
    if len({chunk.ordinal for chunk in chunks}) != len(chunks):
        raise ValueError("retrieval chunk positions must be unique within an index")
    chunk_ids = {chunk.id for chunk in chunks}
    passage_text: dict[str, str] = {}
    for row in connection.execute(
        "SELECT id,text FROM passages WHERE extraction_version_id=?",
        (index.extraction_version_id,),
    ):
        passage_text[row["id"]] = row["text"]
    for chunk in chunks:
        validate_id(chunk.id)
        if (
            chunk.index_version_id != index.id
            or chunk.source_id != index.source_id
            or chunk.source_version_id != index.source_version_id
            or chunk.extraction_version_id != index.extraction_version_id
        ):
            raise ValueError("retrieval chunk ownership does not match its index")
        _integer(chunk.ordinal, "retrieval chunk position")
        if not chunk.text:
            raise ValueError("retrieval chunk position and text must be valid")
    seen_positions: set[tuple[str, int]] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for mapping in mappings:
        validate_id(mapping.retrieval_chunk_id)
        validate_id(mapping.passage_id)
        if (
            mapping.index_version_id != index.id
            or mapping.source_id != index.source_id
            or mapping.source_version_id != index.source_version_id
            or mapping.extraction_version_id != index.extraction_version_id
        ):
            raise ValueError("chunk mapping ownership does not match its index")
        if mapping.retrieval_chunk_id not in chunk_ids:
            raise ValueError("chunk mapping references a chunk outside this index")
        if mapping.passage_id not in passage_text:
            raise ValueError("chunk mapping references a passage outside this extraction")
        _integer(mapping.ordinal, "chunk mapping position")
        pair = (mapping.retrieval_chunk_id, mapping.passage_id)
        position = (mapping.retrieval_chunk_id, mapping.ordinal)
        if pair in seen_pairs or position in seen_positions:
            raise ValueError("chunk mappings must have unique passage and order per chunk")
        seen_pairs.add(pair)
        seen_positions.add(position)
        if (mapping.start_offset is None) != (mapping.end_offset is None):
            raise ValueError("partial passage offsets must be supplied as a pair")
        if mapping.start_offset is not None:
            _integer(mapping.start_offset, "partial passage start offset")
            if mapping.end_offset is None:
                raise ValueError("partial passage offsets must be supplied as a pair")
            _integer(mapping.end_offset, "partial passage end offset")
            if mapping.end_offset < mapping.start_offset:
                raise ValueError("partial passage offsets are invalid")
            if mapping.end_offset > len(passage_text[mapping.passage_id]):
                raise ValueError("partial passage offsets exceed the saved passage text")


def _validate_job_targets(
    source_id: str,
    source_version_id: str | None,
    extraction_version_id: str | None,
    index_version_id: str | None,
) -> None:
    validate_id(source_id)
    for identifier in (source_version_id, extraction_version_id, index_version_id):
        if identifier is not None:
            validate_id(identifier)
    if source_version_id is None and extraction_version_id is not None:
        raise ValueError("an extraction job target requires a source-version target")
    if extraction_version_id is None and index_version_id is not None:
        raise ValueError("an index job target requires an extraction-version target")


def _save_managed_file(connection: sqlite3.Connection, item: ManagedFile) -> None:
    values = {
        "id": item.id,
        "relative_path": item.relative_path,
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
        "created_at": item.created_at,
    }
    existing = _fetchone(connection, "SELECT * FROM managed_files WHERE id=?", (item.id,))
    if existing is not None:
        if any(existing[name] != value for name, value in values.items() if name != "created_at"):
            raise ConflictingWriteError("managed-file identity already has different content")
        return
    try:
        connection.execute(
            """INSERT INTO managed_files(id,relative_path,size_bytes,sha256,created_at)
               VALUES(:id,:relative_path,:size_bytes,:sha256,:created_at)""",
            values,
        )
    except sqlite3.IntegrityError as error:
        raise ConflictingWriteError("managed-file reference is already assigned") from error


def _save_source_version(connection: sqlite3.Connection, item: SourceVersion) -> None:
    values = {
        "source_id": item.source_id,
        "managed_file_id": item.managed_file_id,
        "content_sha256": item.content_sha256,
    }
    existing = _fetchone(connection, "SELECT * FROM source_versions WHERE id=?", (item.id,))
    if existing is not None:
        if any(existing[name] != value for name, value in values.items()):
            raise ConflictingWriteError("source-version identity already has different content")
        return
    connection.execute(
        """INSERT INTO source_versions(id,source_id,managed_file_id,content_sha256,created_at)
           VALUES(?,?,?,?,?)""",
        (item.id, item.source_id, item.managed_file_id, item.content_sha256, item.created_at),
    )


def _validate_extraction_result(
    extraction: ExtractionVersion,
    passages: Sequence[Passage],
    pages: Sequence[ExtractionPage],
    visuals: Sequence[VisualLocation],
    outcome: ExtractionOutcome,
    review: ExtractionReview | None,
    *,
    is_reuse: bool = False,
) -> None:
    _validate_extraction(extraction, passages)
    if (
        outcome.extraction_version_id != extraction.id
        or outcome.source_id != extraction.source_id
        or outcome.source_version_id != extraction.source_version_id
    ):
        raise ValueError("extraction outcome ownership does not match its extraction")
    if outcome.outcome not in ("complete", "partial"):
        raise ValueError("extraction outcome must be 'complete' or 'partial'")
    validate_utc_timestamp(outcome.created_at)
    if (outcome.outcome == "partial") != (review is not None):
        raise ValueError(
            "partial extractions require a review state; complete extractions must not carry one"
        )
    if review is not None:
        if (
            review.extraction_version_id != extraction.id
            or review.source_id != extraction.source_id
            or review.source_version_id != extraction.source_version_id
        ):
            raise ValueError("extraction review ownership does not match its extraction")
        if review.state not in ("pending", "confirmed", "declined"):
            raise ValueError("review state is not a known state")
        _integer(review.generation, "review generation", minimum=1)
        if not is_reuse:
            # A NEW extraction starts with a pending review at generation 1. A reused
            # extraction carries its saved decision unchanged (confirmed, declined, or a
            # reopened generation) and must never be rejected for it.
            if review.state != "pending":
                raise ValueError("a new extraction's review starts pending")
            if review.generation != 1:
                raise ValueError("a new extraction's review starts at generation 1")
            if review.decided_at is not None:
                raise ValueError("a pending review has no decision time")
        if review.decided_at is not None:
            validate_utc_timestamp(review.decided_at)
        validate_utc_timestamp(review.updated_at)
    if len({page.id for page in pages}) != len(pages):
        raise ValueError("page identities must be unique within an extraction")
    if len({page.page_number for page in pages}) != len(pages):
        raise ValueError("page numbers must be unique within an extraction")
    for page in pages:
        validate_id(page.id)
        if (
            page.source_id != extraction.source_id
            or page.source_version_id != extraction.source_version_id
            or page.extraction_version_id != extraction.id
        ):
            raise ValueError("page ownership does not match its extraction")
        _integer(page.page_number, "page number", minimum=1)
        if not isinstance(page.text_drawn, bool):
            raise ValueError("page text-drawn flag must be a boolean")
        _integer(page.extracted_chars, "extracted characters")
        for label, ratio in (
            ("unmapped ratio", page.unmapped_ratio),
            ("image area ratio", page.image_area_ratio),
        ):
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError(f"page {label} must be numeric")
            if not 0.0 <= float(ratio) <= 1.0:
                raise ValueError(f"page {label} must be between 0 and 1")
        if page.classification not in (
            "extracted",
            "suspect",
            "image-only",
            "unreadable",
            "blank",
        ):
            raise ValueError("page classification is not a known state")
        if page.extracted_chars > 0 and not page.text_drawn:
            raise ValueError("decoded characters require drawn text")
        if page.classification == "extracted" and page.extracted_chars == 0:
            raise ValueError("an extracted page has decoded text")
        if page.classification == "blank" and (page.text_drawn or page.extracted_chars):
            raise ValueError("a blank page draws no text and decodes no characters")
        if page.classification == "unreadable" and not page.text_drawn:
            raise ValueError("an unreadable page has drawn text that did not decode")
        validate_utc_timestamp(page.created_at)
    if len({visual.id for visual in visuals}) != len(visuals):
        raise ValueError("visual identities must be unique within an extraction")
    seen: set[tuple[int, int]] = set()
    for visual in visuals:
        validate_id(visual.id)
        if (
            visual.source_id != extraction.source_id
            or visual.source_version_id != extraction.source_version_id
            or visual.extraction_version_id != extraction.id
        ):
            raise ValueError("visual location ownership does not match its extraction")
        _integer(visual.page_number, "visual page number", minimum=1)
        _integer(visual.ordinal, "visual position")
        if (visual.page_number, visual.ordinal) in seen:
            raise ValueError("visual positions must be unique within a page")
        seen.add((visual.page_number, visual.ordinal))
        if visual.kind not in ("image", "figure"):
            raise ValueError("visual location kind is not a known kind")
        encode_json(visual.bbox)
        validate_utc_timestamp(visual.created_at)


def _save_extraction_result(
    connection: sqlite3.Connection,
    extraction: ExtractionVersion,
    passages: Sequence[Passage],
    pages: Sequence[ExtractionPage],
    visuals: Sequence[VisualLocation],
    outcome: ExtractionOutcome,
    review: ExtractionReview | None,
) -> None:
    existing = _fetchone(
        connection,
        "SELECT outcome FROM extraction_outcomes WHERE extraction_version_id=?",
        (extraction.id,),
    )
    _validate_extraction_result(
        extraction,
        passages,
        pages,
        visuals,
        outcome,
        review,
        is_reuse=existing is not None,
    )
    _save_extraction(connection, extraction, passages)
    if existing is not None:
        _verify_saved_result_matches(connection, extraction.id, pages, visuals, outcome, review)
        return
    connection.execute(
        """INSERT INTO extraction_outcomes(
               extraction_version_id,source_id,source_version_id,outcome,created_at
           ) VALUES(?,?,?,?,?)""",
        (
            outcome.extraction_version_id,
            outcome.source_id,
            outcome.source_version_id,
            outcome.outcome,
            outcome.created_at,
        ),
    )
    for page in pages:
        connection.execute(
            """INSERT INTO extraction_pages(
                   id,source_id,source_version_id,extraction_version_id,page_number,
                   text_drawn,extracted_chars,unmapped_ratio,image_area_ratio,
                   classification,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                page.id,
                page.source_id,
                page.source_version_id,
                page.extraction_version_id,
                page.page_number,
                int(page.text_drawn),
                page.extracted_chars,
                page.unmapped_ratio,
                page.image_area_ratio,
                page.classification,
                page.created_at,
            ),
        )
    for visual in visuals:
        connection.execute(
            """INSERT INTO extraction_visuals(
                   id,source_id,source_version_id,extraction_version_id,page_number,
                   ordinal,kind,bbox_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                visual.id,
                visual.source_id,
                visual.source_version_id,
                visual.extraction_version_id,
                visual.page_number,
                visual.ordinal,
                visual.kind,
                encode_json(visual.bbox),
                visual.created_at,
            ),
        )
    if review is not None:
        connection.execute(
            """INSERT INTO extraction_reviews(
                   extraction_version_id,source_id,source_version_id,state,generation,
                   decided_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                review.extraction_version_id,
                review.source_id,
                review.source_version_id,
                review.state,
                review.generation,
                review.decided_at,
                review.updated_at,
            ),
        )


def _verify_saved_result_matches(
    connection: sqlite3.Connection,
    extraction_id: str,
    pages: Sequence[ExtractionPage],
    visuals: Sequence[VisualLocation],
    outcome: ExtractionOutcome,
    review: ExtractionReview | None,
) -> None:
    """Retry discipline: a re-saved result must match the committed records exactly."""
    existing_outcome = _fetchone(
        connection,
        "SELECT * FROM extraction_outcomes WHERE extraction_version_id=?",
        (extraction_id,),
    )
    if existing_outcome is None or existing_outcome["outcome"] != outcome.outcome:
        raise ConflictingWriteError("extraction retry contains a different outcome")
    existing_pages = {
        row["id"]
        for row in connection.execute(
            "SELECT id FROM extraction_pages WHERE extraction_version_id=?",
            (extraction_id,),
        )
    }
    if existing_pages != {page.id for page in pages}:
        raise ConflictingWriteError("extraction retry contains a different page set")
    existing_visuals = {
        row["id"]
        for row in connection.execute(
            "SELECT id FROM extraction_visuals WHERE extraction_version_id=?",
            (extraction_id,),
        )
    }
    if existing_visuals != {visual.id for visual in visuals}:
        raise ConflictingWriteError("extraction retry contains a different visual set")
    existing_review = _fetchone(
        connection,
        "SELECT * FROM extraction_reviews WHERE extraction_version_id=?",
        (extraction_id,),
    )
    if (existing_review is None) != (review is None):
        raise ConflictingWriteError("extraction retry contains a different review state")


def _verify_review_currency(connection: sqlite3.Connection, review: ExtractionReview) -> None:
    """Confirm only against the candidate the user actually reviewed.

    Mirrors readiness publication plus the candidate's own import and processing state:
    the extraction's source version must still be current (otherwise it was superseded),
    neither the source nor the candidate import may be trashed, the candidate import must
    still be ready (cancelled or failed processing leaves it `failed`), and the candidate's
    processing job must not have been cancelled.
    """
    current_row = connection.execute(
        """SELECT id FROM source_versions WHERE source_id=?
           ORDER BY created_at DESC, rowid DESC LIMIT 1""",
        (review.source_id,),
    ).fetchone()
    if current_row is None or current_row[0] != review.source_version_id:
        raise ConflictingWriteError(
            "extraction source version is not current; confirmation rejected"
        )
    trash_row = _fetchone(
        connection, "SELECT 1 FROM source_trash WHERE source_id=?", (review.source_id,)
    )
    if trash_row is not None:
        raise ConflictingWriteError("source is trashed; confirmation rejected")
    import_row = _fetchone(
        connection,
        "SELECT id, state FROM pdf_imports WHERE source_version_id=?",
        (review.source_version_id,),
    )
    if import_row is not None:
        if import_row["state"] != "ready":
            raise ConflictingWriteError(
                f"candidate import is not ready ({import_row['state']}); confirmation rejected"
            )
        import_trash = _fetchone(
            connection, "SELECT 1 FROM pdf_import_trash WHERE import_id=?", (import_row["id"],)
        )
        if import_trash is not None:
            raise ConflictingWriteError("candidate import is trashed; confirmation rejected")
    job_row = _fetchone(
        connection,
        """SELECT job_status.status AS state FROM job_status
           JOIN jobs ON jobs.id = job_status.job_id
           WHERE jobs.source_version_id=? AND jobs.kind='pdf-extract'
           ORDER BY jobs.created_at DESC, jobs.rowid DESC LIMIT 1""",
        (review.source_version_id,),
    )
    if job_row is not None and job_row["state"] == "cancelled":
        raise ConflictingWriteError("candidate processing was cancelled; confirmation rejected")


def _verify_extraction_decision(
    connection: sqlite3.Connection, index_version_ids: Sequence[str]
) -> None:
    """Publication verifies the decision of the exact extraction the indexes came from.

    Each published index resolves to its extraction; when that extraction carries a
    decision record it must be `complete` or confirmed. An older complete extraction can
    never bless a newer unconfirmed partial one. Extractions without a decision record are
    pre-M1.5 records (1.2's shape, used by synthetic fixtures) and have nothing to enforce.
    """
    if not index_version_ids:
        return
    extraction_ids = {
        row[0]
        for row in connection.execute(
            f"""SELECT extraction_version_id FROM index_versions
                WHERE id IN ({",".join("?" for _ in index_version_ids)})""",
            tuple(index_version_ids),
        ).fetchall()
    }
    for extraction_id in extraction_ids:
        outcome = _fetchone(
            connection,
            "SELECT outcome FROM extraction_outcomes WHERE extraction_version_id=?",
            (extraction_id,),
        )
        if outcome is None:
            continue
        if outcome["outcome"] == "complete":
            continue
        review_row = _fetchone(
            connection,
            "SELECT state FROM extraction_reviews WHERE extraction_version_id=?",
            (extraction_id,),
        )
        if review_row is None or review_row["state"] != "confirmed":
            raise ConflictingWriteError(
                "extraction decision is missing or declined; publication rejected"
            )


def _save_extraction(
    connection: sqlite3.Connection,
    extraction: ExtractionVersion,
    passages: Sequence[Passage],
) -> None:
    _validate_extraction(extraction, passages)
    settings_json = encode_json(extraction.settings)
    values = {
        "source_id": extraction.source_id,
        "source_version_id": extraction.source_version_id,
        "parser_name": extraction.parser_name,
        "parser_version": extraction.parser_version,
        "normalization_version": extraction.normalization_version,
        "settings_format_version": extraction.settings_format_version,
        "settings_json": settings_json,
    }
    existing = connection.execute(
        "SELECT * FROM extraction_versions WHERE id=?", (extraction.id,)
    ).fetchone()
    if existing is not None:
        if any(existing[name] != value for name, value in values.items()):
            raise ConflictingWriteError("extraction identity already has different content")
        existing_ids = {
            row[0]
            for row in connection.execute(
                "SELECT id FROM passages WHERE extraction_version_id=?", (extraction.id,)
            )
        }
        if existing_ids != {passage.id for passage in passages}:
            raise ConflictingWriteError("extraction retry contains a different passage set")
    else:
        connection.execute(
            """INSERT INTO extraction_versions(
                   id,source_id,source_version_id,parser_name,parser_version,
                   normalization_version,settings_format_version,settings_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                extraction.id,
                extraction.source_id,
                extraction.source_version_id,
                extraction.parser_name,
                extraction.parser_version,
                extraction.normalization_version,
                extraction.settings_format_version,
                settings_json,
                extraction.created_at,
            ),
        )
    for passage in passages:
        _save_passage(connection, passage)


def _save_passage(connection: sqlite3.Connection, item: Passage) -> None:
    values: dict[str, object] = {
        "source_id": item.source_id,
        "source_version_id": item.source_version_id,
        "extraction_version_id": item.extraction_version_id,
        "ordinal": item.ordinal,
        "text": item.text,
        "location_format_version": item.location_format_version,
        "location_json": encode_json(item.location),
    }
    existing = _fetchone(connection, "SELECT * FROM passages WHERE id=?", (item.id,))
    if existing is not None:
        if any(existing[name] != value for name, value in values.items()):
            raise ConflictingWriteError("passage identity already has different evidence")
        return
    connection.execute(
        """INSERT INTO passages(
               id,source_id,source_version_id,extraction_version_id,ordinal,text,
               location_format_version,location_json,created_at
           ) VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            item.id,
            item.source_id,
            item.source_version_id,
            item.extraction_version_id,
            item.ordinal,
            item.text,
            item.location_format_version,
            values["location_json"],
            item.created_at,
        ),
    )


def _save_index(
    connection: sqlite3.Connection,
    index: IndexVersion,
    chunks: Sequence[RetrievalChunk],
    mappings: Sequence[ChunkPassage],
) -> None:
    _validate_index(index, chunks, mappings, connection)
    values: dict[str, object] = {
        "source_id": index.source_id,
        "source_version_id": index.source_version_id,
        "extraction_version_id": index.extraction_version_id,
        "index_kind": index.index_kind,
        "chunker_version": index.chunker_version,
        "settings_format_version": index.settings_format_version,
        "settings_json": encode_json(index.settings),
        "embedding_model": index.embedding_model,
        "embedding_revision": index.embedding_revision,
    }
    existing = _fetchone(connection, "SELECT * FROM index_versions WHERE id=?", (index.id,))
    if existing is not None:
        if any(existing[name] != value for name, value in values.items()):
            raise ConflictingWriteError("index identity already has different content")
        existing_chunks = {
            row[0]
            for row in connection.execute(
                "SELECT id FROM retrieval_chunks WHERE index_version_id=?", (index.id,)
            )
        }
        if existing_chunks != {chunk.id for chunk in chunks}:
            raise ConflictingWriteError("index retry contains a different chunk set")
        existing_mappings = {
            tuple(row)
            for row in connection.execute(
                """SELECT retrieval_chunk_id,passage_id,ordinal,start_offset,end_offset
                   FROM chunk_passages WHERE index_version_id=?""",
                (index.id,),
            )
        }
        expected_mappings = {
            (
                item.retrieval_chunk_id,
                item.passage_id,
                item.ordinal,
                item.start_offset,
                item.end_offset,
            )
            for item in mappings
        }
        if existing_mappings != expected_mappings:
            raise ConflictingWriteError("index retry contains different passage mappings")
    else:
        connection.execute(
            """INSERT INTO index_versions(
                   id,source_id,source_version_id,extraction_version_id,index_kind,
                   chunker_version,settings_format_version,settings_json,embedding_model,
                   embedding_revision,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                index.id,
                index.source_id,
                index.source_version_id,
                index.extraction_version_id,
                index.index_kind,
                index.chunker_version,
                index.settings_format_version,
                values["settings_json"],
                index.embedding_model,
                index.embedding_revision,
                index.created_at,
            ),
        )
    for chunk in chunks:
        _save_chunk(connection, chunk)
    for mapping in mappings:
        _save_mapping(connection, mapping)


def _save_chunk(connection: sqlite3.Connection, item: RetrievalChunk) -> None:
    values = (
        item.index_version_id,
        item.source_id,
        item.source_version_id,
        item.extraction_version_id,
        item.ordinal,
        item.text,
    )
    existing = _fetchone(connection, "SELECT * FROM retrieval_chunks WHERE id=?", (item.id,))
    if existing is not None:
        actual = tuple(
            existing[field]
            for field in (
                "index_version_id",
                "source_id",
                "source_version_id",
                "extraction_version_id",
                "ordinal",
                "text",
            )
        )
        if actual != values:
            raise ConflictingWriteError("retrieval-chunk identity already has different content")
        return
    connection.execute(
        """INSERT INTO retrieval_chunks(
               id,index_version_id,source_id,source_version_id,extraction_version_id,ordinal,text
           ) VALUES(?,?,?,?,?,?,?)""",
        (item.id, *values),
    )


def _save_mapping(connection: sqlite3.Connection, item: ChunkPassage) -> None:
    values = (
        item.index_version_id,
        item.source_id,
        item.source_version_id,
        item.extraction_version_id,
        item.ordinal,
        item.start_offset,
        item.end_offset,
    )
    existing = connection.execute(
        "SELECT * FROM chunk_passages WHERE retrieval_chunk_id=? AND passage_id=?",
        (item.retrieval_chunk_id, item.passage_id),
    ).fetchone()
    if existing is not None:
        actual = tuple(
            existing[field]
            for field in (
                "index_version_id",
                "source_id",
                "source_version_id",
                "extraction_version_id",
                "ordinal",
                "start_offset",
                "end_offset",
            )
        )
        if actual != values:
            raise ConflictingWriteError("chunk-to-passage mapping already has different content")
        return
    connection.execute(
        """INSERT INTO chunk_passages(
               retrieval_chunk_id,passage_id,index_version_id,source_id,source_version_id,
               extraction_version_id,ordinal,start_offset,end_offset
           ) VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            item.retrieval_chunk_id,
            item.passage_id,
            item.index_version_id,
            item.source_id,
            item.source_version_id,
            item.extraction_version_id,
            item.ordinal,
            item.start_offset,
            item.end_offset,
        ),
    )


def _save_job_step(
    connection: sqlite3.Connection,
    job_id: str,
    step_key: str,
    checkpoint_format_version: int,
    checkpoint: JSONValue,
    inputs_sha256: str | None = None,
    output_source_version_id: str | None = None,
    output_extraction_version_id: str | None = None,
    output_index_version_id: str | None = None,
) -> None:
    validate_id(job_id)
    step_key = _required_text(step_key, "job step key")
    _integer(checkpoint_format_version, "checkpoint format version", minimum=1)
    if inputs_sha256 is not None:
        validate_sha256(inputs_sha256)
    checkpoint_json = encode_json(checkpoint)
    existing = connection.execute(
        "SELECT * FROM job_steps WHERE job_id=? AND step_key=?", (job_id, step_key)
    ).fetchone()
    output_values = (
        output_source_version_id,
        output_extraction_version_id,
        output_index_version_id,
    )
    if existing is not None:
        actual = (
            existing["checkpoint_format_version"],
            existing["checkpoint_json"],
            existing["inputs_sha256"],
            existing["output_source_version_id"],
            existing["output_extraction_version_id"],
            existing["output_index_version_id"],
        )
        expected = (checkpoint_format_version, checkpoint_json, inputs_sha256, *output_values)
        if actual != expected:
            raise ConflictingWriteError("job step already has a different committed checkpoint")
        return
    connection.execute(
        """INSERT INTO job_steps(
               job_id,step_key,checkpoint_format_version,checkpoint_json,inputs_sha256,
               output_source_version_id,output_extraction_version_id,output_index_version_id,committed_at
           ) VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            job_id,
            step_key,
            checkpoint_format_version,
            checkpoint_json,
            inputs_sha256,
            *output_values,
            utc_timestamp(),
        ),
    )


def _verify_attempt_current(
    connection: sqlite3.Connection,
    job_id: str,
    step_key: str,
    guard: AttemptGuard,
) -> None:
    """Same-transaction proof that a commit or publication may be recorded.

    The job must be running, owned by the guarding instance, and the guarded
    attempt must be the newest unfinished attempt for that step.
    """
    if guard.step_key != step_key:
        raise ValueError("attempt guard does not match the step")
    status_row = _fetchone(
        connection, "SELECT status, owner_instance_id FROM job_status WHERE job_id=?", (job_id,)
    )
    if status_row is None or status_row["status"] != "running":
        raise ConflictingWriteError("job is not running; commit rejected")
    if status_row["owner_instance_id"] != guard.owner_instance_id:
        raise ConflictingWriteError("job is owned by another instance; commit rejected")
    attempt = _fetchone(
        connection,
        "SELECT * FROM job_step_attempts WHERE job_id=? AND step_key=? AND attempt_number=?",
        (job_id, step_key, guard.expected_attempt_number),
    )
    if attempt is None or attempt["finished_at"] is not None:
        raise ConflictingWriteError("step attempt is not open; commit rejected")
    latest = connection.execute(
        "SELECT COALESCE(MAX(attempt_number), 0) FROM job_step_attempts"
        " WHERE job_id=? AND step_key=?",
        (job_id, step_key),
    ).fetchone()[0]
    if int(latest) != guard.expected_attempt_number:
        raise ConflictingWriteError("a newer attempt supersedes this one; commit rejected")


def _load_extraction(
    connection: sqlite3.Connection,
    extraction_id: str,
) -> ExtractionGraph:
    row = _fetchone(
        connection,
        "SELECT * FROM extraction_versions WHERE id=?",
        (extraction_id,),
    )
    if row is None:
        raise PersistenceError("extraction version does not exist")
    extraction = _extraction(row)
    passages = tuple(
        _passage(item)
        for item in connection.execute(
            """SELECT * FROM passages
               WHERE extraction_version_id=? ORDER BY ordinal,id""",
            (extraction_id,),
        )
    )
    return ExtractionGraph(extraction, passages)


def _extraction_page(row: sqlite3.Row) -> ExtractionPage:
    return ExtractionPage(
        row["id"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["page_number"],
        bool(row["text_drawn"]),
        row["extracted_chars"],
        row["unmapped_ratio"],
        row["image_area_ratio"],
        row["classification"],
        row["created_at"],
    )


def _visual_location(row: sqlite3.Row) -> VisualLocation:
    return VisualLocation(
        row["id"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["page_number"],
        row["ordinal"],
        row["kind"],
        decode_json(row["bbox_json"]),
        row["created_at"],
    )


def _extraction_outcome(row: sqlite3.Row) -> ExtractionOutcome:
    return ExtractionOutcome(
        row["extraction_version_id"],
        row["source_id"],
        row["source_version_id"],
        row["outcome"],
        row["created_at"],
    )


def _extraction_review(row: sqlite3.Row) -> ExtractionReview:
    return ExtractionReview(
        row["extraction_version_id"],
        row["source_id"],
        row["source_version_id"],
        row["state"],
        row["generation"],
        row["decided_at"],
        row["updated_at"],
    )


def _load_extraction_report(
    connection: sqlite3.Connection,
    extraction_id: str,
) -> ExtractionReport:
    graph = _load_extraction(connection, extraction_id)
    pages = tuple(
        _extraction_page(item)
        for item in connection.execute(
            """SELECT * FROM extraction_pages
               WHERE extraction_version_id=? ORDER BY page_number,id""",
            (extraction_id,),
        )
    )
    visuals = tuple(
        _visual_location(item)
        for item in connection.execute(
            """SELECT * FROM extraction_visuals
               WHERE extraction_version_id=? ORDER BY page_number,ordinal,id""",
            (extraction_id,),
        )
    )
    outcome_row = _fetchone(
        connection,
        "SELECT * FROM extraction_outcomes WHERE extraction_version_id=?",
        (extraction_id,),
    )
    review_row = _fetchone(
        connection,
        "SELECT * FROM extraction_reviews WHERE extraction_version_id=?",
        (extraction_id,),
    )
    return ExtractionReport(
        graph.extraction,
        graph.passages,
        pages,
        visuals,
        None if outcome_row is None else _extraction_outcome(outcome_row),
        None if review_row is None else _extraction_review(review_row),
    )


def _load_index(connection: sqlite3.Connection, index_id: str) -> IndexGraph:
    row = connection.execute("SELECT * FROM index_versions WHERE id=?", (index_id,)).fetchone()
    if row is None:
        raise PersistenceError("index version does not exist")
    index = _index(row)
    chunks = tuple(
        _chunk(item)
        for item in connection.execute(
            """SELECT * FROM retrieval_chunks
               WHERE index_version_id=? ORDER BY ordinal,id""",
            (index_id,),
        )
    )
    mappings = tuple(
        _mapping(item)
        for item in connection.execute(
            """SELECT * FROM chunk_passages WHERE index_version_id=?
               ORDER BY retrieval_chunk_id,ordinal,passage_id""",
            (index_id,),
        )
    )
    return IndexGraph(index, chunks, mappings)


def _load_job_step(connection: sqlite3.Connection, job_id: str, step_key: str) -> JobStep:
    row = connection.execute(
        "SELECT * FROM job_steps WHERE job_id=? AND step_key=?", (job_id, step_key)
    ).fetchone()
    if row is None:
        raise PersistenceError("job step does not exist")
    return _job_step(row)


def _notebook(row: sqlite3.Row) -> Notebook:
    return Notebook(
        row["id"],
        row["title"],
        row["created_at"],
        row["updated_at"],
    )


def _source(row: sqlite3.Row) -> Source:
    return Source(
        row["id"],
        row["notebook_id"],
        row["display_name"],
        row["created_at"],
        row["updated_at"],
    )


def _managed_file(row: sqlite3.Row) -> ManagedFile:
    return ManagedFile(
        row["id"],
        row["relative_path"],
        row["size_bytes"],
        row["sha256"],
        row["created_at"],
    )


def _managed_file_from_join(row: sqlite3.Row) -> ManagedFile:
    return ManagedFile(
        row["managed_file_id"],
        row["relative_path"],
        row["size_bytes"],
        row["managed_sha256"],
        row["managed_created_at"],
    )


def _pdf_acquisition(row: sqlite3.Row) -> PdfAcquisition:
    return PdfAcquisition(
        row["id"],
        row["retry_key"],
        row["notebook_id"],
        row["intent"],
        row["target_source_id"],
        row["expected_current_import_id"],
        row["display_filename"],
        row["state"],
        int(row["finalization_generation"]),
        row["owner_instance_id"],
        row["staging_relative_path"],
        row["input_sha256"],
        row["input_size_bytes"],
        int(row["received_size_bytes"]),
        row["input_complete_at"],
        row["last_write_at"],
        row["planned_reuse_managed_file_id"],
        row["planned_managed_file_id"],
        row["planned_relative_path"],
        row["planned_sha256"],
        row["planned_size_bytes"],
        row["placement_state"],
        row["outcome_kind"],
        row["outcome_source_id"],
        row["outcome_import_id"],
        row["failure_code"],
        row["created_at"],
        row["updated_at"],
    )


def _pdf_import(row: sqlite3.Row) -> PdfImport:
    return PdfImport(
        row["id"],
        row["source_id"],
        row["source_version_id"],
        row["acquisition_id"],
        row["display_filename"],
        row["state"],
        row["failure_code"],
        row["created_at"],
        row["updated_at"],
    )


def _pdf_import_from_join(row: sqlite3.Row) -> PdfImport:
    return PdfImport(
        row["import_id"],
        row["source_id"],
        row["source_version_id"],
        row["acquisition_id"],
        row["display_filename"],
        row["import_state"],
        row["failure_code"],
        row["import_created_at"],
        row["import_updated_at"],
    )


def _source_pdf_state(row: sqlite3.Row) -> SourcePdfState:
    return SourcePdfState(
        row["source_id"],
        row["current_import_id"],
        row["pending_import_id"],
        int(row["revision"]),
        row["updated_at"],
    )


def _pdf_import_trash(row: sqlite3.Row) -> PdfImportTrash:
    return PdfImportTrash(
        row["import_id"],
        row["source_id"],
        row["trashed_at"],
        row["purge_deadline_at"],
        row["file_removed_at"],
        row["snapshots_removed_at"],
        row["updated_at"],
    )


def _source_version(row: sqlite3.Row) -> SourceVersion:
    return SourceVersion(
        row["id"],
        row["source_id"],
        row["managed_file_id"],
        row["content_sha256"],
        row["created_at"],
    )


def _extraction(row: sqlite3.Row) -> ExtractionVersion:
    return ExtractionVersion(
        row["id"],
        row["source_id"],
        row["source_version_id"],
        row["parser_name"],
        row["parser_version"],
        row["normalization_version"],
        row["settings_format_version"],
        decode_json(row["settings_json"]),
        row["created_at"],
    )


def _passage(row: sqlite3.Row) -> Passage:
    return Passage(
        row["id"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["ordinal"],
        row["text"],
        row["location_format_version"],
        decode_json(row["location_json"]),
        row["created_at"],
    )


def _index(row: sqlite3.Row) -> IndexVersion:
    return IndexVersion(
        row["id"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["index_kind"],
        row["chunker_version"],
        row["settings_format_version"],
        decode_json(row["settings_json"]),
        row["embedding_model"],
        row["embedding_revision"],
        row["created_at"],
    )


def _chunk(row: sqlite3.Row) -> RetrievalChunk:
    return RetrievalChunk(
        row["id"],
        row["index_version_id"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["ordinal"],
        row["text"],
    )


def _mapping(row: sqlite3.Row) -> ChunkPassage:
    return ChunkPassage(
        row["retrieval_chunk_id"],
        row["passage_id"],
        row["index_version_id"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["ordinal"],
        row["start_offset"],
        row["end_offset"],
    )


def _job(row: sqlite3.Row) -> Job:
    return Job(
        row["id"],
        row["logical_work_key"],
        row["kind"],
        row["source_id"],
        row["source_version_id"],
        row["extraction_version_id"],
        row["index_version_id"],
        row["created_at"],
        row["updated_at"],
    )


def _job_step(row: sqlite3.Row) -> JobStep:
    return JobStep(
        row["job_id"],
        row["step_key"],
        row["checkpoint_format_version"],
        decode_json(row["checkpoint_json"]),
        row["inputs_sha256"],
        row["output_source_version_id"],
        row["output_extraction_version_id"],
        row["output_index_version_id"],
        row["committed_at"],
    )


def _job_status(row: sqlite3.Row) -> JobStatus:
    policy_json = row["policy_json"]
    return JobStatus(
        row["job_id"],
        row["status"],
        row["current_step_key"],
        int(row["attempt_count"]),
        row["owner_instance_id"],
        row["claimed_at"],
        row["heartbeat_at"],
        row["failure_reason"],
        row["superseded_by_job_id"],
        None if policy_json is None else decode_json(policy_json),
        row["updated_at"],
        row["failure_code"],
    )


def _managed_file_removal(row: sqlite3.Row) -> ManagedFileRemoval:
    return ManagedFileRemoval(
        row["managed_file_id"],
        row["relative_path"],
        row["requested_at"],
        row["removed_at"],
        int(row["attempts"]),
        row["last_error"],
    )


def _job_step_attempt(row: sqlite3.Row) -> JobStepAttempt:
    return JobStepAttempt(
        row["job_id"],
        row["step_key"],
        int(row["attempt_number"]),
        row["started_at"],
        row["finished_at"],
        row["failure_reason"],
        row["outcome"],
    )


def _load_readiness(connection: sqlite3.Connection, source_id: str) -> SourceReadiness:
    row = _require_row(
        _fetchone(connection, "SELECT * FROM source_readiness WHERE source_id=?", (source_id,))
    )
    index_rows = connection.execute(
        """SELECT index_version_id FROM source_readiness_indexes
           WHERE source_id=? ORDER BY index_version_id""",
        (source_id,),
    ).fetchall()
    return SourceReadiness(
        row["source_id"],
        row["source_version_id"],
        tuple(index_row[0] for index_row in index_rows),
        row["published_by_job_id"],
        row["published_at"],
        row["updated_at"],
    )


def _source_trash(row: sqlite3.Row) -> SourceTrash:
    return SourceTrash(
        row["source_id"],
        row["trashed_at"],
        row["purge_deadline_at"],
        row["files_removed_at"],
        row["snapshots_removed_at"],
        row["updated_at"],
    )
