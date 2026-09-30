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
    PersistenceSnapshot,
    RetrievalChunk,
    Source,
    SourceReadiness,
    SourceTrash,
    SourceVersion,
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
    ) -> JobStep:
        with self._transaction() as connection:
            _verify_attempt_current(connection, job_id, step_key, guard)
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
    ) -> JobStatus:
        if status not in ("completed", "failed", "cancelled"):
            raise ValueError("job can only finish as completed, failed, or cancelled")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_status SET status=?, failure_reason=?, updated_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (status, failure_reason, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("job is not active; finish rejected")
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
    ) -> JobStepAttempt:
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_step_attempts SET finished_at=?, failure_reason=?
                   WHERE job_id=? AND step_key=? AND attempt_number=? AND finished_at IS NULL""",
                (timestamp, failure_reason, job_id, step_key, attempt_number),
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
    ) -> SourceReadiness:
        """Publish readiness only after verifying the publishing job in-transaction.

        The job must still be running, belong to this source, target the
        source's current version, and have committed every required index as
        one of its own step outputs. A trashed source can never publish, and
        cancelling the job later hides exactly the readiness it published.
        """
        validate_id(job_id)
        validate_id(source_id)
        validate_id(source_version_id)
        cleaned_indexes = tuple(dict.fromkeys(validate_id(item) for item in index_version_ids))
        if not cleaned_indexes:
            raise ValueError("readiness requires at least one finished index")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
            status_row = _fetchone(
                connection, "SELECT status FROM job_status WHERE job_id=?", (job_id,)
            )
            if status_row is None or status_row["status"] != "running":
                raise ConflictingWriteError("job is not running; publication rejected")
            job_row = _fetchone(
                connection,
                "SELECT source_id, source_version_id FROM jobs WHERE id=?",
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
            managed_file_ids = [
                row[0]
                for row in connection.execute(
                    """SELECT DISTINCT managed_file_id FROM source_versions
                       WHERE source_id=? AND managed_file_id IS NOT NULL""",
                    (source_id,),
                ).fetchall()
            ]
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
            connection.execute("DELETE FROM extraction_versions WHERE source_id=?", (source_id,))
            connection.execute("DELETE FROM source_versions WHERE source_id=?", (source_id,))
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

    def managed_file_shared_with_other_sources(self, managed_file_id: str, source_id: str) -> bool:
        """True when any other source's versions still reference this file."""
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM source_versions
                   WHERE managed_file_id=? AND source_id<>? LIMIT 1""",
                (managed_file_id, source_id),
            ).fetchone()
            return row is not None

    def ensure_managed_file_removal(
        self, managed_file_id: str, relative_path: str
    ) -> ManagedFileRemoval:
        """Create the durable removal marker that blocks new references.

        Idempotent: an existing marker (including a failed earlier attempt)
        keeps its history and state.
        """
        relative_path = _required_text(relative_path, "relative path")
        timestamp = utc_timestamp()
        with self._transaction() as connection:
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
    """Same-transaction proof that a commit may be recorded.

    The job must be running, owned by the guarding instance, and the guarded
    attempt must be the newest unfinished attempt for that step.
    """
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
        None if policy_json is None else decode_json(policy_json),
        row["updated_at"],
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
