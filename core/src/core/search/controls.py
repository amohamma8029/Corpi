"""Explicit source-bound cancellation/resumption; neither grants a fresh source budget."""

from __future__ import annotations

import json
from types import SimpleNamespace

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import ExtractionReview, new_id, utc_timestamp
from core.persistence.store import PersistenceStore, _verify_review_currency
from core.search.handoff import invalidate_handoffs
from core.search.profiles import profiles
from core.search.storage import generation_identity


class PreparationControl:
    def __init__(self, store: PersistenceStore, *, notify=lambda: None) -> None:
        self.store, self.notify = store, notify

    @staticmethod
    def _current(connection, source_id: str, job_id: str):
        row = connection.execute(
            """SELECT h.*,s.status,s.failure_code,j.source_id,j.source_version_id,
                      i.profile_id,i.settings_json,i.spent_seconds
               FROM search_handoffs h JOIN jobs j ON j.id=h.job_id
               JOIN job_status s ON s.job_id=j.id JOIN search_job_inputs i ON i.job_id=j.id
               WHERE j.source_id=? AND h.purpose='import'
               ORDER BY j.created_at DESC,j.rowid DESC LIMIT 1""",
            (source_id,),
        ).fetchone()
        if row is None or row["job_id"] != job_id:
            raise ConflictingWriteError("search preparation changed; refresh its progress")
        return row

    def cancel(self, source_id: str, job_id: str) -> str:
        self.store.load_source(source_id)
        with self.store._transaction() as connection:
            row = self._current(connection, source_id, job_id)
            if row["state"] == "cancelled":
                return job_id
            if row["status"] not in {"queued", "running"}:
                raise ConflictingWriteError("search preparation is no longer cancellable")
            invalidate_handoffs(connection, job_id=job_id, reason="search preparation cancelled")
        self.notify()
        return job_id

    @classmethod
    def can_resume(cls, connection, source_id: str, job_id: str) -> bool:
        try:
            cls._resumable(connection, source_id, job_id)
        except ConflictingWriteError:
            return False
        return True

    @classmethod
    def _resumable(cls, connection, source_id: str, job_id: str):
        row = cls._current(connection, source_id, job_id)
        if row["status"] not in {"cancelled", "failed"}:
            raise ConflictingWriteError("search preparation is not stopped")
        if row["failure_code"] in {
            "search_time_limit",
            "embedding_attempt_limit",
            "embedding_timeout",
        }:
            raise ConflictingWriteError("saved search allowance or attempt limit is exhausted")
        current = connection.execute(
            """SELECT p.active_manifest_id,p.selection_generation,
                          e.source_version_id,o.outcome,r.state,
                          COALESCE(r.generation,0) AS review_generation
                   FROM extraction_versions e JOIN sources s ON s.id=e.source_id
                   JOIN notebook_profiles p ON p.notebook_id=s.notebook_id
                   JOIN extraction_outcomes o ON o.extraction_version_id=e.id
                   LEFT JOIN extraction_reviews r ON r.extraction_version_id=e.id
                   WHERE e.id=? AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)
                     AND e.id=(SELECT latest.id FROM extraction_versions latest
                        WHERE latest.source_version_id=e.source_version_id
                        ORDER BY latest.created_at DESC,latest.rowid DESC LIMIT 1)""",
            (row["extraction_version_id"],),
        ).fetchone()
        if (
            current is None
            or current["active_manifest_id"] != row["manifest_id"]
            or current["selection_generation"] != row["selection_generation"]
            or current["source_version_id"] != row["source_version_id"]
            or current["review_generation"] != row["review_generation"]
            or not (
                current["outcome"] == "complete"
                or (current["outcome"] == "partial" and current["state"] == "confirmed")
            )
        ):
            raise ConflictingWriteError("source, review or notebook model changed")
        _verify_review_currency(
            connection,
            ExtractionReview(
                row["extraction_version_id"],
                source_id,
                row["source_version_id"],
                "confirmed",
                max(1, row["review_generation"]),
                None,
                utc_timestamp(),
            ),
        )
        settings = json.loads(row["settings_json"])
        identity = generation_identity(
            SimpleNamespace(
                source_id=source_id,
                source_version_id=row["source_version_id"],
                extraction_version_id=row["extraction_version_id"],
            ),
            profiles()[row["profile_id"]],
            settings,
        )
        generation = connection.execute(
            "SELECT index_version_id,spent_seconds FROM search_generations WHERE identity_sha256=?",
            (identity,),
        ).fetchone()
        spent = row["spent_seconds"] + (generation["spent_seconds"] if generation else 0)
        if spent >= settings["total_seconds"]:
            raise ConflictingWriteError("saved search preparation allowance is exhausted")
        if (
            generation
            and connection.execute(
                "SELECT 1 FROM search_batches WHERE index_version_id=? "
                "AND complete=0 AND attempts>=2 LIMIT 1",
                (generation["index_version_id"],),
            ).fetchone()
        ):
            raise ConflictingWriteError("saved search batch attempts are exhausted")
        return row

    def resume(self, source_id: str, job_id: str, request_key: str) -> str:
        self.store.load_source(source_id)
        key = f"pdf-search-resume:{job_id}:{request_key}"
        with self.store._transaction() as connection:
            previous = connection.execute(
                "SELECT id,source_id FROM jobs WHERE logical_work_key=?", (key,)
            ).fetchone()
            if previous is not None:
                if previous["source_id"] != source_id:
                    raise ConflictingWriteError("resume request belongs to another source")
                return previous["id"]
            row = self._resumable(connection, source_id, job_id)
            resumed, timestamp = new_id(), utc_timestamp()
            connection.execute(
                """INSERT INTO jobs(id,logical_work_key,kind,source_id,source_version_id,
                   extraction_version_id,created_at,updated_at)
                   VALUES(?,?,'pdf-search',?,?,?,?,?)""",
                (
                    resumed,
                    key,
                    source_id,
                    row["source_version_id"],
                    row["extraction_version_id"],
                    timestamp,
                    timestamp,
                ),
            )
            # Keep immutable old inputs and conservative pre-planning charges.
            # Existing generations/batches carry their own ledger; no row is reset.
            connection.execute(
                """INSERT INTO search_job_inputs(job_id,profile_id,settings_json,review_generation,
                   spent_seconds) VALUES(?,?,?,?,?)""",
                (
                    resumed,
                    row["profile_id"],
                    row["settings_json"],
                    row["review_generation"],
                    row["spent_seconds"],
                ),
            )
            old_policy = connection.execute(
                "SELECT policy_json FROM job_status WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO job_status(job_id,status,policy_json,updated_at) "
                "VALUES(?,'queued',?,?)",
                (resumed, old_policy, timestamp),
            )
            connection.execute(
                "UPDATE search_handoffs SET job_id=?,state='scheduled',waiting_reason=NULL "
                "WHERE job_id=?",
                (resumed, job_id),
            )
        self.notify()
        return resumed
