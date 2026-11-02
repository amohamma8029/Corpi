"""Extraction-to-search ownership and scheduling in existing store transactions.

Missing models leave source-bound jobs queued; the existing runner skips them.
No inference, download, or model-file hashing occurs while holding a database lock.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import ExtractionReview, Job, utc_timestamp
from core.search.preparation import SearchLimits, enqueue_search_job
from core.search.profiles import ModelProfile, profiles
from core.search.representation import ARABIC, CANONICAL

# Implementation choices from measured comparisons; final acceptance still
# requires the broader benchmark and platform evidence recorded in the Task.
DEFAULT_PROFILE_ID = "bekko-a8m-quint8-avx2"
ENGLISH_PROFILE_ID = "minilm-en"
ALTERNATIVE_PROFILE_ID = "multilingual-e5"
SUPPORTED_PROFILE_IDS = (DEFAULT_PROFILE_ID, ENGLISH_PROFILE_ID, ALTERNATIVE_PROFILE_ID)


def preparation_limits(profile: ModelProfile) -> SearchLimits:
    return SearchLimits(
        batch_size=1
        if profile.id == DEFAULT_PROFILE_ID
        else 8
        if profile.id == ALTERNATIVE_PROFILE_ID
        else 32
    )


def preparation_representation(profile: ModelProfile) -> str:
    return ARABIC if profile.id == DEFAULT_PROFILE_ID else CANONICAL


def ensure_notebook_profile(connection: sqlite3.Connection, notebook_id: str) -> sqlite3.Row:
    connection.execute(
        """INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)
           VALUES(?,?,?) ON CONFLICT(notebook_id) DO NOTHING""",
        (notebook_id, profiles()[DEFAULT_PROFILE_ID].identity, utc_timestamp()),
    )
    return connection.execute(
        "SELECT * FROM notebook_profiles WHERE notebook_id=?", (notebook_id,)
    ).fetchone()


def _origin(connection: sqlite3.Connection, extraction_id: str) -> str | None:
    row = connection.execute(
        """SELECT j.id FROM jobs j JOIN job_steps s ON s.job_id=j.id
           WHERE s.output_extraction_version_id=? AND j.kind='pdf-extract'
           ORDER BY j.created_at DESC,j.rowid DESC LIMIT 1""",
        (extraction_id,),
    ).fetchone()
    return None if row is None else row[0]


def record_handoff(
    connection: sqlite3.Connection,
    extraction_id: str,
    *,
    origin_job_id: str | None = None,
) -> str | None:
    """Atomically enqueue eligible work; cancelled origins never resurrect it."""
    from core.persistence.store import _verify_review_currency

    row = connection.execute(
        """SELECT e.*,s.notebook_id,o.outcome,r.state AS review_state,
                  COALESCE(r.generation,0) AS review_generation
           FROM extraction_versions e JOIN sources s ON s.id=e.source_id
           JOIN extraction_outcomes o ON o.extraction_version_id=e.id
           LEFT JOIN extraction_reviews r ON r.extraction_version_id=e.id WHERE e.id=?""",
        (extraction_id,),
    ).fetchone()
    if row is None or not (
        row["outcome"] == "complete"
        or (row["outcome"] == "partial" and row["review_state"] == "confirmed")
    ):
        return None
    latest = connection.execute(
        "SELECT id FROM extraction_versions WHERE source_version_id=? "
        "ORDER BY created_at DESC,rowid DESC LIMIT 1",
        (row["source_version_id"],),
    ).fetchone()
    if latest is None or latest[0] != extraction_id:
        return None
    try:
        _verify_review_currency(
            connection,
            ExtractionReview(
                extraction_id,
                row["source_id"],
                row["source_version_id"],
                "confirmed",
                max(1, row["review_generation"]),
                None,
                utc_timestamp(),
            ),
        )
    except ConflictingWriteError:
        return None
    origin_job_id = origin_job_id or _origin(connection, extraction_id)
    if origin_job_id is not None:
        origin = connection.execute(
            "SELECT s.status,s.superseded_by_job_id,j.kind FROM job_status s"
            " JOIN jobs j ON j.id=s.job_id"
            " WHERE j.id=?",
            (origin_job_id,),
        ).fetchone()
        if (
            origin is None
            or origin["kind"] != "pdf-extract"
            or origin["status"] in ("failed", "cancelled")
            or origin["superseded_by_job_id"]
        ):
            return None
    notebook = ensure_notebook_profile(connection, row["notebook_id"])
    candidates = profiles()
    profile = next(
        (
            candidates[key]
            for key in SUPPORTED_PROFILE_IDS
            if candidates[key].identity == notebook["active_manifest_id"]
        ),
        None,
    )
    if profile is None:
        return None
    key = (
        extraction_id,
        row["review_generation"],
        profile.identity,
        notebook["selection_generation"],
        "import",
    )
    existing = connection.execute(
        """SELECT * FROM search_handoffs WHERE extraction_version_id=? AND review_generation=?
           AND manifest_id=? AND selection_generation=? AND purpose=?""",
        key,
    ).fetchone()
    if existing is not None:
        if existing["state"] == "cancelled":
            # Only an explicit, newer extraction retry may restart cancelled work.
            new_origin = connection.execute(
                "SELECT logical_work_key FROM jobs WHERE id=?", (origin_job_id,)
            ).fetchone()
            if (
                origin_job_id == existing["origin_job_id"]
                or new_origin is None
                or not new_origin[0].startswith("pdf-extract-retry:")
            ):
                return None
        elif existing["job_id"] is not None:
            return existing["job_id"]
    limits = preparation_limits(profile)
    job_id = enqueue_search_job(
        connection,
        row["source_id"],
        row["source_version_id"],
        extraction_id,
        profile,
        row["review_generation"],
        limits=limits,
        representation=preparation_representation(profile),
        work_suffix=origin_job_id or "saved-extraction",
    )
    installed = connection.execute(
        "SELECT state FROM model_installations WHERE manifest_id=?",
        (profile.identity,),
    ).fetchone()
    reason = (
        None if installed is not None and installed[0] == "installed" else "model_not_installed"
    )
    if existing is None:
        connection.execute(
            """INSERT INTO search_handoffs(extraction_version_id,review_generation,manifest_id,
               selection_generation,purpose,state,job_id,waiting_reason,origin_job_id)
               VALUES(?,?,?,?,?,'scheduled',?,?,?)""",
            (*key, job_id, reason, origin_job_id),
        )
    else:
        connection.execute(
            """UPDATE search_handoffs
               SET state='scheduled',job_id=?,waiting_reason=?,origin_job_id=?
               WHERE extraction_version_id=? AND review_generation=? AND manifest_id=?
                 AND selection_generation=? AND purpose=?""",
            (job_id, reason, origin_job_id, *key),
        )
    return job_id


def invalidate_handoffs(
    connection: sqlite3.Connection,
    *,
    extraction_id: str | None = None,
    source_version_id: str | None = None,
    job_id: str | None = None,
    reason: str = "search input invalidated",
) -> None:
    conditions = []
    parameters = []
    if extraction_id:
        conditions.append("h.extraction_version_id=?")
        parameters.append(extraction_id)
    if source_version_id:
        conditions.append("e.source_version_id=?")
        parameters.append(source_version_id)
    if job_id:
        conditions.append("(h.job_id=? OR h.origin_job_id=?)")
        parameters.extend((job_id, job_id))
    if not conditions:
        raise ValueError("search invalidation requires an exact target")
    rows = connection.execute(
        f"""SELECT h.* FROM search_handoffs h
            JOIN extraction_versions e ON e.id=h.extraction_version_id
            WHERE {" AND ".join(conditions)}""",
        tuple(parameters),
    ).fetchall()
    for row in rows:
        connection.execute(
            """UPDATE search_handoffs SET state='cancelled',waiting_reason=?
               WHERE extraction_version_id=? AND review_generation=? AND manifest_id=?
                 AND selection_generation=? AND purpose=?""",
            (
                reason,
                row["extraction_version_id"],
                row["review_generation"],
                row["manifest_id"],
                row["selection_generation"],
                row["purpose"],
            ),
        )
        if row["job_id"] is not None:
            connection.execute(
                """UPDATE job_status SET status='cancelled',failure_reason=?,updated_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (reason, utc_timestamp(), row["job_id"]),
            )


class SearchCoordinator:
    def __init__(self, store, models_directory: Path) -> None:
        self.store = store
        self.models_directory = models_directory

    def reconcile(self) -> None:
        """Startup catch-up for eligible saved extractions, preserving tombstones."""
        with self.store._transaction() as connection:
            identities = connection.execute(
                """SELECT e.id FROM extraction_versions e
                   JOIN extraction_outcomes o ON o.extraction_version_id=e.id
                   LEFT JOIN extraction_reviews r ON r.extraction_version_id=e.id
                   WHERE o.outcome='complete' OR (o.outcome='partial' AND r.state='confirmed')""",
            ).fetchall()
            for row in identities:
                record_handoff(connection, row[0])
            from core.search.switching import sync_pending_switch

            notebooks = connection.execute(
                "SELECT notebook_id FROM notebook_profiles WHERE pending_manifest_id IS NOT NULL"
            ).fetchall()
            for notebook in notebooks:
                sync_pending_switch(connection, notebook[0])

    def claim_blocking_reason(self, job: Job) -> str | None:
        if job.kind != "pdf-search":
            return None
        with self.store._transaction() as connection:
            row = connection.execute(
                """SELECT h.*,m.state AS model_state FROM search_handoffs h
                   LEFT JOIN model_installations m ON m.manifest_id=h.manifest_id
                   WHERE h.job_id=?""",
                (job.id,),
            ).fetchone()
            if row is None:
                return "search_handoff_missing"
            if row["state"] == "cancelled":
                return "search_handoff_cancelled"
            if row["purpose"] == "model-switch":
                from core.search.switching import switch_current

                if not switch_current(connection, row):
                    invalidate_handoffs(connection, job_id=job.id, reason="model selection changed")
                    return "search_input_changed"
                reason = None if row["model_state"] == "installed" else "model_not_installed"
                connection.execute(
                    "UPDATE search_handoffs SET waiting_reason=? WHERE job_id=?", (reason, job.id)
                )
                return reason
            from core.persistence.store import _verify_review_currency

            extraction = connection.execute(
                "SELECT * FROM extraction_versions WHERE id=?", (row["extraction_version_id"],)
            ).fetchone()
            if extraction is None:
                return "search_handoff_missing"
            review = connection.execute(
                "SELECT state,generation FROM extraction_reviews WHERE extraction_version_id=?",
                (row["extraction_version_id"],),
            ).fetchone()
            current_profile = connection.execute(
                "SELECT p.active_manifest_id FROM notebook_profiles p"
                " JOIN sources s ON s.notebook_id=p.notebook_id WHERE s.id=?",
                (extraction["source_id"],),
            ).fetchone()
            try:
                _verify_review_currency(
                    connection,
                    ExtractionReview(
                        extraction["id"],
                        extraction["source_id"],
                        extraction["source_version_id"],
                        "confirmed",
                        max(1, row["review_generation"]),
                        None,
                        utc_timestamp(),
                    ),
                )
                if row["review_generation"] and (
                    review is None
                    or review["state"] != "confirmed"
                    or review["generation"] != row["review_generation"]
                ):
                    raise ConflictingWriteError("review changed")
                if current_profile is None or current_profile[0] != row["manifest_id"]:
                    raise ConflictingWriteError("model changed")
                latest = connection.execute(
                    "SELECT id FROM extraction_versions WHERE source_version_id=? "
                    "ORDER BY created_at DESC,rowid DESC LIMIT 1",
                    (extraction["source_version_id"],),
                ).fetchone()
                if latest is None or latest[0] != extraction["id"]:
                    raise ConflictingWriteError("extraction changed")
            except ConflictingWriteError:
                invalidate_handoffs(connection, job_id=job.id, reason="search input changed")
                return "search_input_changed"
            reason = None if row["model_state"] == "installed" else "model_not_installed"
            connection.execute(
                "UPDATE search_handoffs SET waiting_reason=? WHERE job_id=?", (reason, job.id)
            )
            return reason
