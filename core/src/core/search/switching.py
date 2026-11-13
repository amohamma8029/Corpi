"""Coexisting model rebuilds and one guarded notebook-level activation."""

from __future__ import annotations

import sqlite3
import time

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard, JSONValue, utc_timestamp
from core.persistence.store import (
    PersistenceStore,
    _save_job_step,
    _verify_attempt_current,
    _verify_extraction_decision,
)
from core.search.handoff import (
    SUPPORTED_PROFILE_IDS,
    ensure_notebook_profile,
    invalidate_handoffs,
    preparation_limits,
    preparation_representation,
)
from core.search.preparation import enqueue_search_job
from core.search.profiles import profiles
from core.search.storage import verify_generation


def ready_sources(connection: sqlite3.Connection, notebook_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        """SELECT DISTINCT r.source_id,r.source_version_id,i.extraction_version_id
           FROM source_readiness r JOIN sources s ON s.id=r.source_id
           JOIN source_readiness_indexes ri ON ri.source_id=r.source_id
           JOIN index_versions i ON i.id=ri.index_version_id
           JOIN extraction_outcomes o ON o.extraction_version_id=i.extraction_version_id
           LEFT JOIN extraction_reviews review
             ON review.extraction_version_id=i.extraction_version_id
           WHERE s.notebook_id=? AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)
             AND (o.outcome='complete' OR (o.outcome='partial' AND review.state='confirmed'))""",
        (notebook_id,),
    ).fetchall()


def switch_current(connection: sqlite3.Connection, handoff: sqlite3.Row) -> bool:
    row = connection.execute(
        """SELECT p.pending_manifest_id,p.selection_generation,r.source_version_id
           FROM extraction_versions e JOIN sources s ON s.id=e.source_id
           JOIN notebook_profiles p ON p.notebook_id=s.notebook_id
           LEFT JOIN source_readiness r ON r.source_id=s.id WHERE e.id=?""",
        (handoff["extraction_version_id"],),
    ).fetchone()
    extraction = connection.execute(
        "SELECT source_version_id FROM extraction_versions WHERE id=?",
        (handoff["extraction_version_id"],),
    ).fetchone()
    exact_ready = connection.execute(
        """SELECT 1 FROM source_readiness_indexes ri
           JOIN index_versions i ON i.id=ri.index_version_id
           JOIN extraction_versions e ON e.id=? WHERE ri.source_id=e.source_id
             AND i.extraction_version_id=e.id""",
        (handoff["extraction_version_id"],),
    ).fetchone()
    review = connection.execute(
        "SELECT state,generation FROM extraction_reviews WHERE extraction_version_id=?",
        (handoff["extraction_version_id"],),
    ).fetchone()
    return bool(
        row
        and extraction
        and exact_ready
        and row["pending_manifest_id"] == handoff["manifest_id"]
        and row["selection_generation"] == handoff["selection_generation"]
        and row["source_version_id"] == extraction[0]
        and (
            not handoff["review_generation"]
            or (review and review[0] == "confirmed" and review[1] == handoff["review_generation"])
        )
    )


def activate_if_complete(connection: sqlite3.Connection, notebook_id: str) -> bool:
    notebook = connection.execute(
        "SELECT * FROM notebook_profiles WHERE notebook_id=?", (notebook_id,)
    ).fetchone()
    if notebook is None or notebook["pending_manifest_id"] is None:
        return False
    sources = ready_sources(connection, notebook_id)
    publications = []
    for source in sources:
        publication = connection.execute(
            """SELECT p.*,s.status FROM profile_search_publications p
               JOIN job_status s ON s.job_id=p.published_by_job_id
               JOIN index_versions i ON i.id=p.index_version_id
               WHERE p.source_id=? AND p.source_version_id=? AND p.manifest_id=?
                 AND p.selection_generation=? AND i.extraction_version_id=?
                 AND s.status IN ('running','completed')""",
            (
                source["source_id"],
                source["source_version_id"],
                notebook["pending_manifest_id"],
                notebook["selection_generation"],
                source["extraction_version_id"],
            ),
        ).fetchone()
        if publication is None:
            return False
        handoff = connection.execute(
            "SELECT * FROM search_handoffs WHERE job_id=? AND purpose='model-switch'",
            (publication["published_by_job_id"],),
        ).fetchone()
        # A completed job's status survives later input invalidation. Recheck
        # its handoff and exact review generation before counting its output.
        if (
            handoff is None
            or handoff["state"] == "cancelled"
            or not switch_current(connection, handoff)
        ):
            return False
        verify_generation(connection, publication["index_version_id"])
        _verify_extraction_decision(connection, [publication["index_version_id"]])
        publications.append(publication)
    model = connection.execute(
        "SELECT state FROM model_installations WHERE manifest_id=?",
        (notebook["pending_manifest_id"],),
    ).fetchone()
    if model is None or model[0] != "installed":
        return False
    for publication in publications:
        connection.execute(
            "DELETE FROM source_readiness_indexes WHERE source_id=?", (publication["source_id"],)
        )
        connection.execute(
            "INSERT INTO source_readiness_indexes(source_id,index_version_id) VALUES(?,?)",
            (publication["source_id"], publication["index_version_id"]),
        )
        connection.execute(
            "UPDATE source_readiness SET published_by_job_id=?,updated_at=?"
            " WHERE source_id=? AND source_version_id=?",
            (
                publication["published_by_job_id"],
                utc_timestamp(),
                publication["source_id"],
                publication["source_version_id"],
            ),
        )
    connection.execute(
        "UPDATE notebook_profiles SET active_manifest_id=pending_manifest_id,"
        "pending_manifest_id=NULL,updated_at=? WHERE notebook_id=?",
        (utc_timestamp(), notebook_id),
    )
    connection.execute(
        "UPDATE notebook_model_requests SET state='active'"
        " WHERE notebook_id=? AND generation=? AND state='pending'",
        (notebook_id, notebook["selection_generation"]),
    )
    # Unready imports scheduled under the prior active profile need current-model work.
    imports = connection.execute(
        """SELECT h.job_id FROM search_handoffs h
           JOIN extraction_versions e ON e.id=h.extraction_version_id
           JOIN sources s ON s.id=e.source_id WHERE s.notebook_id=? AND h.purpose='import'
             AND h.manifest_id<>? AND h.state<>'cancelled'""",
        (notebook_id, notebook["pending_manifest_id"]),
    ).fetchall()
    for handoff in imports:
        status = connection.execute(
            "SELECT status FROM job_status WHERE job_id=?", (handoff[0],)
        ).fetchone()
        if status and status[0] in ("queued", "running"):
            invalidate_handoffs(connection, job_id=handoff[0], reason="notebook model activated")
    return True


def sync_pending_switch(connection: sqlite3.Connection, notebook_id: str) -> None:
    notebook = connection.execute(
        "SELECT * FROM notebook_profiles WHERE notebook_id=?", (notebook_id,)
    ).fetchone()
    if notebook is None or notebook["pending_manifest_id"] is None:
        return
    profile = next(
        (
            profiles()[key]
            for key in SUPPORTED_PROFILE_IDS
            if profiles()[key].identity == notebook["pending_manifest_id"]
        ),
        None,
    )
    if profile is None:
        return
    generation = notebook["selection_generation"]
    for source in ready_sources(connection, notebook_id):
        review = connection.execute(
            "SELECT generation FROM extraction_reviews WHERE extraction_version_id=?",
            (source["extraction_version_id"],),
        ).fetchone()
        review_generation = review[0] if review else 0
        key = (
            source["extraction_version_id"],
            review_generation,
            profile.identity,
            generation,
            "model-switch",
        )
        existing = connection.execute(
            "SELECT state FROM search_handoffs WHERE extraction_version_id=?"
            " AND review_generation=?"
            " AND manifest_id=? AND selection_generation=? AND purpose=?",
            key,
        ).fetchone()
        if existing is not None:
            continue
        job = enqueue_search_job(
            connection,
            source["source_id"],
            source["source_version_id"],
            source["extraction_version_id"],
            profile,
            review_generation,
            limits=preparation_limits(profile),
            representation=preparation_representation(profile),
            work_suffix=f"model-switch:{notebook_id}:{generation}",
        )
        connection.execute(
            "INSERT INTO search_handoffs(extraction_version_id,review_generation,manifest_id,"
            "selection_generation,purpose,state,job_id,waiting_reason)"
            " VALUES(?,?,?,?,?,'scheduled',?,'model_rebuild')",
            (*key, job),
        )
    activate_if_complete(connection, notebook_id)


def publish_replacement(
    store: PersistenceStore,
    job_id: str,
    guard: AttemptGuard,
    index_id: str,
    *,
    deadline: float | None = None,
    checkpoint: tuple[int, JSONValue, str] | None = None,
) -> None:
    with store._transaction() as connection:
        _verify_attempt_current(connection, job_id, guard.step_key, guard)
        handoff = connection.execute(
            "SELECT * FROM search_handoffs WHERE job_id=? AND purpose='model-switch'", (job_id,)
        ).fetchone()
        if (
            handoff is None
            or handoff["state"] == "cancelled"
            or not switch_current(connection, handoff)
        ):
            raise ConflictingWriteError("model selection or ready source changed")
        verify_generation(connection, index_id)
        _verify_extraction_decision(connection, [index_id])
        index = connection.execute(
            "SELECT * FROM index_versions WHERE id=?", (index_id,)
        ).fetchone()
        job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        manifest = connection.execute(
            "SELECT manifest_id FROM search_generations WHERE index_version_id=?", (index_id,)
        ).fetchone()
        if (
            index is None
            or manifest is None
            or manifest[0] != handoff["manifest_id"]
            or any(
                index[key] != job[key]
                for key in ("source_id", "source_version_id", "extraction_version_id")
            )
        ):
            raise ConflictingWriteError("model replacement output mismatch")
        committed = connection.execute(
            "SELECT 1 FROM job_steps WHERE job_id=? AND output_index_version_id=?",
            (job_id, index_id),
        ).fetchone()
        if committed is None:
            raise ConflictingWriteError("model replacement was not committed by this job")
        connection.execute(
            "INSERT INTO profile_search_publications(source_id,source_version_id,manifest_id,"
            "index_version_id,selection_generation,published_by_job_id)"
            " VALUES(?,?,?,?,?,?) ON CONFLICT(source_id,manifest_id)"
            " DO UPDATE SET source_version_id=excluded.source_version_id,"
            "index_version_id=excluded.index_version_id,selection_generation=excluded.selection_generation,published_by_job_id=excluded.published_by_job_id",
            (
                index["source_id"],
                index["source_version_id"],
                handoff["manifest_id"],
                index_id,
                handoff["selection_generation"],
                job_id,
            ),
        )
        notebook = connection.execute(
            "SELECT notebook_id FROM sources WHERE id=?", (index["source_id"],)
        ).fetchone()[0]
        activate_if_complete(connection, notebook)
        if checkpoint is not None:
            checkpoint_version, payload, inputs_sha256 = checkpoint
            _save_job_step(
                connection,
                job_id,
                guard.step_key,
                checkpoint_version,
                payload,
                inputs_sha256,
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise ConflictingWriteError("model replacement publication deadline expired")


class NotebookModelService:
    def __init__(self, store: PersistenceStore, *, notify=lambda: None) -> None:
        self.store = store
        self.notify = notify

    def state(self, notebook_id: str) -> dict:
        self.store.load_notebook(notebook_id)
        with self.store._transaction() as connection:
            row = ensure_notebook_profile(connection, notebook_id)
            sources = ready_sources(connection, notebook_id)
            completed = connection.execute(
                "SELECT count(*) FROM profile_search_publications p"
                " JOIN sources s ON s.id=p.source_id JOIN source_readiness r"
                " ON r.source_id=p.source_id AND r.source_version_id=p.source_version_id"
                " WHERE s.notebook_id=? AND p.manifest_id=? AND p.selection_generation=?",
                (notebook_id, row["pending_manifest_id"], row["selection_generation"]),
            ).fetchone()[0]
            failed: set[str] = set()
            cancelled: set[str] = set()
            if row["pending_manifest_id"] is not None:
                ready = {
                    (
                        source["source_id"],
                        source["source_version_id"],
                        source["extraction_version_id"],
                    )
                    for source in sources
                }
                terminal = connection.execute(
                    """SELECT h.*,j.source_id,j.source_version_id,s.status
                       FROM search_handoffs h JOIN jobs j ON j.id=h.job_id
                       JOIN job_status s ON s.job_id=j.id
                       JOIN sources source ON source.id=j.source_id
                       WHERE source.notebook_id=? AND h.purpose='model-switch'
                         AND h.manifest_id=? AND h.selection_generation=?
                         AND (s.status IN ('failed','cancelled') OR h.state='cancelled')""",
                    (notebook_id, row["pending_manifest_id"], row["selection_generation"]),
                ).fetchall()
                for handoff in terminal:
                    identity = (
                        handoff["source_id"],
                        handoff["source_version_id"],
                        handoff["extraction_version_id"],
                    )
                    if identity not in ready or not switch_current(connection, handoff):
                        continue
                    if handoff["state"] == "cancelled" or handoff["status"] == "cancelled":
                        cancelled.add(handoff["source_id"])
                    else:
                        failed.add(handoff["source_id"])
            return {
                **dict(row),
                "ready_sources": len(sources),
                "rebuilt_sources": completed,
                "failed_sources": len(failed - cancelled),
                "cancelled_sources": len(cancelled),
            }

    def choose(self, notebook_id: str, profile_id: str, generation: int, request_key: str) -> dict:
        self.store.load_notebook(notebook_id)
        if profile_id not in SUPPORTED_PROFILE_IDS:
            raise ValueError("unsupported_model")
        profile = profiles()[profile_id]
        with self.store._transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM notebook_model_requests WHERE request_key=?", (request_key,)
            ).fetchone()
            if previous is not None:
                if (
                    previous["notebook_id"] != notebook_id
                    or previous["manifest_id"] != profile.identity
                ):
                    raise ConflictingWriteError("model request belongs to different work")
            else:
                row = ensure_notebook_profile(connection, notebook_id)
                if row["selection_generation"] != generation:
                    raise ConflictingWriteError("notebook selection generation changed")
                if row["pending_manifest_id"] is not None:
                    self._cancel_pending(connection, notebook_id, row["selection_generation"])
                generation += 1
                same_active = profile.identity == row["active_manifest_id"]
                connection.execute(
                    "UPDATE notebook_profiles SET pending_manifest_id=?,selection_generation=?,"
                    "updated_at=? WHERE notebook_id=?",
                    (
                        None if same_active else profile.identity,
                        generation,
                        utc_timestamp(),
                        notebook_id,
                    ),
                )
                connection.execute(
                    "INSERT INTO notebook_model_requests VALUES(?,?,?,?,?,?)",
                    (
                        request_key,
                        notebook_id,
                        profile.identity,
                        generation,
                        "active" if same_active else "pending",
                        utc_timestamp(),
                    ),
                )
                if not same_active:
                    sync_pending_switch(connection, notebook_id)
        self.notify()
        return self.state(notebook_id)

    def _cancel_pending(self, connection, notebook_id, generation):
        rows = connection.execute(
            "SELECT h.job_id FROM search_handoffs h"
            " JOIN extraction_versions e ON e.id=h.extraction_version_id"
            " JOIN sources s ON s.id=e.source_id WHERE s.notebook_id=?"
            " AND h.selection_generation=? AND h.purpose='model-switch'",
            (notebook_id, generation),
        ).fetchall()
        for row in rows:
            invalidate_handoffs(connection, job_id=row[0], reason="model selection superseded")
        connection.execute(
            "UPDATE notebook_model_requests SET state='cancelled'"
            " WHERE notebook_id=? AND generation=? AND state='pending'",
            (notebook_id, generation),
        )

    def cancel(self, notebook_id: str, generation: int) -> dict:
        with self.store._transaction() as connection:
            row = ensure_notebook_profile(connection, notebook_id)
            if row["pending_manifest_id"] is None or row["selection_generation"] != generation:
                raise ConflictingWriteError("no matching pending model selection")
            self._cancel_pending(connection, notebook_id, generation)
            connection.execute(
                "UPDATE notebook_profiles SET pending_manifest_id=NULL,"
                "selection_generation=selection_generation+1,updated_at=? WHERE notebook_id=?",
                (utc_timestamp(), notebook_id),
            )
        self.notify()
        return self.state(notebook_id)
