"""Runner and lifecycle records: job status/attempts, readiness, and trash/purge."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import (
    ConflictingWriteError,
    IntegrityViolationError,
    PersistenceError,
)
from core.persistence.migrations import load_migrations
from core.persistence.models import (
    ChunkPassage,
    ExtractionVersion,
    IndexVersion,
    ManagedFile,
    Passage,
    RetrievalChunk,
    SourceVersion,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore

HASH_A = "a" * 64
DEADLINE = "2026-10-27T00:00:00.000000Z"
AFTER_DEADLINE = "2026-10-28T00:00:00.000000Z"
BEFORE_DEADLINE = "2026-10-26T00:00:00.000000Z"


def seed_source(store: PersistenceStore, name: str = "report.pdf") -> tuple[str, SourceVersion]:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Notebook {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    managed_file = ManagedFile(new_id(), f"originals/{name}", 17, HASH_A, utc_timestamp())
    store.save_managed_file(managed_file)
    source_version = store.create_source_version(source_id, HASH_A, managed_file.id)
    return source_id, source_version


def make_extraction(
    source_id: str, source_version: SourceVersion
) -> tuple[ExtractionVersion, tuple[Passage, ...]]:
    extraction_id = new_id()
    extraction = ExtractionVersion(
        extraction_id,
        source_id,
        source_version.id,
        "pdf",
        "1.0",
        "unicode-nfc-v1",
        1,
        {"keep_hyphens": True},
        utc_timestamp(),
    )
    passages = (
        Passage(
            new_id(),
            source_id,
            source_version.id,
            extraction_id,
            0,
            "The saved evidence remains stable.",
            1,
            {"page": 1},
            utc_timestamp(),
        ),
    )
    return extraction, passages


def make_index(
    extraction: ExtractionVersion, passages: tuple[Passage, ...]
) -> tuple[IndexVersion, tuple[RetrievalChunk, ...], tuple[ChunkPassage, ...]]:
    index_id = new_id()
    chunks = (
        RetrievalChunk(
            new_id(),
            index_id,
            extraction.source_id,
            extraction.source_version_id,
            extraction.id,
            0,
            passages[0].text,
        ),
    )
    mappings = (
        ChunkPassage(
            chunks[0].id,
            passages[0].id,
            index_id,
            extraction.source_id,
            extraction.source_version_id,
            extraction.id,
            0,
            0,
            len(passages[0].text),
        ),
    )
    index = IndexVersion(
        index_id,
        extraction.source_id,
        extraction.source_version_id,
        extraction.id,
        "keyword",
        "chunker-v1",
        1,
        {"maximum_characters": 900},
        None,
        None,
        utc_timestamp(),
    )
    return index, chunks, mappings


def seed_graph(
    store: PersistenceStore, name: str = "report.pdf"
) -> tuple[str, SourceVersion, IndexVersion, str]:
    source_id, source_version = seed_source(store, name)
    extraction, passages = make_extraction(source_id, source_version)
    store.save_extraction(extraction, passages)
    index, chunks, mappings = make_index(extraction, passages)
    store.save_index(index, chunks, mappings)
    job = store.create_job(
        f"extract:{source_version.id}",
        "extract",
        source_id,
        source_version_id=source_version.id,
    )
    store.commit_extraction_step(job.id, "extract-text", extraction, passages, 1, {"page": 1})
    store.create_job_status(job.id)
    return source_id, source_version, index, job.id


def test_populated_upgrade_to_revision_two_preserves_records(data_dir: Path) -> None:
    migrations = load_migrations()
    assert [migration.revision for migration in migrations] == [1, 2]

    legacy = Database(DatabaseConfig(data_dir), migrations=(migrations[0],))
    legacy.initialize()
    source_id, _ = seed_source(PersistenceStore(legacy))

    upgraded = Database(DatabaseConfig(data_dir))
    upgraded.initialize()
    store = PersistenceStore(upgraded)
    assert store.load_source(source_id).id == source_id
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_job_status_claim_heartbeat_and_finish(store: PersistenceStore, database: Database) -> None:
    source_id, source_version = seed_source(store)
    job = store.create_job(
        "status:lifecycle", "extract", source_id, source_version_id=source_version.id
    )

    status = store.create_job_status(job.id)
    assert status.status == "queued"
    assert status.attempt_count == 0
    assert status.owner_instance_id is None

    claimed = store.claim_job(job.id, "core-a")
    assert claimed.status == "running"
    assert claimed.owner_instance_id == "core-a"
    assert claimed.claimed_at is not None
    assert claimed.heartbeat_at is not None

    with pytest.raises(ConflictingWriteError):
        store.heartbeat_job(job.id, "core-b")
    store.heartbeat_job(job.id, "core-a")

    with pytest.raises(ConflictingWriteError):
        store.claim_job(job.id, "core-b")

    finished = store.finish_job(job.id, "completed")
    assert finished.status == "completed"

    with pytest.raises(ValueError):
        store.finish_job(job.id, "running")

    with pytest.raises(ConflictingWriteError):
        store.claim_job(job.id, "core-a")

    with database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE job_status SET status='nonsense' WHERE job_id=?", (job.id,))


def test_step_attempts_number_finish_and_count(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    job = store.create_job(
        "attempts:lifecycle", "extract", source_id, source_version_id=source_version.id
    )
    store.create_job_status(job.id)

    first = store.start_step_attempt(job.id, "extract-text")
    assert first.attempt_number == 1
    assert first.finished_at is None
    store.finish_step_attempt(job.id, "extract-text", 1, failure_reason="temporary failure")

    with pytest.raises(ConflictingWriteError):
        store.finish_step_attempt(job.id, "extract-text", 1)

    second = store.start_step_attempt(job.id, "extract-text")
    assert second.attempt_number == 2

    attempts = store.load_step_attempts(job.id, "extract-text")
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert attempts[0].failure_reason == "temporary failure"
    assert attempts[0].finished_at is not None
    assert attempts[1].finished_at is None

    status = store.load_job_status(job.id)
    assert status is not None
    assert status.attempt_count == 2
    assert status.current_step_key == "extract-text"


def test_readiness_publish_load_replace_and_remove(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction, passages = make_extraction(source_id, source_version)
    store.save_extraction(extraction, passages)
    first_index, first_chunks, first_mappings = make_index(extraction, passages)
    store.save_index(first_index, first_chunks, first_mappings)
    second_index, second_chunks, second_mappings = make_index(extraction, passages)
    store.save_index(second_index, second_chunks, second_mappings)

    readiness = store.publish_readiness(source_id, source_version.id, [first_index.id])
    assert readiness.source_version_id == source_version.id
    assert readiness.index_version_ids == (first_index.id,)
    assert store.load_readiness(source_id) == readiness

    # Republishing replaces the required-index set in one transaction.
    replaced = store.publish_readiness(source_id, source_version.id, [second_index.id])
    assert replaced.index_version_ids == (second_index.id,)

    store.remove_readiness(source_id)
    assert store.load_readiness(source_id) is None

    with pytest.raises(ValueError):
        store.publish_readiness(source_id, source_version.id, [])


def test_readiness_rejects_an_index_from_another_source(store: PersistenceStore) -> None:
    source_a, version_a, _, _ = seed_graph(store, "source-a.pdf")
    _, _, index_b, _ = seed_graph(store, "source-b.pdf")

    with pytest.raises(IntegrityViolationError):
        store.publish_readiness(source_a, version_a.id, [index_b.id])


def test_publication_recheck_blocks_cancelled_jobs_and_trashed_sources(
    store: PersistenceStore,
) -> None:
    # Cancel-before-publish: the recheck refuses; no readiness is recorded.
    source_a, version_a, index_a, job_a = seed_graph(store, "publication-a.pdf")
    store.claim_job(job_a, "core-a")
    store.cancel_job(job_a, "cancelled before publication")
    with pytest.raises(ConflictingWriteError):
        store.publish_readiness(source_a, version_a.id, [index_a.id], job_id=job_a)
    assert store.load_readiness(source_a) is None

    # Publish-before-cancel: readiness commits, then deletion hides it.
    source_b, version_b, index_b, job_b = seed_graph(store, "publication-b.pdf")
    store.claim_job(job_b, "core-a")
    published = store.publish_readiness(source_b, version_b.id, [index_b.id], job_id=job_b)
    assert published.source_version_id == version_b.id
    store.trash_source(source_b, DEADLINE)
    assert store.load_readiness(source_b) is None

    # Trash-before-publish: no publication path accepts a deleted source.
    source_c, version_c, index_c, job_c = seed_graph(store, "publication-c.pdf")
    store.claim_job(job_c, "core-a")
    store.trash_source(source_c, DEADLINE)
    with pytest.raises(ConflictingWriteError):
        store.publish_readiness(source_c, version_c.id, [index_c.id], job_id=job_c)
    assert store.load_readiness(source_c) is None


def test_duplicate_completion_with_different_content_is_rejected(
    store: PersistenceStore,
) -> None:
    source_id, source_version = seed_source(store)
    job = store.create_job(
        "completion:conflict", "extract", source_id, source_version_id=source_version.id
    )
    store.create_job_status(job.id)
    store.claim_job(job.id, "core-a")

    first = store.commit_step(job.id, "one", 1, {"value": 1})
    assert first.checkpoint == {"value": 1}

    # The same key cannot store a different result.
    with pytest.raises(ConflictingWriteError):
        store.commit_step(job.id, "one", 1, {"value": 2})
    assert store.load_job_steps(job.id)[0].checkpoint == {"value": 1}

    # An identical commit is idempotent rather than an error.
    store.commit_step(job.id, "one", 1, {"value": 1})


def test_trash_hides_readiness_cancels_jobs_and_lists_expiry(store: PersistenceStore) -> None:
    source_id, source_version, index, job_id = seed_graph(store)
    store.publish_readiness(source_id, source_version.id, [index.id])
    store.claim_job(job_id, "core-a")

    trash = store.trash_source(source_id, DEADLINE)
    assert trash.purge_deadline_at == DEADLINE
    assert trash.files_removed_at is None
    assert trash.snapshots_removed_at is None

    assert store.load_readiness(source_id) is None
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.status == "cancelled"
    assert status.failure_reason == "source deleted"

    assert store.list_expired_trash(BEFORE_DEADLINE) == ()
    expired = store.list_expired_trash(AFTER_DEADLINE)
    assert [item.source_id for item in expired] == [source_id]


def test_purge_phases_resume_and_final_removal_is_atomic(
    store: PersistenceStore, database: Database
) -> None:
    source_id, source_version, index, job_id = seed_graph(store)
    store.publish_readiness(source_id, source_version.id, [index.id])
    store.trash_source(source_id, DEADLINE)

    # Phase marks survive a restart: only the remaining phases need to rerun.
    marked = store.mark_trash_files_removed(source_id)
    assert marked.files_removed_at is not None
    reopened_database = Database(DatabaseConfig(store.database.config.data_dir))
    reopened_database.initialize()
    reopened = PersistenceStore(reopened_database)
    still = reopened.load_trash(source_id)
    assert still is not None
    assert still.files_removed_at is not None
    assert reopened.mark_trash_snapshots_removed(source_id).snapshots_removed_at is not None

    reopened.remove_trashed_records(source_id)

    with pytest.raises(PersistenceError):
        reopened.load_source(source_id)
    assert reopened.load_trash(source_id) is None
    assert reopened.load_readiness(source_id) is None
    assert reopened.load_job_status(job_id) is None

    with database.connect() as connection:
        for table, column in (
            ("jobs", "source_id"),
            ("source_versions", "source_id"),
            ("extraction_versions", "source_id"),
            ("index_versions", "source_id"),
            ("passages", "source_id"),
            ("retrieval_chunks", "source_id"),
            ("chunk_passages", "source_id"),
            ("source_trash", "source_id"),
        ):
            count = connection.execute(
                f"SELECT count(*) FROM {table} WHERE {column}=?", (source_id,)
            ).fetchone()[0]
            assert count == 0, table

    # The tracker left with the source row: purging again is a clear error.
    with pytest.raises(PersistenceError):
        reopened.remove_trashed_records(source_id)


def test_purge_keeps_a_managed_file_another_source_still_references(
    store: PersistenceStore, database: Database
) -> None:
    shared_file = ManagedFile(new_id(), "originals/shared.pdf", 17, HASH_A, utc_timestamp())
    store.save_managed_file(shared_file)

    notebook_id = new_id()
    store.create_notebook(notebook_id, "Shared")
    first_source = new_id()
    second_source = new_id()
    store.create_source(first_source, notebook_id, "first.pdf")
    store.create_source(second_source, notebook_id, "second.pdf")
    store.create_source_version(first_source, HASH_A, shared_file.id)
    store.create_source_version(second_source, HASH_A, shared_file.id)

    store.trash_source(first_source, DEADLINE)
    store.remove_trashed_records(first_source)
    assert store.load_source(second_source).id == second_source
    with database.connect() as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM managed_files WHERE id=?", (shared_file.id,)
        ).fetchone()[0]
        assert remaining == 1

    store.trash_source(second_source, DEADLINE)
    store.remove_trashed_records(second_source)
    with pytest.raises(PersistenceError):
        store.load_source(second_source)
    with database.connect() as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM managed_files WHERE id=?", (shared_file.id,)
        ).fetchone()[0]
        assert remaining == 0
