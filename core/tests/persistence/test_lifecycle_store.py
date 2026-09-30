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
    AttemptGuard,
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
HASH_B = "b" * 64
DEADLINE = "2026-10-27T00:00:00.000000Z"
AFTER_DEADLINE = "2026-10-28T00:00:00.000000Z"
BEFORE_DEADLINE = "2026-10-26T00:00:00.000000Z"
SEED_OWNER = "core-seed"


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


def commit_with_guard(store: PersistenceStore, job_id: str, step_key: str) -> AttemptGuard:
    """Start a step attempt as the seed owner and return its commit guard."""
    attempt = store.start_step_attempt(job_id, step_key)
    return AttemptGuard(SEED_OWNER, step_key, attempt.attempt_number)


def begin_publication(
    store: PersistenceStore, job_id: str, step_key: str = "publish-source"
) -> AttemptGuard:
    """Open a publication attempt as the seed owner and return its guard."""
    attempt = store.start_step_attempt(job_id, step_key)
    return AttemptGuard(SEED_OWNER, step_key, attempt.attempt_number)


def publish(
    store: PersistenceStore,
    job_id: str,
    source_id: str,
    source_version_id: str,
    index_version_ids: list[str],
):
    return store.publish_readiness(
        job_id,
        source_id,
        source_version_id,
        index_version_ids,
        guard=begin_publication(store, job_id),
    )


def seed_graph(
    store: PersistenceStore,
    name: str = "report.pdf",
    *,
    index_count: int = 1,
) -> tuple[str, SourceVersion, tuple[IndexVersion, ...], str]:
    source_id, source_version = seed_source(store, name)
    extraction, passages = make_extraction(source_id, source_version)
    store.save_extraction(extraction, passages)
    job = store.create_job(
        f"extract:{source_version.id}",
        "extract",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=extraction.id,
    )
    store.create_job_status(job.id)
    store.claim_job(job.id, SEED_OWNER)

    guard = commit_with_guard(store, job.id, "extract-text")
    store.commit_extraction_step(
        job.id, "extract-text", extraction, passages, 1, {"page": 1}, guard=guard
    )
    store.finish_step_attempt(job.id, "extract-text", guard.expected_attempt_number)

    indexes: list[IndexVersion] = []
    for number in range(1, index_count + 1):
        index, chunks, mappings = make_index(extraction, passages)
        step_key = f"build-index-{number}"
        index_guard = commit_with_guard(store, job.id, step_key)
        store.commit_index_step(
            job.id, step_key, index, chunks, mappings, 1, {"index": number}, guard=index_guard
        )
        store.finish_step_attempt(job.id, step_key, index_guard.expected_attempt_number)
        indexes.append(index)
    return source_id, source_version, tuple(indexes), job.id


def test_populated_upgrade_to_revision_three_preserves_records(data_dir: Path) -> None:
    migrations = load_migrations()
    assert [migration.revision for migration in migrations] == [1, 2, 3]

    legacy = Database(DatabaseConfig(data_dir), migrations=migrations[:1])
    legacy.initialize()
    source_id, _ = seed_source(PersistenceStore(legacy))

    upgraded = Database(DatabaseConfig(data_dir))
    upgraded.initialize()
    store = PersistenceStore(upgraded)
    assert store.load_source(source_id).id == source_id
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_job_status_claim_heartbeat_and_finish(store: PersistenceStore, database: Database) -> None:
    source_id, source_version = seed_source(store)
    job = store.create_job(
        "status:lifecycle", "extract", source_id, source_version_id=source_version.id
    )

    status = store.create_job_status(job.id)
    assert status.status == "queued"
    assert status.attempt_count == 0
    assert status.owner_instance_id is None
    assert status.policy is None

    claimed = store.claim_job(job.id, "core-a", policy={"generic": {"attempt_cap": 3}})
    assert claimed.status == "running"
    assert claimed.owner_instance_id == "core-a"
    assert claimed.claimed_at is not None
    assert claimed.heartbeat_at is not None
    assert claimed.policy == {"generic": {"attempt_cap": 3}}

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
        store.finish_job(job.id, "completed")

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
    source_id, source_version, indexes, job_id = seed_graph(store, index_count=2)

    readiness = publish(store, job_id, source_id, source_version.id, [indexes[0].id])
    assert readiness.source_version_id == source_version.id
    assert readiness.index_version_ids == (indexes[0].id,)
    assert readiness.published_by_job_id == job_id
    assert store.load_readiness(source_id) == readiness

    # Republishing replaces the required-index set in one transaction.
    replaced = publish(store, job_id, source_id, source_version.id, [indexes[1].id])
    assert replaced.index_version_ids == (indexes[1].id,)

    store.remove_readiness(source_id)
    assert store.load_readiness(source_id) is None

    with pytest.raises(ValueError):
        publish(store, job_id, source_id, source_version.id, [])


def test_publication_verifies_the_publishing_job(store: PersistenceStore) -> None:
    source_a, version_a, indexes_a, job_a = seed_graph(store, "source-a.pdf")
    _, _, indexes_b, job_b = seed_graph(store, "source-b.pdf")

    # An unrelated job cannot publish this source.
    with pytest.raises(ConflictingWriteError):
        publish(store, job_b, source_a, version_a.id, [indexes_a[0].id])

    # An index this job never committed cannot be published.
    other_index, _, _ = make_index(
        ExtractionVersion(
            new_id(), source_a, version_a.id, "pdf", "1.0", "unicode-nfc-v1", 1, {}, utc_timestamp()
        ),
        (
            Passage(
                new_id(),
                source_a,
                version_a.id,
                new_id(),
                0,
                "text",
                1,
                {"page": 1},
                utc_timestamp(),
            ),
        ),
    )
    with pytest.raises(ConflictingWriteError):
        publish(store, job_a, source_a, version_a.id, [other_index.id])

    # A job targeting a superseded version cannot publish once a newer version
    # exists for the source.
    store.create_source_version(source_a, HASH_B, None)
    with pytest.raises(ConflictingWriteError):
        publish(store, job_a, source_a, version_a.id, [indexes_a[0].id])
    assert store.load_readiness(source_a) is None
    assert indexes_b  # silence unused-variable lint for the second graph


def test_publication_recheck_blocks_cancelled_jobs_and_trashed_sources(
    store: PersistenceStore,
) -> None:
    # Cancel-before-publish: the recheck refuses; no readiness is recorded.
    source_a, version_a, indexes_a, job_a = seed_graph(store, "publication-a.pdf")
    store.cancel_job(job_a, "cancelled before publication")
    with pytest.raises(ConflictingWriteError):
        publish(store, job_a, source_a, version_a.id, [indexes_a[0].id])
    assert store.load_readiness(source_a) is None

    # Publish-before-cancel: readiness commits, then cancellation hides it.
    source_b, version_b, indexes_b, job_b = seed_graph(store, "publication-b.pdf")
    published = publish(store, job_b, source_b, version_b.id, [indexes_b[0].id])
    assert published.source_version_id == version_b.id
    store.cancel_job(job_b, "cancelled after publication")
    assert store.load_readiness(source_b) is None

    # Trash-before-publish: no publication path accepts a deleted source.
    source_c, version_c, indexes_c, job_c = seed_graph(store, "publication-c.pdf")
    store.trash_source(source_c, DEADLINE)
    with pytest.raises(ConflictingWriteError):
        publish(store, job_c, source_c, version_c.id, [indexes_c[0].id])
    assert store.load_readiness(source_c) is None


def test_cancelling_an_older_job_keeps_a_newer_jobs_readiness(
    store: PersistenceStore,
) -> None:
    source_id, source_version, indexes, job_a = seed_graph(store, "newer.pdf")
    publish(store, job_a, source_id, source_version.id, [indexes[0].id])

    # A second job publishes over the first; cancelling the first must not
    # hide the newer result.
    graph = store.load_extraction(indexes[0].extraction_version_id)
    new_index, chunks, mappings = make_index(graph.extraction, graph.passages)
    job_b = store.create_job(
        "extract:newer-secondary",
        "extract",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=graph.extraction.id,
    )
    store.create_job_status(job_b.id)
    store.claim_job(job_b.id, SEED_OWNER)
    guard = commit_with_guard(store, job_b.id, "build-index-new")
    store.commit_index_step(
        job_b.id, "build-index-new", new_index, chunks, mappings, 1, {}, guard=guard
    )
    store.finish_step_attempt(job_b.id, "build-index-new", guard.expected_attempt_number)
    publish(store, job_b.id, source_id, source_version.id, [new_index.id])

    store.cancel_job(job_a, "cancelled older job")
    readiness = store.load_readiness(source_id)
    assert readiness is not None
    assert readiness.published_by_job_id == job_b.id
    assert readiness.index_version_ids == (new_index.id,)


def test_reservation_refuses_when_another_source_references_the_file(
    store: PersistenceStore,
) -> None:
    """The reference check and the removal reservation share one transaction,
    so a reference cannot slip in between them."""
    shared = ManagedFile(new_id(), "originals/race.pdf", 17, HASH_A, utc_timestamp())
    store.save_managed_file(shared)
    notebook_id = new_id()
    store.create_notebook(notebook_id, "Race")
    first_source = new_id()
    second_source = new_id()
    store.create_source(first_source, notebook_id, "first.pdf")
    store.create_source(second_source, notebook_id, "second.pdf")
    store.create_source_version(first_source, HASH_A, shared.id)
    store.create_source_version(second_source, HASH_A, shared.id)

    # The other source's reference is visible inside the reservation's
    # transaction, so nothing is reserved and no marker is left behind.
    assert store.reserve_managed_file_removal(shared.id, "originals/race.pdf", first_source) is None
    assert store.load_managed_file_removal(shared.id) is None

    # After the other reference is gone, reservation succeeds and the marker
    # blocks anything new.
    store.trash_source(second_source, DEADLINE)
    store.mark_trash_files_removed(second_source)
    store.mark_trash_snapshots_removed(second_source)
    store.remove_trashed_records(second_source)
    reserved = store.reserve_managed_file_removal(shared.id, "originals/race.pdf", first_source)
    assert reserved is not None
    assert reserved.removed_at is None


def test_older_jobs_cannot_replace_newer_readiness(store: PersistenceStore) -> None:
    source_id, source_version, indexes, job_a = seed_graph(store, "versions.pdf")
    publish(store, job_a, source_id, source_version.id, [indexes[0].id])

    # A job created later may replace the older result.
    graph = store.load_extraction(indexes[0].extraction_version_id)
    new_index, chunks, mappings = make_index(graph.extraction, graph.passages)
    job_b = store.create_job(
        "extract:newer-publisher",
        "extract",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=graph.extraction.id,
    )
    store.create_job_status(job_b.id)
    store.claim_job(job_b.id, SEED_OWNER)
    guard = commit_with_guard(store, job_b.id, "build-index-new")
    store.commit_index_step(
        job_b.id, "build-index-new", new_index, chunks, mappings, 1, {}, guard=guard
    )
    store.finish_step_attempt(job_b.id, "build-index-new", guard.expected_attempt_number)
    publish(store, job_b.id, source_id, source_version.id, [new_index.id])
    readiness = store.load_readiness(source_id)
    assert readiness is not None
    assert readiness.published_by_job_id == job_b.id

    # The older job cannot replace the newer result.
    with pytest.raises(ConflictingWriteError):
        publish(store, job_a, source_id, source_version.id, [indexes[0].id])
    readiness = store.load_readiness(source_id)
    assert readiness is not None
    assert readiness.published_by_job_id == job_b.id
    assert readiness.index_version_ids == (new_index.id,)


def test_publication_rejects_finished_or_superseded_attempts(
    store: PersistenceStore,
) -> None:
    source_id, source_version, indexes, job_id = seed_graph(store, "attempts.pdf")

    first = begin_publication(store, job_id)
    store.finish_step_attempt(job_id, first.step_key, first.expected_attempt_number, "finished")
    with pytest.raises(ConflictingWriteError):
        store.publish_readiness(job_id, source_id, source_version.id, [indexes[0].id], guard=first)

    second = begin_publication(store, job_id)
    with pytest.raises(ConflictingWriteError):
        store.publish_readiness(job_id, source_id, source_version.id, [indexes[0].id], guard=first)
    published = store.publish_readiness(
        job_id, source_id, source_version.id, [indexes[0].id], guard=second
    )
    assert published.published_by_job_id == job_id


def test_duplicate_completion_with_different_content_is_rejected(
    store: PersistenceStore,
) -> None:
    source_id, source_version = seed_source(store)
    job = store.create_job(
        "completion:conflict", "extract", source_id, source_version_id=source_version.id
    )
    store.create_job_status(job.id)
    store.claim_job(job.id, SEED_OWNER)
    guard = commit_with_guard(store, job.id, "one")

    first = store.commit_step(job.id, "one", 1, {"value": 1}, guard=guard)
    assert first.checkpoint == {"value": 1}

    # The same key cannot store a different result.
    with pytest.raises(ConflictingWriteError):
        store.commit_step(job.id, "one", 1, {"value": 2}, guard=guard)
    assert store.load_job_steps(job.id)[0].checkpoint == {"value": 1}

    # An identical commit is idempotent rather than an error.
    store.commit_step(job.id, "one", 1, {"value": 1}, guard=guard)


def test_commits_require_the_current_attempt_and_owner(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    job = store.create_job(
        "completion:guard", "extract", source_id, source_version_id=source_version.id
    )
    store.create_job_status(job.id)
    store.claim_job(job.id, SEED_OWNER)
    attempt = store.start_step_attempt(job.id, "one")

    # Another instance cannot save.
    with pytest.raises(ConflictingWriteError):
        store.commit_step(
            job.id, "one", 1, {"value": 1}, guard=AttemptGuard("another-core", "one", 1)
        )

    # A superseded attempt cannot save.
    store.finish_step_attempt(job.id, "one", attempt.attempt_number)
    newer = store.start_step_attempt(job.id, "one")
    with pytest.raises(ConflictingWriteError):
        store.commit_step(job.id, "one", 1, {"value": 1}, guard=AttemptGuard(SEED_OWNER, "one", 1))
    # An expired (already finished) attempt cannot save either.
    with pytest.raises(ConflictingWriteError):
        store.commit_step(job.id, "one", 1, {"value": 1}, guard=AttemptGuard(SEED_OWNER, "one", 1))
    # The current attempt saves.
    saved = store.commit_step(
        job.id, "one", 1, {"value": 1}, guard=AttemptGuard(SEED_OWNER, "one", newer.attempt_number)
    )
    assert saved.checkpoint == {"value": 1}

    # Cancellation rejects a later save even for the current attempt.
    # Cancellation rejects a later save regardless of the guard.
    store.cancel_job(job.id, "cancelled")
    with pytest.raises(ConflictingWriteError):
        store.commit_step(job.id, "two", 1, {"value": 2}, guard=AttemptGuard(SEED_OWNER, "two", 1))


def test_trash_hides_readiness_cancels_jobs_and_lists_expiry(store: PersistenceStore) -> None:
    source_id, source_version, indexes, job_id = seed_graph(store)
    publish(store, job_id, source_id, source_version.id, [indexes[0].id])

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


def test_managed_file_removal_marker_blocks_new_references(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    managed_file_id = source_version.managed_file_id
    assert managed_file_id is not None

    removal = store.reserve_managed_file_removal(managed_file_id, "originals/report.pdf", source_id)
    assert removal is not None
    assert removal.removed_at is None
    assert removal.attempts == 0

    # A new reference cannot be created while the removal is pending.
    with pytest.raises(IntegrityViolationError):
        store.create_source_version(source_id, HASH_B, managed_file_id)

    failed = store.record_managed_file_removal_failure(managed_file_id, "PermissionError")
    assert failed.attempts == 1
    assert failed.last_error == "PermissionError"

    removed = store.mark_managed_file_removed(managed_file_id)
    assert removed.removed_at is not None
    assert removed.last_error is None
    assert store.load_managed_file_removal(managed_file_id) == removed


def test_records_purge_keeps_rows_with_pending_removal_markers(
    store: PersistenceStore, database: Database
) -> None:
    source_id, source_version, indexes, job_id = seed_graph(store)
    managed_file_id = source_version.managed_file_id
    assert managed_file_id is not None
    store.trash_source(source_id, DEADLINE)
    store.mark_trash_files_removed(source_id)
    store.mark_trash_snapshots_removed(source_id)
    # The file removal is still pending (marker without removed_at).
    store.reserve_managed_file_removal(managed_file_id, "originals/report.pdf", source_id)

    store.remove_trashed_records(source_id)

    # The records purge kept the file row and marker for the next sweep.
    with database.connect() as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM managed_files WHERE id=?", (managed_file_id,)
        ).fetchone()[0]
        assert remaining == 1
    removal = store.load_managed_file_removal(managed_file_id)
    assert removal is not None
    assert removal.removed_at is None


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
    store.mark_trash_files_removed(first_source)
    store.mark_trash_snapshots_removed(first_source)
    store.remove_trashed_records(first_source)
    assert store.load_source(second_source).id == second_source
    with database.connect() as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM managed_files WHERE id=?", (shared_file.id,)
        ).fetchone()[0]
        assert remaining == 1

    store.trash_source(second_source, DEADLINE)
    # No removal marker exists for the shared file (it was shared at the first
    # source's purge), so the records removal deletes it now that the last
    # reference is gone.
    store.mark_trash_files_removed(second_source)
    store.mark_trash_snapshots_removed(second_source)
    store.remove_trashed_records(second_source)
    with pytest.raises(PersistenceError):
        store.load_source(second_source)
    with database.connect() as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM managed_files WHERE id=?", (shared_file.id,)
        ).fetchone()[0]
        assert remaining == 0
