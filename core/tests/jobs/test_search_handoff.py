import sqlite3
from pathlib import Path

import pytest
from pdf_fixtures import native_text_pdf
from test_pdf_extract_steps import (
    RunnerHarness,
    enqueue_extract,
    job_status,
    partial_pdf,
    seed_extraction_source,
    wait_until,
)

from core.jobs.steps import build_pdf_extract_steps
from core.persistence.errors import PersistenceError
from core.persistence.models import utc_timestamp
from core.persistence.store import PersistenceStore
from core.search.handoff import DEFAULT_PROFILE_ID, SearchCoordinator
from core.search.profiles import profiles


def extract(store: PersistenceStore, *, partial: bool = False) -> tuple[str, str, str]:
    source, version = seed_extraction_source(
        store,
        "partial.pdf" if partial else "complete.pdf",
        partial_pdf() if partial else native_text_pdf("Readable search handoff body"),
    )
    coordinator = SearchCoordinator(store, store.database.config.data_dir / "models")
    runner = RunnerHarness(
        store,
        {"pdf-extract": build_pdf_extract_steps(store)},
        claim_blocking_reason=coordinator.claim_blocking_reason,
    )
    job = enqueue_extract(store, "extract:" + source, source, version)
    runner.start()
    try:
        runner.enqueue_and_notify(job)
        assert wait_until(lambda: job_status(store, job) == "completed")
    finally:
        runner.stop()
    report = store.load_extraction_report_for_source(source)
    assert report is not None
    return source, report.extraction.id, job


def handoffs(store: PersistenceStore, extraction: str):
    with store.database.connect() as connection:
        return connection.execute(
            "SELECT * FROM search_handoffs WHERE extraction_version_id=?"
            " ORDER BY review_generation",
            (extraction,),
        ).fetchall()


def test_complete_extraction_atomically_queues_one_source_bound_search(
    store: PersistenceStore,
) -> None:
    source, extraction, parent = extract(store)
    rows = handoffs(store, extraction)
    assert len(rows) == 1
    handoff = rows[0]
    job = store.load_job_by_id(handoff["job_id"])
    assert handoff["origin_job_id"] == parent
    assert (
        job.kind == "pdf-search"
        and job.source_id == source
        and job.extraction_version_id == extraction
    )
    assert job_status(store, job.id) == "queued"
    assert handoff["waiting_reason"] == "model_not_installed"
    SearchCoordinator(store, Path("unused-model-directory")).reconcile()
    assert [row["job_id"] for row in handoffs(store, extraction)] == [job.id]
    assert store.load_readiness(source) is None


def test_partial_confirmation_is_unique_and_reopening_invalidates_old_work(
    store: PersistenceStore,
) -> None:
    source, extraction, _ = extract(store, partial=True)
    assert handoffs(store, extraction) == []
    store.record_review_decision(extraction, "confirmed", 1)
    first = handoffs(store, extraction)[0]["job_id"]
    store.record_review_decision(extraction, "confirmed", 1)
    SearchCoordinator(store, Path("unused")).reconcile()
    assert [row["job_id"] for row in handoffs(store, extraction)] == [first]
    review = store.reopen_review(extraction)
    assert review.generation == 2
    assert job_status(store, first) == "cancelled"
    store.record_review_decision(extraction, "confirmed", 2)
    rows = handoffs(store, extraction)
    assert len(rows) == 2 and rows[0]["state"] == "cancelled"
    assert rows[1]["job_id"] != first and job_status(store, rows[1]["job_id"]) == "queued"
    assert store.load_readiness(source) is None


def test_cancelled_handoff_is_not_rescheduled_on_startup(store: PersistenceStore) -> None:
    _, extraction, _ = extract(store)
    job = handoffs(store, extraction)[0]["job_id"]
    store.cancel_job(job)
    coordinator = SearchCoordinator(store, Path("unused"))
    coordinator.reconcile()
    coordinator.reconcile()
    rows = handoffs(store, extraction)
    assert len(rows) == 1 and rows[0]["job_id"] == job and rows[0]["state"] == "cancelled"
    assert job_status(store, job) == "cancelled"


def test_failed_queue_write_rolls_back_the_confirm_decision(store: PersistenceStore) -> None:
    _, extraction, _ = extract(store, partial=True)
    with store.database.connect() as connection:
        connection.execute("""CREATE TRIGGER reject_test_search_inputs
                            BEFORE INSERT ON search_job_inputs
                            BEGIN SELECT RAISE(ABORT,'test queue write failure'); END""")
    with pytest.raises(PersistenceError):
        store.record_review_decision(extraction, "confirmed", 1)
    report = store.load_extraction_report(extraction)
    assert report.review is not None and report.review.state == "pending"
    assert handoffs(store, extraction) == []
    with store.database.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM jobs WHERE kind='pdf-search'").fetchone()[0]
            == 0
        )


def test_missing_model_claim_gate_unblocks_after_verified_install_state(
    store: PersistenceStore,
) -> None:
    _, extraction, _ = extract(store)
    job = store.load_job_by_id(handoffs(store, extraction)[0]["job_id"])
    coordinator = SearchCoordinator(store, Path("unused"))
    assert coordinator.claim_blocking_reason(job) == "model_not_installed"
    profile = profiles()[DEFAULT_PROFILE_ID]
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
            " VALUES(?,?,'installed',?)",
            (profile.identity, profile.id, utc_timestamp()),
        )
    assert coordinator.claim_blocking_reason(job) is None
    assert handoffs(store, extraction)[0]["waiting_reason"] is None


def test_new_source_version_cancels_stale_search_before_claim(store: PersistenceStore) -> None:
    source, extraction, _ = extract(store)
    job = store.load_job_by_id(handoffs(store, extraction)[0]["job_id"])
    store.create_source_version(source, "b" * 64)
    assert (
        SearchCoordinator(store, Path("unused")).claim_blocking_reason(job)
        == "search_input_changed"
    )
    assert job_status(store, job.id) == "cancelled"


def test_search_input_snapshot_cannot_be_changed_in_place(store: PersistenceStore) -> None:
    _, extraction, _ = extract(store)
    job = handoffs(store, extraction)[0]["job_id"]
    with store.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE search_job_inputs SET settings_json='{}' WHERE job_id=?", (job,)
            )


def test_missing_model_does_not_hold_up_the_next_extraction(store: PersistenceStore) -> None:
    coordinator = SearchCoordinator(store, Path("unused"))
    runner = RunnerHarness(
        store,
        {"pdf-extract": build_pdf_extract_steps(store)},
        claim_blocking_reason=coordinator.claim_blocking_reason,
    )
    first_source, first_version = seed_extraction_source(
        store, "one.pdf", native_text_pdf("First readable body")
    )
    second_source, second_version = seed_extraction_source(
        store, "two.pdf", native_text_pdf("Second readable body")
    )
    first = enqueue_extract(store, "extract:first", first_source, first_version)
    second = enqueue_extract(store, "extract:second", second_source, second_version)
    runner.start()
    try:
        runner.enqueue_and_notify(first)
        assert wait_until(lambda: job_status(store, first) == "completed")
        runner.enqueue_and_notify(second)
        assert wait_until(lambda: job_status(store, second) == "completed")
        with store.database.connect() as connection:
            rows = connection.execute(
                "SELECT j.id,s.status FROM jobs j JOIN job_status s ON s.job_id=j.id"
                " WHERE j.kind='pdf-search'"
            ).fetchall()
        assert len(rows) == 2 and all(row["status"] == "queued" for row in rows)
    finally:
        runner.stop()
