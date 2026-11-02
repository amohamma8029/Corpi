"""Fresh cancellation polling reuses only its own verified, transaction-free connection."""

import sqlite3
import threading
from contextlib import contextmanager

import pytest
from test_runner import FAST_POLICY, RunnerHarness, enqueue, wait_until

from core.jobs.runner import StepContext, StepDefinition, StepOutcome
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore


def queued_job(store: PersistenceStore) -> tuple[str, str]:
    job, source = enqueue(store, "polling", "polling:" + new_id(), "polling.pdf")
    store.create_job_status(job)
    return job, source


def capture_connections(store: PersistenceStore, monkeypatch: pytest.MonkeyPatch):
    opened = []
    original = store.database._connect

    def connect(read_only):
        connection = original(read_only)
        opened.append((threading.get_ident(), connection))
        return connection

    monkeypatch.setattr(store.database, "_connect", connect)
    return opened


def test_reader_opens_once_keeps_runtime_policy_and_cannot_write(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, _ = queued_job(store)
    opened = capture_connections(store, monkeypatch)
    with store.job_cancellation_reader(job) as cancelled:
        for _ in range(100):
            assert not cancelled()
        assert len(opened) == 1
        connection = opened[0][1]
        assert not connection.in_transaction
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT vec_version()").fetchone()[0] == "v0.1.9"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("UPDATE job_status SET status='cancelled' WHERE job_id=?", (job,))
        assert not cancelled()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


@pytest.mark.parametrize("action", ["cancel", "trash"])
def test_reader_sees_another_connection_commit_without_retaining_a_snapshot(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    job, source = queued_job(store)
    opened = capture_connections(store, monkeypatch)
    other = PersistenceStore(store.database)
    with store.job_cancellation_reader(job) as cancelled:
        assert not cancelled()
        reader = opened[0][1]
        if action == "cancel":
            other.cancel_job(job, "user cancelled")
        else:
            other.trash_source(source, "2099-01-01T00:00:00.000000Z")
        assert not reader.in_transaction
        assert cancelled()
        assert cancelled()


def test_reader_treats_missing_status_as_cancelled(store: PersistenceStore) -> None:
    with store.job_cancellation_reader(new_id()) as cancelled:
        assert cancelled()


def test_reader_does_not_hold_back_wal_checkpoint_between_polls(store: PersistenceStore) -> None:
    job, _ = queued_job(store)
    with store.job_cancellation_reader(job) as cancelled:
        assert not cancelled()
        store.cancel_job(job, "writer commits while reader stays open")
        with store.database.connect() as writer:
            checkpoint = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            assert tuple(checkpoint) == (0, 0, 0)
        assert cancelled()


def test_reader_closes_after_step_failure(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, _ = queued_job(store)
    opened = capture_connections(store, monkeypatch)
    with pytest.raises(RuntimeError, match="controlled step failure"):
        with store.job_cancellation_reader(job) as cancelled:
            assert not cancelled()
            raise RuntimeError("controlled step failure")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0][1].execute("SELECT 1")


def test_runner_polling_does_not_reopen_connections_and_retries_get_fresh_readers(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = capture_connections(store, monkeypatch)
    observed = []
    closed = []
    original_reader = store.job_cancellation_reader

    @contextmanager
    def tracked_reader(job_id):
        connection = None
        try:
            with original_reader(job_id) as cancelled:
                connection = [item for thread, item in opened if thread == threading.get_ident()][
                    -1
                ]
                yield cancelled
        finally:
            assert connection is not None
            # Verify closure on its owning worker; a cross-thread call would
            # reject even a live connection and would not prove cleanup.
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
            closed.append(True)

    monkeypatch.setattr(store, "job_cancellation_reader", tracked_reader)

    def run(context: StepContext) -> StepOutcome:
        identity = threading.get_ident()
        initial = [connection for thread, connection in opened if thread == identity]
        assert initial
        for _ in range(100):
            assert not context.is_cancelled()
            assert not context.should_stop()
        final = [connection for thread, connection in opened if thread == identity]
        assert len(final) == len(initial), "polls reopened/configured the database"
        observed.append(initial[-1])
        if context.attempt_number == 1:
            raise RuntimeError("retry with another reader")
        return StepOutcome(1, {})

    harness = RunnerHarness(
        store,
        {"polling": [StepDefinition("poll", run)]},
        pool_size=1,
        policies={"generic": FAST_POLICY},
        backoff_sleep=lambda _: None,
    )
    job, _ = queued_job(store)
    harness.start()
    try:
        harness.runner.notify()
        assert wait_until(
            lambda: (
                (status := store.load_job_status(job)) is not None and status.status == "completed"
            )
        ), store.load_job_status(job)
    finally:
        harness.stop()
    assert len(observed) == 2 and observed[0] is not observed[1]
    assert closed == [True, True]


def test_runner_watcher_thread_reads_fresh_cancellation_on_its_own_connection(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = capture_connections(store, monkeypatch)
    watcher_polled = threading.Event()
    cancelled_committed = threading.Event()
    watcher_done = threading.Event()
    observations = []

    def run(context: StepContext) -> StepOutcome:
        def watch():
            try:
                assert not context.is_cancelled()
                assert not context.should_stop()
                watcher_polled.set()
                assert cancelled_committed.wait(5)
                observations.append((context.is_cancelled(), context.should_stop()))
            finally:
                watcher_done.set()

        watcher = threading.Thread(target=watch)
        watcher.start()
        try:
            assert watcher_done.wait(10)
        finally:
            watcher.join(5)
        assert not watcher.is_alive()
        assert context.is_cancelled()
        return StepOutcome(1, {})  # The ordinary guarded commit must reject this.

    harness = RunnerHarness(store, {"polling": [StepDefinition("poll", run)]})
    job, _ = queued_job(store)
    harness.start()
    try:
        harness.runner.notify()
        assert watcher_polled.wait(5)
        store.cancel_job(job, "cancel while separate watcher is polling")
        cancelled_committed.set()
        assert watcher_done.wait(5)
        assert wait_until(
            lambda: (
                bool(attempts := store.load_step_attempts(job, "poll"))
                and attempts[0].finished_at is not None
            )
        )
    finally:
        cancelled_committed.set()
        harness.stop()
    assert observations == [(True, True)]
    assert store.load_job_steps(job) == ()
    # Worker reuse and watcher fallback must have distinct thread-owned connections.
    assert len({thread for thread, _ in opened}) >= 3
