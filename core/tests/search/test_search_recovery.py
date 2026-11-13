"""Actual process exits before/after native search stage commits and cross-job reuse."""

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_storage import seed

from core.jobs.runner import PermanentStepFailure, StepContext
from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard
from core.persistence.store import PersistenceStore
from core.search.chunking import make_chunks
from core.search.preparation import _budget
from core.search.profiles import profiles
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage, verify_generation
from core.search.supervisor import stop_owned_process_tree, worker_environment

PROFILE = profiles()["minilm-en"]
SETTINGS = {"batch_size": 32, "separator": PROFILE.separator, "control": "search-restart"}
VECTOR = vector_bytes([1.0] + [0.0] * 383, 384)


def plan(store, job, guard):
    item = store.load_job_by_id(job)
    report = store.load_extraction_report(item.extraction_version_id)
    chunks = make_chunks(report.passages, PROFILE, lambda _: 1, lambda text: text)
    return SearchStorage(store).create_generation(job, guard, PROFILE, chunks, SETTINGS)


def apply(store, job, guard, stage):
    storage = SearchStorage(store)
    if stage == "plan":
        plan(store, job, guard)
    else:
        index = storage.lookup_generation(job, PROFILE, SETTINGS)
        assert index is not None
        if stage == "batch":
            graph = store.load_index(index)
            storage.commit_batch(
                job, guard, index, 0, [c.id for c in graph.chunks], [VECTOR for _ in graph.chunks]
            )
        else:
            storage.finish_components(job, guard, index)


def worker(root, job, stage, boundary):
    database = Database(DatabaseConfig(Path(root)))
    database.initialize()
    store = PersistenceStore(database)
    store.claim_job(job, "crashing-search")
    attempt = store.start_step_attempt(job, stage)
    guard = AttemptGuard("crashing-search", stage, attempt.attempt_number)
    if boundary == "before":
        transaction = store._transaction

        @contextmanager
        def crash_before_commit(write=True):
            with transaction(write=write) as connection:
                yield connection
                os._exit(73)

        store._transaction = crash_before_commit
    apply(store, job, guard, stage)
    os._exit(73)


def budget_worker(root, job, index):
    database = Database(DatabaseConfig(Path(root)))
    database.initialize()
    store = PersistenceStore(database)
    guard = AttemptGuard("test-worker", "prepare", 1)
    context = StepContext(job, "prepare", 1, lambda: False, lambda: False, guard)
    with _budget(store, context, 10, 10, index):
        os._exit(73)


def test_actual_crash_keeps_time_reservation_and_restart_cannot_reset_budget(store):
    notebook, source, version, index, job, _ = seed(
        store, "rainfall budget", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    store.remove_readiness(source)
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--budget-worker",
            str(store.database.config.data_dir),
            job,
            index,
        ],
        env=worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = child.communicate(timeout=25)
        assert child.returncode == 73, (stdout, stderr)
    finally:
        stop_owned_process_tree(child)
    database = Database(store.database.config)
    database.initialize()
    restarted = PersistenceStore(database)
    with restarted.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT spent_seconds FROM search_generations WHERE index_version_id=?", (index,)
            ).fetchone()[0]
            == 10
        )
    restarted.release_running_jobs()
    restarted.claim_job(job, "budget-restarted")
    attempt = restarted.start_step_attempt(job, "prepare")
    guard = AttemptGuard("budget-restarted", "prepare", attempt.attempt_number)
    context = StepContext(
        job, "prepare", attempt.attempt_number, lambda: False, lambda: False, guard
    )
    with pytest.raises(PermanentStepFailure) as failure:
        with _budget(restarted, context, 10, 10, index):
            pytest.fail("crashed budget was reset")
    assert failure.value.code == "search_time_limit"
    assert SearchStorage(restarted).search(notebook, [version], query="rainfall") == ()


@pytest.mark.parametrize("stage", ["plan", "batch", "components"])
@pytest.mark.parametrize("boundary", ["before", "after"])
def test_search_stage_process_exit_reopens_native_state_and_reuses_exact_output(
    store: PersistenceStore, stage, boundary
):
    notebook, source, version, original, prior, _ = seed(
        store, "rainfall immutable", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    store.finish_job(prior, "completed")
    store.remove_readiness(source)
    extraction = store.load_index(original).index.extraction_version_id
    job = store.create_job(
        "search-crash:" + stage + boundary, "pdf-search", source, version, extraction
    )
    store.create_job_status(job.id)
    # Earlier stages commit normally;only the selected stage belongs to the crash child.
    store.claim_job(job.id, "preparing-search")
    attempt = store.start_step_attempt(job.id, "prepare")
    guard = AttemptGuard("preparing-search", "prepare", attempt.attempt_number)
    if stage != "plan":
        plan(store, job.id, guard)
    if stage == "components":
        apply(store, job.id, guard, "batch")
    store.release_running_jobs()
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--crash-worker",
            str(store.database.config.data_dir),
            job.id,
            stage,
            boundary,
        ],
        env=worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = child.communicate(timeout=25)
        assert child.returncode == 73, (stdout, stderr)
    finally:
        stop_owned_process_tree(child)
    database = Database(store.database.config)
    database.initialize()
    restarted = PersistenceStore(database)
    storage = SearchStorage(restarted)
    existing = storage.lookup_generation(job.id, PROFILE, SETTINGS)
    if stage == "plan" and boundary == "before":
        assert existing is None
    else:
        assert existing is not None
        with restarted.database.connect() as connection:
            complete = connection.execute(
                "SELECT complete FROM search_batches WHERE index_version_id=?",
                (existing,),
            ).fetchone()[0]
            components = connection.execute(
                "SELECT count(*) FROM search_components WHERE index_version_id=?",
                (existing,),
            ).fetchone()[0]
        assert complete == (stage == "components" or (stage == "batch" and boundary == "after"))
        assert components == (2 if stage == "components" and boundary == "after" else 0)
    assert storage.search(notebook, [version], query="rainfall") == ()
    restarted.release_running_jobs()
    restarted.claim_job(job.id, "resumed-search")
    attempt = restarted.start_step_attempt(job.id, "resume")
    current = AttemptGuard("resumed-search", "resume", attempt.attempt_number)
    index = plan(restarted, job.id, current)
    if existing:
        assert index == existing
    apply(restarted, job.id, current, "batch")
    apply(restarted, job.id, current, "components")
    graph = restarted.load_index(index)
    restarted.commit_index_step(
        job.id, "resume", graph.index, graph.chunks, graph.mappings, 1, {}, guard=current
    )
    restarted.publish_readiness(job.id, source, version, [index], guard=current)
    assert storage.search(notebook, [version], query="rainfall")[0].index_version_id == index
    with restarted.database.connect() as connection:
        verify_generation(connection, index)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute(
                "SELECT count(*) FROM search_generations WHERE index_version_id=?",
                (index,),
            ).fetchone()[0]
            == 1
        )
    # A different matching job attaches the exact same generation;changed settings do not.
    next_job = restarted.create_job(
        "cross-job:" + stage + boundary, "pdf-search", source, version, extraction
    )
    restarted.create_job_status(next_job.id)
    assert storage.lookup_generation(next_job.id, PROFILE, SETTINGS) == index
    assert (
        storage.lookup_generation(next_job.id, PROFILE, {**SETTINGS, "overlap_tokens": 7}) is None
    )
    assert storage.lookup_generation(next_job.id, profiles()["multilingual-e5"], SETTINGS) is None
    restarted.claim_job(next_job.id, "cross-job-owner")
    next_attempt = restarted.start_step_attempt(next_job.id, "attach")
    next_guard = AttemptGuard("cross-job-owner", "attach", next_attempt.attempt_number)
    assert plan(restarted, next_job.id, next_guard) == index
    reused = restarted.load_index(index)
    assert reused == graph
    restarted.commit_index_step(
        next_job.id, "attach", reused.index, reused.chunks, reused.mappings, 1, {}, guard=next_guard
    )
    restarted.publish_readiness(next_job.id, source, version, [index], guard=next_guard)
    ready = restarted.load_readiness(source)
    assert ready is not None and ready.published_by_job_id == next_job.id
    with pytest.raises(ConflictingWriteError):
        restarted.publish_readiness(job.id, source, version, [index], guard=current)
    with restarted.database.connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM search_rows WHERE index_version_id=?", (index,)
        ).fetchone()[0] == len(graph.chunks)


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--crash-worker":
    worker(*sys.argv[2:])
elif __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--budget-worker":
    budget_worker(*sys.argv[2:])
