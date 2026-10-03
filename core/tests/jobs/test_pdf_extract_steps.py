"""M1.5 `pdf-extract` steps on the 1.3 runner: records, reuse, failure codes, stuck child."""

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path

from pdf_fixtures import (
    PageSpec,
    blank_pdf,
    build_pdf,
    image_draw_stream,
    native_text_pdf,
    native_text_stream,
    page_number_stream,
)

from core.extract.supervisor import ExtractionCancelled, run_extraction
from core.jobs.runner import (
    JobRunner,
    StepContext,
    StepDefinition,
    StepPolicy,
)
from core.jobs.steps import (
    EXTRACTION_SETTINGS,
    NORMALIZATION_VERSION,
    PDF_EXTRACT_STEP_KEY,
    build_pdf_extract_steps,
    register_pdf_extract_steps,
)
from core.persistence.models import ManagedFile, new_id, utc_timestamp
from core.persistence.store import PersistenceStore

FAST_POLICY = StepPolicy(timeout_seconds=5.0, attempt_cap=2, backoff_base_seconds=0.0)


def wait_until(predicate: Callable[[], bool], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class RunnerHarness:
    """Runs the runner's event loop on a background thread for sync tests."""

    def __init__(self, store: PersistenceStore, step_sets, **kwargs) -> None:
        self.runner = JobRunner(store, step_sets, **kwargs)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._main, daemon=True)
        self._started = threading.Event()

    def _main(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.runner.start())
        self._started.set()
        self.loop.run_forever()

    def start(self) -> None:
        self.thread.start()
        assert self._started.wait(5)

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.runner.stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)

    def enqueue_and_notify(self, job_id: str) -> None:
        self.runner._store.create_job_status(job_id)  # noqa: SLF001 - harness wiring
        self.runner.notify()


def seed_extraction_source(store: PersistenceStore, name: str, pdf: bytes) -> tuple[str, str]:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Notebook {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    managed = ManagedFile(
        new_id(),
        f"originals/{name}",
        len(pdf),
        "c" * 64,
        utc_timestamp(),
    )
    store.save_managed_file(managed)
    path = store.database.config.data_dir / managed.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)
    source_version = store.create_source_version(source_id, "c" * 64, managed.id)
    return source_id, source_version.id


def enqueue_extract(
    store: PersistenceStore, key: str, source_id: str, source_version_id: str
) -> str:
    job = store.create_job(key, "pdf-extract", source_id, source_version_id=source_version_id)
    return job.id


def job_status(store: PersistenceStore, job_id: str) -> str:
    status = store.load_job_status(job_id)
    assert status is not None
    return status.status


def partial_pdf() -> bytes:
    return build_pdf(
        [
            PageSpec(native_text_stream("Readable body text for the first page"), helvetica=True),
            PageSpec(
                image_draw_stream() + b"\n" + page_number_stream(2),
                helvetica=True,
                image=True,
            ),
        ]
    )


def test_extraction_job_commits_records_and_completes(store: PersistenceStore) -> None:
    source_id, version_id = seed_extraction_source(
        store, "native.pdf", native_text_pdf("Hello Corpi extraction body")
    )
    steps = register_pdf_extract_steps(store)
    runner = RunnerHarness(store, steps)
    job_id = enqueue_extract(store, "pdf-extract:native", source_id, version_id)
    runner.start()
    try:
        runner.enqueue_and_notify(job_id)
        assert wait_until(lambda: job_status(store, job_id) == "completed")
    finally:
        runner.stop()
    job_steps = store.load_job_steps(job_id)
    assert [row.step_key for row in job_steps] == [PDF_EXTRACT_STEP_KEY]
    extraction_id = job_steps[0].output_extraction_version_id
    assert extraction_id is not None
    report = store.load_extraction_report(extraction_id)
    assert report.outcome is not None and report.outcome.outcome == "complete"
    assert report.review is None
    assert [page.classification for page in report.pages] == ["extracted"]
    assert any("Hello Corpi extraction body" in passage.text for passage in report.passages)
    location = report.passages[0].location
    assert isinstance(location, dict)
    assert location["page"] == 1


def test_partial_document_requires_review(store: PersistenceStore) -> None:
    source_id, version_id = seed_extraction_source(store, "mixed.pdf", partial_pdf())
    register_pdf_extract_steps(store)
    runner = RunnerHarness(store, {"pdf-extract": build_pdf_extract_steps(store)})
    job_id = enqueue_extract(store, "pdf-extract:mixed", source_id, version_id)
    runner.start()
    try:
        runner.enqueue_and_notify(job_id)
        assert wait_until(lambda: job_status(store, job_id) == "completed")
    finally:
        runner.stop()
    extraction_id = store.load_job_steps(job_id)[0].output_extraction_version_id
    assert extraction_id is not None
    report = store.load_extraction_report(extraction_id)
    assert report.outcome is not None and report.outcome.outcome == "partial"
    assert report.review is not None
    assert (report.review.state, report.review.generation) == ("pending", 1)
    assert [page.classification for page in report.pages] == ["extracted", "image-only"]


def test_no_usable_text_fails_the_job_with_the_code(store: PersistenceStore) -> None:
    source_id, version_id = seed_extraction_source(store, "blank.pdf", blank_pdf())
    register_pdf_extract_steps(store)
    runner = RunnerHarness(store, {"pdf-extract": build_pdf_extract_steps(store)})
    job_id = enqueue_extract(store, "pdf-extract:blank", source_id, version_id)
    runner.start()
    try:
        runner.enqueue_and_notify(job_id)
        assert wait_until(lambda: job_status(store, job_id) == "failed")
    finally:
        runner.stop()
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.failure_reason is not None
    assert "no_usable_text" in status.failure_reason
    assert store.load_job_steps(job_id) == ()


def test_page_limit_fails_with_the_code_without_committing(store: PersistenceStore) -> None:
    over_limit = build_pdf(
        [PageSpec(native_text_stream(f"page {index}"), helvetica=True) for index in range(201)]
    )
    source_id, version_id = seed_extraction_source(store, "long.pdf", over_limit)
    register_pdf_extract_steps(store)
    runner = RunnerHarness(store, {"pdf-extract": build_pdf_extract_steps(store)})
    job_id = enqueue_extract(store, "pdf-extract:long", source_id, version_id)
    runner.start()
    try:
        runner.enqueue_and_notify(job_id)
        assert wait_until(lambda: job_status(store, job_id) == "failed")
    finally:
        runner.stop()
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.failure_reason is not None
    assert "page_limit_exceeded" in status.failure_reason
    assert store.load_job_steps(job_id) == ()


def test_cross_job_reuse_preserves_the_saved_extraction(store: PersistenceStore) -> None:
    source_id, version_id = seed_extraction_source(
        store, "retry.pdf", native_text_pdf("Stable extraction body")
    )
    register_pdf_extract_steps(store)
    runner = RunnerHarness(store, {"pdf-extract": build_pdf_extract_steps(store)})
    first_job = enqueue_extract(store, "pdf-extract:retry:1", source_id, version_id)
    runner.start()
    try:
        runner.enqueue_and_notify(first_job)
        assert wait_until(lambda: job_status(store, first_job) == "completed")
        second_job = enqueue_extract(store, "pdf-extract:retry:2", source_id, version_id)
        runner.enqueue_and_notify(second_job)
        assert wait_until(lambda: job_status(store, second_job) == "completed")
    finally:
        runner.stop()
    first_extraction = store.load_job_steps(first_job)[0].output_extraction_version_id
    second_extraction = store.load_job_steps(second_job)[0].output_extraction_version_id
    assert first_extraction == second_extraction


def test_retry_before_commit_reruns_and_commits_once(store: PersistenceStore) -> None:
    source_id, version_id = seed_extraction_source(
        store, "retry-once.pdf", native_text_pdf("Recoverable extraction body")
    )
    attempts: list[int] = []
    real_run = build_pdf_extract_steps(store)[0].run

    flaky_steps = build_pdf_extract_steps(store)

    def flaky_run(context):
        attempts.append(context.attempt_number)
        if len(attempts) == 1:
            raise RuntimeError("simulated crash before commit")
        return real_run(context)

    step = flaky_steps[0]
    steps = (
        StepDefinition(
            step_key=step.step_key,
            run=flaky_run,
            policy_class=step.policy_class,
            commit=step.commit,
            inputs=step.inputs,
        ),
    )
    runner = RunnerHarness(store, {"pdf-extract": steps})
    job_id = enqueue_extract(store, "pdf-extract:flaky", source_id, version_id)
    runner.start()
    try:
        runner.enqueue_and_notify(job_id)
        assert wait_until(lambda: job_status(store, job_id) == "completed")
    finally:
        runner.stop()
    assert len(attempts) == 2
    assert len(store.load_job_steps(job_id)) == 1


def test_stuck_child_keeps_status_responsive_and_frees_the_runner(
    store: PersistenceStore,
) -> None:
    source_id, version_id = seed_extraction_source(
        store, "stuck.pdf", native_text_pdf("Stuck extraction body")
    )
    second_source, second_version = seed_extraction_source(
        store, "queued.pdf", native_text_pdf("Queued extraction body")
    )
    started = threading.Event()
    observed_cancel = threading.Event()

    def blocking_extract(pdf_path: Path, *, timeout_s: float, cancel=None, **kwargs):
        if "stuck" in pdf_path.name:
            started.set()
            while cancel is None or not cancel.is_set():
                time.sleep(0.02)
            observed_cancel.set()
            raise ExtractionCancelled("stopped")
        return run_extraction(pdf_path, timeout_s=timeout_s, cancel=cancel, **kwargs)

    register_pdf_extract_steps(store)
    runner = RunnerHarness(
        store,
        {"pdf-extract": build_pdf_extract_steps(store, extract=blocking_extract)},
        policies={"extract": FAST_POLICY},
    )
    stuck_job = enqueue_extract(store, "pdf-extract:stuck", source_id, version_id)
    queued_job = enqueue_extract(store, "pdf-extract:queued", second_source, second_version)
    runner.start()
    try:
        runner.enqueue_and_notify(stuck_job)
        assert started.wait(5)
        snapshot = runner.runner.snapshot()
        assert snapshot.state in ("running", "stalled")
        assert store.load_job_status(stuck_job) is not None
        runner.enqueue_and_notify(queued_job)
        assert wait_until(lambda: job_status(store, stuck_job) == "failed", timeout=30)
        assert wait_until(lambda: job_status(store, queued_job) == "completed", timeout=30)
    finally:
        runner.stop()
    assert observed_cancel.is_set()
    stuck_status = store.load_job_status(stuck_job)
    assert stuck_status is not None and stuck_status.status == "failed"


def test_inputs_fingerprint_includes_rule_versions_and_settings(store: PersistenceStore) -> None:
    steps = build_pdf_extract_steps(store)
    probe = StepContext(
        job_id="x",
        step_key="s",
        attempt_number=0,
        is_cancelled=lambda: False,
        should_stop=lambda: False,
    )
    inputs_fn = steps[0].inputs
    assert inputs_fn is not None
    payload = inputs_fn(probe)
    assert isinstance(payload, dict)
    assert payload["normalization_version"] == NORMALIZATION_VERSION
    assert payload["settings"] == EXTRACTION_SETTINGS
