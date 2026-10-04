"""Extraction report and review API (M1.5): pages, failure codes, Confirm/Decline/Review Again."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from core.api.app import create_app
from core.persistence.database import Database, DatabaseConfig
from core.persistence.models import (
    ExtractionOutcome,
    ExtractionPage,
    ExtractionReview,
    ExtractionVersion,
    ManagedFile,
    Passage,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore

TOKEN = "extraction-api-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SETTINGS = {"rules_version": "corpi-rules-1", "min_chars": 24}


def make_store(tmp_path: Path) -> PersistenceStore:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    return PersistenceStore(database)


def make_client(store: PersistenceStore) -> TestClient:
    app = create_app(token=TOKEN, store=store)
    return TestClient(app, base_url="http://127.0.0.1")


def seed_source(store: PersistenceStore, name: str = "report.pdf") -> tuple[str, str]:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Notebook {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    managed = ManagedFile(new_id(), f"originals/{name}", 17, "a" * 64, utc_timestamp())
    store.save_managed_file(managed)
    version = store.create_source_version(source_id, "a" * 64, managed.id)
    return source_id, version.id


def seed_source_with_pdf(store: PersistenceStore, name: str, pdf: bytes) -> tuple[str, str]:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Notebook {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    managed = ManagedFile(new_id(), f"originals/{name}", len(pdf), "a" * 64, utc_timestamp())
    store.save_managed_file(managed)
    path = store.database.config.data_dir / managed.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)
    version = store.create_source_version(source_id, "a" * 64, managed.id)
    return source_id, version.id


def save_partial_extraction(store: PersistenceStore, source_id: str, version_id: str) -> str:
    extraction = ExtractionVersion(
        new_id(),
        source_id,
        version_id,
        "pdfminer.six",
        "20260107",
        "faithfully-normalized-text-v1",
        1,
        SETTINGS,
        utc_timestamp(),
    )
    passage = Passage(
        new_id(),
        source_id,
        version_id,
        extraction.id,
        0,
        "Readable body",
        1,
        {"page": 1, "block": "text", "bbox": [0, 0, 1, 1], "reading_order": 0},
        utc_timestamp(),
    )
    page = ExtractionPage(
        new_id(),
        source_id,
        version_id,
        extraction.id,
        1,
        True,
        40,
        0.0,
        0.0,
        "extracted",
        utc_timestamp(),
    )
    outcome = ExtractionOutcome(extraction.id, source_id, version_id, "partial", utc_timestamp())
    review = ExtractionReview(
        extraction.id, source_id, version_id, "pending", 1, None, utc_timestamp()
    )
    store.save_extraction_result(extraction, (passage,), (page,), (), outcome, review)
    return extraction.id


def test_extraction_routes_require_authentication(tmp_path: Path) -> None:
    client = make_client(make_store(tmp_path))
    assert client.get(f"/sources/{new_id()}/extraction").status_code == 401
    assert client.post(f"/extractions/{new_id()}/review").status_code == 401


def test_report_returns_pages_outcome_and_review(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source_id, version_id = seed_source(store)
    save_partial_extraction(store, source_id, version_id)
    response = make_client(store).get(f"/sources/{source_id}/extraction", headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "partial"
    assert payload["passage_count"] == 1
    assert payload["pages"][0]["classification"] == "extracted"
    assert payload["review"] == {
        "state": "pending",
        "generation": 1,
        "decided_at": None,
    }
    assert payload["failure_code"] is None


def test_failed_extraction_surfaces_the_failure_code(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source_id, version_id = seed_source(store)
    job = store.create_job("extract:fail", "pdf-extract", source_id, source_version_id=version_id)
    store.create_job_status(job.id)
    store.finish_job(
        job.id,
        "failed",
        "step extract-text failed permanently: no_usable_text: the document has no usable text",
        failure_code="no_usable_text",
    )
    payload = make_client(store).get(f"/sources/{source_id}/extraction", headers=AUTH).json()
    assert payload["extraction_id"] is None
    assert payload["failure_code"] == "no_usable_text"
    assert "no usable text" in payload["failure_detail"]


def test_failure_codes_reach_the_report_endpoint_through_the_runner(tmp_path: Path) -> None:
    import asyncio
    import threading

    from pdf_fixtures import PageSpec, blank_pdf, build_pdf, native_text_stream

    from core.jobs.runner import JobRunner
    from core.jobs.steps import build_pdf_extract_steps

    store = make_store(tmp_path)
    client = make_client(store)
    runner = JobRunner(store, {"pdf-extract": build_pdf_extract_steps(store)})
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def main() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.start())
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=main, daemon=True)
    thread.start()
    assert started.wait(5)
    try:
        over_limit = build_pdf(
            [PageSpec(native_text_stream(f"page {i}"), helvetica=True) for i in range(201)]
        )
        for name, pdf, code in (
            ("blank.pdf", blank_pdf(), "no_usable_text"),
            ("over.pdf", over_limit, "page_limit_exceeded"),
        ):
            source_id, version_id = seed_source_with_pdf(store, name, pdf)
            job = store.create_job(
                f"extract:api:{name}", "pdf-extract", source_id, source_version_id=version_id
            )
            store.create_job_status(job.id)
            runner.notify()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status = store.load_job_status(job.id)
                if status is not None and status.status == "failed":
                    break
                time.sleep(0.02)
            status = store.load_job_status(job.id)
            assert status is not None and status.status == "failed"
            assert status.failure_code == code
            payload = client.get(f"/sources/{source_id}/extraction", headers=AUTH).json()
            assert payload["failure_code"] == code
            assert code in payload["failure_detail"]
    finally:
        asyncio.run_coroutine_threadsafe(runner.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)


def test_source_without_extraction_reports_empty(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source_id, _ = seed_source(store)
    payload = make_client(store).get(f"/sources/{source_id}/extraction", headers=AUTH).json()
    assert payload["extraction_id"] is None
    assert payload["outcome"] is None
    assert payload["pages"] == []


def test_confirm_reopen_and_decline_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source_id, version_id = seed_source(store)
    extraction_id = save_partial_extraction(store, source_id, version_id)
    client = make_client(store)

    confirmed = client.post(
        f"/extractions/{extraction_id}/review",
        headers=AUTH,
        json={"decision": "confirmed", "generation": 1},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["state"] == "confirmed"

    reopened = client.post(f"/extractions/{extraction_id}/review/reopen", headers=AUTH)
    assert reopened.status_code == 200
    assert (reopened.json()["state"], reopened.json()["generation"]) == ("pending", 2)

    declined = client.post(
        f"/extractions/{extraction_id}/review",
        headers=AUTH,
        json={"decision": "declined", "generation": 2},
    )
    assert declined.status_code == 200
    assert declined.json()["state"] == "declined"


def test_confirm_rechecks_currency_and_rejects_conflicts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source_id, version_id = seed_source(store)
    extraction_id = save_partial_extraction(store, source_id, version_id)
    managed = ManagedFile(new_id(), "originals/newer.pdf", 17, "b" * 64, utc_timestamp())
    store.save_managed_file(managed)
    store.create_source_version(source_id, "b" * 64, managed.id)
    response = make_client(store).post(
        f"/extractions/{extraction_id}/review",
        headers=AUTH,
        json={"decision": "confirmed", "generation": 1},
    )
    assert response.status_code == 409
    assert "not current" in response.json()["detail"]

    stale = make_client(store).post(
        f"/extractions/{extraction_id}/review",
        headers=AUTH,
        json={"decision": "confirmed", "generation": 9},
    )
    assert stale.status_code in (400, 409)


def test_unknown_decision_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source_id, version_id = seed_source(store)
    extraction_id = save_partial_extraction(store, source_id, version_id)
    response = make_client(store).post(
        f"/extractions/{extraction_id}/review",
        headers=AUTH,
        json={"decision": "maybe", "generation": 1},
    )
    assert response.status_code == 400


def test_real_import_extraction_confirm_flow(tmp_path: Path) -> None:
    """A real import stays `pending` through extraction and must still be confirmable."""
    import asyncio
    import threading

    from pdf_fixtures import (
        PageSpec,
        build_pdf,
        image_draw_stream,
        native_text_stream,
        page_number_stream,
    )

    from core.ingest.service import PdfIngestService
    from core.jobs.runner import JobRunner
    from core.jobs.steps import build_pdf_extract_steps

    origin = "http://tauri.localhost"
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(store, "core-test")
    app = create_app(TOKEN, allowed_origins=(origin,), store=store, pdf_ingest=ingest)
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": origin}
    pdf = build_pdf(
        [
            PageSpec(native_text_stream("Readable body text for the first page"), helvetica=True),
            PageSpec(
                image_draw_stream() + b"\n" + page_number_stream(2),
                helvetica=True,
                image=True,
            ),
        ]
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        notebook_id = new_id()
        store.create_notebook(notebook_id, "Real flow")
        created = client.post(
            f"/notebooks/{notebook_id}/pdf-acquisitions",
            headers=headers,
            json={"retry_key": "real-flow", "filename": "real.pdf"},
        )
        assert created.status_code == 201, created.text
        uploaded = client.put(
            f"/pdf-acquisitions/{created.json()['id']}/content",
            headers={**headers, "Content-Type": "application/pdf"},
            content=pdf,
        )
        assert uploaded.status_code == 200, uploaded.text
        outcome = uploaded.json()
        source_id = outcome["source_id"]
        import_id = outcome["import_id"]
        assert source_id is not None and import_id is not None
        assert store.load_pdf_import(import_id).state == "pending"

        runner = JobRunner(store, {"pdf-extract": build_pdf_extract_steps(store)})
        loop = asyncio.new_event_loop()
        started = threading.Event()

        def main() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner.start())
            started.set()
            loop.run_forever()

        thread = threading.Thread(target=main, daemon=True)
        thread.start()
        assert started.wait(5)
        try:
            runner.notify()
            deadline = time.monotonic() + 30
            report = None
            while time.monotonic() < deadline:
                report = client.get(f"/sources/{source_id}/extraction", headers=headers).json()
                if report.get("outcome"):
                    break
                time.sleep(0.05)
            assert report is not None and report["outcome"] == "partial"
            assert store.load_pdf_import(import_id).state == "pending"
            decided = client.post(
                f"/extractions/{report['extraction_id']}/review",
                headers=headers,
                json={"decision": "confirmed", "generation": 1},
            )
            assert decided.status_code == 200, decided.text
            assert decided.json()["state"] == "confirmed"
        finally:
            asyncio.run_coroutine_threadsafe(runner.stop(), loop).result(10)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
