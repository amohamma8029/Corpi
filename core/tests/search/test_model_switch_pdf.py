"""A switch failure must not rewrite a saved PDF's successful import.

Real PDF extraction, persisted native indexes and authenticated APIs are exercised;
document vectors are controlled by the existing lifecycle fixture, not new inference.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import native_text_pdf
from test_import_expiry import HEADERS, TOKEN
from test_model_switch import pending_jobs
from test_replacement_safety import extract_candidate, upload
from test_replacement_safety import original as original
from test_search_expiry import reopen

from core.api.app import create_app
from core.ingest.service import PdfIngestService
from core.persistence.errors import ConflictingWriteError
from core.search.query import QueryService
from core.search.switching import NotebookModelService


def accounting(store):
    with store.database.connect() as connection:
        return {
            table: [
                tuple(row)
                for row in connection.execute("SELECT * FROM " + table + " ORDER BY rowid")
            ]
            for table in (
                "search_generations",
                "search_batches",
                "search_job_inputs",
                "job_step_attempts",
                "source_readiness",
                "source_readiness_indexes",
            )
        }


def assert_saved_result(store, notebook, old):
    app = create_app(
        TOKEN,
        store=store,
        pdf_ingest=PdfIngestService(store, "switch-pdf-reader"),
        notebook_models=NotebookModelService(store),
        search_queries=QueryService(store, Path("unused-no-inference")),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            f"/notebooks/{notebook}/search",
            headers=HEADERS,
            json={"query": "rainfall", "mode": "words"},
        )
        assert response.status_code == 200, response.text
        hits = response.json()["results"]
        assert len(hits) == 1
        hit = hits[0]
        assert hit["source_id"] == old.item.source_id
        assert hit["source_version_id"] == old.item.source_version_id
        assert hit["index_version_id"] == old.index
        assert hit["import_id"] == old.item.id
        assert [citation["page_number"] for citation in hit["citations"]] == [1]
        saved = client.get(
            f"/sources/{hit['source_id']}/imports/{hit['import_id']}/pdf",
            headers=HEADERS,
        )
        assert saved.status_code == 200
        assert saved.headers["content-type"] == "application/pdf"
        assert saved.content == old.content


@pytest.mark.parametrize(
    "outcome", ["failed", "finished-cancelled", "cancel", "selection-cancel", "failed-then-cancel"]
)
def test_switch_terminal_state_preserves_saved_pdf_and_exact_search_page(store, original, outcome):
    notebook, _, old = original
    before_import = store.load_pdf_import(old.item.id)
    before_pdf_state = store.load_source_pdf_state(old.item.source_id)
    assert before_import.state == "ready"
    assert_saved_result(store, notebook, old)
    service = NotebookModelService(store)
    selection = service.choose(notebook, "multilingual-e5", 0, "switch-pdf-control")
    job = pending_jobs(store, notebook)[0][0]
    before = accounting(store)
    if outcome.startswith("failed"):
        store.finish_job(
            job, "failed", "controlled worker failure", failure_code="embedding_worker_failed"
        )
    elif outcome == "finished-cancelled":
        store.finish_job(job, "cancelled", "controlled cancellation")
    elif outcome == "cancel":
        store.cancel_job(job)
    if outcome in {"selection-cancel", "failed-then-cancel"}:
        service.cancel(notebook, selection["selection_generation"])
    assert store.load_pdf_import(old.item.id) == before_import
    assert store.load_source_pdf_state(old.item.source_id) == before_pdf_state
    assert accounting(store) == before
    assert old.original.read_bytes() == old.content
    assert_saved_result(store, notebook, old)
    restarted = reopen(store)
    assert restarted.load_pdf_import(old.item.id) == before_import
    assert accounting(restarted) == before
    assert_saved_result(restarted, notebook, old)
    with pytest.raises(ConflictingWriteError):
        restarted.finish_job(job, "failed", "late terminal write")
    assert restarted.load_pdf_import(old.item.id) == before_import


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("stage", ["extract", "search"])
@pytest.mark.parametrize("outcome", ["failed", "finished-cancelled", "cancel", "explicit-cancel"])
def test_actual_import_terminal_state_still_fails_only_its_import(
    store, original, replacement, stage, outcome
):
    notebook, client, old = original
    content = native_text_pdf("New rainfall candidate must not hide the previous saved PDF.")
    import_id = upload(client, notebook, content, old.item.source_id if replacement else None)
    candidate = store.load_pdf_import(import_id)
    if stage == "search":
        job = extract_candidate(store, import_id)
    else:
        with store.database.connect() as connection:
            job = connection.execute(
                "SELECT id FROM jobs WHERE source_version_id=? AND kind='pdf-extract'",
                (candidate.source_version_id,),
            ).fetchone()[0]
            assert (
                connection.execute(
                    "SELECT 1 FROM search_handoffs WHERE job_id=?", (job,)
                ).fetchone()
                is None
            )
    if outcome == "failed":
        store.finish_job(
            job, "failed", "genuine import failure", failure_code="import_control_failed"
        )
    elif outcome == "finished-cancelled":
        store.finish_job(job, "cancelled", "genuine import cancellation")
    elif outcome == "cancel":
        store.cancel_job(job)
    else:
        assert store.cancel_pdf_import_processing(import_id) == job
    failed = store.load_pdf_import(import_id)
    assert failed.state == "failed"
    expected_code = {
        "failed": "import_control_failed",
        "finished-cancelled": "processing_failed",
        "cancel": "processing_cancelled",
        "explicit-cancel": "processing_cancelled",
    }[outcome]
    assert failed.failure_code == expected_code
    state = store.load_source_pdf_state(candidate.source_id)
    assert state is not None and state.pending_import_id is None
    assert store.load_pdf_import(old.item.id).state == "ready"
    assert old.original.read_bytes() == old.content
    assert_saved_result(reopen(store), notebook, old)
