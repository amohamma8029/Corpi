"""Import cancellation owns the exact pending PDF, not its completed extraction or old search."""

import pytest
from pdf_fixtures import PageSpec, build_pdf, image_draw_stream, native_text_pdf, native_text_stream
from test_import_expiry import HEADERS
from test_model_switch import pending_jobs
from test_model_switch_pdf import accounting, assert_saved_result
from test_replacement_safety import extract_candidate, finish_extraction, upload
from test_replacement_safety import original as original
from test_search_expiry import reopen

from core.search.handoff import SearchCoordinator
from core.search.switching import NotebookModelService


def partial_pdf():
    return build_pdf(
        [
            PageSpec(
                native_text_stream("Replacement rainfall evidence requires review."), helvetica=True
            ),
            PageSpec(image_draw_stream(), image=True),
        ]
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_cancel_review_waiting_import_preserves_completed_job_and_previous_pdf(
    store, original, replacement
):
    notebook, client, old = original
    import_id = upload(client, notebook, partial_pdf(), old.item.source_id if replacement else None)
    candidate = store.load_pdf_import(import_id)
    extraction = finish_extraction(store, import_id)
    report = store.load_extraction_report(extraction)
    assert report.outcome.outcome == "partial" and report.review.state == "pending"
    with store.database.connect() as connection:
        jobs = [tuple(row) for row in connection.execute("SELECT * FROM job_status ORDER BY rowid")]
        assert not connection.execute(
            "SELECT 1 FROM search_handoffs WHERE extraction_version_id=?", (extraction,)
        ).fetchall()
    before = accounting(store)
    path = f"/sources/{candidate.source_id}/imports/{import_id}/cancel"
    response = client.post(path, headers={"Authorization": "Bearer replacement-test-token"})
    # Use the fixture's authenticated boundary, not a direct store substitute.
    assert response.status_code == 401
    response = client.post(path, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json() == {"job_id": None, "status": "cancelled"}
    failed = store.load_pdf_import(import_id)
    assert failed.state == "failed" and failed.failure_code == "processing_cancelled"
    assert store.load_source_pdf_state(candidate.source_id).pending_import_id is None
    with store.database.connect() as connection:
        assert [
            tuple(row) for row in connection.execute("SELECT * FROM job_status ORDER BY rowid")
        ] == jobs
    assert accounting(store) == before
    assert client.post(path, headers=HEADERS).status_code == 200
    assert store.load_pdf_import(import_id) == failed
    confirmation = client.post(
        f"/extractions/{extraction}/review",
        headers=HEADERS,
        json={"decision": "confirmed", "generation": report.review.generation},
    )
    assert confirmation.status_code == 409
    restarted = reopen(store)
    SearchCoordinator(restarted, store.database.config.data_dir / "unused-models").reconcile()
    assert restarted.load_pdf_import(import_id) == failed
    assert accounting(restarted) == before
    assert_saved_result(restarted, notebook, old)


@pytest.mark.parametrize(
    "stage", ["queued-extract", "running-extract", "queued-search", "running-search"]
)
def test_cancel_active_candidate_returns_the_exact_job(store, original, stage):
    notebook, client, old = original
    import_id = upload(
        client, notebook, native_text_pdf("Replacement rainfall."), old.item.source_id
    )
    candidate = store.load_pdf_import(import_id)
    if stage.endswith("search"):
        job = extract_candidate(store, import_id)
    else:
        with store.database.connect() as connection:
            job = connection.execute(
                "SELECT id FROM jobs WHERE source_version_id=?", (candidate.source_version_id,)
            ).fetchone()[0]
    if stage.startswith("running"):
        store.claim_job(job, "cancel-control")
    response = client.post(
        f"/sources/{candidate.source_id}/imports/{import_id}/cancel", headers=HEADERS
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"job_id": job, "status": "cancelled"}
    assert store.load_job_status(job).status == "cancelled"
    assert store.load_pdf_import(import_id).failure_code == "processing_cancelled"
    assert_saved_result(reopen(store), notebook, old)


def test_import_cancel_cannot_cancel_a_ready_pdfs_model_switch(store, original):
    notebook, client, old = original
    NotebookModelService(store).choose(notebook, "multilingual-e5", 0, "ready-cancel-control")
    job = pending_jobs(store, notebook)[0][0]
    before = store.load_pdf_import(old.item.id)
    response = client.post(
        f"/sources/{old.item.source_id}/imports/{old.item.id}/cancel", headers=HEADERS
    )
    assert response.status_code == 409
    assert store.load_job_status(job).status == "queued"
    assert store.load_pdf_import(old.item.id) == before
    assert_saved_result(store, notebook, old)


def test_import_cancel_rejects_a_superseded_pending_candidate(store, original):
    notebook, client, old = original
    first = upload(client, notebook, partial_pdf(), old.item.source_id)
    finish_extraction(store, first)
    second = upload(
        client, notebook, native_text_pdf("Newer replacement rainfall."), old.item.source_id
    )
    before = store.load_pdf_import(first)
    response = client.post(f"/sources/{old.item.source_id}/imports/{first}/cancel", headers=HEADERS)
    assert response.status_code == 409
    assert store.load_pdf_import(first) == before
    assert store.load_source_pdf_state(old.item.source_id).pending_import_id == second
    assert store.load_pdf_import(second).state == "pending"
    assert_saved_result(store, notebook, old)
