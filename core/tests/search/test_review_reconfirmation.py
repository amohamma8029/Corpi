"""Full authenticated Review Again/Confirm lifecycle with real PDF/index publication."""

import pytest
from test_import_cancellation import partial_pdf
from test_import_expiry import HEADERS
from test_model_switch import complete, mark_installed, pending_jobs
from test_replacement_safety import finish_extraction, upload
from test_replacement_safety import original as original
from test_search_expiry import reopen
from test_search_safety import prepare

from core.persistence.errors import ConflictingWriteError
from core.search.handoff import SearchCoordinator
from core.search.query import QueryService
from core.search.switching import NotebookModelService


def test_review_again_confirm_reschedules_exact_generation_and_reuses_saved_vectors(
    store, original
):
    notebook, client, old = original
    candidate = store.load_pdf_import(upload(client, notebook, partial_pdf(), old.item.source_id))
    extraction = finish_extraction(store, candidate.id)
    path = f"/extractions/{extraction}/review"
    assert (
        client.post(
            path, headers=HEADERS, json={"decision": "confirmed", "generation": 1}
        ).status_code
        == 200
    )
    with store.database.connect() as connection:
        job = connection.execute(
            "SELECT job_id FROM search_handoffs WHERE extraction_version_id=?", (extraction,)
        ).fetchone()[0]
    index, guard = prepare(store, job)
    store.publish_readiness(
        job, candidate.source_id, candidate.source_version_id, [index], guard=guard
    )
    store.finish_step_attempt(job, "prepare", guard.expected_attempt_number, None)
    store.finish_job(job, "completed")
    service = QueryService(store, store.database.config.data_dir / "unused-models")
    assert service.search(notebook, "rainfall", "words")["results"]
    response = client.post(path + "/reopen", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["generation"] == 2 and response.json()["state"] == "pending"
    assert not service.search(notebook, "rainfall", "words")["results"]
    assert (
        client.post(
            path, headers=HEADERS, json={"decision": "confirmed", "generation": 1}
        ).status_code
        == 409
    )
    response = client.post(path, headers=HEADERS, json={"decision": "confirmed", "generation": 2})
    assert response.status_code == 200, response.text
    with store.database.connect() as connection:
        fresh = connection.execute(
            "SELECT job_id FROM search_handoffs WHERE extraction_version_id=? "
            "AND review_generation=2",
            (extraction,),
        ).fetchone()[0]
        batches = [
            tuple(row) for row in connection.execute("SELECT * FROM search_batches ORDER BY rowid")
        ]
    assert fresh != job
    progress = client.get(f"/sources/{candidate.source_id}/search-progress", headers=HEADERS).json()
    assert progress["profile_id"] == "minilm-en"
    assert progress["filename"] == candidate.display_filename and progress["pdf_role"] == "current"
    restarted = reopen(store)
    SearchCoordinator(restarted, store.database.config.data_dir / "unused-models").reconcile()
    fresh_index, fresh_guard = prepare(restarted, fresh)
    assert fresh_index == index
    restarted.publish_readiness(
        fresh, candidate.source_id, candidate.source_version_id, [fresh_index], guard=fresh_guard
    )
    with pytest.raises(ConflictingWriteError):
        restarted.publish_readiness(
            job, candidate.source_id, candidate.source_version_id, [index], guard=guard
        )
    with restarted.database.connect() as connection:
        assert [
            tuple(row) for row in connection.execute("SELECT * FROM search_batches ORDER BY rowid")
        ] == batches
    results = QueryService(restarted, store.database.config.data_dir / "unused-models").search(
        notebook, "rainfall", "words"
    )["results"]
    assert results and results[0]["import_id"] == candidate.id
    assert results[0]["citations"][0]["page_number"] == 1
    assert old.original.read_bytes() == old.content
    assert_no_import_rebuild_after_switch(restarted, notebook)


def assert_no_import_rebuild_after_switch(store, notebook):
    mark_installed(store)
    models = NotebookModelService(store)
    models.choose(notebook, "multilingual-e5", 0, "no-duplicate-import-after-switch")
    for row in pending_jobs(store, notebook):
        complete(store, row[0])
    assert models.state(notebook)["pending_manifest_id"] is None
    with store.database.connect() as connection:
        before = [tuple(row) for row in connection.execute("SELECT * FROM jobs ORDER BY rowid")]
    SearchCoordinator(store, store.database.config.data_dir / "unused-models").reconcile()
    with store.database.connect() as connection:
        assert [
            tuple(row) for row in connection.execute("SELECT * FROM jobs ORDER BY rowid")
        ] == before


def test_completed_pdf_switch_does_not_enqueue_an_import_rebuild_on_reconcile(store, original):
    notebook, _, _ = original
    assert_no_import_rebuild_after_switch(store, notebook)
