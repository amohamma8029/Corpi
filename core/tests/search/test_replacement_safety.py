"""Real PDF replacement actions with controlled native indexes and pending model rebuilds."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import native_text_pdf
from test_import_expiry import HEADERS, TOKEN, publish_import
from test_model_switch import complete, mark_installed, pending_jobs
from test_search_expiry import reopen
from test_search_safety import VALUES, native_hits, prepare
from test_storage import seed

from core.api.app import create_app
from core.ingest.service import PdfIngestService
from core.jobs.runner import StepCommitContext, StepContext
from core.jobs.steps import build_pdf_extract_steps
from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard, new_id, utc_timestamp
from core.search.handoff import SearchCoordinator
from core.search.profiles import profiles
from core.search.switching import NotebookModelService, publish_replacement


def upload(client, notebook, content, target=None, *, filename="replacement.pdf"):
    created = client.post(
        f"/notebooks/{notebook}/pdf-acquisitions",
        headers=HEADERS,
        json={
            "retry_key": new_id(),
            "filename": filename,
            "intent": "replace" if target else "add",
            "target_source_id": target,
        },
    )
    assert created.status_code == 201, created.text
    result = client.put(
        f"/pdf-acquisitions/{created.json()['id']}/content",
        headers={**HEADERS, "Content-Type": "application/pdf"},
        content=content,
    )
    assert result.status_code == 200, result.text
    return result.json()["import_id"]


def finish_extraction(store, import_id):
    """Complete real extraction, including partial PDFs that cannot schedule search yet."""
    item = store.load_pdf_import(import_id)
    with store.database.connect() as connection:
        job = connection.execute(
            "SELECT id FROM jobs WHERE source_version_id=? AND kind='pdf-extract'",
            (item.source_version_id,),
        ).fetchone()[0]
    step = build_pdf_extract_steps(store)[0]
    store.claim_job(job, "replacement-extract")
    attempt = store.start_step_attempt(job, step.step_key)
    guard = AttemptGuard("replacement-extract", step.step_key, attempt.attempt_number)
    context = StepContext(
        job, step.step_key, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    assert step.inputs is not None
    fingerprint = hashlib.sha256(
        json.dumps(step.inputs(context), sort_keys=True).encode()
    ).hexdigest()
    step.commit(store, StepCommitContext(job, step.step_key, guard, fingerprint, step.run(context)))
    store.finish_step_attempt(job, step.step_key, attempt.attempt_number, None)
    store.finish_job(job, "completed")
    extraction = store.load_job_steps(job)[0].output_extraction_version_id
    assert extraction is not None
    return extraction


def extract_candidate(store, import_id):
    extraction = finish_extraction(store, import_id)
    with store.database.connect() as connection:
        return connection.execute(
            "SELECT job_id FROM search_handoffs WHERE extraction_version_id=? AND purpose='import'",
            (extraction,),
        ).fetchone()[0]


@pytest.fixture
def original(store):
    notebook_id = new_id()
    notebook = store.create_notebook(notebook_id, notebook_id)
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
            " VALUES(?,?,?)",
            (notebook.id, profiles()["minilm-en"].identity, utc_timestamp()),
        )
    content = native_text_pdf("Original rainfall evidence must survive a failed replacement.")
    ingest = PdfIngestService(store, "replacement-tests")
    with TestClient(
        create_app(TOKEN, store=store, pdf_ingest=ingest), base_url="http://127.0.0.1"
    ) as client:
        old = publish_import(store, upload(client, notebook.id, content), content)
        yield notebook.id, client, old


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "ready"])
def test_native_replacement_failure_cancel_success_and_stale_publication(store, original, outcome):
    notebook, client, old = original
    content = native_text_pdf(
        "New rainfall evidence belongs only to its exact replacement version."
    )
    candidate = store.load_pdf_import(upload(client, notebook, content, old.item.source_id))
    job = extract_candidate(store, candidate.id)
    index, guard = prepare(store, job)
    assert all(native_hits(store, notebook, [old.item.source_version_id]))
    assert native_hits(store, notebook, [candidate.source_version_id]) == ((), ())
    if outcome == "failed":
        store.finish_job(job, "failed", "controlled failure before publication")
    elif outcome == "cancelled":
        store.cancel_job(job)
    else:
        store.publish_readiness(
            job, candidate.source_id, candidate.source_version_id, [index], guard=guard
        )
    restarted = reopen(store)
    if outcome != "ready":
        with pytest.raises(ConflictingWriteError):
            restarted.publish_readiness(
                job, candidate.source_id, candidate.source_version_id, [index], guard=guard
            )
        assert restarted.load_pdf_import(old.item.id).state == "ready"
        ready = restarted.load_readiness(old.item.source_id)
        assert ready is not None and ready.source_version_id == old.item.source_version_id
        assert all(native_hits(restarted, notebook, [old.item.source_version_id]))
        assert native_hits(restarted, notebook, [candidate.source_version_id]) == ((), ())
    else:
        assert restarted.load_pdf_import(candidate.id).state == "ready"
        assert native_hits(restarted, notebook, [old.item.source_version_id]) == ((), ())
        assert all(native_hits(restarted, notebook, [candidate.source_version_id]))
        with restarted.database.connect() as connection:
            old_job = connection.execute(
                "SELECT job_id FROM job_steps WHERE output_index_version_id=?",
                (old.index,),
            ).fetchone()[0]
        with pytest.raises(ConflictingWriteError):
            restarted.publish_readiness(
                old_job,
                old.item.source_id,
                old.item.source_version_id,
                [old.index],
                guard=AttemptGuard("history-test", "prepare", 1),
            )
    assert old.original.read_bytes() == old.content
    assert store.load_index(old.index).chunks == restarted.load_index(old.index).chunks


def test_ready_replacement_during_pending_switch_requires_the_new_version_rebuild(store, original):
    notebook, client, old = original
    seed(store, "rainfall second source", VALUES, notebook_id=notebook, profile_id="minilm-en")
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "replace-during-switch")
    jobs = [r[0] for r in pending_jobs(store, notebook)]
    old_rebuild = next(j for j in jobs if store.load_job_by_id(j).source_id == old.item.source_id)
    stale_index, stale_guard = complete(store, old_rebuild)
    content = native_text_pdf("Replacement rainfall evidence changes the ready source version.")
    candidate = store.load_pdf_import(upload(client, notebook, content, old.item.source_id))
    job = extract_candidate(store, candidate.id)
    index, guard = prepare(store, job)
    store.publish_readiness(
        job, candidate.source_id, candidate.source_version_id, [index], guard=guard
    )
    restarted = reopen(store)
    SearchCoordinator(restarted, Path("unused")).reconcile()
    with pytest.raises(ConflictingWriteError):
        publish_replacement(restarted, old_rebuild, stale_guard, stale_index)
    complete(restarted, next(j for j in jobs if j != old_rebuild))
    assert NotebookModelService(restarted).state(notebook)["pending_manifest_id"] is not None
    fresh = next(
        r[0]
        for r in pending_jobs(restarted, notebook)
        if restarted.load_job_by_id(r[0]).source_version_id == candidate.source_version_id
    )
    complete(restarted, fresh)
    assert NotebookModelService(restarted).state(notebook)["pending_manifest_id"] is None
    ready = restarted.load_readiness(candidate.source_id)
    assert ready is not None
    assert ready.source_version_id == candidate.source_version_id
    assert all(native_hits(restarted, notebook, [candidate.source_version_id]))
    assert all(
        all(h.manifest_id == profiles()["multilingual-e5"].identity for h in method)
        for method in native_hits(restarted, notebook, [candidate.source_version_id])
    )
    assert old.original.read_bytes() == old.content


def test_pending_pdf_and_model_choices_do_not_relabel_or_invalidate_saved_results(store, original):
    from core.search.query import QueryService

    notebook, client, old = original
    service = QueryService(store, store.database.config.data_dir / "unused-models")
    before = service.state(notebook)
    candidate = upload(
        client,
        notebook,
        native_text_pdf("New rainfall pending evidence."),
        old.item.source_id,
        filename="new-name.pdf",
    )
    NotebookModelService(store).choose(notebook, "multilingual-e5", 0, "provenance-control")
    result = service.search(notebook, "rainfall", "words")
    assert result["revision"] == before["revision"]
    hit = result["results"][0]
    assert hit["filename"] == old.item.display_filename != "new-name.pdf"
    assert (
        hit["import_id"] == old.item.id and hit["source_version_id"] == old.item.source_version_id
    )
    assert hit["profile_id"] == "minilm-en" and hit["pdf_role"] == "current"
    assert store.load_pdf_import(candidate).display_filename == "new-name.pdf"
