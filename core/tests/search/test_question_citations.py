"""A historical question citation resolves its saved passage and PDF."""

from dataclasses import replace

import pytest
from pdf_fixtures import native_text_pdf
from test_import_cancellation import partial_pdf
from test_import_expiry import publish_import
from test_model_switch import complete, mark_installed, pending_jobs
from test_replacement_safety import finish_extraction, upload
from test_replacement_safety import original as original
from test_search_expiry import reopen
from test_search_safety import prepare

from core.ingest.service import PdfIngestService
from core.persistence.models import new_id
from core.questions.citations import (
    SavedPassageRef,
    SavedPassageUnavailable,
    lookup_saved_pdf_passage,
)
from core.search.switching import NotebookModelService


def lookup(store, notebook, reference):
    with store._transaction(write=False) as connection:
        return lookup_saved_pdf_passage(connection, notebook, reference)


def original_reference(store, item):
    passage = store.load_extraction_report(item.extraction).passages[0]
    return SavedPassageRef(item.item.source_id, item.item.source_version_id, passage.id)


def replace_pdf(store, client, notebook, source_id):
    content = native_text_pdf("Replacement rainfall evidence must not be used for older citations.")
    return publish_import(store, upload(client, notebook, content, source_id), content)


def test_old_citation_keeps_saved_page_and_exact_pdf_after_replacement_and_rebuild(store, original):
    notebook, client, old = original
    original_ref = original_reference(store, old)
    saved = lookup(store, notebook, original_ref)
    assert saved.page_number == 1
    assert saved.import_id == old.item.id
    assert saved.extraction_version_id == old.extraction
    assert saved.location_format_version == 1
    assert saved.location["bbox"]

    newer = replace_pdf(store, client, notebook, old.item.source_id)
    mark_installed(store)
    models = NotebookModelService(store)
    models.choose(notebook, "multilingual-e5", 0, "cite-previous-after-rebuild")
    for job in pending_jobs(store, notebook):
        complete(store, job[0])
    assert models.state(notebook)["pending_manifest_id"] is None
    readiness = store.load_readiness(old.item.source_id)
    assert readiness is not None and readiness.source_version_id == newer.item.source_version_id

    reopened = reopen(store)
    resolved = lookup(reopened, notebook, original_ref)
    assert resolved == saved
    content, _ = PdfIngestService(reopened, "saved-citation-test").read_verified_import(
        resolved.import_id
    )
    assert content == old.content
    with pytest.raises(SavedPassageUnavailable):
        lookup(
            reopened,
            notebook,
            replace(original_ref, source_version_id=newer.item.source_version_id),
        )


def test_lookup_checks_notebook_and_passage_offsets_without_guessing(store, original):
    notebook, _, old = original
    reference = original_reference(store, old)
    saved = lookup(store, notebook, reference)
    other_id = new_id()
    store.create_notebook(other_id, "Other notebook")

    with pytest.raises(SavedPassageUnavailable):
        lookup(store, other_id, reference)
    with pytest.raises(SavedPassageUnavailable):
        lookup(store, notebook, replace(reference, text_range=(0, len(saved.quote) + 1)))
    with pytest.raises(SavedPassageUnavailable):
        lookup(store, notebook, replace(reference, passage_id=new_id()))
    assert lookup(store, notebook, replace(reference, text_range=(0, 8))).quote == saved.quote[:8]


def test_failed_reindex_after_review_does_not_break_saved_citation(store, original):
    notebook, client, old = original
    pdf = partial_pdf()
    item = store.load_pdf_import(upload(client, notebook, pdf, old.item.source_id))
    extraction = finish_extraction(store, item.id)

    def confirm(generation):
        store.record_review_decision(extraction, "confirmed", generation)
        with store.database.connect() as connection:
            return connection.execute(
                "SELECT job_id FROM search_handoffs WHERE extraction_version_id=? "
                "AND review_generation=?",
                (extraction, generation),
            ).fetchone()[0]

    job = confirm(1)
    index, guard = prepare(store, job)
    store.publish_readiness(job, item.source_id, item.source_version_id, [index], guard=guard)
    store.finish_step_attempt(job, "prepare", guard.expected_attempt_number, None)
    store.finish_job(job, "completed")
    passage = store.load_extraction_report(extraction).passages[0]
    reference = SavedPassageRef(item.source_id, item.source_version_id, passage.id)
    saved = lookup(store, notebook, reference)

    reopened_review = store.reopen_review(extraction)
    rebuild_job = confirm(reopened_review.generation)
    store.finish_job(rebuild_job, "failed", "controlled search rebuild failure")
    assert store.load_pdf_import(item.id).state == "failed"
    content, _ = PdfIngestService(store, "saved-citation-test").read_verified_import(item.id)
    assert content == pdf
    assert lookup(store, notebook, reference) == saved


def test_cancelled_reprocessing_does_not_break_citation_to_older_pdf(store, original):
    notebook, client, old = original
    reference = original_reference(store, old)
    saved = lookup(store, notebook, reference)
    replace_pdf(store, client, notebook, old.item.source_id)

    retry_job = store.retry_pdf_import_processing(old.item.id)
    assert store.cancel_pdf_import_processing(old.item.id) == retry_job
    assert store.load_pdf_import(old.item.id).failure_code == "processing_cancelled"
    content, _ = PdfIngestService(store, "saved-citation-test").read_verified_import(old.item.id)
    assert content == old.content
    assert lookup(store, notebook, reference) == saved


@pytest.mark.parametrize("trash", ["older_pdf", "whole_source"])
def test_citation_lookup_excludes_trashed_pdf_or_source_before_expiry(store, original, trash):
    notebook, client, old = original
    reference = original_reference(store, old)
    replace_pdf(store, client, notebook, old.item.source_id)
    assert lookup(store, notebook, reference).import_id == old.item.id
    deadline = "2099-01-01T00:00:00.000000Z"
    if trash == "older_pdf":
        store.trash_pdf_import(old.item.id, deadline)
    else:
        store.trash_source(old.item.source_id, deadline)
    assert old.original.is_file()
    with pytest.raises(SavedPassageUnavailable):
        lookup(store, notebook, reference)
