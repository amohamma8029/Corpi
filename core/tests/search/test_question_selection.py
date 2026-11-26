"""Question selection uses real PDF readiness, not the newest imported version."""

import json

import pytest
from pdf_fixtures import native_text_pdf
from test_import_expiry import publish_import
from test_replacement_safety import extract_candidate, upload
from test_replacement_safety import original as original
from test_search_safety import prepare

from core.ingest.service import PdfIngestService
from core.persistence.models import new_id, utc_timestamp
from core.questions.selection import (
    QuestionSelectionError,
    SelectedPdf,
    resolve_question_selection,
)
from core.search import representation
from core.search.profiles import profiles
from core.search.query import QueryService, SearchFailure


def resolve(store, notebook, *selected):
    with store._transaction(write=False) as connection:
        return resolve_question_selection(connection, notebook, selected)


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "ready"])
def test_selection_uses_previous_ready_pdf_until_replacement_publishes(store, original, outcome):
    notebook, client, old = original
    selection = SelectedPdf(old.item.source_id, old.item.source_version_id)
    (ready,) = resolve(store, notebook, selection)
    assert ready.import_id == old.item.id
    assert ready.index_version_ids == (old.index,)
    assert ready.manifest_id == profiles()["minilm-en"].identity
    assert ready.review_generation is None

    content = native_text_pdf("A newly imported PDF is not automatically ready for questions.")
    candidate = store.load_pdf_import(upload(client, notebook, content, selection.source_id))
    replacement = SelectedPdf(candidate.source_id, candidate.source_version_id)
    assert resolve(store, notebook, selection) == (ready,)
    with pytest.raises(QuestionSelectionError, match="selected_version_changed"):
        resolve(store, notebook, replacement)

    job = extract_candidate(store, candidate.id)
    index, guard = prepare(store, job)
    if outcome == "failed":
        store.finish_job(job, "failed", "controlled failure before publication")
    elif outcome == "cancelled":
        store.cancel_job(job)
    else:
        store.publish_readiness(
            job, candidate.source_id, candidate.source_version_id, [index], guard=guard
        )

    if outcome != "ready":
        assert resolve(store, notebook, selection) == (ready,)
        with pytest.raises(QuestionSelectionError, match="selected_version_changed"):
            resolve(store, notebook, replacement)
        return

    with pytest.raises(QuestionSelectionError, match="selected_version_changed"):
        resolve(store, notebook, selection)
    (new_ready,) = resolve(store, notebook, replacement)
    assert new_ready.import_id == candidate.id
    assert new_ready.source_version_id == candidate.source_version_id
    assert new_ready.extraction_version_id != ready.extraction_version_id
    assert new_ready.index_version_ids == (index,)


@pytest.mark.parametrize("outcome", ["pending", "failed", "cancelled"])
def test_selection_keeps_existing_publication_when_current_pdf_is_retried(store, original, outcome):
    notebook, _, old = original
    selection = SelectedPdf(old.item.source_id, old.item.source_version_id)
    saved = resolve(store, notebook, selection)
    retry = store.retry_pdf_import_processing(old.item.id)
    if outcome == "failed":
        store.finish_job(retry, "failed", "controlled failure before extraction")
    elif outcome == "cancelled":
        assert store.cancel_pdf_import_processing(old.item.id) == retry
    assert store.load_pdf_import(old.item.id).state == (
        "pending" if outcome == "pending" else "failed"
    )
    service = QueryService(store, store.database.config.data_dir / "unused-models")
    hit = service.search(notebook, "rainfall", "words")["results"][0]
    assert hit["source_version_id"] == old.item.source_version_id
    assert hit["index_version_id"] == old.index
    content, _ = PdfIngestService(store, "selection-test").read_verified_import(old.item.id)
    assert content == old.content
    assert resolve(store, notebook, selection) == saved


def test_selection_rejects_search_generation_unusable_by_current_runtime(
    store, original, monkeypatch
):
    _, client, _ = original
    notebook = store.create_notebook(new_id(), "Versioned search notebook").id
    content = native_text_pdf("Rainfall evidence in a PDF with versioned search input rules.")
    item = store.load_pdf_import(upload(client, notebook, content))
    job = extract_candidate(store, item.id)
    with store.database.connect() as connection:
        settings = json.loads(
            connection.execute(
                "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
            ).fetchone()[0]
        )
    assert settings["search_representation"] == representation.ARABIC
    index, guard = prepare(store, job, profile_id="bekko-a8m-quint8-avx2", settings=settings)
    store.publish_readiness(job, item.source_id, item.source_version_id, [index], guard=guard)
    selection = SelectedPdf(item.source_id, item.source_version_id)
    assert resolve(store, notebook, selection)[0].index_version_ids == (index,)

    # Open the same persisted index under simulated newer Unicode data.
    monkeypatch.setattr(representation.unicodedata, "unidata_version", "simulated-new-version")
    service = QueryService(store, store.database.config.data_dir / "unused-models")
    with pytest.raises(SearchFailure, match="unsupported input"):
        service.search(notebook, "rainfall", "words")
    with pytest.raises(QuestionSelectionError, match="selected_source_unavailable"):
        resolve(store, notebook, selection)


def test_selection_rejects_entire_mixed_set_and_keeps_notebooks_separate(store, original):
    notebook, client, old = original
    own = SelectedPdf(old.item.source_id, old.item.source_version_id)
    other_id = new_id()
    store.create_notebook(other_id, other_id)
    # The index fixture builds MiniLM vectors, not the new default representation.
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
            " VALUES(?,?,?)",
            (other_id, profiles()["minilm-en"].identity, utc_timestamp()),
        )
    content = native_text_pdf("This PDF can be ready in its own notebook only.")
    other = publish_import(store, upload(client, other_id, content), content)
    foreign = SelectedPdf(other.item.source_id, other.item.source_version_id)
    assert resolve(store, other_id, foreign)[0].import_id == other.item.id

    with pytest.raises(QuestionSelectionError, match="selected_source_unavailable") as error:
        resolve(store, notebook, own, foreign)
    assert error.value.source_id == foreign.source_id
    with pytest.raises(QuestionSelectionError, match="duplicate_source"):
        resolve(store, notebook, own, own)
    assert resolve(store, notebook, own)[0].import_id == old.item.id


def test_selection_distinguishes_no_selection_from_no_ready_pdfs(store, original):
    notebook, _, old = original
    with pytest.raises(QuestionSelectionError, match="no_sources_selected"):
        resolve(store, notebook)

    store.trash_source(old.item.source_id, "2099-01-01T00:00:00.000000Z")
    with pytest.raises(QuestionSelectionError, match="no_ready_sources"):
        resolve(store, notebook, SelectedPdf(old.item.source_id, old.item.source_version_id))
