"""Question selection uses real PDF readiness, not the newest imported version."""

import pytest
from pdf_fixtures import native_text_pdf
from test_import_expiry import publish_import
from test_replacement_safety import extract_candidate, upload
from test_replacement_safety import original as original
from test_search_safety import prepare

from core.persistence.models import new_id, utc_timestamp
from core.questions.selection import (
    QuestionSelectionError,
    SelectedPdf,
    resolve_question_selection,
)
from core.search.profiles import profiles


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
