"""Conversation and question acceptance against a disposable migrated database."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.resources import files
from threading import Barrier

import pytest
from pdf_fixtures import native_text_pdf
from test_import_expiry import publish_import
from test_replacement_safety import original as original
from test_replacement_safety import upload

from core.ingest.service import PdfIngestService
from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import ConflictingWriteError
from core.persistence.migrations import Migration, load_migrations
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore
from core.questions.conversations import ConversationStore, ConversationUnavailable
from core.questions.selection import QuestionSelectionError, SelectedPdf


def candidate_database(config):
    migrations = load_migrations()
    sql = (
        files("core.persistence.migrations")
        .joinpath("question_lifecycle_candidate.sql")
        .read_text(encoding="utf-8")
    )
    revision = len(migrations) + 1
    return Database(
        config, migrations + (Migration(revision, f"{revision:04}_questions.sql", sql),)
    )


@pytest.fixture
def store(tmp_path):
    database = candidate_database(DatabaseConfig(tmp_path / "synthetic-library"))
    database.initialize()
    return PersistenceStore(database)


def test_candidate_upgrade_is_explicit_and_keeps_existing_records(tmp_path):
    config = DatabaseConfig(tmp_path / "upgrade-test")
    baseline = Database(config)
    baseline.initialize()
    persistence = PersistenceStore(baseline)
    notebook = persistence.create_notebook(new_id(), "Already saved")
    source = persistence.create_source(new_id(), notebook.id, "Earlier evidence")
    version = persistence.create_source_version(source.id, "f" * 64)
    with baseline.connect() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='qa_conversations'"
            ).fetchone()
            is None
        )
        old_revision = connection.execute("PRAGMA user_version").fetchone()[0]

    upgraded = candidate_database(config)
    upgraded.initialize()
    persistence = PersistenceStore(upgraded)
    assert persistence.load_notebook(notebook.id) == notebook
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == old_revision + 1
        assert (
            connection.execute(
                "SELECT id FROM source_versions WHERE source_id=?", (source.id,)
            ).fetchone()[0]
            == version.id
        )
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
    assert ConversationStore(persistence).list_conversations(notebook.id) == ()


def new_conversation_with_selection(questions, notebook_id, item, title):
    conversation = questions.create_conversation(notebook_id, title)
    updated = questions.update_picker(
        notebook_id, conversation.id, expected_revision=0, source_ids=[item.source_id]
    )
    return updated, SelectedPdf(item.source_id, item.source_version_id)


def test_two_conversations_reopen_with_distinct_pickers_and_ordered_questions(store, original):
    notebook, client, first = original
    content = native_text_pdf("The second source belongs to a separate conversation picker.")
    second = publish_import(store, upload(client, notebook, content), content)
    questions = ConversationStore(store)
    one, first_pdf = new_conversation_with_selection(questions, notebook, first.item, "First")
    two, second_pdf = new_conversation_with_selection(questions, notebook, second.item, "Second")

    owner = f"core-{new_id()}"
    submissions = []
    for conversation, selected, text in (
        (one, first_pdf, "  First question, verbatim.  "),
        (two, second_pdf, "Separate thread?"),
        (one, first_pdf, "Following up?"),
    ):
        submissions.append(
            questions.submit_question(
                notebook,
                conversation.id,
                submission_id=new_id(),
                owner_instance_id=owner,
                text=text,
                picker_revision=conversation.picker_revision,
                selected=[selected],
            )
        )
    assert all(submission.created for submission in submissions)

    database = candidate_database(store.database.config)
    database.initialize()
    reopened = ConversationStore(PersistenceStore(database))
    assert {record.id for record in reopened.list_conversations(notebook)} == {one.id, two.id}
    first_view = reopened.open_conversation(notebook, one.id)
    second_view = reopened.open_conversation(notebook, two.id)
    assert first_view.selected_source_ids == (first_pdf.source_id,)
    assert second_view.selected_source_ids == (second_pdf.source_id,)
    assert [message.user_text for message in first_view.messages] == [
        "  First question, verbatim.  ",
        "Following up?",
    ]
    assert [message.user_text for message in second_view.messages] == ["Separate thread?"]
    assert [s.attempt.history_through_ordinal for s in submissions] == [0, 0, 1]
    saved, sources = reopened.load_attempt(notebook, one.id, submissions[2].attempt.id)
    assert saved.question_message_id == first_view.messages[1].id
    assert saved.owner_instance_id == owner and saved.status == "accepted"
    assert sources[0].source_version_id == first_pdf.source_version_id
    assert sources[0].import_id == first.item.id
    assert sources[0].index_version_ids == (first.index,)

    other_notebook = store.create_notebook(new_id(), "Separate notebook").id
    with pytest.raises(ConversationUnavailable):
        reopened.open_conversation(other_notebook, one.id)
    with pytest.raises(ConversationUnavailable):
        reopened.load_attempt(notebook, two.id, submissions[2].attempt.id)
    other = reopened.create_conversation(other_notebook, "Not permitted to select the first PDF")
    with pytest.raises(ValueError, match="not an accessible PDF"):
        reopened.update_picker(
            other_notebook, other.id, expected_revision=0, source_ids=[first_pdf.source_id]
        )


def test_concurrent_retry_returns_original_without_reusing_current_picker(store, original):
    notebook, _, pdf = original
    questions = ConversationStore(store)
    conversation, selected = new_conversation_with_selection(questions, notebook, pdf.item, "One")
    submission_id = new_id()
    go = Barrier(2, timeout=10)

    def submit(owner, text):
        return questions.submit_question(
            notebook,
            conversation.id,
            submission_id=submission_id,
            owner_instance_id=owner,
            text=text,
            picker_revision=conversation.picker_revision,
            selected=[selected],
        )

    def attempt():
        owner = new_id()
        go.wait()
        return submit(owner, "What is the evidence?")

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(attempt)
        right = executor.submit(attempt)
    first, second = left.result(), right.result()
    assert first.attempt == second.attempt
    assert first.created != second.created
    assert len(questions.open_conversation(notebook, conversation.id).messages) == 1

    questions.update_picker(
        notebook, conversation.id, expected_revision=conversation.picker_revision, source_ids=[]
    )
    with pytest.raises(ConflictingWriteError, match="picker changed"):
        questions.update_picker(
            notebook,
            conversation.id,
            expected_revision=conversation.picker_revision,
            source_ids=[selected.source_id],
        )
    repeated = submit(new_id(), "What is the evidence?")
    assert not repeated.created and repeated.attempt == first.attempt
    with pytest.raises(ConflictingWriteError, match="different question"):
        submit(new_id(), "Different question")
    saved, sources = questions.load_attempt(notebook, conversation.id, first.attempt.id)
    assert saved == first.attempt and sources[0].source_version_id == selected.source_version_id
    assert questions.open_conversation(notebook, conversation.id).selected_source_ids == ()


def test_stale_selection_cannot_partially_accept_and_selected_only_does_not_pin_pdf(
    store, original
):
    notebook, client, old = original
    questions = ConversationStore(store)
    conversation, first = new_conversation_with_selection(questions, notebook, old.item, "One")
    sub = new_id()

    def submit(selected, submission_id):
        return questions.submit_question(
            notebook,
            conversation.id,
            submission_id=submission_id,
            owner_instance_id=new_id(),
            text="Read the selected PDF.",
            picker_revision=conversation.picker_revision,
            selected=[selected],
        )

    attempt = submit(first, sub).attempt
    replacement_content = native_text_pdf("The current publication is a different version now.")
    replacement = publish_import(
        store, upload(client, notebook, replacement_content, first.source_id), replacement_content
    )
    rejected = new_id()
    with pytest.raises(QuestionSelectionError, match="selected_version_changed"):
        submit(first, rejected)
    assert len(questions.open_conversation(notebook, conversation.id).messages) == 1
    new_selection = replace(first, source_version_id=replacement.item.source_version_id)
    accepted = submit(new_selection, rejected)
    assert accepted.created and accepted.attempt.history_through_ordinal == 1
    assert (
        questions.load_attempt(notebook, conversation.id, attempt.id)[1][0].import_id == old.item.id
    )

    store.trash_pdf_import(old.item.id, "2000-01-01T00:00:00.000000Z")
    PdfIngestService(store, "question-selection-expiry").purge_expired_pdf_history()
    assert not old.original.exists()
    assert not questions.begin_attempt(
        notebook, conversation.id, attempt.id, owner_instance_id=attempt.owner_instance_id
    )
    saved, sources = questions.load_attempt(notebook, conversation.id, attempt.id)
    assert saved.id == attempt.id and sources == ()
    assert (saved.status, saved.reason_code) == ("interrupted", "source_changed")
    assert not submit(first, sub).created
    assert len(questions.open_conversation(notebook, conversation.id).messages) == 2
    with store.database.connect() as connection:
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


def test_stop_prevents_start_and_picker_changes_do_not_stop_an_accepted_question(store, original):
    notebook, _, pdf = original
    questions = ConversationStore(store)
    conversation, selected = new_conversation_with_selection(questions, notebook, pdf.item, "Work")
    other = questions.create_conversation(notebook, "Other thread")
    owner = f"core-{new_id()}"

    def accept():
        return questions.submit_question(
            notebook,
            conversation.id,
            submission_id=new_id(),
            owner_instance_id=owner,
            text="What do the sources say?",
            picker_revision=conversation.picker_revision,
            selected=[selected],
        ).attempt

    stop_first, start_first = accept(), accept()
    with pytest.raises(ConversationUnavailable):
        questions.stop_attempt(notebook, other.id, stop_first.id)
    stopped = questions.stop_attempt(notebook, conversation.id, stop_first.id)
    assert (stopped.status, stopped.reason_code) == ("stopped", "user_stopped")
    assert questions.stop_attempt(notebook, conversation.id, stop_first.id) == stopped
    assert not questions.begin_attempt(
        notebook, conversation.id, stop_first.id, owner_instance_id=owner
    )

    questions.update_picker(
        notebook, conversation.id, expected_revision=conversation.picker_revision, source_ids=[]
    )
    assert not questions.begin_attempt(
        notebook, conversation.id, start_first.id, owner_instance_id=new_id()
    )
    assert questions.begin_attempt(
        notebook, conversation.id, start_first.id, owner_instance_id=owner
    )
    assert not questions.begin_attempt(
        notebook, conversation.id, start_first.id, owner_instance_id=owner
    )
    assert questions.stop_attempt(notebook, conversation.id, start_first.id).status == "stopped"


def test_restart_interrupts_accepted_and_running_attempts_without_replaying(store, original):
    notebook, _, pdf = original
    questions = ConversationStore(store)
    conversation, selected = new_conversation_with_selection(questions, notebook, pdf.item, "Work")
    previous_owner = f"core-{new_id()}"

    def submit(submission_id, owner):
        return questions.submit_question(
            notebook,
            conversation.id,
            submission_id=submission_id,
            owner_instance_id=owner,
            text="What do the sources say?",
            picker_revision=conversation.picker_revision,
            selected=[selected],
        )

    first_id, second_id = new_id(), new_id()
    accepted = submit(first_id, previous_owner).attempt
    started = submit(second_id, previous_owner).attempt
    assert questions.begin_attempt(
        notebook, conversation.id, started.id, owner_instance_id=previous_owner
    )
    replacement_owner = f"core-{new_id()}"
    database = candidate_database(store.database.config)
    database.initialize()
    questions = ConversationStore(PersistenceStore(database))
    assert questions.interrupt_abandoned_attempts(replacement_owner) == 2
    assert questions.interrupt_abandoned_attempts(replacement_owner) == 0
    for submission_id, original_attempt in ((first_id, accepted), (second_id, started)):
        replay = submit(submission_id, replacement_owner)
        assert not replay.created and replay.attempt.id == original_attempt.id
        assert (replay.attempt.status, replay.attempt.reason_code) == (
            "interrupted",
            "core_restarted",
        )
        assert questions.stop_attempt(notebook, conversation.id, original_attempt.id) == (
            replay.attempt
        )
        assert not questions.begin_attempt(
            notebook, conversation.id, original_attempt.id, owner_instance_id=previous_owner
        )
