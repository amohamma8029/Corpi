"""Notebook-scoped conversation reads and atomic question acceptance."""

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import new_id, utc_timestamp, validate_id
from core.persistence.store import PersistenceStore
from core.questions.selection import (
    QuestionSelectionError,
    ReadyQuestionPdf,
    SelectedPdf,
    resolve_question_selection,
)


class ConversationUnavailable(LookupError):
    def __init__(self) -> None:
        super().__init__("The conversation or question is unavailable in this notebook.")


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    notebook_id: str
    title: str
    picker_revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    id: str
    ordinal: int
    role: str
    user_text: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class ConversationView:
    conversation: Conversation
    selected_source_ids: tuple[str, ...]
    messages: tuple[ConversationMessage, ...]


@dataclass(frozen=True, slots=True)
class QuestionAttempt:
    id: str
    conversation_id: str
    question_message_id: str
    owner_instance_id: str
    history_through_ordinal: int
    status: str
    reason_code: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class QuestionSubmission:
    attempt: QuestionAttempt
    # Only the caller that created the attempt may dispatch it, after commit.
    created: bool


class ConversationStore:
    def __init__(self, persistence: PersistenceStore) -> None:
        self._persistence = persistence

    def create_conversation(self, notebook_id: str, title: str) -> Conversation:
        validate_id(notebook_id)
        title = title.strip()
        if not title:
            raise ValueError("conversation title must not be empty")
        conversation_id, now = new_id(), utc_timestamp()
        with self._persistence._transaction() as connection:
            _require_notebook(connection, notebook_id)
            connection.execute(
                """INSERT INTO qa_conversations(id,notebook_id,title,created_at,updated_at)
                   VALUES(?,?,?,?,?)""",
                (conversation_id, notebook_id, title, now, now),
            )
        return Conversation(conversation_id, notebook_id, title, 0, now, now)

    def list_conversations(self, notebook_id: str) -> tuple[Conversation, ...]:
        validate_id(notebook_id)
        with self._persistence._transaction(write=False) as connection:
            _require_notebook(connection, notebook_id)
            rows = connection.execute(
                """SELECT * FROM qa_conversations WHERE notebook_id=?
                   ORDER BY updated_at DESC,id""",
                (notebook_id,),
            )
            return tuple(_conversation(row) for row in rows)

    def open_conversation(self, notebook_id: str, conversation_id: str) -> ConversationView:
        _validate_conversation_ids(notebook_id, conversation_id)
        with self._persistence._transaction(write=False) as connection:
            row = _require_conversation(connection, notebook_id, conversation_id)
            messages = connection.execute(
                """SELECT id,ordinal,role,user_text,created_at FROM qa_messages
                   WHERE conversation_id=? ORDER BY ordinal""",
                (conversation_id,),
            )
            return ConversationView(
                _conversation(row),
                _picker_ids(connection, conversation_id),
                tuple(ConversationMessage(*message) for message in messages),
            )

    def update_picker(
        self,
        notebook_id: str,
        conversation_id: str,
        *,
        expected_revision: int,
        source_ids: Sequence[str],
    ) -> Conversation:
        _validate_conversation_ids(notebook_id, conversation_id)
        for source_id in source_ids:
            validate_id(source_id)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("a picker source cannot be selected twice")

        with self._persistence._transaction() as connection:
            current = _require_conversation(connection, notebook_id, conversation_id)
            if current["picker_revision"] != expected_revision:
                raise ConflictingWriteError(
                    "The saved source picker changed. Reopen it and try again."
                )
            if tuple(source_ids) == _picker_ids(connection, conversation_id):
                return _conversation(current)
            sources = connection.execute(
                """SELECT s.id FROM sources s
                   WHERE s.notebook_id=? AND s.id IN (SELECT value FROM json_each(?))
                     AND EXISTS(SELECT 1 FROM pdf_imports p WHERE p.source_id=s.id)
                     AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)""",
                (notebook_id, json.dumps(list(source_ids))),
            )
            if {row["id"] for row in sources} != set(source_ids):
                raise ValueError("A picker source is not an accessible PDF in this notebook.")
            connection.execute(
                "DELETE FROM qa_picker_sources WHERE conversation_id=?", (conversation_id,)
            )
            connection.executemany(
                """INSERT INTO qa_picker_sources(conversation_id,notebook_id,source_id,ordinal)
                   VALUES(?,?,?,?)""",
                (
                    (conversation_id, notebook_id, source_id, ordinal)
                    for ordinal, source_id in enumerate(source_ids)
                ),
            )
            updated = connection.execute(
                """UPDATE qa_conversations SET picker_revision=picker_revision+1,updated_at=?
                   WHERE id=? RETURNING *""",
                (utc_timestamp(), conversation_id),
            ).fetchone()
            return _conversation(updated)

    def submit_question(
        self,
        notebook_id: str,
        conversation_id: str,
        *,
        submission_id: str,
        owner_instance_id: str,
        text: str,
        picker_revision: int,
        selected: Sequence[SelectedPdf],
    ) -> QuestionSubmission:
        """Save the question, history boundary and exact versions before returning."""
        _validate_conversation_ids(notebook_id, conversation_id)
        validate_id(submission_id)
        _validate_attempt_owner(owner_instance_id)
        if not text.strip():
            raise ValueError("question text must not be empty")
        selected = tuple(selected)
        for pdf in selected:
            validate_id(pdf.source_id)
            validate_id(pdf.source_version_id)
        fingerprint = _submission_fingerprint(text, picker_revision, selected)

        with self._persistence._transaction() as connection:
            conversation = _require_conversation(connection, notebook_id, conversation_id)
            previous = connection.execute(
                "SELECT * FROM qa_attempts WHERE conversation_id=? AND submission_id=?",
                (conversation_id, submission_id),
            ).fetchone()
            if previous is not None:
                if previous["submission_sha256"] != fingerprint:
                    raise ConflictingWriteError(
                        "The submission ID already belongs to a different question."
                    )
                return QuestionSubmission(_attempt(previous), created=False)

            if conversation["picker_revision"] != picker_revision or tuple(
                pdf.source_id for pdf in selected
            ) != _picker_ids(connection, conversation_id):
                raise ConflictingWriteError(
                    "The saved source picker changed. Reopen it and try again."
                )
            exact = resolve_question_selection(connection, notebook_id, selected)
            now = utc_timestamp()
            message_id, attempt_id = new_id(), new_id()
            history_end = conversation["last_message_ordinal"]
            connection.execute(
                """INSERT INTO qa_messages(id,conversation_id,ordinal,role,user_text,created_at)
                   VALUES(?,?,?,'user',?,?)""",
                (message_id, conversation_id, history_end + 1, text, now),
            )
            row = connection.execute(
                """INSERT INTO qa_attempts(
                       id,conversation_id,notebook_id,question_message_id,submission_id,
                       submission_sha256,owner_instance_id,history_through_ordinal,
                       selected_source_count,status,
                       created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,'accepted',?,?) RETURNING *""",
                (
                    attempt_id,
                    conversation_id,
                    notebook_id,
                    message_id,
                    submission_id,
                    fingerprint,
                    owner_instance_id,
                    history_end,
                    len(exact),
                    now,
                    now,
                ),
            ).fetchone()
            _save_selection(connection, notebook_id, attempt_id, exact)
            connection.execute(
                "UPDATE qa_conversations SET last_message_ordinal=?,updated_at=? WHERE id=?",
                (history_end + 1, now, conversation_id),
            )
            return QuestionSubmission(_attempt(row), created=True)

    def load_attempt(
        self, notebook_id: str, conversation_id: str, attempt_id: str
    ) -> tuple[QuestionAttempt, tuple[ReadyQuestionPdf, ...]]:
        _validate_conversation_ids(notebook_id, conversation_id)
        validate_id(attempt_id)
        with self._persistence._transaction(write=False) as connection:
            row = _require_attempt(connection, notebook_id, conversation_id, attempt_id)
            return _attempt(row), _attempt_sources(connection, attempt_id)

    def begin_attempt(
        self,
        notebook_id: str,
        conversation_id: str,
        attempt_id: str,
        *,
        owner_instance_id: str,
    ) -> bool:
        """Give the accepting owner permission to start this attempt at most once."""
        _validate_conversation_ids(notebook_id, conversation_id)
        validate_id(attempt_id)
        _validate_attempt_owner(owner_instance_id)
        with self._persistence._transaction() as connection:
            row = _require_attempt(connection, notebook_id, conversation_id, attempt_id)
            if row["status"] != "accepted" or row["owner_instance_id"] != owner_instance_id:
                return False
            saved = _attempt_sources(connection, attempt_id)
            if len(saved) != row["selected_source_count"] or not _selection_is_current(
                connection, notebook_id, saved
            ):
                _set_attempt_status(connection, attempt_id, "interrupted", "source_changed")
                return False
            _set_attempt_status(connection, attempt_id, "running", None)
            return True

    def stop_attempt(
        self, notebook_id: str, conversation_id: str, attempt_id: str
    ) -> QuestionAttempt:
        """Stop this question, or report the outcome if it already finished."""
        _validate_conversation_ids(notebook_id, conversation_id)
        validate_id(attempt_id)
        with self._persistence._transaction() as connection:
            row = _require_attempt(connection, notebook_id, conversation_id, attempt_id)
            if row["status"] in ("accepted", "running"):
                row = _set_attempt_status(connection, attempt_id, "stopped", "user_stopped")
            return _attempt(row)

    def interrupt_abandoned_attempts(self, current_owner_instance_id: str) -> int:
        """Interrupt prior work at startup, with exclusive ownership of the library."""
        _validate_attempt_owner(current_owner_instance_id)
        with self._persistence._transaction() as connection:
            result = connection.execute(
                """UPDATE qa_attempts
                   SET status='interrupted',reason_code='core_restarted',updated_at=?
                   WHERE status IN ('accepted','running') AND owner_instance_id<>?""",
                (utc_timestamp(), current_owner_instance_id),
            )
            return result.rowcount


def _validate_conversation_ids(notebook_id: str, conversation_id: str) -> None:
    validate_id(notebook_id)
    validate_id(conversation_id)


def _validate_attempt_owner(owner_instance_id: str) -> None:
    # The application passes its "core-<uuid>" instance identity, not a row ID.
    if not owner_instance_id.strip():
        raise ValueError("question owner instance must not be empty")


def _require_notebook(connection: sqlite3.Connection, notebook_id: str) -> None:
    if connection.execute("SELECT 1 FROM notebooks WHERE id=?", (notebook_id,)).fetchone() is None:
        raise ConversationUnavailable()


def _require_conversation(
    connection: sqlite3.Connection, notebook_id: str, conversation_id: str
) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM qa_conversations WHERE notebook_id=? AND id=?",
        (notebook_id, conversation_id),
    ).fetchone()
    if row is None:
        raise ConversationUnavailable()
    return row


def _require_attempt(
    connection: sqlite3.Connection, notebook_id: str, conversation_id: str, attempt_id: str
) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM qa_attempts WHERE notebook_id=? AND conversation_id=? AND id=?",
        (notebook_id, conversation_id, attempt_id),
    ).fetchone()
    if row is None:
        raise ConversationUnavailable()
    return row


def _attempt_sources(
    connection: sqlite3.Connection, attempt_id: str
) -> tuple[ReadyQuestionPdf, ...]:
    rows = connection.execute(
        "SELECT * FROM qa_attempt_sources WHERE attempt_id=? ORDER BY ordinal", (attempt_id,)
    )
    return tuple(
        ReadyQuestionPdf(
            row["source_id"],
            row["source_version_id"],
            row["import_id"],
            row["extraction_version_id"],
            row["review_generation"],
            row["manifest_id"],
            tuple(json.loads(row["index_version_ids_json"])),
        )
        for row in rows
    )


def _selection_is_current(
    connection: sqlite3.Connection, notebook_id: str, saved: Sequence[ReadyQuestionPdf]
) -> bool:
    selected = tuple(SelectedPdf(pdf.source_id, pdf.source_version_id) for pdf in saved)
    try:
        return resolve_question_selection(connection, notebook_id, selected) == tuple(saved)
    except QuestionSelectionError:
        return False


def _set_attempt_status(
    connection: sqlite3.Connection, attempt_id: str, status: str, reason: str | None
) -> sqlite3.Row:
    # The caller owns the write transaction and has checked the previous status.
    return connection.execute(
        """UPDATE qa_attempts SET status=?,reason_code=?,updated_at=?
           WHERE id=? RETURNING *""",
        (status, reason, utc_timestamp(), attempt_id),
    ).fetchone()


def _picker_ids(connection: sqlite3.Connection, conversation_id: str) -> tuple[str, ...]:
    return tuple(
        row["source_id"]
        for row in connection.execute(
            "SELECT source_id FROM qa_picker_sources WHERE conversation_id=? ORDER BY ordinal",
            (conversation_id,),
        )
    )


def _conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        row["id"],
        row["notebook_id"],
        row["title"],
        row["picker_revision"],
        row["created_at"],
        row["updated_at"],
    )


def _attempt(row: sqlite3.Row) -> QuestionAttempt:
    return QuestionAttempt(
        row["id"],
        row["conversation_id"],
        row["question_message_id"],
        row["owner_instance_id"],
        row["history_through_ordinal"],
        row["status"],
        row["reason_code"],
        row["created_at"],
        row["updated_at"],
    )


def _submission_fingerprint(
    text: str, picker_revision: int, selected: Sequence[SelectedPdf]
) -> str:
    payload = [text, picker_revision, [[pdf.source_id, pdf.source_version_id] for pdf in selected]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _save_selection(
    connection: sqlite3.Connection,
    notebook_id: str,
    attempt_id: str,
    selected: Sequence[ReadyQuestionPdf],
) -> None:
    connection.executemany(
        """INSERT INTO qa_attempt_sources(
               attempt_id,notebook_id,source_id,source_version_id,ordinal,import_id,
               extraction_version_id,review_generation,manifest_id,index_version_ids_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            (
                attempt_id,
                notebook_id,
                pdf.source_id,
                pdf.source_version_id,
                ordinal,
                pdf.import_id,
                pdf.extraction_version_id,
                pdf.review_generation,
                pdf.manifest_id,
                json.dumps(pdf.index_version_ids),
            )
            for ordinal, pdf in enumerate(selected)
        ),
    )
