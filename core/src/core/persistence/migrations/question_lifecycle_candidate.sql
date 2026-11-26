-- M2.2 work-ahead: loaded explicitly by disposable database tests only.
-- Assign the final revision after the provider migrations are integrated.
CREATE TABLE qa_conversations (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE RESTRICT,
    title TEXT NOT NULL CHECK(length(trim(title)) > 0),
    picker_revision INTEGER NOT NULL DEFAULT 0 CHECK(picker_revision >= 0),
    last_message_ordinal INTEGER NOT NULL DEFAULT 0 CHECK(last_message_ordinal >= 0),
    created_at TEXT NOT NULL CHECK(
        length(created_at)=27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK(
        length(updated_at)=27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE(id,notebook_id)
) STRICT;
CREATE INDEX qa_conversations_notebook_idx
    ON qa_conversations(notebook_id,updated_at DESC,id);

CREATE TABLE qa_picker_sources (
    conversation_id TEXT NOT NULL,
    notebook_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    PRIMARY KEY(conversation_id,source_id),
    UNIQUE(conversation_id,ordinal),
    FOREIGN KEY(conversation_id,notebook_id)
        REFERENCES qa_conversations(id,notebook_id) ON DELETE CASCADE,
    FOREIGN KEY(source_id,notebook_id) REFERENCES sources(id,notebook_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX qa_picker_sources_source_idx ON qa_picker_sources(source_id,notebook_id);

CREATE TABLE qa_messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES qa_conversations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK(ordinal > 0),
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    user_text TEXT,
    created_at TEXT NOT NULL CHECK(
        length(created_at)=27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    CHECK((role='user' AND user_text IS NOT NULL AND length(trim(user_text)) > 0)
          OR (role='assistant' AND user_text IS NULL)),
    UNIQUE(id,conversation_id),
    UNIQUE(conversation_id,ordinal)
) STRICT;

CREATE TABLE qa_attempts (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    notebook_id TEXT NOT NULL,
    question_message_id TEXT NOT NULL UNIQUE,
    submission_id TEXT NOT NULL,
    submission_sha256 TEXT NOT NULL CHECK(
        length(submission_sha256)=64 AND submission_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    owner_instance_id TEXT NOT NULL,
    history_through_ordinal INTEGER NOT NULL CHECK(history_through_ordinal >= 0),
    status TEXT NOT NULL CHECK(status IN (
        'accepted','running','completed','failed','stopped','interrupted'
    )),
    reason_code TEXT,
    created_at TEXT NOT NULL CHECK(
        length(created_at)=27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK(
        length(updated_at)=27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE(id,notebook_id),
    UNIQUE(conversation_id,submission_id),
    FOREIGN KEY(conversation_id,notebook_id)
        REFERENCES qa_conversations(id,notebook_id) ON DELETE CASCADE,
    FOREIGN KEY(question_message_id,conversation_id)
        REFERENCES qa_messages(id,conversation_id) ON DELETE CASCADE
) STRICT;

-- These are recorded choices, not dependencies that keep an original alive.
-- Removing a source version removes this snapshot row, not its question.
CREATE TABLE qa_attempt_sources (
    attempt_id TEXT NOT NULL,
    notebook_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    import_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    review_generation INTEGER CHECK(review_generation > 0),
    manifest_id TEXT NOT NULL,
    index_version_ids_json TEXT NOT NULL CHECK(
        json_valid(index_version_ids_json) AND json_type(index_version_ids_json)='array'
        AND json_array_length(index_version_ids_json) > 0
    ),
    PRIMARY KEY(attempt_id,source_id),
    UNIQUE(attempt_id,ordinal),
    FOREIGN KEY(attempt_id,notebook_id) REFERENCES qa_attempts(id,notebook_id) ON DELETE CASCADE,
    FOREIGN KEY(source_id,notebook_id) REFERENCES sources(id,notebook_id) ON DELETE CASCADE,
    FOREIGN KEY(source_version_id,source_id)
        REFERENCES source_versions(id,source_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX qa_attempt_sources_source_idx ON qa_attempt_sources(source_id,notebook_id);
CREATE INDEX qa_attempt_sources_version_idx ON qa_attempt_sources(source_version_id,source_id);
