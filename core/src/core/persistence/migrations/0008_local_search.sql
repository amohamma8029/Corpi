-- Model identity is independent of notebook evidence and source deletion.
CREATE TABLE model_installations (
    manifest_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('missing','downloading','installed','failed','cancelled')),
    generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
    received_bytes INTEGER NOT NULL DEFAULT 0 CHECK(received_bytes >= 0),
    spent_seconds REAL NOT NULL DEFAULT 0 CHECK(spent_seconds >= 0),
    transfer_attempts INTEGER NOT NULL DEFAULT 0 CHECK(transfer_attempts >= 0),
    staging_path TEXT,
    failure_code TEXT,
    updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE notebook_profiles (
    notebook_id TEXT PRIMARY KEY REFERENCES notebooks(id) ON DELETE RESTRICT,
    active_manifest_id TEXT NOT NULL,
    pending_manifest_id TEXT,
    selection_generation INTEGER NOT NULL DEFAULT 0 CHECK(selection_generation >= 0),
    updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE search_handoffs (
    extraction_version_id TEXT NOT NULL REFERENCES extraction_versions(id) ON DELETE RESTRICT,
    review_generation INTEGER NOT NULL CHECK(review_generation >= 0),
    manifest_id TEXT NOT NULL,
    selection_generation INTEGER NOT NULL CHECK(selection_generation >= 0),
    purpose TEXT NOT NULL CHECK(purpose IN ('import','model-switch')),
    state TEXT NOT NULL CHECK(state IN ('waiting','scheduled','cancelled')),
    job_id TEXT REFERENCES jobs(id) ON DELETE RESTRICT,
    waiting_reason TEXT,
    PRIMARY KEY(extraction_version_id,review_generation,manifest_id,selection_generation,purpose)
) STRICT;

CREATE TABLE search_generations (
    index_version_id TEXT PRIMARY KEY REFERENCES index_versions(id) ON DELETE RESTRICT,
    manifest_id TEXT NOT NULL,
    identity_sha256 TEXT NOT NULL,
    settings_json TEXT NOT NULL CHECK(json_valid(settings_json)),
    expected_chunks INTEGER NOT NULL CHECK(expected_chunks > 0),
    spent_seconds REAL NOT NULL DEFAULT 0 CHECK(spent_seconds >= 0),
    published_by_job_id TEXT REFERENCES jobs(id) ON DELETE RESTRICT,
    UNIQUE(identity_sha256)
) STRICT;

CREATE TABLE search_rows (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL UNIQUE REFERENCES retrieval_chunks(id) ON DELETE RESTRICT,
    index_version_id TEXT NOT NULL REFERENCES search_generations(index_version_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX search_rows_generation ON search_rows(index_version_id);

CREATE VIRTUAL TABLE search_words USING fts5(text, tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE search_vectors USING vec0(embedding float[384] distance_metric=cosine);

CREATE TABLE search_batches (
    index_version_id TEXT NOT NULL REFERENCES search_generations(index_version_id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    chunk_ids_json TEXT NOT NULL CHECK(json_valid(chunk_ids_json)),
    vectors_sha256 TEXT,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    reserved_seconds REAL NOT NULL DEFAULT 0 CHECK(reserved_seconds >= 0),
    complete INTEGER NOT NULL DEFAULT 0 CHECK(complete IN (0,1)),
    PRIMARY KEY(index_version_id,ordinal)
) STRICT;

CREATE TABLE search_components (
    index_version_id TEXT NOT NULL REFERENCES search_generations(index_version_id) ON DELETE RESTRICT,
    component TEXT NOT NULL CHECK(component IN ('word','vector')),
    chunk_ids_sha256 TEXT NOT NULL,
    PRIMARY KEY(index_version_id,component)
) STRICT;

-- These remain hidden until a guarded notebook-level activation.
CREATE TABLE profile_search_publications (
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(id) ON DELETE RESTRICT,
    manifest_id TEXT NOT NULL,
    index_version_id TEXT NOT NULL REFERENCES search_generations(index_version_id) ON DELETE RESTRICT,
    selection_generation INTEGER NOT NULL,
    PRIMARY KEY(source_id,manifest_id)
) STRICT;
