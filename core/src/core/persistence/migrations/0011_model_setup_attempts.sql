CREATE TABLE model_manifests (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json))
) STRICT;
CREATE TRIGGER model_manifests_are_immutable BEFORE UPDATE ON model_manifests
BEGIN SELECT RAISE(ABORT, 'model manifests are immutable'); END;

CREATE TABLE model_setup_attempts (
    id TEXT PRIMARY KEY,
    request_key TEXT NOT NULL UNIQUE,
    manifest_id TEXT NOT NULL REFERENCES model_manifests(id) ON DELETE RESTRICT,
    generation INTEGER NOT NULL CHECK(generation > 0),
    state TEXT NOT NULL CHECK(state IN ('running','installed','failed','cancelled','interrupted')),
    staging_path TEXT NOT NULL UNIQUE,
    received_bytes INTEGER NOT NULL DEFAULT 0 CHECK(received_bytes >= 0),
    spent_seconds REAL NOT NULL DEFAULT 0 CHECK(spent_seconds >= 0),
    transfer_attempts INTEGER NOT NULL DEFAULT 0 CHECK(transfer_attempts >= 0),
    cleanup_pending INTEGER NOT NULL DEFAULT 1 CHECK(cleanup_pending IN (0,1)),
    failure_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(manifest_id,generation)
) STRICT;
CREATE TRIGGER model_setup_identity_is_immutable
BEFORE UPDATE OF id,request_key,manifest_id,generation,staging_path,created_at ON model_setup_attempts
BEGIN SELECT RAISE(ABORT, 'model setup identity is immutable'); END;
