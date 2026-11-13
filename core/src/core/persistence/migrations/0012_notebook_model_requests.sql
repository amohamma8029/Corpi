ALTER TABLE profile_search_publications ADD COLUMN published_by_job_id TEXT REFERENCES jobs(id) ON DELETE RESTRICT;

CREATE TABLE notebook_model_requests (
    request_key TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE RESTRICT,
    manifest_id TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK(generation > 0),
    state TEXT NOT NULL CHECK(state IN ('pending','active','cancelled')),
    created_at TEXT NOT NULL,
    UNIQUE(notebook_id,generation)
) STRICT;
CREATE TRIGGER notebook_model_request_identity_is_immutable
BEFORE UPDATE OF request_key,notebook_id,manifest_id,generation,created_at ON notebook_model_requests
BEGIN SELECT RAISE(ABORT, 'notebook model request identity is immutable'); END;
