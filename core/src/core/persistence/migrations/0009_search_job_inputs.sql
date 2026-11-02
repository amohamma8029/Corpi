CREATE TABLE search_job_inputs (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE RESTRICT,
    profile_id TEXT NOT NULL,
    settings_json TEXT NOT NULL CHECK(json_valid(settings_json)),
    review_generation INTEGER NOT NULL CHECK(review_generation >= 0),
    spent_seconds REAL NOT NULL DEFAULT 0 CHECK(spent_seconds >= 0)
) STRICT;
