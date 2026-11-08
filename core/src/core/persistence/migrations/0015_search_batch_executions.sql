-- Actual runtime of new attempts, separate from immutable generation settings.
-- Existing completed batches are deliberately not backfilled or relabelled.
CREATE TABLE search_batch_executions (
    index_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    batch_attempt_number INTEGER NOT NULL CHECK(batch_attempt_number BETWEEN 1 AND 2),
    job_id TEXT NOT NULL,
    step_key TEXT NOT NULL CHECK(step_key='embed-search'),
    step_attempt_number INTEGER NOT NULL CHECK(step_attempt_number > 0),
    runtime_json TEXT NOT NULL CHECK(
        length(runtime_json) <= 2048 AND json_valid(runtime_json)
        AND json_type(runtime_json)='object'
    ),
    created_at TEXT NOT NULL CHECK(
        length(created_at)=27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    PRIMARY KEY(index_version_id,ordinal,batch_attempt_number),
    FOREIGN KEY(index_version_id,ordinal)
        REFERENCES search_batches(index_version_id,ordinal) ON DELETE RESTRICT,
    FOREIGN KEY(job_id,step_key,step_attempt_number)
        REFERENCES job_step_attempts(job_id,step_key,attempt_number) ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER search_batch_executions_are_immutable
BEFORE UPDATE ON search_batch_executions
BEGIN
    SELECT RAISE(ABORT, 'search batch execution is immutable');
END;
