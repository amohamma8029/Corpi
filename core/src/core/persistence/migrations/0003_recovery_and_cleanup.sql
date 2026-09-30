-- Review-fix foundations for the job runner and cleanup guarantees.
--
-- Adds, in support of the recovery/cancellation/cleanup corrections:
-- * source_readiness.published_by_job_id: records which job published a ready
--   version, so cancelling that job can hide exactly its own result and never
--   a newer job's result.
-- * job_steps.inputs_sha256: the inputs/settings fingerprint each committed
--   step ran with, so changed work cannot silently reuse an old result.
-- * job_status.policy_json: the retry policy recorded when a job is claimed,
--   so a resumed job keeps the same allowance.
-- * managed_file_removals: a durable marker for managed files that must be
--   unlinked. It survives failed removals for a later sweep, and the trigger
--   below refuses new source-version references to a marked file so a purge
--   cannot race a fresh reference. Managed-file rows marked here are kept by
--   the records purge until their file is actually gone.

ALTER TABLE source_readiness
ADD COLUMN published_by_job_id TEXT REFERENCES jobs(id) ON DELETE RESTRICT;

ALTER TABLE job_steps
ADD COLUMN inputs_sha256 TEXT CHECK (
    inputs_sha256 IS NULL
    OR (length(inputs_sha256) = 64 AND inputs_sha256 NOT GLOB '*[^0-9a-f]*')
);

ALTER TABLE job_status
ADD COLUMN policy_json TEXT CHECK (policy_json IS NULL OR json_valid(policy_json));

CREATE TABLE managed_file_removals (
    managed_file_id TEXT PRIMARY KEY REFERENCES managed_files(id) ON DELETE RESTRICT,
    relative_path TEXT NOT NULL CHECK (length(trim(relative_path)) > 0),
    requested_at TEXT NOT NULL CHECK (
        length(requested_at) = 27 AND requested_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    removed_at TEXT CHECK (
        removed_at IS NULL
        OR (length(removed_at) = 27 AND removed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT
) STRICT;

CREATE TRIGGER source_versions_reject_removal_marked_files
BEFORE INSERT ON source_versions
WHEN NEW.managed_file_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1
        FROM managed_file_removals
        WHERE managed_file_id = NEW.managed_file_id
    );
END;
