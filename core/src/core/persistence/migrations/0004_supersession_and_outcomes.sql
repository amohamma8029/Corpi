-- Durable supersession marks and paid-call attempt outcomes (review round 8).
--
-- * job_step_attempts.outcome records what happened to an attempt: 'ok'
--   (finished normally), 'failed' (finished with a known failure), or
--   'unknown' (a paid call whose outcome is unknown). Recovery checks this
--   even when the attempt has a finish timestamp, so a crash between the
--   attempt write and the job write can never make an unknown paid call
--   replayable.
-- * job_status.superseded_by_job_id records that newer work replaced this
--   job's result, independently of the removable readiness record: cancelling
--   the newer result must not restore the older job's right to publish.

ALTER TABLE job_step_attempts
ADD COLUMN outcome TEXT CHECK (outcome IS NULL OR outcome IN ('ok', 'failed', 'unknown'));

ALTER TABLE job_status
ADD COLUMN superseded_by_job_id TEXT REFERENCES jobs(id) ON DELETE RESTRICT;
