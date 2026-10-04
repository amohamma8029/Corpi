-- M1.5 review round: structured failure codes on job failures.
--
-- Extraction failures carry machine-readable codes (`no_usable_text`,
-- `page_limit_exceeded`) alongside the human-readable failure reason, so the API and
-- the Files UI report specific causes instead of parsing prose.

ALTER TABLE job_status ADD COLUMN failure_code TEXT CHECK (
    failure_code IS NULL OR length(trim(failure_code)) > 0
);
