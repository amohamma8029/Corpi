ALTER TABLE search_handoffs ADD COLUMN origin_job_id TEXT REFERENCES jobs(id) ON DELETE RESTRICT;

CREATE TRIGGER search_handoff_origin_matches_extraction
BEFORE INSERT ON search_handoffs WHEN NEW.origin_job_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'search handoff origin mismatch') WHERE NOT EXISTS (
        SELECT 1 FROM jobs j JOIN extraction_versions e ON e.id=NEW.extraction_version_id
        WHERE j.id=NEW.origin_job_id AND j.source_id=e.source_id
          AND j.source_version_id=e.source_version_id AND j.kind='pdf-extract'
    );
END;

CREATE TRIGGER search_handoff_origin_update_matches_extraction
BEFORE UPDATE OF origin_job_id ON search_handoffs WHEN NEW.origin_job_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'search handoff origin mismatch') WHERE NOT EXISTS (
        SELECT 1 FROM jobs j JOIN extraction_versions e ON e.id=NEW.extraction_version_id
        WHERE j.id=NEW.origin_job_id AND j.source_id=e.source_id
          AND j.source_version_id=e.source_version_id AND j.kind='pdf-extract'
    );
END;

CREATE TRIGGER search_job_inputs_are_immutable
BEFORE UPDATE OF job_id,profile_id,settings_json,review_generation ON search_job_inputs
BEGIN
    SELECT RAISE(ABORT, 'search job inputs are immutable');
END;
