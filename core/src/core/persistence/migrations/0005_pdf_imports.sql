-- Durable local PDF acquisitions, per-source file history, and canonical originals.

CREATE TABLE pdf_acquisitions (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    retry_key TEXT NOT NULL UNIQUE CHECK (length(trim(retry_key)) > 0),
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE RESTRICT,
    intent TEXT NOT NULL CHECK (intent IN ('add','replace')),
    target_source_id TEXT REFERENCES sources(id) ON DELETE RESTRICT,
    expected_current_import_id TEXT,
    display_filename TEXT NOT NULL CHECK (length(trim(display_filename)) > 0),
    state TEXT NOT NULL CHECK (
        state IN ('awaiting_upload','receiving','staged','finalizing','waiting_cleanup',
                  'completed','cancelled','failed')
    ),
    finalization_generation INTEGER NOT NULL DEFAULT 1 CHECK (finalization_generation > 0),
    owner_instance_id TEXT NOT NULL CHECK (length(trim(owner_instance_id)) > 0),
    staging_relative_path TEXT NOT NULL UNIQUE CHECK (length(trim(staging_relative_path)) > 0),
    input_sha256 TEXT CHECK (
        input_sha256 IS NULL OR
        (length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*')
    ),
    input_size_bytes INTEGER CHECK (input_size_bytes IS NULL OR input_size_bytes >= 0),
    received_size_bytes INTEGER NOT NULL DEFAULT 0 CHECK (received_size_bytes >= 0),
    input_complete_at TEXT CHECK (
        input_complete_at IS NULL OR
        (length(input_complete_at) = 27 AND input_complete_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    last_write_at TEXT NOT NULL CHECK (
        length(last_write_at) = 27 AND last_write_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    planned_reuse_managed_file_id TEXT REFERENCES managed_files(id) ON DELETE RESTRICT,
    planned_managed_file_id TEXT CHECK (
        planned_managed_file_id IS NULL OR
        (planned_managed_file_id GLOB '????????-????-4???-[89ab]???-????????????'
         AND planned_managed_file_id NOT GLOB '*[^0-9a-f-]*')
    ),
    planned_relative_path TEXT,
    planned_sha256 TEXT CHECK (
        planned_sha256 IS NULL OR
        (length(planned_sha256) = 64 AND planned_sha256 NOT GLOB '*[^0-9a-f]*')
    ),
    planned_size_bytes INTEGER CHECK (planned_size_bytes IS NULL OR planned_size_bytes >= 0),
    placement_state TEXT NOT NULL DEFAULT 'none' CHECK (
        placement_state IN ('none','planned','placed','cleanup_pending','removed')
    ),
    outcome_kind TEXT CHECK (
        outcome_kind IS NULL OR outcome_kind IN ('created','duplicate','replaced','unchanged')
    ),
    outcome_source_id TEXT REFERENCES sources(id) ON DELETE RESTRICT,
    outcome_import_id TEXT,
    failure_code TEXT,
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    CHECK ((input_sha256 IS NULL) = (input_size_bytes IS NULL)),
    CHECK ((input_complete_at IS NULL) = (input_sha256 IS NULL)),
    CHECK ((planned_managed_file_id IS NULL) = (planned_relative_path IS NULL)),
    CHECK ((planned_sha256 IS NULL) = (planned_size_bytes IS NULL)),
    CHECK (intent = 'add' OR target_source_id IS NOT NULL),
    UNIQUE (id, target_source_id)
) STRICT;
CREATE INDEX pdf_acquisitions_state_idx ON pdf_acquisitions(state,updated_at);
CREATE INDEX pdf_acquisitions_target_source_idx ON pdf_acquisitions(target_source_id,state);
CREATE INDEX pdf_acquisitions_reuse_hold_idx ON pdf_acquisitions(planned_reuse_managed_file_id);

CREATE TABLE pdf_imports (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL UNIQUE,
    acquisition_id TEXT REFERENCES pdf_acquisitions(id) ON DELETE RESTRICT,
    display_filename TEXT NOT NULL CHECK (length(trim(display_filename)) > 0),
    state TEXT NOT NULL CHECK (state IN ('pending','ready','failed')),
    failure_code TEXT,
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (id,source_id),
    FOREIGN KEY (source_version_id,source_id)
        REFERENCES source_versions(id,source_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX pdf_imports_source_id_idx ON pdf_imports(source_id,created_at,id);
CREATE INDEX pdf_imports_acquisition_idx ON pdf_imports(acquisition_id);

-- Older managed paths identify likely PDFs by their saved name. Record only
-- database metadata; never read, merge, move, or delete a legacy file here.
INSERT INTO pdf_imports(
    id,source_id,source_version_id,acquisition_id,display_filename,state,
    failure_code,created_at,updated_at
)
SELECT lower(
           substr(hex(randomblob(4)),1,8) || '-' ||
           substr(hex(randomblob(2)),1,4) || '-4' ||
           substr(hex(randomblob(2)),2,3) || '-' ||
           substr('89ab',(random() & 3)+1,1) ||
           substr(hex(randomblob(2)),2,3) || '-' ||
           substr(hex(randomblob(6)),1,12)
       ),
       source_version.source_id,
       source_version.id,
       NULL,
       'Earlier PDF (filename unknown)',
       CASE
           WHEN readiness.source_version_id=source_version.id THEN 'ready'
           WHEN EXISTS (
               SELECT 1 FROM jobs AS completed_job
               JOIN job_status AS completed_status ON completed_status.job_id=completed_job.id
               WHERE completed_job.source_version_id=source_version.id
                 AND completed_job.kind='pdf-extract'
                 AND completed_status.status='completed'
           ) THEN 'ready'
           WHEN EXISTS (
               SELECT 1 FROM jobs AS failed_job
               JOIN job_status AS failed_status ON failed_status.job_id=failed_job.id
               WHERE failed_job.source_version_id=source_version.id
                 AND failed_job.kind='pdf-extract'
                 AND failed_status.status IN ('failed','cancelled')
           ) THEN 'failed'
           ELSE 'pending'
       END,
       CASE WHEN EXISTS (
           SELECT 1 FROM jobs AS failed_job
           JOIN job_status AS failed_status ON failed_status.job_id=failed_job.id
           WHERE failed_job.source_version_id=source_version.id
             AND failed_job.kind='pdf-extract'
             AND failed_status.status IN ('failed','cancelled')
       ) THEN 'legacy_processing_failed' ELSE NULL END,
       source_version.created_at,
       source_version.created_at
FROM source_versions AS source_version
JOIN sources AS source ON source.id=source_version.source_id
JOIN managed_files AS managed_file ON managed_file.id=source_version.managed_file_id
LEFT JOIN source_readiness AS readiness
       ON readiness.source_id=source_version.source_id
WHERE lower(managed_file.relative_path) LIKE '%.pdf'
   OR lower(source.display_name) LIKE '%.pdf';

CREATE TABLE source_pdf_state (
    source_id TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE RESTRICT,
    current_import_id TEXT,
    pending_import_id TEXT,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    CHECK (current_import_id IS NULL OR pending_import_id IS NULL OR
           current_import_id <> pending_import_id),
    FOREIGN KEY (current_import_id,source_id)
        REFERENCES pdf_imports(id,source_id) ON DELETE RESTRICT,
    FOREIGN KEY (pending_import_id,source_id)
        REFERENCES pdf_imports(id,source_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX source_pdf_state_pending_idx ON source_pdf_state(pending_import_id);

INSERT INTO source_pdf_state(
    source_id,current_import_id,pending_import_id,revision,updated_at
)
SELECT source.id,
       COALESCE(
           (SELECT item.id FROM pdf_imports AS item
            JOIN source_readiness AS readiness
              ON readiness.source_id=item.source_id
             AND readiness.source_version_id=item.source_version_id
            WHERE item.source_id=source.id LIMIT 1),
           (SELECT item.id FROM pdf_imports AS item
            JOIN source_versions AS version ON version.id=item.source_version_id
            WHERE item.source_id=source.id
            ORDER BY version.created_at DESC,version.rowid DESC LIMIT 1)
       ),
       NULL,
       1,
       strftime('%Y-%m-%dT%H:%M:%f','now') || '000Z'
FROM sources AS source
WHERE EXISTS (SELECT 1 FROM pdf_imports WHERE source_id=source.id);

CREATE TABLE original_registry (
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    managed_file_id TEXT NOT NULL UNIQUE REFERENCES managed_files(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    PRIMARY KEY (sha256,size_bytes)
) STRICT;
CREATE INDEX original_registry_file_idx ON original_registry(managed_file_id);

CREATE TABLE pdf_import_trash (
    import_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    trashed_at TEXT NOT NULL CHECK (
        length(trashed_at) = 27 AND trashed_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    purge_deadline_at TEXT NOT NULL CHECK (
        length(purge_deadline_at) = 27 AND purge_deadline_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    file_removed_at TEXT CHECK (
        file_removed_at IS NULL OR
        (length(file_removed_at) = 27 AND file_removed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    snapshots_removed_at TEXT CHECK (
        snapshots_removed_at IS NULL OR
        (length(snapshots_removed_at) = 27 AND snapshots_removed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    FOREIGN KEY (import_id,source_id)
        REFERENCES pdf_imports(id,source_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX pdf_import_trash_deadline_idx ON pdf_import_trash(purge_deadline_at);

CREATE TABLE acquisition_file_cleanup (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    acquisition_id TEXT NOT NULL REFERENCES pdf_acquisitions(id) ON DELETE RESTRICT,
    relative_path TEXT NOT NULL CHECK (length(trim(relative_path)) > 0),
    file_kind TEXT NOT NULL CHECK (file_kind IN ('staging','planned_original')),
    expected_sha256 TEXT CHECK (
        expected_sha256 IS NULL OR
        (length(expected_sha256) = 64 AND expected_sha256 NOT GLOB '*[^0-9a-f]*')
    ),
    expected_size_bytes INTEGER CHECK (expected_size_bytes IS NULL OR expected_size_bytes >= 0),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT,
    removed_at TEXT CHECK (
        removed_at IS NULL OR
        (length(removed_at) = 27 AND removed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (acquisition_id,relative_path)
) STRICT;
CREATE INDEX acquisition_file_cleanup_pending_idx
    ON acquisition_file_cleanup(removed_at,created_at);

CREATE TRIGGER pdf_acquisitions_reject_removal_marked_hold_insert
BEFORE INSERT ON pdf_acquisitions
WHEN NEW.planned_reuse_managed_file_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (SELECT 1 FROM managed_file_removals
                  WHERE managed_file_id=NEW.planned_reuse_managed_file_id);
END;

CREATE TRIGGER pdf_acquisitions_reject_removal_marked_hold_update
BEFORE UPDATE OF planned_reuse_managed_file_id ON pdf_acquisitions
WHEN NEW.planned_reuse_managed_file_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (SELECT 1 FROM managed_file_removals
                  WHERE managed_file_id=NEW.planned_reuse_managed_file_id);
END;

CREATE TRIGGER original_registry_validate_target_insert
BEFORE INSERT ON original_registry
BEGIN
    SELECT RAISE(ABORT, 'registry fingerprint does not match managed file')
    WHERE NOT EXISTS (
        SELECT 1 FROM managed_files
        WHERE id=NEW.managed_file_id AND sha256=NEW.sha256 AND size_bytes=NEW.size_bytes
    );
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (SELECT 1 FROM managed_file_removals
                  WHERE managed_file_id=NEW.managed_file_id);
END;

CREATE TRIGGER original_registry_validate_target_update
BEFORE UPDATE OF sha256,size_bytes,managed_file_id ON original_registry
BEGIN
    SELECT RAISE(ABORT, 'registry fingerprint does not match managed file')
    WHERE NOT EXISTS (
        SELECT 1 FROM managed_files
        WHERE id=NEW.managed_file_id AND sha256=NEW.sha256 AND size_bytes=NEW.size_bytes
    );
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (SELECT 1 FROM managed_file_removals
                  WHERE managed_file_id=NEW.managed_file_id);
END;

CREATE TRIGGER source_versions_reject_removal_marked_files_update
BEFORE UPDATE OF managed_file_id ON source_versions
WHEN NEW.managed_file_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (SELECT 1 FROM managed_file_removals
                  WHERE managed_file_id=NEW.managed_file_id);
END;

CREATE TRIGGER pdf_imports_validate_acquisition_source
BEFORE INSERT ON pdf_imports
WHEN NEW.acquisition_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'PDF acquisition target mismatch')
    WHERE NOT EXISTS (
        SELECT 1 FROM pdf_acquisitions AS acquisition
        JOIN sources AS source ON source.id=NEW.source_id
        WHERE acquisition.id=NEW.acquisition_id
          AND acquisition.notebook_id=source.notebook_id
          AND (acquisition.target_source_id IS NULL OR
               acquisition.target_source_id=NEW.source_id)
    );
END;

CREATE TRIGGER pdf_imports_reject_removal_marked_file_insert
BEFORE INSERT ON pdf_imports
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM source_versions AS version
        JOIN managed_file_removals AS removal
          ON removal.managed_file_id=version.managed_file_id
        WHERE version.id=NEW.source_version_id AND version.source_id=NEW.source_id
    );
END;

CREATE TRIGGER source_pdf_state_validate_imports_insert
BEFORE INSERT ON source_pdf_state
BEGIN
    SELECT RAISE(ABORT, 'current PDF import belongs to another source')
    WHERE NEW.current_import_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pdf_imports WHERE id=NEW.current_import_id AND source_id=NEW.source_id
    );
    SELECT RAISE(ABORT, 'pending PDF import belongs to another source')
    WHERE NEW.pending_import_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pdf_imports WHERE id=NEW.pending_import_id AND source_id=NEW.source_id
    );
    SELECT RAISE(ABORT, 'trashed PDF import cannot become current')
    WHERE NEW.current_import_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM pdf_import_trash WHERE import_id=NEW.current_import_id
    );
    SELECT RAISE(ABORT, 'trashed PDF import cannot become pending')
    WHERE NEW.pending_import_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM pdf_import_trash WHERE import_id=NEW.pending_import_id
    );
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM pdf_imports AS item
        JOIN source_versions AS version
          ON version.id=item.source_version_id AND version.source_id=item.source_id
        JOIN managed_file_removals AS removal
          ON removal.managed_file_id=version.managed_file_id
        WHERE item.source_id=NEW.source_id
          AND item.id IN (NEW.current_import_id,NEW.pending_import_id)
    );
END;

CREATE TRIGGER source_pdf_state_validate_imports_update
BEFORE UPDATE OF current_import_id,pending_import_id,source_id ON source_pdf_state
BEGIN
    SELECT RAISE(ABORT, 'current PDF import belongs to another source')
    WHERE NEW.current_import_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pdf_imports WHERE id=NEW.current_import_id AND source_id=NEW.source_id
    );
    SELECT RAISE(ABORT, 'pending PDF import belongs to another source')
    WHERE NEW.pending_import_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pdf_imports WHERE id=NEW.pending_import_id AND source_id=NEW.source_id
    );
    SELECT RAISE(ABORT, 'trashed PDF import cannot become current')
    WHERE NEW.current_import_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM pdf_import_trash WHERE import_id=NEW.current_import_id
    );
    SELECT RAISE(ABORT, 'trashed PDF import cannot become pending')
    WHERE NEW.pending_import_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM pdf_import_trash WHERE import_id=NEW.pending_import_id
    );
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM pdf_imports AS item
        JOIN source_versions AS version
          ON version.id=item.source_version_id AND version.source_id=item.source_id
        JOIN managed_file_removals AS removal
          ON removal.managed_file_id=version.managed_file_id
        WHERE item.source_id=NEW.source_id
          AND item.id IN (NEW.current_import_id,NEW.pending_import_id)
    );
END;

CREATE TRIGGER pdf_import_trash_reject_in_use
BEFORE INSERT ON pdf_import_trash
BEGIN
    SELECT RAISE(ABORT, 'current or pending PDF import cannot enter history trash')
    WHERE EXISTS (
        SELECT 1 FROM source_pdf_state
        WHERE source_id=NEW.source_id
          AND NEW.import_id IN (current_import_id,pending_import_id)
    );
    SELECT RAISE(ABORT, 'PDF import with active processing cannot enter history trash')
    WHERE EXISTS (
        SELECT 1 FROM pdf_imports AS item
        JOIN jobs ON jobs.source_version_id=item.source_version_id
        JOIN job_status ON job_status.job_id=jobs.id
        WHERE item.id=NEW.import_id AND job_status.status IN ('queued','running')
    );
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM pdf_imports AS item
        JOIN source_versions AS version
          ON version.id=item.source_version_id AND version.source_id=item.source_id
        JOIN managed_file_removals AS removal
          ON removal.managed_file_id=version.managed_file_id
        WHERE item.id=NEW.import_id AND item.source_id=NEW.source_id
    );
END;

CREATE TRIGGER pdf_imports_reject_removal_marked_file_update
BEFORE UPDATE OF source_version_id ON pdf_imports
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM source_versions AS version
        JOIN managed_file_removals AS removal ON removal.managed_file_id=version.managed_file_id
        WHERE version.id=NEW.source_version_id AND version.source_id=NEW.source_id
    );
END;

CREATE TRIGGER jobs_reject_trashed_pdf_source_version_insert
BEFORE INSERT ON jobs
WHEN NEW.source_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'PDF import is in history trash')
    WHERE EXISTS (
        SELECT 1 FROM pdf_imports AS item
        JOIN pdf_import_trash AS trash ON trash.import_id=item.id
        WHERE item.source_version_id=NEW.source_version_id
    );
END;

CREATE TRIGGER jobs_reject_trashed_pdf_source_version_update
BEFORE UPDATE OF source_version_id ON jobs
WHEN NEW.source_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'PDF import is in history trash')
    WHERE EXISTS (
        SELECT 1 FROM pdf_imports AS item
        JOIN pdf_import_trash AS trash ON trash.import_id=item.id
        WHERE item.source_version_id=NEW.source_version_id
    );
END;

CREATE TRIGGER jobs_reject_removal_marked_pdf_file_insert
BEFORE INSERT ON jobs
WHEN NEW.source_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM source_versions AS version
        JOIN managed_file_removals AS removal
          ON removal.managed_file_id=version.managed_file_id
        WHERE version.id=NEW.source_version_id AND version.source_id=NEW.source_id
    );
END;

CREATE TRIGGER jobs_reject_removal_marked_pdf_file_update
BEFORE UPDATE OF source_version_id ON jobs
WHEN NEW.source_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'managed file is pending removal')
    WHERE EXISTS (
        SELECT 1 FROM source_versions AS version
        JOIN managed_file_removals AS removal
          ON removal.managed_file_id=version.managed_file_id
        WHERE version.id=NEW.source_version_id AND version.source_id=NEW.source_id
    );
END;
