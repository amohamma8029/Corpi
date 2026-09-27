-- Mutable runner and source-lifecycle records (subproblem 1.3, issue #9).
--
-- Committed evidence stays in jobs/job_steps (immutable once written); this
-- migration adds the records the runner updates while work runs: job status,
-- ownership, and attempts; source readiness; and the trash/purge tracker.
-- The trash tracker keeps a cascading key to its source and the final purge
-- transaction removes the source row and tracker together, so a tracker row
-- always implies its source row still exists.

CREATE TABLE job_status (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (
        status IN ('queued','running','completed','failed','cancelled')
    ),
    current_step_key TEXT CHECK (
        current_step_key IS NULL OR length(trim(current_step_key)) > 0
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    owner_instance_id TEXT CHECK (
        owner_instance_id IS NULL OR length(trim(owner_instance_id)) > 0
    ),
    claimed_at TEXT CHECK (
        claimed_at IS NULL
        OR (length(claimed_at) = 27 AND claimed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    heartbeat_at TEXT CHECK (
        heartbeat_at IS NULL
        OR (length(heartbeat_at) = 27 AND heartbeat_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    failure_reason TEXT,
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT;
CREATE INDEX job_status_status_idx ON job_status(status);

CREATE TABLE job_step_attempts (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE RESTRICT,
    step_key TEXT NOT NULL CHECK (length(trim(step_key)) > 0),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    started_at TEXT NOT NULL CHECK (
        length(started_at) = 27 AND started_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    finished_at TEXT CHECK (
        finished_at IS NULL
        OR (length(finished_at) = 27 AND finished_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    failure_reason TEXT,
    PRIMARY KEY (job_id, step_key, attempt_number)
) STRICT;

CREATE TABLE source_readiness (
    source_id TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL,
    published_at TEXT NOT NULL CHECK (
        length(published_at) = 27 AND published_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    FOREIGN KEY (source_version_id, source_id)
        REFERENCES source_versions(id, source_id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE source_readiness_indexes (
    source_id TEXT NOT NULL REFERENCES source_readiness(source_id) ON DELETE RESTRICT,
    index_version_id TEXT NOT NULL REFERENCES index_versions(id) ON DELETE RESTRICT,
    PRIMARY KEY (source_id, index_version_id)
) STRICT;

CREATE TRIGGER source_readiness_indexes_validate_target
BEFORE INSERT ON source_readiness_indexes
BEGIN
    SELECT RAISE(ABORT, 'readiness index target mismatch')
    WHERE NOT EXISTS (
        SELECT 1
        FROM source_readiness AS readiness
        JOIN index_versions AS index_version
          ON index_version.id = NEW.index_version_id
         AND index_version.source_id = readiness.source_id
         AND index_version.source_version_id = readiness.source_version_id
        WHERE readiness.source_id = NEW.source_id
    );
END;

CREATE TABLE source_trash (
    source_id TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    trashed_at TEXT NOT NULL CHECK (
        length(trashed_at) = 27 AND trashed_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    purge_deadline_at TEXT NOT NULL CHECK (
        length(purge_deadline_at) = 27 AND purge_deadline_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    files_removed_at TEXT CHECK (
        files_removed_at IS NULL
        OR (length(files_removed_at) = 27 AND files_removed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    snapshots_removed_at TEXT CHECK (
        snapshots_removed_at IS NULL
        OR (length(snapshots_removed_at) = 27 AND snapshots_removed_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT;
CREATE INDEX source_trash_deadline_idx ON source_trash(purge_deadline_at);
