CREATE TABLE notebooks (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT;

CREATE TABLE sources (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE RESTRICT,
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (id, notebook_id)
) STRICT;
CREATE INDEX sources_notebook_id_idx ON sources(notebook_id);

CREATE TABLE managed_files (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    relative_path TEXT NOT NULL UNIQUE CHECK (length(relative_path) > 0),
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT;

CREATE TABLE source_versions (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
    managed_file_id TEXT REFERENCES managed_files(id) ON DELETE RESTRICT,
    content_sha256 TEXT NOT NULL CHECK (
        length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (id, source_id)
) STRICT;
CREATE INDEX source_versions_source_id_idx ON source_versions(source_id);
CREATE INDEX source_versions_managed_file_id_idx ON source_versions(managed_file_id);

CREATE TABLE extraction_versions (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    parser_name TEXT NOT NULL CHECK (length(trim(parser_name)) > 0),
    parser_version TEXT NOT NULL CHECK (length(trim(parser_version)) > 0),
    normalization_version TEXT NOT NULL CHECK (length(trim(normalization_version)) > 0),
    settings_format_version INTEGER NOT NULL CHECK (settings_format_version > 0),
    settings_json TEXT NOT NULL CHECK (json_valid(settings_json)),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (id, source_id, source_version_id),
    FOREIGN KEY (source_version_id, source_id)
        REFERENCES source_versions(id, source_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX extraction_versions_source_version_id_idx
    ON extraction_versions(source_version_id, source_id);

CREATE TABLE passages (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL CHECK (length(text) > 0),
    location_format_version INTEGER NOT NULL CHECK (location_format_version > 0),
    location_json TEXT NOT NULL CHECK (json_valid(location_json)),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (source_id, extraction_version_id, ordinal),
    UNIQUE (source_id, id, source_version_id),
    UNIQUE (id, extraction_version_id, source_id, source_version_id),
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX passages_source_version_lookup_idx
    ON passages(source_id, id, source_version_id);
CREATE INDEX passages_extraction_id_idx ON passages(extraction_version_id, source_id, source_version_id);

CREATE TRIGGER passages_are_immutable
BEFORE UPDATE OF id, source_id, source_version_id, extraction_version_id, ordinal, text,
                 location_format_version, location_json ON passages
BEGIN
    SELECT RAISE(ABORT, 'citation passages are immutable');
END;

CREATE TABLE index_versions (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    index_kind TEXT NOT NULL CHECK (length(trim(index_kind)) > 0),
    chunker_version TEXT NOT NULL CHECK (length(trim(chunker_version)) > 0),
    settings_format_version INTEGER NOT NULL CHECK (settings_format_version > 0),
    settings_json TEXT NOT NULL CHECK (json_valid(settings_json)),
    embedding_model TEXT,
    embedding_revision TEXT,
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    CHECK ((embedding_model IS NULL) = (embedding_revision IS NULL)),
    UNIQUE (id, extraction_version_id),
    UNIQUE (id, source_id, source_version_id, extraction_version_id),
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX index_versions_extraction_id_idx
    ON index_versions(extraction_version_id, source_id, source_version_id);

CREATE TABLE retrieval_chunks (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    index_version_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL CHECK (length(text) > 0),
    UNIQUE (index_version_id, ordinal),
    UNIQUE (id, index_version_id, extraction_version_id, source_id, source_version_id),
    FOREIGN KEY (index_version_id, source_id, source_version_id, extraction_version_id)
        REFERENCES index_versions(id, source_id, source_version_id, extraction_version_id)
        ON DELETE RESTRICT
) STRICT;
CREATE INDEX retrieval_chunks_extraction_id_idx
    ON retrieval_chunks(extraction_version_id, source_id, source_version_id);

CREATE TABLE chunk_passages (
    retrieval_chunk_id TEXT NOT NULL,
    passage_id TEXT NOT NULL,
    index_version_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    start_offset INTEGER,
    end_offset INTEGER,
    CHECK ((start_offset IS NULL) = (end_offset IS NULL)),
    CHECK (start_offset IS NULL OR (start_offset >= 0 AND end_offset >= start_offset)),
    PRIMARY KEY (retrieval_chunk_id, passage_id),
    UNIQUE (retrieval_chunk_id, ordinal),
    FOREIGN KEY (retrieval_chunk_id, index_version_id, extraction_version_id, source_id, source_version_id)
        REFERENCES retrieval_chunks(id, index_version_id, extraction_version_id, source_id, source_version_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (passage_id, extraction_version_id, source_id, source_version_id)
        REFERENCES passages(id, extraction_version_id, source_id, source_version_id)
        ON DELETE RESTRICT
) STRICT;
CREATE INDEX chunk_passages_passage_id_idx ON chunk_passages(passage_id, source_id, source_version_id);
CREATE INDEX chunk_passages_index_version_id_idx ON chunk_passages(index_version_id, extraction_version_id);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    logical_work_key TEXT NOT NULL UNIQUE CHECK (length(trim(logical_work_key)) > 0),
    kind TEXT NOT NULL CHECK (length(trim(kind)) > 0),
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
    source_version_id TEXT,
    extraction_version_id TEXT,
    index_version_id TEXT,
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    CHECK (source_version_id IS NOT NULL OR extraction_version_id IS NULL),
    CHECK (extraction_version_id IS NOT NULL OR index_version_id IS NULL),
    UNIQUE (id, source_id, source_version_id, extraction_version_id),
    FOREIGN KEY (source_version_id, source_id)
        REFERENCES source_versions(id, source_id) ON DELETE RESTRICT,
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT,
    FOREIGN KEY (index_version_id, source_id, source_version_id, extraction_version_id)
        REFERENCES index_versions(id, source_id, source_version_id, extraction_version_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX jobs_source_id_idx ON jobs(source_id);
CREATE INDEX jobs_source_version_id_idx ON jobs(source_version_id, source_id);
CREATE INDEX jobs_extraction_version_id_idx ON jobs(extraction_version_id, source_id, source_version_id);
CREATE INDEX jobs_index_version_id_idx
    ON jobs(index_version_id, source_id, source_version_id, extraction_version_id);

CREATE TABLE job_steps (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE RESTRICT,
    step_key TEXT NOT NULL CHECK (length(trim(step_key)) > 0),
    checkpoint_format_version INTEGER NOT NULL CHECK (checkpoint_format_version > 0),
    checkpoint_json TEXT NOT NULL CHECK (json_valid(checkpoint_json)),
    output_source_version_id TEXT REFERENCES source_versions(id) ON DELETE RESTRICT,
    output_extraction_version_id TEXT REFERENCES extraction_versions(id) ON DELETE RESTRICT,
    output_index_version_id TEXT REFERENCES index_versions(id) ON DELETE RESTRICT,
    committed_at TEXT NOT NULL CHECK (
        length(committed_at) = 27 AND committed_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    CHECK (
        (output_source_version_id IS NOT NULL)
        + (output_extraction_version_id IS NOT NULL)
        + (output_index_version_id IS NOT NULL) <= 1
    ),
    PRIMARY KEY (job_id, step_key)
) STRICT;
CREATE INDEX job_steps_output_source_version_idx ON job_steps(output_source_version_id);
CREATE INDEX job_steps_output_extraction_version_idx ON job_steps(output_extraction_version_id);
CREATE INDEX job_steps_output_index_version_idx ON job_steps(output_index_version_id);

CREATE TRIGGER job_steps_validate_source_version_output
BEFORE INSERT ON job_steps
WHEN NEW.output_source_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'job step source version target mismatch')
    WHERE NOT EXISTS (
        SELECT 1 FROM jobs AS job
        JOIN source_versions AS output ON output.id = NEW.output_source_version_id
        WHERE job.id = NEW.job_id AND job.source_id = output.source_id
    );
END;

CREATE TRIGGER job_steps_validate_extraction_output
BEFORE INSERT ON job_steps
WHEN NEW.output_extraction_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'job step extraction target mismatch')
    WHERE NOT EXISTS (
        SELECT 1 FROM jobs AS job
        JOIN extraction_versions AS output ON output.id = NEW.output_extraction_version_id
        WHERE job.id = NEW.job_id
          AND job.source_id = output.source_id
          AND job.source_version_id = output.source_version_id
    );
END;

CREATE TRIGGER job_steps_validate_index_output
BEFORE INSERT ON job_steps
WHEN NEW.output_index_version_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'job step index target mismatch')
    WHERE NOT EXISTS (
        SELECT 1 FROM jobs AS job
        JOIN index_versions AS output ON output.id = NEW.output_index_version_id
        WHERE job.id = NEW.job_id
          AND job.source_id = output.source_id
          AND job.source_version_id = output.source_version_id
          AND job.extraction_version_id = output.extraction_version_id
    );
END;

CREATE TRIGGER job_steps_validate_update
BEFORE UPDATE ON job_steps
BEGIN
    SELECT RAISE(ABORT, 'job steps are immutable once committed');
END;
