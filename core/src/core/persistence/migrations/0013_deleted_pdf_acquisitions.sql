-- Minimal replay protection after whole-source expiry. The action fingerprint
-- covers notebook, intent, requested target and filename; no source/file foreign
-- keys, filename, PDF hash, placement plan or research payload is retained.
CREATE TABLE deleted_pdf_acquisitions (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    retry_key TEXT NOT NULL UNIQUE CHECK (length(trim(retry_key)) > 0),
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE RESTRICT,
    request_sha256 TEXT NOT NULL CHECK (
        length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    deleted_at TEXT NOT NULL CHECK (
        length(deleted_at) = 27 AND deleted_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT;

CREATE TRIGGER pdf_acquisitions_reject_deleted_action_insert
BEFORE INSERT ON pdf_acquisitions
BEGIN
    SELECT RAISE(ABORT, 'PDF acquisition was deleted')
    WHERE EXISTS (
        SELECT 1 FROM deleted_pdf_acquisitions WHERE id=NEW.id OR retry_key=NEW.retry_key
    );
END;

CREATE TRIGGER pdf_acquisitions_reject_deleted_action_update
BEFORE UPDATE OF id,retry_key ON pdf_acquisitions
BEGIN
    SELECT RAISE(ABORT, 'PDF acquisition was deleted')
    WHERE EXISTS (
        SELECT 1 FROM deleted_pdf_acquisitions WHERE id=NEW.id OR retry_key=NEW.retry_key
    );
END;
