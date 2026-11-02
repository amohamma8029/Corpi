-- Canonical chunks and their citation mappings remain authoritative and unchanged.
CREATE TABLE search_text_inputs (
    row_id INTEGER PRIMARY KEY REFERENCES search_rows(row_id) ON DELETE RESTRICT,
    representation TEXT NOT NULL,
    canonical_sha256 TEXT NOT NULL CHECK(length(canonical_sha256)=64),
    search_sha256 TEXT NOT NULL CHECK(length(search_sha256)=64),
    search_text TEXT NOT NULL
) STRICT;

ALTER TABLE search_batches ADD COLUMN search_input_sha256 TEXT
    CHECK(search_input_sha256 IS NULL OR length(search_input_sha256)=64);

-- Budget/publication fields remain mutable. Input identity cannot be downgraded.
CREATE TRIGGER search_generation_inputs_are_immutable
BEFORE UPDATE OF index_version_id,manifest_id,identity_sha256,settings_json,expected_chunks
ON search_generations
BEGIN
    SELECT RAISE(ABORT, 'search generation inputs are immutable');
END;
