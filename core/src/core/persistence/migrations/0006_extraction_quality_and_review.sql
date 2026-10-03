-- M1.5: extraction page quality, visual metadata, and the partial-source review gate.
--
-- These are the records extraction itself needs beyond 1.2's extraction_versions/passages:
--   * extraction_pages    - per-page readability metrics and classification (evidence, immutable)
--   * extraction_visuals  - image/figure locations kept separate from citation passages (immutable)
--   * extraction_outcomes - the committed extraction outcome (complete|partial) recorded on the extraction
--   * extraction_reviews  - the durable Confirm/Decline/Review Again gate for partial extractions
--
-- A failed extraction (no_usable_text, page_limit_exceeded) commits no extraction version at
-- all; its failure code travels on the job failure state instead (see 1.3's job lifecycle).
-- Only 'complete' and 'partial' outcomes exist here because only committed extractions have
-- outcome rows.

CREATE TABLE extraction_pages (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    text_drawn INTEGER NOT NULL CHECK (text_drawn IN (0, 1)),
    extracted_chars INTEGER NOT NULL CHECK (extracted_chars >= 0),
    unmapped_ratio REAL NOT NULL CHECK (unmapped_ratio >= 0 AND unmapped_ratio <= 1),
    image_area_ratio REAL NOT NULL CHECK (image_area_ratio >= 0 AND image_area_ratio <= 1),
    classification TEXT NOT NULL CHECK (
        classification IN ('extracted', 'suspect', 'image-only', 'unreadable', 'blank')
    ),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (source_id, extraction_version_id, page_number),
    UNIQUE (id, extraction_version_id, source_id, source_version_id),
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX extraction_pages_extraction_id_idx
    ON extraction_pages(extraction_version_id, source_id, source_version_id);

CREATE TRIGGER extraction_pages_are_immutable
BEFORE UPDATE OF id, source_id, source_version_id, extraction_version_id, page_number,
                 text_drawn, extracted_chars, unmapped_ratio, image_area_ratio,
                 classification, created_at ON extraction_pages
BEGIN
    SELECT RAISE(ABORT, 'extraction page records are immutable');
END;

CREATE TABLE extraction_visuals (
    id TEXT PRIMARY KEY CHECK (
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    extraction_version_id TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    kind TEXT NOT NULL CHECK (kind IN ('image', 'figure')),
    bbox_json TEXT NOT NULL CHECK (json_valid(bbox_json)),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (source_id, extraction_version_id, page_number, ordinal),
    UNIQUE (id, extraction_version_id, source_id, source_version_id),
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT
) STRICT;
CREATE INDEX extraction_visuals_extraction_id_idx
    ON extraction_visuals(extraction_version_id, source_id, source_version_id);

CREATE TRIGGER extraction_visuals_are_immutable
BEFORE UPDATE OF id, source_id, source_version_id, extraction_version_id, page_number,
                 ordinal, kind, bbox_json, created_at ON extraction_visuals
BEGIN
    SELECT RAISE(ABORT, 'extraction visual records are immutable');
END;

CREATE TABLE extraction_outcomes (
    extraction_version_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('complete', 'partial')),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (extraction_version_id, source_id, source_version_id),
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER extraction_outcomes_are_immutable
BEFORE UPDATE OF extraction_version_id, source_id, source_version_id, outcome, created_at
ON extraction_outcomes
BEGIN
    SELECT RAISE(ABORT, 'extraction outcomes are immutable');
END;

CREATE TABLE extraction_reviews (
    extraction_version_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'confirmed', 'declined')),
    generation INTEGER NOT NULL CHECK (generation > 0),
    decided_at TEXT CHECK (
        decided_at IS NULL
        OR (length(decided_at) = 27 AND decided_at GLOB '????-??-??T??:??:??.??????Z')
    ),
    updated_at TEXT NOT NULL CHECK (
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    UNIQUE (extraction_version_id, source_id, source_version_id),
    FOREIGN KEY (extraction_version_id, source_id, source_version_id)
        REFERENCES extraction_versions(id, source_id, source_version_id) ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER extraction_review_ownership_is_immutable
BEFORE UPDATE OF extraction_version_id, source_id, source_version_id ON extraction_reviews
BEGIN
    SELECT RAISE(ABORT, 'extraction review ownership is immutable');
END;
