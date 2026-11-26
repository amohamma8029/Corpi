-- M2.1 follow-up: preserve uncertain provider outcomes, both request identities,
-- and the discovered model catalog. Credentials remain only in the OS keychain.
CREATE TABLE model_call_usage_new (
    id TEXT PRIMARY KEY CHECK(
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    request_id TEXT NOT NULL CHECK(length(request_id) BETWEEN 1 AND 200),
    provider_request_id TEXT CHECK(
        provider_request_id IS NULL
        OR length(provider_request_id) BETWEEN 1 AND 512
    ),
    parent_request_id TEXT CHECK(
        parent_request_id IS NULL
        OR length(parent_request_id) BETWEEN 1 AND 200
    ),
    call_type TEXT NOT NULL CHECK(call_type IN (
        'generation', 'support_check', 'summary', 'format_repair',
        'citation_retry', 'settings_test'
    )),
    provider_id TEXT NOT NULL,
    model TEXT NOT NULL CHECK(length(model) BETWEEN 1 AND 200),
    input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
    thought_tokens INTEGER CHECK(thought_tokens IS NULL OR thought_tokens >= 0),
    cached_tokens INTEGER CHECK(cached_tokens IS NULL OR cached_tokens >= 0),
    total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens >= 0),
    outcome TEXT NOT NULL CHECK(outcome IN (
        'pending', 'completed', 'failed', 'cancelled', 'unknown'
    )),
    error_code TEXT CHECK(error_code IS NULL OR length(error_code) BETWEEN 1 AND 120),
    started_at TEXT NOT NULL CHECK(
        length(started_at) = 27 AND started_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    finished_at TEXT CHECK(
        finished_at IS NULL OR (
            length(finished_at) = 27 AND finished_at GLOB '????-??-??T??:??:??.??????Z'
        )
    ),
    created_at TEXT NOT NULL CHECK(
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    FOREIGN KEY(provider_id) REFERENCES provider_settings(provider_id) ON DELETE RESTRICT
) STRICT;

INSERT INTO model_call_usage_new(
    id, request_id, provider_request_id, parent_request_id, call_type, provider_id,
    model, input_tokens, output_tokens, thought_tokens, cached_tokens, total_tokens,
    outcome, error_code, started_at, finished_at, created_at
)
SELECT
    id, request_id, NULL, NULL, call_type, provider_id,
    model, input_tokens, output_tokens, thought_tokens, cached_tokens, total_tokens,
    outcome, error_code, started_at, finished_at, created_at
FROM model_call_usage;

DROP TABLE model_call_usage;
ALTER TABLE model_call_usage_new RENAME TO model_call_usage;

CREATE INDEX model_call_usage_request_idx ON model_call_usage(request_id, created_at);
CREATE INDEX model_call_usage_provider_request_idx
    ON model_call_usage(provider_request_id, created_at);
CREATE INDEX model_call_usage_created_idx ON model_call_usage(created_at);

CREATE TABLE provider_model_catalog (
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL CHECK(length(model_id) BETWEEN 1 AND 200),
    discovered_at TEXT NOT NULL CHECK(
        length(discovered_at) = 27 AND discovered_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    PRIMARY KEY(provider_id, model_id)
) STRICT;
