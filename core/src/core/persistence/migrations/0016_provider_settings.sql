-- M2.1: provider configuration and model-call usage. Credentials live in the
-- OS keychain; this database stores only non-secret settings and call records.
CREATE TABLE provider_settings (
    provider_id TEXT PRIMARY KEY CHECK(length(provider_id) BETWEEN 1 AND 80),
    settings_format_version INTEGER NOT NULL CHECK(settings_format_version > 0),
    generation_model TEXT NOT NULL CHECK(length(generation_model) BETWEEN 1 AND 200),
    checker_model TEXT NOT NULL CHECK(length(checker_model) BETWEEN 1 AND 200),
    free_tier_acknowledged INTEGER NOT NULL DEFAULT 0
        CHECK(free_tier_acknowledged IN (0, 1)),
    output_token_allowance INTEGER NOT NULL
        CHECK(output_token_allowance BETWEEN 1 AND 1000000),
    safety_margin_tokens INTEGER NOT NULL
        CHECK(safety_margin_tokens BETWEEN 0 AND 1000000),
    call_policies_json TEXT NOT NULL CHECK(
        length(call_policies_json) <= 8192
        AND json_valid(call_policies_json)
        AND json_type(call_policies_json) = 'object'
    ),
    created_at TEXT NOT NULL CHECK(
        length(created_at) = 27 AND created_at GLOB '????-??-??T??:??:??.??????Z'
    ),
    updated_at TEXT NOT NULL CHECK(
        length(updated_at) = 27 AND updated_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT;

CREATE TABLE model_call_usage (
    id TEXT PRIMARY KEY CHECK(
        id GLOB '????????-????-4???-[89ab]???-????????????'
        AND id NOT GLOB '*[^0-9a-f-]*'
    ),
    request_id TEXT NOT NULL CHECK(length(request_id) BETWEEN 1 AND 200),
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
    outcome TEXT NOT NULL CHECK(outcome IN ('completed', 'failed', 'cancelled', 'unknown')),
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

CREATE INDEX model_call_usage_request_idx ON model_call_usage(request_id, created_at);
CREATE INDEX model_call_usage_created_idx ON model_call_usage(created_at);
