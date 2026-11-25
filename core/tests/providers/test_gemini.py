import asyncio
import json
from typing import Any

import httpx
import pytest

from core.jobs.runner import PaidCallOutcomeUnknown
from core.providers.errors import ProviderCancelled, ProviderFailure
from core.providers.gemini import GeminiClient, _retry_delay


def call(client: GeminiClient, **overrides):
    values = {
        "model": "gemini-3.8-flash",
        "instructions": "Use only the supplied evidence.",
        "input_text": "What is the result?",
        "schema": None,
        "max_output_tokens": 128,
        "timeout_seconds": 5,
        "attempts": 1,
        "request_id": "local-request",
    }
    values.update(overrides)
    return asyncio.run(client.generate(**values))


def test_generate_uses_interactions_contract_and_parses_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers["x-goog-api-key"]
        seen["revision"] = request.headers["Api-Revision"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "interaction-1",
                "output_text": "Supported answer.",
                "usage": {
                    "total_input_tokens": 120,
                    "total_output_tokens": 30,
                    "total_thought_tokens": 4,
                    "total_cached_tokens": 8,
                    "total_tokens": 162,
                },
            },
        )

    result = call(
        GeminiClient("secret-key", transport=httpx.MockTransport(handler)),
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    assert seen["path"] == "/v1beta/interactions"
    assert seen["key"] == "secret-key"
    assert seen["revision"] == "2026-05-20"
    body = seen["body"]
    assert body["store"] is False
    assert body["generation_config"] == {
        "max_output_tokens": 128,
        "thinking_level": "low",
    }
    assert body["response_format"]["type"] == "text"
    assert body["response_format"]["mime_type"] == "application/json"
    assert result.output_text == "Supported answer."
    assert result.request_id == "interaction-1"
    assert (result.input_tokens, result.output_tokens, result.thought_tokens) == (120, 30, 4)
    assert (result.cached_tokens, result.total_tokens) == (8, 162)


def test_generate_retries_429_and_503_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "slow down"}},
            )
        if calls == 2:
            return httpx.Response(
                503,
                headers={"retry-after": "0"},
                json={"error": {"message": "temporary"}},
            )
        return httpx.Response(200, json={"id": "ok", "output_text": "done"})

    result = call(
        GeminiClient("secret-key", transport=httpx.MockTransport(handler)),
        attempts=3,
    )

    assert calls == 3
    assert result.output_text == "done"


def test_generate_does_not_replay_a_timeout() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(PaidCallOutcomeUnknown):
        call(
            GeminiClient("secret-key", transport=httpx.MockTransport(handler)),
            attempts=2,
        )
    assert calls == 1


def test_generate_retries_connect_errors_then_reports_unavailable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(ProviderFailure) as failure:
        call(
            GeminiClient("secret-key", transport=httpx.MockTransport(handler)),
            attempts=2,
        )
    assert calls == 2
    assert failure.value.code == "provider_unavailable"


@pytest.mark.parametrize(
    ("status", "message", "code"),
    [
        (401, "bad key", "invalid_credentials"),
        (429, "quota exceeded", "quota_exhausted"),
        (429, "rate limit", "rate_limited"),
        (400, "context window exceeded", "context_length_exceeded"),
        (400, "invalid schema", "provider_rejected"),
        (500, "server error", "provider_unavailable"),
    ],
)
def test_generate_maps_provider_errors(status: int, message: str, code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message}})

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))
    assert failure.value.code == code


def test_error_details_map_invalid_key_and_entitlement_failures() -> None:
    def invalid_key(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "API key not valid.",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(invalid_key)))
    assert failure.value.code == "invalid_credentials"

    def project_denied(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": {
                    "message": "The project is not allowed to use this service.",
                    "details": [{"reason": "SERVICE_DISABLED"}],
                }
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(project_denied)))
    assert failure.value.code == "provider_rejected"


def test_error_details_distinguish_rate_limit_from_quota_exhaustion() -> None:
    def per_minute(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "Resource exhausted.",
                    "details": [
                        {
                            "violations": [
                                {
                                    "quotaId": (
                                        "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
                                    )
                                }
                            ]
                        }
                    ],
                }
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(per_minute)))
    assert failure.value.code == "rate_limited"

    def per_day(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "Resource exhausted.",
                    "details": [
                        {
                            "violations": [
                                {"quotaId": ("GenerateRequestsPerDayPerProjectPerModel-FreeTier")}
                            ]
                        }
                    ],
                }
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(per_day)))
    assert failure.value.code == "quota_exhausted"


def test_incomplete_or_truncated_response_is_not_accepted() -> None:
    def incomplete(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "partial", "status": "incomplete", "output_text": "truncated"},
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(incomplete)))
    assert failure.value.code == "incomplete_response"

    def max_tokens(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "partial", "finish_reason": "MAX_TOKENS", "output_text": "truncated"},
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(max_tokens)))
    assert failure.value.code == "incomplete_response"


def test_retry_delay_honors_provider_directives() -> None:
    header = httpx.Response(429, headers={"retry-after": "60"})
    assert _retry_delay(header) == 60.0

    payload = httpx.Response(
        429,
        json={"error": {"details": [{"retryDelay": "46.130728812s"}]}},
    )
    assert _retry_delay(payload) == pytest.approx(46.130728812)

    array_payload = httpx.Response(
        429,
        json=[{"error": {"details": [{"retryDelay": "12.5s"}]}}],
    )
    assert _retry_delay(array_payload) == pytest.approx(12.5)


def test_error_array_preserves_google_message_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json=[
                {
                    "error": {
                        "code": 400,
                        "message": "Request contains an invalid argument.",
                        "status": "INVALID_ARGUMENT",
                        "details": [
                            {"reason": "INVALID_ARGUMENT"},
                            {
                                "fieldViolations": [
                                    {
                                        "field": "generation_config.max_output_tokens",
                                        "description": "must be greater than zero",
                                    }
                                ]
                            },
                        ],
                    }
                }
            ],
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "provider_rejected"
    text = str(failure.value)
    assert "INVALID_ARGUMENT" in text
    assert "Request contains an invalid argument." in text
    assert "generation_config.max_output_tokens" in text


def test_unstructured_rejection_includes_http_and_google_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"code": 403, "status": "PERMISSION_DENIED"}},
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "provider_rejected"
    assert "403" in str(failure.value)
    assert "PERMISSION_DENIED" in str(failure.value)


def test_provider_error_does_not_echo_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key secret-key"}})

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "invalid_credentials"
    assert "secret-key" not in str(failure.value)


def test_list_models_filters_text_models_and_follows_pages() -> None:
    pages: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("pageToken")
        pages.append(token)
        if token is None:
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3.8-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/text-embedding-004",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                        {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]},
                    ],
                    "nextPageToken": "next-page",
                },
            )
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.5-flash-lite",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )

    models = asyncio.run(
        GeminiClient("secret-key", transport=httpx.MockTransport(handler)).list_models()
    )

    assert models == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    assert pages == [None, "next-page"]


def test_count_tokens_uses_the_same_instructions_and_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"totalTokens": 42})

    count = asyncio.run(
        GeminiClient("secret-key", transport=httpx.MockTransport(handler)).count_tokens(
            model="gemini-3.8-flash",
            instructions="Use only evidence.",
            input_text="Question",
            timeout_seconds=5,
        )
    )

    assert count == 42
    assert seen["path"] == "/v1beta/models/gemini-3.8-flash:countTokens"
    assert "Use only evidence." in seen["body"]["contents"][0]["parts"][0]["text"]
    assert "Question" in seen["body"]["contents"][0]["parts"][0]["text"]


def test_empty_model_output_is_invalid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "empty", "steps": []})

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))
    assert failure.value.code == "invalid_response"


def test_cancellation_is_not_a_completed_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    with pytest.raises(ProviderCancelled):
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))
