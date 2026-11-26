import asyncio
import contextlib
import json
import time
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
        "deadline_seconds": 5,
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
    assert result.local_request_id == "local-request"
    assert result.provider_request_id == "interaction-1"
    assert (result.input_tokens, result.output_tokens, result.thought_tokens) == (120, 30, 4)
    assert (result.cached_tokens, result.total_tokens) == (8, 162)


def test_current_model_output_ignores_input_and_earlier_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "interaction-current",
                "status": "completed",
                "output_text": "leaked input or earlier output",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "earlier answer"}],
                    },
                    {
                        "type": "user_input",
                        "content": [{"type": "text", "text": "private question"}],
                    },
                    {
                        "type": "thought",
                        "content": [{"type": "text", "text": "internal thought"}],
                    },
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": '{"answer":"current"}'}],
                    },
                ],
            },
        )

    result = call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert result.output_text == '{"answer":"current"}'


def test_multiple_current_model_output_steps_join_in_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "interaction-multiple",
                "status": "completed",
                "steps": [
                    {
                        "type": "user_input",
                        "content": [{"type": "text", "text": "private question"}],
                    },
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "first "}],
                    },
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "second"}],
                    },
                ],
            },
        )

    result = call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert result.output_text == "first second"


def test_response_without_current_model_text_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "no-current-answer",
                "status": "completed",
                "output_text": "leaked input or earlier output",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "earlier answer"}],
                    },
                    {
                        "type": "user_input",
                        "content": [{"type": "text", "text": "private question"}],
                    },
                    {
                        "type": "thought",
                        "content": [{"type": "text", "text": "internal thought"}],
                    },
                ],
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "invalid_response"


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


def test_incomplete_response_retains_reported_identity_and_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "provider-partial",
                "status": "incomplete",
                "output_text": "truncated",
                "usage": {
                    "total_input_tokens": 77,
                    "total_output_tokens": 9,
                    "total_thought_tokens": 3,
                    "total_cached_tokens": 2,
                    "total_tokens": 91,
                },
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "incomplete_response"
    assert failure.value.provider_request_id == "provider-partial"
    assert failure.value.input_tokens == 77
    assert failure.value.output_tokens == 9
    assert failure.value.thought_tokens == 3
    assert failure.value.cached_tokens == 2
    assert failure.value.total_tokens == 91


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


def test_error_array_maps_to_controlled_text_without_provider_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json=[
                {
                    "error": {
                        "code": 400,
                        "message": (
                            "Request contains secret-key and private source text: "
                            "the patient has a rare diagnosis."
                        ),
                        "status": "INVALID_ARGUMENT",
                        "details": [
                            {"reason": "INVALID_ARGUMENT"},
                            {
                                "fieldViolations": [
                                    {
                                        "field": "source.secret-key",
                                        "description": "private source text is invalid",
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
    assert "check the selected model and request settings" in text.lower()
    assert "INVALID_ARGUMENT" not in text
    assert "secret-key" not in text
    assert "private source text" not in text
    assert "generation_config" not in text


def test_unstructured_rejection_is_controlled_and_does_not_echo_the_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            content=(
                b"private source text and secret-key appeared in the provider body "
                b"PERMISSION_DENIED"
            ),
            headers={"content-type": "text/plain"},
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "provider_rejected"
    text = str(failure.value)
    assert "denied access" in text.lower()
    assert "private source text" not in text
    assert "secret-key" not in text
    assert "PERMISSION_DENIED" not in text


def test_provider_error_does_not_echo_the_key_or_source_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "bad key secret-key for source private source text",
                    "details": [{"violations": [{"quotaId": "secret-key-private-source-text"}]}],
                }
            },
        )

    with pytest.raises(ProviderFailure) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.code == "invalid_credentials"
    assert "secret-key" not in str(failure.value)
    assert "private source text" not in str(failure.value)


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


def test_total_deadline_cancels_a_slow_response() -> None:
    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(1)
            return httpx.Response(200, json={"id": "late", "output_text": "late"})

    started = time.monotonic()
    with pytest.raises(PaidCallOutcomeUnknown) as failure:
        call(
            GeminiClient("secret-key", transport=SlowTransport()),
            timeout_seconds=5,
            attempts=1,
            deadline_seconds=0.05,
        )

    assert time.monotonic() - started < 0.5
    assert "total time limit" in str(failure.value)


def test_total_deadline_bounds_a_trickling_http_response(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    body = json.dumps(
        {
            "id": "trickle",
            "output_text": "late" * 80,
            "usage": {},
        }
    ).encode()

    async def main() -> None:
        async def serve(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                )
                await writer.drain()
                for index in range(0, len(body), 10):
                    writer.write(body[index : index + 10])
                    await writer.drain()
                    await asyncio.sleep(0.1)
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        sockets = server.sockets
        assert sockets is not None
        port = sockets[0].getsockname()[1]
        started = time.monotonic()
        try:
            with pytest.raises(PaidCallOutcomeUnknown) as failure:
                await GeminiClient(
                    "secret-key",
                    base_url=f"http://127.0.0.1:{port}",
                ).generate(
                    model="gemini-3.8-flash",
                    instructions="Use only evidence.",
                    input_text="Question",
                    schema=None,
                    max_output_tokens=10,
                    timeout_seconds=5,
                    attempts=1,
                    deadline_seconds=1.0,
                    request_id="local-request",
                )
        finally:
            server.close()
            await server.wait_closed()

        assert time.monotonic() - started < 2.0
        assert "total time limit" in str(failure.value)

    asyncio.run(main())


def test_slow_client_close_is_bounded_and_preserves_result() -> None:
    class SlowCloseTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "slow-close", "output_text": "answer"})

        async def aclose(self) -> None:
            await asyncio.sleep(3)

    started = time.monotonic()
    result = call(
        GeminiClient("secret-key", transport=SlowCloseTransport()),
        timeout_seconds=5,
        attempts=1,
        deadline_seconds=0.5,
    )

    assert result.output_text == "answer"
    assert time.monotonic() - started < 1.0


def test_cancelled_client_close_preserves_known_result() -> None:
    class CancelCloseTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "known", "output_text": "answer"})

        async def aclose(self) -> None:
            raise asyncio.CancelledError

    result = call(
        GeminiClient("secret-key", transport=CancelCloseTransport()),
        timeout_seconds=5,
        attempts=1,
        deadline_seconds=2,
    )

    assert result.output_text == "answer"


def test_cancelled_client_close_after_no_reply_is_possibly_processed() -> None:
    class CancelRequestAndCloseTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise asyncio.CancelledError

        async def aclose(self) -> None:
            raise asyncio.CancelledError

    with pytest.raises(ProviderCancelled) as failure:
        call(
            GeminiClient("secret-key", transport=CancelRequestAndCloseTransport()),
            timeout_seconds=5,
            attempts=1,
            deadline_seconds=2,
        )

    assert failure.value.possibly_processed is True


def test_model_discovery_deadline_covers_pagination() -> None:
    calls = 0

    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "models/gemini-3.8-flash",
                                "supportedGenerationMethods": ["generateContent"],
                            }
                        ],
                        "nextPageToken": "next-page",
                    },
                )
            await asyncio.sleep(0.2)
            return httpx.Response(200, json={"models": []})

    with pytest.raises(PaidCallOutcomeUnknown) as failure:
        asyncio.run(
            GeminiClient("secret-key", transport=SlowTransport()).list_models(
                timeout_seconds=1,
                deadline_seconds=0.1,
            )
        )

    assert "total time limit" in str(failure.value)
    assert calls == 2


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


def test_cancellation_during_a_request_is_possibly_processed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    with pytest.raises(ProviderCancelled) as failure:
        call(GeminiClient("secret-key", transport=httpx.MockTransport(handler)))

    assert failure.value.possibly_processed is True
    assert failure.value.code == "cancelled_outcome_unknown"


def test_cancellation_during_retry_wait_is_known_unsent(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"message": "slow down"}},
        )

    async def cancel(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancel)
    with pytest.raises(ProviderCancelled) as failure:
        call(
            GeminiClient("secret-key", transport=httpx.MockTransport(handler)),
            attempts=2,
        )

    assert failure.value.possibly_processed is False
    assert failure.value.code == "cancelled"


def test_total_deadline_rejects_a_retry_wait_past_the_limit() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"retry-after": "3600"},
            json={"error": {"message": "slow down"}},
        )

    with pytest.raises(ProviderFailure) as failure:
        call(
            GeminiClient("secret-key", transport=httpx.MockTransport(handler)),
            timeout_seconds=5,
            attempts=3,
            deadline_seconds=2,
        )

    assert failure.value.code == "provider_timeout"
    assert calls == 1
    assert "total time limit" in str(failure.value)


def test_total_deadline_caps_the_http_timeout_budget() -> None:
    class TimeoutRecordingTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.timeout: dict[str, float] | None = None

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            timeout = request.extensions.get("timeout")
            self.timeout = timeout if isinstance(timeout, dict) else None
            return httpx.Response(200, json={"id": "ok", "output_text": "ok"})

    transport = TimeoutRecordingTransport()
    call(
        GeminiClient("secret-key", transport=transport),
        timeout_seconds=5,
        attempts=1,
        deadline_seconds=0.01,
    )

    assert transport.timeout is not None
    assert transport.timeout["read"] <= 0.01
    assert transport.timeout["connect"] <= 0.01
