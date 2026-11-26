import asyncio
import json
from typing import Any, cast

import httpx
import pytest

from core.jobs.runner import PaidCallOutcomeUnknown
from core.persistence.models import ModelCallUsage, ProviderSettings, new_id, utc_timestamp
from core.persistence.store import PersistenceStore
from core.providers.credentials import InMemoryCredentialStore
from core.providers.errors import ProviderCancelled, ProviderFailure
from core.providers.gemini import GeminiClient
from core.providers.protocol import ProviderCallResult
from core.providers.service import (
    DEFAULT_CHECKER_MODEL,
    DEFAULT_GENERATION_MODEL,
    SETTINGS_TEST_OUTPUT_TOKENS,
    ProviderService,
    ProviderSettingsUpdate,
)


def configured_service(
    store: PersistenceStore,
    handler,
    *,
    api_key: str = "test-key",
) -> ProviderService:
    credentials = InMemoryCredentialStore()
    service = ProviderService(
        store,
        credentials,
        client_factory=lambda key: GeminiClient(
            key,
            transport=httpx.MockTransport(handler),
        ),
    )
    service.save(
        ProviderSettingsUpdate(
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        ),
        api_key=api_key,
    )
    return service


def success_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "provider-result",
            "output_text": "Grounded answer.",
            "usage": {
                "total_input_tokens": 80,
                "total_output_tokens": 20,
                "total_tokens": 100,
            },
        },
    )


def test_settings_require_consent_and_never_return_the_secret(
    store: PersistenceStore,
) -> None:
    credentials = InMemoryCredentialStore()
    service = ProviderService(store, credentials)
    update = ProviderSettingsUpdate(
        generation_model=DEFAULT_GENERATION_MODEL,
        checker_model=DEFAULT_CHECKER_MODEL,
        free_tier_acknowledged=False,
    )

    with pytest.raises(ProviderFailure) as failure:
        service.save(update, api_key="secret")
    assert failure.value.code == "free_tier_acknowledgement_required"

    state = service.save(
        ProviderSettingsUpdate(
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        ),
        api_key="secret",
    )
    assert state["credential_configured"] is True
    assert "api_key" not in state
    assert "secret" not in json.dumps(state)
    assert credentials.get("google-gemini") == "secret"

    cleared = service.clear_credential()
    assert cleared["credential_configured"] is False
    assert credentials.get("google-gemini") is None


def test_call_policy_attempt_cap_is_used(store: PersistenceStore) -> None:
    credentials = InMemoryCredentialStore()
    credentials.set("google-gemini", "test-key")
    now = utc_timestamp()
    store.save_provider_settings(
        ProviderSettings(
            provider_id="google-gemini",
            settings_format_version=1,
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
            output_token_allowance=1024,
            safety_margin_tokens=512,
            call_policies={"generation": {"timeout_seconds": 1, "attempts": 1}},
            created_at=now,
            updated_at=now,
        )
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "temporary"}})

    service = ProviderService(
        store,
        credentials,
        client_factory=lambda key: GeminiClient(
            key,
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(ProviderFailure) as failure:
        asyncio.run(
            service.complete(
                "generation",
                instructions="Use evidence.",
                input_text="Question",
            )
        )

    assert failure.value.code == "provider_unavailable"
    assert calls == 1


def test_model_change_keeps_the_key_until_explicitly_removed(
    store: PersistenceStore,
) -> None:
    service = configured_service(store, success_handler)

    changed = service.save(
        ProviderSettingsUpdate(
            generation_model="gemini-4-hydrogen",
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        )
    )
    assert changed["credential_configured"] is True
    assert changed["generation_model"] == "gemini-4-hydrogen"

    cleared = service.clear_credential()
    assert cleared["credential_configured"] is False
    assert cleared["generation_model"] == "gemini-4-hydrogen"


def test_available_models_uses_the_provider_catalog(store: PersistenceStore) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.8-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-4-hydrogen",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            },
        )

    service = configured_service(store, handler)
    catalog = asyncio.run(service.available_models())

    assert catalog["discovered"] is True
    assert catalog["warning"] is None
    assert catalog["generation_models"] == [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
    ]
    assert catalog["checker_models"] == [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
    ]


def test_generation_uses_the_generation_model_and_records_usage(
    store: PersistenceStore,
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return success_handler(request)

    service = configured_service(store, handler)
    result = asyncio.run(
        service.complete(
            "generation",
            instructions="Use evidence.",
            input_text="What does the paper say?",
        )
    )

    assert result.output_text == "Grounded answer."
    assert result.provider_request_id == "provider-result"
    assert seen["body"]["model"] == DEFAULT_GENERATION_MODEL
    usage = service.usage()
    assert len(usage) == 1
    assert usage[0].call_type == "generation"
    assert usage[0].model == DEFAULT_GENERATION_MODEL
    assert usage[0].outcome == "completed"
    assert usage[0].request_id == result.local_request_id
    assert usage[0].provider_request_id == "provider-result"
    assert usage[0].parent_request_id is None
    assert (usage[0].input_tokens, usage[0].output_tokens) == (80, 20)


def test_pending_usage_is_recorded_before_the_request_is_sent(
    store: PersistenceStore,
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        rows = store.list_model_call_usage()
        seen["during_call"] = rows
        return success_handler(request)

    service = configured_service(store, handler)
    asyncio.run(
        service.complete(
            "generation",
            instructions="Use evidence.",
            input_text="Question",
        )
    )

    during_call = seen["during_call"]
    assert len(during_call) == 1
    assert during_call[0].outcome == "pending"
    assert during_call[0].finished_at is None
    assert during_call[0].provider_request_id is None
    assert service.usage()[0].outcome == "completed"


def test_restart_recovers_an_interrupted_pending_call_without_replay(
    store: PersistenceStore,
) -> None:
    credentials = InMemoryCredentialStore()
    credentials.set("google-gemini", "test-key")
    now = utc_timestamp()
    store.save_provider_settings(
        ProviderSettings(
            provider_id="google-gemini",
            settings_format_version=1,
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
            output_token_allowance=1024,
            safety_margin_tokens=512,
            call_policies={},
            created_at=now,
            updated_at=now,
        )
    )
    request_id = new_id()
    store.record_model_call_usage(
        ModelCallUsage(
            id=new_id(),
            request_id=request_id,
            provider_request_id=None,
            parent_request_id=None,
            call_type="generation",
            provider_id="google-gemini",
            model=DEFAULT_GENERATION_MODEL,
            input_tokens=None,
            output_tokens=None,
            thought_tokens=None,
            cached_tokens=None,
            total_tokens=None,
            outcome="pending",
            error_code=None,
            started_at=now,
            finished_at=None,
            created_at=now,
        )
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success_handler(request)

    service = ProviderService(
        store,
        credentials,
        client_factory=lambda key: GeminiClient(
            key,
            transport=httpx.MockTransport(handler),
        ),
    )

    usage = service.usage()
    assert calls == 0
    assert usage[0].request_id == request_id
    assert usage[0].outcome == "unknown"
    assert usage[0].error_code == "interrupted_before_completion"
    assert usage[0].finished_at is not None


def test_cancel_after_send_is_recorded_as_unknown_not_cancelled(
    store: PersistenceStore,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    service = configured_service(store, handler)
    with pytest.raises(ProviderCancelled) as failure:
        asyncio.run(
            service.complete(
                "generation",
                instructions="Use evidence.",
                input_text="Question",
            )
        )

    assert failure.value.possibly_processed is True
    usage = service.usage()
    assert usage[0].outcome == "unknown"
    assert usage[0].error_code == "cancelled_outcome_unknown"


def test_cancel_before_retry_is_recorded_as_cancelled(
    store: PersistenceStore,
    monkeypatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"message": "slow down"}},
        )

    async def cancel(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancel)
    service = configured_service(store, handler)
    with pytest.raises(ProviderCancelled) as failure:
        asyncio.run(
            service.complete(
                "generation",
                instructions="Use evidence.",
                input_text="Question",
            )
        )

    assert failure.value.possibly_processed is False
    usage = service.usage()
    assert usage[0].outcome == "cancelled"
    assert usage[0].error_code == "cancelled"


def test_incomplete_result_retains_reported_usage_without_marking_success(
    store: PersistenceStore,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "provider-partial",
                "status": "incomplete",
                "output_text": "truncated",
                "usage": {
                    "total_input_tokens": 70,
                    "total_output_tokens": 8,
                    "total_thought_tokens": 2,
                    "total_cached_tokens": 1,
                    "total_tokens": 80,
                },
            },
        )

    service = configured_service(store, handler)
    with pytest.raises(ProviderFailure) as failure:
        asyncio.run(
            service.complete(
                "generation",
                instructions="Use evidence.",
                input_text="Question",
            )
        )

    assert failure.value.code == "incomplete_response"
    usage = service.usage()
    assert usage[0].outcome == "failed"
    assert usage[0].error_code == "incomplete_response"
    assert usage[0].provider_request_id == "provider-partial"
    assert usage[0].input_tokens == 70
    assert usage[0].output_tokens == 8
    assert usage[0].thought_tokens == 2
    assert usage[0].cached_tokens == 1
    assert usage[0].total_tokens == 80


def test_discovered_models_can_be_selected_saved_and_tested(
    store: PersistenceStore,
) -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1beta/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-9-experimental",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-3.8-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                },
            )
        seen_models.append(json.loads(request.content)["model"])
        return success_handler(request)

    service = configured_service(store, handler)
    with pytest.raises(ProviderFailure) as failure:
        service.save(
            ProviderSettingsUpdate(
                generation_model="gemini-9-experimental",
                checker_model=DEFAULT_CHECKER_MODEL,
                free_tier_acknowledged=True,
            )
        )
    assert failure.value.code == "unsupported_model"

    catalog = asyncio.run(service.available_models())
    assert "gemini-9-experimental" in catalog["generation_models"]
    assert "gemini-9-experimental" in service.state()["generation_models"]

    saved = service.save(
        ProviderSettingsUpdate(
            generation_model="gemini-9-experimental",
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        )
    )
    assert saved["generation_model"] == "gemini-9-experimental"

    result = asyncio.run(service.test_connection())
    assert result["model"] == "gemini-9-experimental"
    assert result["provider_request_id"] == "provider-result"
    assert seen_models == ["gemini-9-experimental"]


class RecordingClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def generate(self, **kwargs: Any) -> ProviderCallResult:
        self.kwargs = kwargs
        return ProviderCallResult(
            output_text="recorded",
            local_request_id=kwargs["request_id"],
            provider_request_id="provider-recorded",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
        )


def test_discovery_before_first_save_persists_models_for_validation(
    store: PersistenceStore,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1beta/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-9-experimental",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            )
        return success_handler(request)

    credentials = InMemoryCredentialStore()
    credentials.set("google-gemini", "test-key")
    service = ProviderService(
        store,
        credentials,
        client_factory=lambda key: GeminiClient(
            key,
            transport=httpx.MockTransport(handler),
        ),
    )
    assert service.state()["credential_configured"] is True
    assert store.load_provider_settings("google-gemini") is None

    catalog = asyncio.run(service.available_models())
    assert "gemini-9-experimental" in catalog["generation_models"]

    saved = service.save(
        ProviderSettingsUpdate(
            generation_model="gemini-9-experimental",
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        )
    )
    assert saved["generation_model"] == "gemini-9-experimental"


def test_service_passes_the_total_deadline_to_the_client(
    store: PersistenceStore,
) -> None:
    recording = RecordingClient()
    credentials = InMemoryCredentialStore()
    service = ProviderService(
        store,
        credentials,
        client_factory=lambda key: cast(GeminiClient, recording),
    )
    service.save(
        ProviderSettingsUpdate(
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        ),
        api_key="test-key",
    )

    result = asyncio.run(
        service.complete(
            "generation",
            instructions="Use evidence.",
            input_text="Question",
        )
    )

    assert result.output_text == "recorded"
    assert recording.kwargs["deadline_seconds"] == 390.0
    assert recording.kwargs["request_id"] == result.local_request_id


def test_support_check_uses_the_checker_model(store: PersistenceStore) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return success_handler(request)

    service = configured_service(store, handler)
    asyncio.run(
        service.complete(
            "support_check",
            instructions="Check the claim.",
            input_text="The claim and passage.",
        )
    )

    assert seen["body"]["model"] == DEFAULT_CHECKER_MODEL


def test_unknown_paid_outcome_is_recorded_and_not_replayed(
    store: PersistenceStore,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    service = configured_service(store, handler)
    with pytest.raises(PaidCallOutcomeUnknown):
        asyncio.run(
            service.complete(
                "generation",
                instructions="Use evidence.",
                input_text="Question",
            )
        )

    assert calls == 1
    usage = service.usage()
    assert usage[0].outcome == "unknown"
    assert usage[0].error_code == "paid_outcome_unknown"


def test_missing_credential_fails_before_a_provider_call(store: PersistenceStore) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("provider should not be called")

    service = ProviderService(
        store,
        InMemoryCredentialStore(),
        client_factory=lambda key: GeminiClient(
            key,
            transport=httpx.MockTransport(handler),
        ),
    )
    service.save(
        ProviderSettingsUpdate(
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=True,
        )
    )

    with pytest.raises(ProviderFailure) as failure:
        asyncio.run(
            service.complete(
                "generation",
                instructions="Use evidence.",
                input_text="Question",
            )
        )
    assert failure.value.code == "credential_missing"
    assert service.usage() == ()


def test_test_connection_uses_a_model_safe_probe_budget(store: PersistenceStore) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return success_handler(request)

    service = configured_service(store, handler)
    result = asyncio.run(service.test_connection())

    assert result["ok"] is True
    assert result["model"] == DEFAULT_GENERATION_MODEL
    assert result["request_id"] != "provider-result"
    assert result["provider_request_id"] == "provider-result"
    assert seen["body"]["generation_config"]["max_output_tokens"] == (SETTINGS_TEST_OUTPUT_TOKENS)
    usage = service.usage()
    assert usage[0].call_type == "settings_test"
    assert usage[0].outcome == "completed"
    assert usage[0].request_id == result["request_id"]
    assert usage[0].provider_request_id == result["provider_request_id"]


def test_count_tokens_uses_generation_model_and_requires_a_key(
    store: PersistenceStore,
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"totalTokens": 12})

    service = configured_service(store, handler)
    count = asyncio.run(
        service.count_input_tokens(
            instructions="Use evidence.",
            input_text="Question",
        )
    )

    assert count == 12 + 512
    assert seen["path"] == f"/v1beta/models/{DEFAULT_GENERATION_MODEL}:countTokens"
