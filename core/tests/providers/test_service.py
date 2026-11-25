import asyncio
import json
from typing import Any

import httpx
import pytest

from core.jobs.runner import PaidCallOutcomeUnknown
from core.persistence.models import ProviderSettings, utc_timestamp
from core.persistence.store import PersistenceStore
from core.providers.credentials import InMemoryCredentialStore
from core.providers.errors import ProviderFailure
from core.providers.gemini import GeminiClient
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
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
    ]
    assert catalog["checker_models"] == [
        "gemini-3.5-flash-lite",
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
    assert seen["body"]["model"] == DEFAULT_GENERATION_MODEL
    usage = service.usage()
    assert len(usage) == 1
    assert usage[0].call_type == "generation"
    assert usage[0].model == DEFAULT_GENERATION_MODEL
    assert usage[0].outcome == "completed"
    assert (usage[0].input_tokens, usage[0].output_tokens) == (80, 20)


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
    assert seen["body"]["generation_config"]["max_output_tokens"] == (SETTINGS_TEST_OUTPUT_TOKENS)
    usage = service.usage()
    assert usage[0].call_type == "settings_test"
    assert usage[0].outcome == "completed"


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
