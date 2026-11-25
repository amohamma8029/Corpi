from dataclasses import replace

import pytest

from core.persistence.errors import PersistenceError
from core.persistence.models import (
    ModelCallUsage,
    ProviderSettings,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore


def make_settings(**changes) -> ProviderSettings:
    settings = ProviderSettings(
        provider_id="google-gemini",
        settings_format_version=1,
        generation_model="gemini-3.8-flash",
        checker_model="gemini-3.5-flash-lite",
        free_tier_acknowledged=True,
        output_token_allowance=1024,
        safety_margin_tokens=512,
        call_policies={
            "generation": {"timeout_seconds": 180, "attempts": 2},
        },
        created_at=utc_timestamp(),
        updated_at=utc_timestamp(),
    )
    return replace(settings, **changes)


def test_provider_settings_round_trip_and_update(store: PersistenceStore) -> None:
    assert store.load_provider_settings("google-gemini") is None

    saved = store.save_provider_settings(make_settings())
    assert saved.generation_model == "gemini-3.8-flash"
    assert saved.free_tier_acknowledged is True
    assert saved.call_policies == {"generation": {"timeout_seconds": 180, "attempts": 2}}

    updated = store.save_provider_settings(
        make_settings(
            generation_model="gemini-4-hydrogen",
            output_token_allowance=2048,
        )
    )
    loaded = store.load_provider_settings("google-gemini")

    assert loaded is not None
    assert loaded == updated
    assert loaded.created_at == saved.created_at
    assert loaded.generation_model == "gemini-4-hydrogen"
    assert loaded.output_token_allowance == 2048
    assert loaded.updated_at >= saved.updated_at


def test_model_call_usage_is_queryable_and_provider_scoped(store: PersistenceStore) -> None:
    store.save_provider_settings(make_settings())
    usage = ModelCallUsage(
        id=new_id(),
        request_id="provider-request-1",
        call_type="generation",
        provider_id="google-gemini",
        model="gemini-3.8-flash",
        input_tokens=120,
        output_tokens=30,
        thought_tokens=4,
        cached_tokens=None,
        total_tokens=154,
        outcome="completed",
        error_code=None,
        started_at=utc_timestamp(),
        finished_at=utc_timestamp(),
        created_at=utc_timestamp(),
    )

    assert store.record_model_call_usage(usage) == usage
    assert store.list_model_call_usage() == (usage,)
    assert store.list_model_call_usage(1) == (usage,)

    with pytest.raises(ValueError):
        store.list_model_call_usage(0)


def test_model_call_usage_requires_a_configured_provider(store: PersistenceStore) -> None:
    usage = ModelCallUsage(
        id=new_id(),
        request_id="provider-request-2",
        call_type="settings_test",
        provider_id="google-gemini",
        model="gemini-3.8-flash",
        input_tokens=None,
        output_tokens=None,
        thought_tokens=None,
        cached_tokens=None,
        total_tokens=None,
        outcome="failed",
        error_code="invalid_credentials",
        started_at=utc_timestamp(),
        finished_at=utc_timestamp(),
        created_at=utc_timestamp(),
    )

    with pytest.raises(PersistenceError):
        store.record_model_call_usage(usage)
