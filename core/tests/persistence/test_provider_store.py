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


def make_usage(**changes) -> ModelCallUsage:
    usage = ModelCallUsage(
        id=new_id(),
        request_id="local-request-1",
        provider_request_id="provider-request-1",
        parent_request_id=None,
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
    return replace(usage, **changes)


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
    usage = make_usage()

    assert store.record_model_call_usage(usage) == usage
    assert store.list_model_call_usage() == (usage,)
    assert store.list_model_call_usage(1) == (usage,)

    with pytest.raises(ValueError):
        store.list_model_call_usage(0)


def test_model_call_usage_requires_a_configured_provider(store: PersistenceStore) -> None:
    usage = make_usage(
        request_id="local-request-2",
        provider_request_id=None,
        call_type="settings_test",
        input_tokens=None,
        output_tokens=None,
        thought_tokens=None,
        cached_tokens=None,
        total_tokens=None,
        outcome="failed",
        error_code="invalid_credentials",
    )

    with pytest.raises(PersistenceError):
        store.record_model_call_usage(usage)


def test_pending_usage_can_be_finalized_exactly_once(store: PersistenceStore) -> None:
    store.save_provider_settings(make_settings())
    usage = make_usage(
        provider_request_id=None,
        input_tokens=None,
        output_tokens=None,
        thought_tokens=None,
        cached_tokens=None,
        total_tokens=None,
        outcome="pending",
        error_code=None,
        finished_at=None,
    )
    stored = store.record_model_call_usage(usage)
    assert stored.outcome == "pending"
    assert stored.finished_at is None

    finished = store.finalize_model_call_usage(
        usage.id,
        outcome="completed",
        error_code=None,
        provider_request_id="provider-finished",
        input_tokens=10,
        output_tokens=2,
        thought_tokens=1,
        cached_tokens=0,
        total_tokens=13,
        finished_at=utc_timestamp(),
    )
    assert finished.outcome == "completed"
    assert finished.provider_request_id == "provider-finished"
    assert (finished.input_tokens, finished.output_tokens) == (10, 2)
    assert finished.finished_at is not None

    with pytest.raises(PersistenceError):
        store.finalize_model_call_usage(
            usage.id,
            outcome="failed",
            error_code="late_failure",
            provider_request_id=None,
            input_tokens=None,
            output_tokens=None,
            thought_tokens=None,
            cached_tokens=None,
            total_tokens=None,
            finished_at=utc_timestamp(),
        )


def test_recovery_marks_pending_calls_unknown_once(store: PersistenceStore) -> None:
    store.save_provider_settings(make_settings())
    pending = make_usage(
        provider_request_id=None,
        input_tokens=None,
        output_tokens=None,
        thought_tokens=None,
        cached_tokens=None,
        total_tokens=None,
        outcome="pending",
        error_code=None,
        finished_at=None,
    )
    store.record_model_call_usage(pending)

    assert store.recover_pending_model_calls() == 1
    recovered = store.list_model_call_usage()[0]
    assert recovered.outcome == "unknown"
    assert recovered.error_code == "interrupted_before_completion"
    assert recovered.finished_at is not None
    assert store.recover_pending_model_calls() == 0


def test_provider_model_catalog_replaces_and_lists(store: PersistenceStore) -> None:
    store.save_provider_settings(make_settings())

    assert store.replace_provider_model_catalog(
        "google-gemini",
        ["gemini-9-experimental", "gemini-3.8-flash", "gemini-9-experimental"],
    ) == ("gemini-3.8-flash", "gemini-9-experimental")
    assert store.list_provider_model_catalog("google-gemini") == (
        "gemini-3.8-flash",
        "gemini-9-experimental",
    )

    assert store.replace_provider_model_catalog(
        "google-gemini",
        ["gemini-10-experimental"],
    ) == ("gemini-10-experimental",)
    assert store.list_provider_model_catalog("google-gemini") == ("gemini-10-experimental",)


def test_provider_model_catalog_can_be_recorded_before_settings(
    store: PersistenceStore,
) -> None:
    assert store.replace_provider_model_catalog(
        "google-gemini",
        ["gemini-9-experimental"],
    ) == ("gemini-9-experimental",)
    assert store.list_provider_model_catalog("google-gemini") == ("gemini-9-experimental",)
