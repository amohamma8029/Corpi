"""Authorized provider service: settings, credentials, calls, and usage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.jobs.runner import PaidCallOutcomeUnknown
from core.persistence.models import (
    JSONValue,
    ModelCallUsage,
    ProviderSettings,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore
from core.providers.credentials import CredentialStore
from core.providers.errors import ProviderCancelled, ProviderFailure
from core.providers.gemini import GeminiClient
from core.providers.protocol import JSONSchema, ProviderCallResult

PROVIDER_ID = "google-gemini"
PROVIDER_LABEL = "Google Gemini"
SETTINGS_FORMAT_VERSION = 1

DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
DEFAULT_CHECKER_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
)
GENERATION_MODELS = FALLBACK_MODELS
CHECKER_MODELS = FALLBACK_MODELS

DEFAULT_OUTPUT_TOKENS = 1024
DEFAULT_SAFETY_MARGIN = 512
SETTINGS_TEST_OUTPUT_TOKENS = 256
DEFAULT_POLICIES: dict[str, JSONValue] = {
    "generation": {
        "timeout_seconds": 180,
        "attempts": 2,
        "total_timeout_seconds": 390,
    },
    "support_check": {
        "timeout_seconds": 90,
        "attempts": 2,
        "total_timeout_seconds": 210,
    },
    "summary": {
        "timeout_seconds": 90,
        "attempts": 2,
        "total_timeout_seconds": 210,
    },
    "format_repair": {
        "timeout_seconds": 90,
        "attempts": 1,
        "total_timeout_seconds": 90,
    },
    "citation_retry": {
        "timeout_seconds": 180,
        "attempts": 1,
        "total_timeout_seconds": 180,
    },
    "settings_test": {
        "timeout_seconds": 30,
        "attempts": 1,
        "total_timeout_seconds": 30,
    },
}
CALL_TYPES = frozenset(DEFAULT_POLICIES)


@dataclass(frozen=True, slots=True)
class ProviderSettingsUpdate:
    generation_model: str
    checker_model: str
    free_tier_acknowledged: bool
    output_token_allowance: int = DEFAULT_OUTPUT_TOKENS
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN


class ProviderService:
    """The only component that reads a provider credential."""

    def __init__(
        self,
        store: PersistenceStore,
        credentials: CredentialStore,
        *,
        client_factory: Callable[[str], GeminiClient] = GeminiClient,
    ) -> None:
        self._store = store
        self._credentials = credentials
        self._client_factory = client_factory
        # A previous process may have stopped after recording a call but before
        # finalizing it. It is never safe to replay those calls automatically.
        self._store.recover_pending_model_calls()

    def usage(self, limit: int = 20) -> tuple[ModelCallUsage, ...]:
        return self._store.list_model_call_usage(limit)

    async def available_models(self) -> dict[str, Any]:
        settings = self._load_or_default()
        fallback = {
            "generation_models": list(self._catalog(settings)),
            "checker_models": list(self._catalog(settings)),
            "discovered": False,
            "warning": None,
        }
        api_key = self._credential()
        if api_key is None:
            return fallback
        client = self._client_factory(api_key)
        try:
            discovered = await client.list_models(
                timeout_seconds=30,
                attempts=1,
                deadline_seconds=30,
            )
        except (ProviderFailure, PaidCallOutcomeUnknown) as error:
            fallback["warning"] = str(error)
            return fallback
        if not discovered:
            fallback["warning"] = "The provider returned no text-generation models."
            return fallback
        self._store.replace_provider_model_catalog(PROVIDER_ID, discovered)
        catalog = list(self._catalog(settings))
        return {
            "generation_models": catalog,
            "checker_models": catalog,
            "discovered": True,
            "warning": None,
        }

    def state(self) -> dict[str, Any]:
        settings = self._load_or_default()
        catalog = list(self._catalog(settings))
        return {
            "provider_id": settings.provider_id,
            "provider_label": PROVIDER_LABEL,
            "generation_model": settings.generation_model,
            "checker_model": settings.checker_model,
            "free_tier_acknowledged": settings.free_tier_acknowledged,
            "output_token_allowance": settings.output_token_allowance,
            "safety_margin_tokens": settings.safety_margin_tokens,
            "call_policies": settings.call_policies,
            "credential_configured": self._credential() is not None,
            "generation_models": catalog,
            "checker_models": catalog,
        }

    def save(
        self,
        update: ProviderSettingsUpdate,
        *,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not update.free_tier_acknowledged:
            raise ProviderFailure(
                "free_tier_acknowledgement_required",
                "Confirm the provider's free-tier data-use notice before saving settings.",
                status=400,
            )
        existing = self._store.load_provider_settings(PROVIDER_ID)
        current = existing if existing is not None else self._load_or_default()
        catalog = self._catalog(current)
        if update.generation_model not in catalog:
            raise ProviderFailure(
                "unsupported_model",
                "Choose a supported generation model.",
                status=400,
            )
        if update.checker_model not in catalog:
            raise ProviderFailure(
                "unsupported_model",
                "Choose a supported claim-checker model.",
                status=400,
            )
        if not 1 <= update.output_token_allowance <= 1_000_000:
            raise ProviderFailure(
                "invalid_setting",
                "Output token allowance is outside the supported range.",
                status=400,
            )
        if not 0 <= update.safety_margin_tokens <= 1_000_000:
            raise ProviderFailure(
                "invalid_setting",
                "Safety margin is outside the supported range.",
                status=400,
            )
        timestamp = utc_timestamp()
        if api_key is not None:
            cleaned = api_key.strip()
            if not cleaned or len(cleaned) > 4096:
                raise ProviderFailure(
                    "invalid_credential",
                    "Enter a valid API key.",
                    status=400,
                )
            self._credentials.set(PROVIDER_ID, cleaned)
        settings = ProviderSettings(
            provider_id=PROVIDER_ID,
            settings_format_version=SETTINGS_FORMAT_VERSION,
            generation_model=update.generation_model,
            checker_model=update.checker_model,
            free_tier_acknowledged=True,
            output_token_allowance=update.output_token_allowance,
            safety_margin_tokens=update.safety_margin_tokens,
            call_policies=(
                existing.call_policies if existing is not None else dict(DEFAULT_POLICIES)
            ),
            created_at=timestamp if existing is None else existing.created_at,
            updated_at=timestamp,
        )
        self._store.save_provider_settings(settings)
        return self.state()

    def clear_credential(self) -> dict[str, Any]:
        self._credentials.delete(PROVIDER_ID)
        return self.state()

    async def test_connection(self) -> dict[str, Any]:
        result = await self.complete(
            "settings_test",
            instructions="Return a short confirmation that the provider is reachable.",
            input_text="Reply with exactly OK.",
            max_output_tokens=SETTINGS_TEST_OUTPUT_TOKENS,
        )
        return {
            "ok": True,
            "provider_id": PROVIDER_ID,
            "model": self._model_for("settings_test"),
            "request_id": result.local_request_id,
            "provider_request_id": result.provider_request_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
        }

    async def complete(
        self,
        call_type: str,
        *,
        instructions: str,
        input_text: str,
        schema: JSONSchema | None = None,
        max_output_tokens: int | None = None,
        parent_request_id: str | None = None,
    ) -> ProviderCallResult:
        if call_type not in CALL_TYPES:
            raise ValueError("unsupported model call type")
        settings = self._load_or_default()
        if not settings.free_tier_acknowledged:
            raise ProviderFailure(
                "free_tier_acknowledgement_required",
                "Confirm the provider's free-tier data-use notice before using it.",
                status=409,
            )
        api_key = self._credential()
        if api_key is None:
            raise ProviderFailure(
                "credential_missing",
                "Configure a provider API key first.",
                status=409,
            )
        if parent_request_id is not None:
            parent_request_id = parent_request_id.strip()
            if not parent_request_id or len(parent_request_id) > 200:
                raise ValueError("parent request id is invalid")
        model = self._model_for(call_type)
        timeout_seconds, attempts, deadline_seconds = self._policy(settings, call_type)
        output_limit = max_output_tokens or self._default_output(call_type, settings)
        request_id = new_id()
        usage_id = new_id()
        started_at = utc_timestamp()
        self._store.record_model_call_usage(
            ModelCallUsage(
                id=usage_id,
                request_id=request_id,
                provider_request_id=None,
                parent_request_id=parent_request_id,
                call_type=call_type,
                provider_id=PROVIDER_ID,
                model=model,
                input_tokens=None,
                output_tokens=None,
                thought_tokens=None,
                cached_tokens=None,
                total_tokens=None,
                outcome="pending",
                error_code=None,
                started_at=started_at,
                finished_at=None,
                created_at=started_at,
            )
        )
        client = self._client_factory(api_key)
        try:
            result = await client.generate(
                model=model,
                instructions=instructions,
                input_text=input_text,
                schema=schema,
                max_output_tokens=output_limit,
                timeout_seconds=timeout_seconds,
                attempts=attempts,
                deadline_seconds=deadline_seconds,
                request_id=request_id,
            )
        except ProviderCancelled as error:
            self._finalize_usage(
                usage_id,
                outcome="unknown" if error.possibly_processed else "cancelled",
                error_code=error.code,
            )
            raise
        except PaidCallOutcomeUnknown:
            self._finalize_usage(
                usage_id,
                outcome="unknown",
                error_code="paid_outcome_unknown",
            )
            raise
        except ProviderFailure as error:
            self._finalize_usage(
                usage_id,
                outcome="failed",
                error_code=error.code,
                details=error,
            )
            raise
        except Exception:
            self._finalize_usage(
                usage_id,
                outcome="unknown",
                error_code="call_interrupted",
            )
            raise
        self._finalize_usage(
            usage_id,
            outcome="completed",
            error_code=None,
            details=result,
        )
        return result

    async def count_input_tokens(
        self,
        *,
        instructions: str,
        input_text: str,
        model: str | None = None,
    ) -> int:
        settings = self._load_or_default()
        if not settings.free_tier_acknowledged:
            raise ProviderFailure(
                "free_tier_acknowledgement_required",
                "Confirm the provider's free-tier data-use notice before using it.",
                status=409,
            )
        api_key = self._credential()
        if api_key is None:
            raise ProviderFailure(
                "credential_missing",
                "Configure a provider API key first.",
                status=409,
            )
        timeout_seconds, _, deadline_seconds = self._policy(settings, "settings_test")
        client = self._client_factory(api_key)
        count = await client.count_tokens(
            model=model or settings.generation_model,
            instructions=instructions,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
            attempts=1,
            deadline_seconds=deadline_seconds,
        )
        return count + settings.safety_margin_tokens

    def _load_or_default(self) -> ProviderSettings:
        settings = self._store.load_provider_settings(PROVIDER_ID)
        if settings is not None:
            return settings
        timestamp = utc_timestamp()
        return ProviderSettings(
            provider_id=PROVIDER_ID,
            settings_format_version=SETTINGS_FORMAT_VERSION,
            generation_model=DEFAULT_GENERATION_MODEL,
            checker_model=DEFAULT_CHECKER_MODEL,
            free_tier_acknowledged=False,
            output_token_allowance=DEFAULT_OUTPUT_TOKENS,
            safety_margin_tokens=DEFAULT_SAFETY_MARGIN,
            call_policies=dict(DEFAULT_POLICIES),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _catalog(self, settings: ProviderSettings) -> tuple[str, ...]:
        discovered = self._store.list_provider_model_catalog(PROVIDER_ID)
        return tuple(
            sorted(
                set(FALLBACK_MODELS)
                | set(discovered)
                | {settings.generation_model, settings.checker_model}
            )
        )

    def _credential(self) -> str | None:
        return self._credentials.get(PROVIDER_ID)

    def _model_for(self, call_type: str) -> str:
        settings = self._load_or_default()
        if call_type in {"support_check", "summary"}:
            return settings.checker_model
        return settings.generation_model

    def _default_output(self, call_type: str, settings: ProviderSettings) -> int:
        if call_type in {"support_check", "summary"}:
            return min(512, settings.output_token_allowance)
        if call_type == "settings_test":
            return min(SETTINGS_TEST_OUTPUT_TOKENS, settings.output_token_allowance)
        return settings.output_token_allowance

    @staticmethod
    def _policy(settings: ProviderSettings, call_type: str) -> tuple[float, int, float]:
        raw = settings.call_policies.get(call_type)
        if isinstance(raw, dict):
            policy = raw
        else:
            fallback = DEFAULT_POLICIES[call_type]
            policy = fallback if isinstance(fallback, dict) else {}
        timeout = policy.get("timeout_seconds", 90)
        attempts = policy.get("attempts", 1)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 90
        if not isinstance(attempts, int) or attempts < 1:
            attempts = 1
        total = policy.get("total_timeout_seconds")
        if not isinstance(total, (int, float)) or total <= 0:
            total = timeout * attempts
        return float(timeout), min(attempts, 3), float(total)

    def _finalize_usage(
        self,
        usage_id: str,
        *,
        outcome: str,
        error_code: str | None,
        details: ProviderCallResult | ProviderFailure | None = None,
    ) -> None:
        self._store.finalize_model_call_usage(
            usage_id,
            outcome=outcome,
            error_code=error_code,
            provider_request_id=None if details is None else details.provider_request_id,
            input_tokens=None if details is None else details.input_tokens,
            output_tokens=None if details is None else details.output_tokens,
            thought_tokens=None if details is None else details.thought_tokens,
            cached_tokens=None if details is None else details.cached_tokens,
            total_tokens=None if details is None else details.total_tokens,
            finished_at=utc_timestamp(),
        )
