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
    "generation": {"timeout_seconds": 180, "attempts": 2},
    "support_check": {"timeout_seconds": 90, "attempts": 2},
    "summary": {"timeout_seconds": 90, "attempts": 2},
    "format_repair": {"timeout_seconds": 90, "attempts": 1},
    "citation_retry": {"timeout_seconds": 180, "attempts": 1},
    "settings_test": {"timeout_seconds": 30, "attempts": 1},
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

    def usage(self, limit: int = 20) -> tuple[ModelCallUsage, ...]:
        return self._store.list_model_call_usage(limit)

    async def available_models(self) -> dict[str, Any]:
        settings = self._load_or_default()
        fallback = {
            "generation_models": list(GENERATION_MODELS),
            "checker_models": list(CHECKER_MODELS),
            "discovered": False,
            "warning": None,
        }
        api_key = self._credential()
        if api_key is None:
            return fallback
        client = self._client_factory(api_key)
        try:
            discovered = await client.list_models(timeout_seconds=30, attempts=1)
        except (ProviderFailure, PaidCallOutcomeUnknown) as error:
            fallback["warning"] = str(error)
            return fallback
        if not discovered:
            fallback["warning"] = "The provider returned no text-generation models."
            return fallback
        models = sorted(set(discovered) | {settings.generation_model, settings.checker_model})
        return {
            "generation_models": models,
            "checker_models": models,
            "discovered": True,
            "warning": None,
        }

    def state(self) -> dict[str, Any]:
        settings = self._load_or_default()
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
            "generation_models": list(GENERATION_MODELS),
            "checker_models": list(CHECKER_MODELS),
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
        if update.generation_model not in GENERATION_MODELS:
            raise ProviderFailure(
                "unsupported_model",
                "Choose a supported generation model.",
                status=400,
            )
        if update.checker_model not in CHECKER_MODELS:
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
        existing = self._store.load_provider_settings(PROVIDER_ID)
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
            "request_id": result.request_id,
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
        model = self._model_for(call_type)
        policy = self._policy(settings, call_type)
        output_limit = max_output_tokens or self._default_output(call_type, settings)
        request_id = new_id()
        started_at = utc_timestamp()
        client = self._client_factory(api_key)
        try:
            result = await client.generate(
                model=model,
                instructions=instructions,
                input_text=input_text,
                schema=schema,
                max_output_tokens=output_limit,
                timeout_seconds=policy[0],
                attempts=policy[1],
                request_id=request_id,
            )
        except ProviderCancelled as error:
            self._record_usage(
                request_id,
                call_type,
                model,
                started_at,
                outcome="cancelled",
                error_code=error.code,
            )
            raise
        except PaidCallOutcomeUnknown:
            self._record_usage(
                request_id,
                call_type,
                model,
                started_at,
                outcome="unknown",
                error_code="paid_outcome_unknown",
            )
            raise
        except ProviderFailure as error:
            self._record_usage(
                request_id,
                call_type,
                model,
                started_at,
                outcome="failed",
                error_code=error.code,
            )
            raise
        self._record_usage(
            request_id,
            call_type,
            model,
            started_at,
            result=result,
            outcome="completed",
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
        policy = self._policy(settings, "settings_test")
        client = self._client_factory(api_key)
        count = await client.count_tokens(
            model=model or settings.generation_model,
            instructions=instructions,
            input_text=input_text,
            timeout_seconds=policy[0],
            attempts=1,
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
    def _policy(settings: ProviderSettings, call_type: str) -> tuple[float, int]:
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
        return float(timeout), min(attempts, 3)

    def _record_usage(
        self,
        request_id: str,
        call_type: str,
        model: str,
        started_at: str,
        *,
        result: ProviderCallResult | None = None,
        outcome: str,
        error_code: str | None = None,
    ) -> None:
        self._store.record_model_call_usage(
            ModelCallUsage(
                id=new_id(),
                request_id=request_id,
                call_type=call_type,
                provider_id=PROVIDER_ID,
                model=model,
                input_tokens=None if result is None else result.input_tokens,
                output_tokens=None if result is None else result.output_tokens,
                thought_tokens=None if result is None else result.thought_tokens,
                cached_tokens=None if result is None else result.cached_tokens,
                total_tokens=None if result is None else result.total_tokens,
                outcome=outcome,
                error_code=error_code,
                started_at=started_at,
                finished_at=utc_timestamp(),
                created_at=utc_timestamp(),
            )
        )
