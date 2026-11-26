"""Authenticated provider settings, credential, test, and usage routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.jobs.runner import PaidCallOutcomeUnknown
from core.providers.errors import ProviderCancelled, ProviderFailure
from core.providers.service import ProviderService, ProviderSettingsUpdate

router = APIRouter()


class ProviderSettingsPayload(BaseModel):
    provider_id: str
    provider_label: str
    generation_model: str
    checker_model: str
    free_tier_acknowledged: bool
    output_token_allowance: int
    safety_margin_tokens: int
    call_policies: dict[str, Any]
    credential_configured: bool
    generation_models: list[str]
    checker_models: list[str]


class ProviderSettingsUpdatePayload(BaseModel):
    generation_model: str
    checker_model: str
    free_tier_acknowledged: bool
    output_token_allowance: int = Field(default=1024, ge=1, le=1_000_000)
    safety_margin_tokens: int = Field(default=512, ge=0, le=1_000_000)
    # Kept as a plain string so validation errors cannot echo a secret.
    api_key: str | None = None


class ProviderModelsPayload(BaseModel):
    generation_models: list[str]
    checker_models: list[str]
    discovered: bool
    warning: str | None


class ProviderTestPayload(BaseModel):
    ok: bool
    provider_id: str
    model: str
    request_id: str
    provider_request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class ProviderUsagePayload(BaseModel):
    id: str
    request_id: str
    provider_request_id: str | None
    parent_request_id: str | None
    call_type: str
    provider_id: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    cached_tokens: int | None
    total_tokens: int | None
    outcome: str
    error_code: str | None
    started_at: str
    finished_at: str | None
    created_at: str


def _service(request: Request) -> ProviderService:
    service: ProviderService | None = getattr(request.app.state, "provider_service", None)
    if service is None:
        raise HTTPException(503, "provider service is unavailable")
    return service


def _failure(error: ProviderFailure) -> HTTPException:
    return HTTPException(error.status, {"code": error.code, "message": str(error)})


@router.get("/provider/settings", response_model=ProviderSettingsPayload)
def get_provider_settings(request: Request) -> ProviderSettingsPayload:
    try:
        return ProviderSettingsPayload.model_validate(_service(request).state())
    except ProviderFailure as error:
        raise _failure(error) from None


@router.get("/provider/models", response_model=ProviderModelsPayload)
async def get_provider_models(request: Request) -> ProviderModelsPayload:
    try:
        return ProviderModelsPayload.model_validate(await _service(request).available_models())
    except ProviderFailure as error:
        raise _failure(error) from None


@router.put("/provider/settings", response_model=ProviderSettingsPayload)
def save_provider_settings(
    payload: ProviderSettingsUpdatePayload, request: Request
) -> ProviderSettingsPayload:
    try:
        state = _service(request).save(
            ProviderSettingsUpdate(
                generation_model=payload.generation_model,
                checker_model=payload.checker_model,
                free_tier_acknowledged=payload.free_tier_acknowledged,
                output_token_allowance=payload.output_token_allowance,
                safety_margin_tokens=payload.safety_margin_tokens,
            ),
            api_key=payload.api_key,
        )
        return ProviderSettingsPayload.model_validate(state)
    except ProviderFailure as error:
        raise _failure(error) from None
    except ValueError:
        raise HTTPException(400, "Provider settings are invalid.") from None


@router.post("/provider/settings/test", response_model=ProviderTestPayload)
async def test_provider(request: Request) -> ProviderTestPayload:
    try:
        return ProviderTestPayload.model_validate(await _service(request).test_connection())
    except ProviderCancelled as error:
        raise _failure(error) from None
    except PaidCallOutcomeUnknown:
        raise HTTPException(
            504,
            {
                "code": "paid_outcome_unknown",
                "message": "The provider did not confirm the test call; review is required.",
            },
        ) from None
    except ProviderFailure as error:
        raise _failure(error) from None


@router.delete("/provider/credential", response_model=ProviderSettingsPayload)
def clear_provider_credential(request: Request) -> ProviderSettingsPayload:
    try:
        return ProviderSettingsPayload.model_validate(_service(request).clear_credential())
    except ProviderFailure as error:
        raise _failure(error) from None


@router.get("/provider/usage", response_model=list[ProviderUsagePayload])
def list_provider_usage(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ProviderUsagePayload]:
    rows = _service(request).usage(limit)
    return [
        ProviderUsagePayload(
            id=row.id,
            request_id=row.request_id,
            provider_request_id=row.provider_request_id,
            parent_request_id=row.parent_request_id,
            call_type=row.call_type,
            provider_id=row.provider_id,
            model=row.model,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            thought_tokens=row.thought_tokens,
            cached_tokens=row.cached_tokens,
            total_tokens=row.total_tokens,
            outcome=row.outcome,
            error_code=row.error_code,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )
        for row in rows
    ]
