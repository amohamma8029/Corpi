"""Google Gemini Interactions client with bounded retries and sanitized errors."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from core.jobs.runner import PaidCallOutcomeUnknown
from core.providers.errors import ProviderCancelled, ProviderFailure
from core.providers.protocol import JSONSchema, ProviderCallResult

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_REVISION = "2026-05-20"
DEFAULT_THINKING_LEVEL = "low"


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) and value >= 0 else None


def _sanitize(
    message: object,
    *,
    secret: str | None = None,
    default: str = "The provider rejected the request.",
) -> str:
    text = " ".join(str(message).split()) if message is not None else ""
    if secret:
        text = text.replace(secret, "[redacted]")
    return text[:300] or default


class GeminiClient:
    """One narrow adapter; callers do not depend on Gemini wire fields."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = GEMINI_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._thinking_level = thinking_level

    async def generate(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        schema: JSONSchema | None,
        max_output_tokens: int,
        timeout_seconds: float,
        attempts: int,
        request_id: str,
    ) -> ProviderCallResult:
        body: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "system_instruction": instructions,
            "store": False,
            "generation_config": {
                "max_output_tokens": max_output_tokens,
                "thinking_level": self._thinking_level,
            },
        }
        if schema is not None:
            body["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            }
        data = await self._post_with_retries(
            "/interactions",
            body,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
        )
        return self._result(data, fallback_request_id=request_id)

    async def count_tokens(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        timeout_seconds: float,
        attempts: int = 1,
    ) -> int:
        text = input_text if not instructions else f"{instructions}\n\n{input_text}"
        data = await self._post_with_retries(
            f"/models/{model}:countTokens",
            {"contents": [{"parts": [{"text": text}]}]},
            timeout_seconds=timeout_seconds,
            attempts=attempts,
        )
        value = data.get("totalTokens", data.get("total_tokens"))
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProviderFailure(
                "invalid_response",
                "The provider did not return a valid token count.",
                status=502,
            )
        return value

    async def list_models(
        self,
        *,
        timeout_seconds: float = 30,
        attempts: int = 1,
    ) -> list[str]:
        models: set[str] = set()
        page_token: str | None = None
        pages = 0
        while pages < 10:
            params: dict[str, str | int] = {"pageSize": 1000}
            if page_token:
                params["pageToken"] = page_token
            data = await self._request_with_retries(
                "GET",
                "/models",
                params=params,
                timeout_seconds=timeout_seconds,
                attempts=attempts,
            )
            items = data.get("models")
            if not isinstance(items, list):
                raise ProviderFailure(
                    "invalid_response",
                    "The provider returned no model list.",
                    status=502,
                )
            models.update(_text_model_ids(items))
            token = data.get("nextPageToken")
            page_token = token if isinstance(token, str) and token else None
            pages += 1
            if page_token is None:
                break
        return sorted(models)

    async def _post_with_retries(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout_seconds: float,
        attempts: int,
    ) -> dict[str, Any]:
        return await self._request_with_retries(
            "POST",
            path,
            body=body,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
        )

    async def _request_with_retries(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str | int] | None = None,
        timeout_seconds: float,
        attempts: int,
    ) -> dict[str, Any]:
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        headers = {
            "x-goog-api-key": self._api_key,
            "Api-Revision": GEMINI_API_REVISION,
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=min(10.0, timeout_seconds),
        )
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            transport=self._transport,
        ) as client:
            for attempt in range(1, attempts + 1):
                try:
                    response = await client.request(
                        method,
                        path,
                        json=body,
                        params=params,
                        headers=headers,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                    if attempt < attempts:
                        await asyncio.sleep(min(2.0 ** (attempt - 1), 5.0))
                        continue
                    raise ProviderFailure(
                        "provider_unavailable",
                        "The provider could not be reached.",
                        status=503,
                    ) from error
                except httpx.TimeoutException as error:
                    raise PaidCallOutcomeUnknown(
                        "The provider request timed out; its paid outcome is unknown."
                    ) from error
                except httpx.TransportError as error:
                    raise PaidCallOutcomeUnknown(
                        "The provider connection ended after the request was sent; "
                        "its paid outcome is unknown."
                    ) from error
                except asyncio.CancelledError as error:
                    raise ProviderCancelled() from error

                if response.status_code in {429, 503}:
                    if attempt < attempts:
                        await asyncio.sleep(_retry_delay(response))
                        continue
                if response.status_code >= 400:
                    raise _response_failure(response, self._api_key)
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as error:
                    raise ProviderFailure(
                        "invalid_response",
                        "The provider returned an unreadable response.",
                        status=502,
                    ) from error
                if not isinstance(data, dict):
                    raise ProviderFailure(
                        "invalid_response",
                        "The provider returned an unexpected response.",
                        status=502,
                    )
                if "error" in data:
                    raise _payload_failure(response.status_code, data["error"], self._api_key)
                return data
        raise ProviderFailure(
            "provider_unavailable",
            "The provider did not return a usable response.",
            status=503,
        )

    def _result(self, data: dict[str, Any], *, fallback_request_id: str) -> ProviderCallResult:
        status = data.get("status")
        if isinstance(status, str) and status.lower() not in {"completed", "succeeded", "success"}:
            code = (
                "incomplete_response"
                if status.lower() in {"incomplete", "truncated", "max_tokens"}
                else "provider_rejected"
            )
            raise ProviderFailure(
                code,
                f"Gemini interaction finished with status {_sanitize(status)}.",
                status=502 if code == "incomplete_response" else 400,
            )
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            raise ProviderFailure(
                "provider_rejected",
                "Gemini returned an error result.",
                status=400,
            )
        finish = data.get("finish_reason") or data.get("finishReason")
        if isinstance(finish, str) and finish.upper() in {
            "MAX_TOKENS",
            "SAFETY",
            "RECITATION",
            "BLOCKED",
            "OTHER",
        }:
            code = "incomplete_response" if finish.upper() == "MAX_TOKENS" else "provider_rejected"
            raise ProviderFailure(
                code,
                f"Gemini stopped with {_sanitize(finish)}.",
                status=502 if code == "incomplete_response" else 400,
            )
        output_text = data.get("output_text")
        if not isinstance(output_text, str) or not output_text:
            parts: list[str] = []
            steps = data.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    content = step.get("content")
                    if not isinstance(content, list):
                        continue
                    for item in content:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and isinstance(item.get("text"), str)
                        ):
                            parts.append(item["text"])
            output_text = "".join(parts)
        if not output_text:
            raise ProviderFailure(
                "invalid_response",
                "The provider returned no answer text.",
                status=502,
            )
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return ProviderCallResult(
            output_text=output_text,
            request_id=str(data.get("id") or fallback_request_id),
            input_tokens=_optional_int(
                usage.get("total_input_tokens", usage.get("prompt_token_count"))
            ),
            output_tokens=_optional_int(
                usage.get("total_output_tokens", usage.get("candidates_token_count"))
            ),
            thought_tokens=_optional_int(usage.get("total_thought_tokens")),
            cached_tokens=_optional_int(
                usage.get("total_cached_tokens", usage.get("cached_content_token_count"))
            ),
            total_tokens=_optional_int(usage.get("total_tokens")),
        )


MAX_RETRY_DELAY_SECONDS = 300.0


def _text_model_ids(items: list[object]) -> set[str]:
    excluded = (
        "embedding",
        "imagen",
        "image",
        "veo",
        "tts",
        "live",
        "audio",
        "robotics",
        "deep-research",
        "antigravity",
        "lyria",
    )
    result: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        model_id = name.removeprefix("models/")
        lowered = model_id.lower()
        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list) and methods and "generateContent" not in methods:
            continue
        if any(marker in lowered for marker in excluded):
            continue
        if model_id.startswith(("gemini-", "gemma-")):
            result.add(model_id)
    return result


def _error_from_payload(payload: object) -> object:
    if isinstance(payload, dict):
        return payload.get("error")
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("error") is not None:
                return item["error"]
    return None


def _retry_delay(response: httpx.Response) -> float:
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(max(float(header), 0.0), MAX_RETRY_DELAY_SECONDS)
        except ValueError:
            pass
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return 1.0
    error = _error_from_payload(payload)
    details = error.get("details") if isinstance(error, dict) else None
    if isinstance(details, list):
        for detail in details:
            delay = detail.get("retryDelay") if isinstance(detail, dict) else None
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return min(max(float(delay[:-1]), 0.0), MAX_RETRY_DELAY_SECONDS)
                except ValueError:
                    continue
    return 1.0


def _response_failure(response: httpx.Response, secret: str | None = None) -> ProviderFailure:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    error = _error_from_payload(payload)
    if error is None:
        raw = _sanitize(response.text, secret=secret, default="")
        if raw:
            return ProviderFailure(
                "provider_rejected",
                f"Gemini rejected the request (HTTP {response.status_code}): {raw}",
                status=400,
            )
    return _payload_failure(response.status_code, error, secret)


def _payload_failure(status: int, error: object, secret: str | None = None) -> ProviderFailure:
    error = error if isinstance(error, dict) else {}
    raw_message = _sanitize(error.get("message"), secret=secret, default="")
    lowered = raw_message.lower()
    reasons: set[str] = set()
    quota_ids: list[str] = []
    field_violations: list[str] = []
    details = error.get("details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            reason = detail.get("reason")
            if isinstance(reason, str):
                reasons.add(reason.upper())
            violations = detail.get("violations")
            if isinstance(violations, list):
                for violation in violations:
                    if not isinstance(violation, dict):
                        continue
                    quota_id = violation.get("quotaId")
                    if isinstance(quota_id, str):
                        quota_ids.append(quota_id)
            field_errors = detail.get("fieldViolations")
            if isinstance(field_errors, list):
                for field_error in field_errors:
                    if not isinstance(field_error, dict):
                        continue
                    field = _sanitize(field_error.get("field"), secret=secret, default="request")
                    description = _sanitize(
                        field_error.get("description"), secret=secret, default="invalid"
                    )
                    field_violations.append(f"{field}: {description}")
    joined = " ".join([raw_message, *sorted(reasons), *quota_ids]).lower()
    status_label = str(error.get("status") or error.get("code") or status)
    if raw_message:
        message = (
            raw_message
            if status_label.lower() in raw_message.lower()
            else f"Gemini {status_label}: {raw_message}"
        )
    else:
        message = f"Gemini rejected the request (HTTP {status}, {status_label})."
    if reasons:
        message = f"{message} [{', '.join(sorted(reasons))}]"
    if quota_ids:
        message = f"{message} [{', '.join(quota_ids[:2])}]"
    if field_violations:
        message = f"{message} [{'; '.join(field_violations[:2])}]"
    message = message[:300]

    invalid_key = (
        bool(reasons & {"API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_NOT_VALID"})
        or "api key not valid" in lowered
        or "api key expired" in lowered
    )
    if invalid_key or status == 401:
        return ProviderFailure("invalid_credentials", message, status=401)
    if status == 403:
        return ProviderFailure("provider_rejected", message, status=403)
    if status == 400:
        if "context" in lowered or "token" in lowered or "too long" in lowered:
            return ProviderFailure("context_length_exceeded", message, status=400)
        return ProviderFailure("provider_rejected", message, status=400)
    if status == 429:
        per_minute = (
            any(
                marker in quota_id.lower()
                for quota_id in quota_ids
                for marker in ("perminute", "persecond", "per second")
            )
            or "per minute" in joined
            or "RATE_LIMIT_EXCEEDED" in reasons
        )
        if per_minute:
            return ProviderFailure("rate_limited", message, status=429)
        if (
            any("perday" in quota_id.lower() for quota_id in quota_ids)
            or "QUOTA_EXCEEDED" in reasons
            or any(word in joined for word in ("quota", "billing", "credit", "per day"))
        ):
            return ProviderFailure("quota_exhausted", message, status=429)
        return ProviderFailure("rate_limited", message, status=429)
    if status >= 500:
        return ProviderFailure("provider_unavailable", message, status=503)
    return ProviderFailure("provider_rejected", message, status=400)
