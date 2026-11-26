"""Google Gemini Interactions client with bounded retries and sanitized errors."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from core.jobs.runner import PaidCallOutcomeUnknown
from core.providers.errors import ProviderCancelled, ProviderFailure
from core.providers.protocol import JSONSchema, ProviderCallResult

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_REVISION = "2026-05-20"
DEFAULT_THINKING_LEVEL = "low"
MAX_RETRY_DELAY_SECONDS = 300.0
CLIENT_CLOSE_TIMEOUT_SECONDS = 5.0
_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9._-]{1,200}")

FAILURE_MESSAGES = {
    "invalid_credentials": "The provider rejected the API key. Replace the key and try again.",
    "quota_exhausted": (
        "The provider quota for this key is exhausted. "
        "Check the account's quota or try again later."
    ),
    "rate_limited": "The provider is limiting requests right now. Wait a moment and try again.",
    "context_length_exceeded": (
        "The request is too long for the selected model. Shorten the input or choose a model "
        "with a larger context window."
    ),
    "provider_access_denied": (
        "The provider denied access to the selected model or project. "
        "Check the key's project and model access."
    ),
    "provider_rejected": (
        "The provider rejected the request. Check the selected model and request settings."
    ),
    "provider_unavailable": "The provider is temporarily unavailable. Try again later.",
    "provider_timeout": (
        "The provider did not complete within the total time limit. Try again later."
    ),
    "invalid_response": "The provider returned an unreadable response.",
    "incomplete_response": (
        "The provider stopped before completing the answer. Try again or choose a different model."
    ),
}


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) and value >= 0 else None


def _provider_request_id(data: dict[str, Any]) -> str | None:
    value = data.get("id")
    if isinstance(value, str) and 0 < len(value) <= 512:
        return value
    return None


@dataclass(frozen=True, slots=True)
class _Reported:
    provider_request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    cached_tokens: int | None
    total_tokens: int | None

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> _Reported:
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return cls(
            provider_request_id=_provider_request_id(data),
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


def _failure(
    code: str,
    *,
    status: int,
    reported: _Reported | None = None,
    message: str | None = None,
) -> ProviderFailure:
    reported = reported or _Reported(None, None, None, None, None, None)
    return ProviderFailure(
        code,
        message or FAILURE_MESSAGES[code],
        status=status,
        provider_request_id=reported.provider_request_id,
        input_tokens=reported.input_tokens,
        output_tokens=reported.output_tokens,
        thought_tokens=reported.thought_tokens,
        cached_tokens=reported.cached_tokens,
        total_tokens=reported.total_tokens,
    )


def _deadline_failure() -> ProviderFailure:
    return _failure("provider_timeout", status=504)


def _current_model_text(steps: list[object]) -> str:
    """Read the trailing model-output text after the last user input.

    Mirrors the Interactions SDK's output boundary: walk steps backward, stop at
    the last user input, skip thoughts, and collect only trailing text blocks
    from model-output steps. Earlier output and input text cannot become the
    current answer.
    """

    text_parts: list[str] = []
    collecting = False
    for step in reversed(steps):
        step_type = step.get("type") if isinstance(step, dict) else None
        if step_type == "user_input":
            break
        if step_type == "thought":
            continue
        if step_type != "model_output":
            if collecting:
                break
            continue
        content = step.get("content") if isinstance(step, dict) else None
        if not isinstance(content, list):
            continue
        should_stop = False
        for item in reversed(content):
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "text":
                collecting = True
                text = item.get("text") if isinstance(item, dict) else None
                text_parts.append(text if isinstance(text, str) else "")
            elif collecting:
                should_stop = True
                break
        if should_stop:
            break
    return "".join(reversed(text_parts))


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
        deadline_seconds: float | None = None,
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
            deadline_seconds=deadline_seconds,
        )
        return self._result(data, local_request_id=request_id)

    async def count_tokens(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        timeout_seconds: float,
        attempts: int = 1,
        deadline_seconds: float | None = None,
    ) -> int:
        text = input_text if not instructions else f"{instructions}\n\n{input_text}"
        data = await self._post_with_retries(
            f"/models/{model}:countTokens",
            {"contents": [{"parts": [{"text": text}]}]},
            timeout_seconds=timeout_seconds,
            attempts=attempts,
            deadline_seconds=deadline_seconds,
        )
        value = data.get("totalTokens", data.get("total_tokens"))
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _failure("invalid_response", status=502)
        return value

    async def list_models(
        self,
        *,
        timeout_seconds: float = 30,
        attempts: int = 1,
        deadline_seconds: float | None = None,
    ) -> list[str]:
        models: set[str] = set()
        page_token: str | None = None
        pages = 0
        total_seconds = deadline_seconds if deadline_seconds is not None else timeout_seconds
        if total_seconds <= 0:
            raise ValueError("deadline must be positive")
        deadline = time.monotonic() + total_seconds
        while pages < 10:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _deadline_failure()
            params: dict[str, str | int] = {"pageSize": 1000}
            if page_token:
                params["pageToken"] = page_token
            data = await self._request_with_retries(
                "GET",
                "/models",
                params=params,
                timeout_seconds=timeout_seconds,
                attempts=attempts,
                deadline_seconds=remaining,
            )
            items = data.get("models")
            if not isinstance(items, list):
                raise _failure("invalid_response", status=502)
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
        deadline_seconds: float | None,
    ) -> dict[str, Any]:
        return await self._request_with_retries(
            "POST",
            path,
            body=body,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
            deadline_seconds=deadline_seconds,
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
        deadline_seconds: float | None = None,
    ) -> dict[str, Any]:
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        total_seconds = deadline_seconds if deadline_seconds is not None else timeout_seconds
        if total_seconds <= 0:
            raise ValueError("deadline must be positive")
        deadline = time.monotonic() + total_seconds
        headers = {
            "x-goog-api-key": self._api_key,
            "Api-Revision": GEMINI_API_REVISION,
            "Content-Type": "application/json",
        }
        client = httpx.AsyncClient(
            base_url=self._base_url,
            transport=self._transport,
        )
        try:
            for attempt in range(1, attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _deadline_failure()
                attempt_timeout = min(timeout_seconds, remaining)
                timeout = httpx.Timeout(
                    attempt_timeout,
                    connect=min(10.0, attempt_timeout),
                )
                in_flight = False
                try:
                    in_flight = True
                    async with asyncio.timeout(remaining):
                        response = await client.request(
                            method,
                            path,
                            json=body,
                            params=params,
                            headers=headers,
                            timeout=timeout,
                        )
                    in_flight = False
                except TimeoutError as error:
                    raise PaidCallOutcomeUnknown(
                        "The provider request exceeded the total time limit; "
                        "its outcome is unknown."
                    ) from error
                except asyncio.CancelledError:
                    raise ProviderCancelled(possibly_processed=in_flight) from None
                except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                    if attempt < attempts:
                        await _wait_before_retry(min(2.0 ** (attempt - 1), 5.0), deadline)
                        continue
                    raise _failure("provider_unavailable", status=503) from error
                except httpx.TimeoutException as error:
                    raise PaidCallOutcomeUnknown(
                        "The provider request timed out; its paid outcome is unknown."
                    ) from error
                except httpx.TransportError as error:
                    raise PaidCallOutcomeUnknown(
                        "The provider connection ended after the request was sent; "
                        "its paid outcome is unknown."
                    ) from error

                if response.status_code in {429, 503}:
                    if attempt < attempts:
                        await _wait_before_retry(_retry_delay(response), deadline)
                        continue
                if response.status_code >= 400:
                    raise _response_failure(response)
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as error:
                    raise _failure("invalid_response", status=502) from error
                if not isinstance(data, dict):
                    raise _failure("invalid_response", status=502)
                if "error" in data:
                    raise _payload_failure(response.status_code, data["error"])
                return data
            raise _failure("provider_unavailable", status=503)
        finally:
            await _close_client(client, deadline)

    def _result(self, data: dict[str, Any], *, local_request_id: str) -> ProviderCallResult:
        reported = _Reported.from_payload(data)
        status = data.get("status")
        if isinstance(status, str) and status.lower() not in {"completed", "succeeded", "success"}:
            code = (
                "incomplete_response"
                if status.lower() in {"incomplete", "truncated", "max_tokens"}
                else "provider_rejected"
            )
            raise _failure(
                code,
                status=502 if code == "incomplete_response" else 400,
                reported=reported,
            )
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            raise _failure("provider_rejected", status=400, reported=reported)
        finish = data.get("finish_reason") or data.get("finishReason")
        if isinstance(finish, str) and finish.upper() in {
            "MAX_TOKENS",
            "SAFETY",
            "RECITATION",
            "BLOCKED",
            "OTHER",
        }:
            code = "incomplete_response" if finish.upper() == "MAX_TOKENS" else "provider_rejected"
            raise _failure(
                code,
                status=502 if code == "incomplete_response" else 400,
                reported=reported,
            )
        steps = data.get("steps")
        if isinstance(steps, list):
            output_text = _current_model_text(steps)
        else:
            output_text = data.get("output_text")
        if not isinstance(output_text, str) or not output_text:
            raise _failure("invalid_response", status=502, reported=reported)
        return ProviderCallResult(
            output_text=output_text,
            local_request_id=local_request_id,
            provider_request_id=reported.provider_request_id,
            input_tokens=reported.input_tokens,
            output_tokens=reported.output_tokens,
            thought_tokens=reported.thought_tokens,
            cached_tokens=reported.cached_tokens,
            total_tokens=reported.total_tokens,
        )


async def _wait_before_retry(delay: float, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or delay >= remaining:
        raise _deadline_failure()
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        raise ProviderCancelled(possibly_processed=False) from None


async def _close_client(client: httpx.AsyncClient, deadline: float) -> None:
    """Close without letting cleanup replace a known outcome or leak a pending call."""

    remaining = deadline - time.monotonic()
    timeout = max(0.01, min(CLIENT_CLOSE_TIMEOUT_SECONDS, remaining))
    try:
        async with asyncio.timeout(timeout):
            await client.aclose()
    except (Exception, asyncio.CancelledError):
        # The provider result or its controlled failure already carries the
        # outcome. A slow or cancelled close must not turn that into a pending
        # usage row or replace a known answer.
        pass


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
        if model_id.startswith(("gemini-", "gemma-")) and _SAFE_MODEL_ID.fullmatch(model_id):
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


def _response_failure(response: httpx.Response) -> ProviderFailure:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    error = _error_from_payload(payload)
    if error is not None:
        return _payload_failure(response.status_code, error)
    return _status_failure(response.status_code)


def _status_failure(status: int) -> ProviderFailure:
    if status == 401:
        return _failure("invalid_credentials", status=401)
    if status == 403:
        return _failure(
            "provider_rejected",
            status=403,
            message=FAILURE_MESSAGES["provider_access_denied"],
        )
    if status == 429:
        return _failure("rate_limited", status=429)
    if status >= 500:
        return _failure("provider_unavailable", status=503)
    return _failure("provider_rejected", status=400)


def _payload_failure(status: int, error: object) -> ProviderFailure:
    error = error if isinstance(error, dict) else {}
    raw_message = error.get("message")
    raw_message = raw_message if isinstance(raw_message, str) else ""
    lowered = raw_message.lower()
    reasons: set[str] = set()
    quota_ids: list[str] = []
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
    joined = " ".join([lowered, *sorted(reason.lower() for reason in reasons), *quota_ids])
    joined = joined.lower()

    invalid_key = (
        bool(reasons & {"API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_NOT_VALID"})
        or "api key not valid" in lowered
        or "api key expired" in lowered
    )
    if invalid_key or status == 401:
        return _failure("invalid_credentials", status=401)
    if status == 403:
        return _failure(
            "provider_rejected",
            status=403,
            message=FAILURE_MESSAGES["provider_access_denied"],
        )
    if status == 400:
        if "context" in joined or "token" in joined or "too long" in joined:
            return _failure("context_length_exceeded", status=400)
        return _failure("provider_rejected", status=400)
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
            return _failure("rate_limited", status=429)
        if (
            any("perday" in quota_id.lower() for quota_id in quota_ids)
            or "QUOTA_EXCEEDED" in reasons
            or any(word in joined for word in ("quota", "billing", "credit", "per day"))
        ):
            return _failure("quota_exhausted", status=429)
        return _failure("rate_limited", status=429)
    if status >= 500:
        return _failure("provider_unavailable", status=503)
    return _failure("provider_rejected", status=400)
