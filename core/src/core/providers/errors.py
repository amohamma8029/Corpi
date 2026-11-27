"""Stable provider failures shared by the core service and API boundary."""

from __future__ import annotations

from core.providers.protocol import ProviderCallResult


class ProviderFailure(Exception):
    """A sanitized, actionable provider or credential failure.

    Messages are controlled text assembled by Corpi. Raw provider response text,
    provider field names, quota identifiers, and credentials never pass through
    this exception. Token counts and the provider request id are retained when
    the provider reported them, including for incomplete results.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 502,
        provider_request_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        thought_tokens: int | None = None,
        cached_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.provider_request_id = provider_request_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thought_tokens = thought_tokens
        self.cached_tokens = cached_tokens
        self.total_tokens = total_tokens


class ProviderCancelled(ProviderFailure):
    """An interactive call was stopped.

    ``possibly_processed`` is true only when cancellation happened after an HTTP
    attempt began, where Corpi cannot prove the provider never received it.
    """

    def __init__(self, *, possibly_processed: bool = False) -> None:
        self.possibly_processed = possibly_processed
        if possibly_processed:
            super().__init__(
                "cancelled_outcome_unknown",
                "The provider call was stopped after it was sent; "
                "the provider may still process it.",
                status=499,
            )
        else:
            super().__init__(
                "cancelled",
                "The provider call was stopped before it was sent.",
                status=499,
            )


class ProviderCleanupCancelled(Exception):
    """Stop arrived while the HTTP client was closing after an outcome was known."""

    def __init__(
        self,
        *,
        result: ProviderCallResult | None = None,
        failure: ProviderFailure | None = None,
        paid_outcome_unknown: bool = False,
        possibly_processed: bool = False,
    ) -> None:
        super().__init__("The provider call was stopped while its client closed.")
        self.result = result
        self.failure = failure
        self.paid_outcome_unknown = paid_outcome_unknown
        self.possibly_processed = possibly_processed
