"""Resource limits for local PDF receipt and display."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PdfIngestLimits:
    max_upload_bytes: int = 25_000_000
    upload_total_seconds: float = 120.0
    upload_idle_seconds: float = 30.0
    rendered_page_pixels: int = 20_000_000
    abandoned_staging_seconds: float = 86_400.0
    maintenance_interval_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0 or self.rendered_page_pixels <= 0:
            raise ValueError("PDF byte and rendered-pixel limits must be positive")
        if (
            min(
                self.upload_total_seconds,
                self.upload_idle_seconds,
                self.abandoned_staging_seconds,
                self.maintenance_interval_seconds,
            )
            <= 0
        ):
            raise ValueError("PDF time limits must be positive")

    @classmethod
    def from_environment(cls) -> PdfIngestLimits:
        return cls(
            max_upload_bytes=_positive_int("CORPI_PDF_MAX_BYTES", 25_000_000),
            upload_total_seconds=_positive_float("CORPI_PDF_UPLOAD_TIMEOUT_SECONDS", 120.0),
            upload_idle_seconds=_positive_float("CORPI_PDF_UPLOAD_IDLE_SECONDS", 30.0),
            rendered_page_pixels=_positive_int("CORPI_PDF_RENDERED_PAGE_PIXELS", 20_000_000),
            abandoned_staging_seconds=_positive_float("CORPI_PDF_STAGING_AGE_SECONDS", 86_400.0),
            maintenance_interval_seconds=_positive_float(
                "CORPI_MAINTENANCE_INTERVAL_SECONDS", 3_600.0
            ),
        )


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not 0 < value < float("inf"):
        raise ValueError(f"{name} must be a positive finite number")
    return value
