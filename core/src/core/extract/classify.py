"""Page classification rules and document outcome ordering (M1.5 decisions 3 and 4).

`text_drawn` is recorded separately from decode success so unread pages never count as
extracted: drawn-but-undecodable text is `unreadable`, and an image-dominant page is
`image-only` even when a few characters (a page number) decoded. Short correctly read pages
stay `extracted`. Thresholds are named hypotheses to validate against fixture PDFs; the
implementation PR records what held and what was tuned.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

from pdfminer.pdfpage import PDFPage

from core.extract.types import PageMetrics

SHORT_PAGE_CHARS = 24
UNMAPPED_RATIO_LIMIT = 0.05
IMAGE_AREA_LIMIT = 0.4
MAX_EXTRACTION_PAGES = 200

PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
NO_USABLE_TEXT = "no_usable_text"

PAGE_CLASSES = ("extracted", "suspect", "image-only", "unreadable", "blank")
UNRELIABLE_CLASSES = ("image-only", "suspect", "unreadable")


def classify_page(metrics: PageMetrics) -> str:
    """Classify one page along readability and search eligibility, highest rule first."""
    if metrics.text_drawn and metrics.extracted_chars == 0:
        return "unreadable"
    if metrics.unmapped_ratio > UNMAPPED_RATIO_LIMIT:
        return "unreadable"
    if metrics.image_area_ratio > IMAGE_AREA_LIMIT and metrics.extracted_chars < SHORT_PAGE_CHARS:
        return "image-only"
    if not metrics.text_drawn:
        # Nothing drawn and nothing decoded: no text could have been misread. Sparse
        # decoration is tolerated like a blank page — the architecture forbids treating
        # blank/sparse pages as unreadable.
        return "blank"
    if metrics.extracted_chars < SHORT_PAGE_CHARS:
        # Closer check, never a verdict by count: corroboration decides. Garbled glyphs or
        # an image region (a partially scanned body under rule 3's threshold) mean text may
        # be missing; a short correctly read page such as "Appendix" is `extracted`.
        if metrics.unmapped_ratio > 0 or metrics.image_area_ratio > 0:
            return "suspect"
        return "extracted"
    return "extracted"


@dataclass(frozen=True, slots=True)
class DocumentDecision:
    outcome: str  # "complete" | "partial" | "failed"
    failure_code: str | None


def decide_document_outcome(page_classes: Sequence[str]) -> DocumentDecision:
    """Apply decision 4's outcomes in order: no usable text first, then partial, then complete.

    Usable text means search-eligible readable text (`extracted`). Questionable `suspect`
    text stays out of search, so a document with no `extracted` page offers no eligible
    evidence and fails `no_usable_text` — confirmation can never approve an empty hand.
    """
    for classification in page_classes:
        if classification not in PAGE_CLASSES:
            raise ValueError("page classification is not a known state")
    if not any(classification == "extracted" for classification in page_classes):
        return DocumentDecision("failed", NO_USABLE_TEXT)
    if any(classification in UNRELIABLE_CLASSES for classification in page_classes):
        return DocumentDecision("partial", None)
    return DocumentDecision("complete", None)


def count_pages(pdf: bytes, *, stop_after: int | None = None) -> int:
    """Count document pages without parsing content.

    Counting walks the page tree past any extraction limit, so a page beyond the limit is
    proven to exist rather than assumed absent from a truncated read.
    """
    count = 0
    for _ in PDFPage.get_pages(io.BytesIO(pdf)):
        count += 1
        if stop_after is not None and count >= stop_after:
            break
    return count


def check_page_limit(page_count: int, limit: int = MAX_EXTRACTION_PAGES) -> str | None:
    """Return `page_limit_exceeded` when pages past the limit are proven to exist."""
    return PAGE_LIMIT_EXCEEDED if page_count > limit else None
