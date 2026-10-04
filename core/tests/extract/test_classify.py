"""Classification rules and document outcome ordering (M1.5 decisions 3 and 4)."""

import io

from pdf_fixtures import (
    PageSpec,
    blank_pdf,
    build_pdf,
    image_only_pdf,
    native_text_pdf,
    native_text_stream,
    scanned_page_number_pdf,
    undecodable_text_pdf,
)
from pdfminer.pdfpage import PDFPage

from core.extract import ExtractionAdapter
from core.extract.classify import (
    IMAGE_AREA_LIMIT,
    MAX_EXTRACTION_PAGES,
    NO_USABLE_TEXT,
    PAGE_LIMIT_EXCEEDED,
    SHORT_PAGE_CHARS,
    DocumentDecision,
    check_page_limit,
    classify_page,
    count_pages,
    decide_document_outcome,
)
from core.extract.types import PageMetrics


def metrics(
    *,
    text_drawn: bool = True,
    extracted_chars: int = 100,
    unmapped_ratio: float = 0.0,
    image_area_ratio: float = 0.0,
) -> PageMetrics:
    return PageMetrics(text_drawn, extracted_chars, unmapped_ratio, image_area_ratio)


def classify_fixture(pdf: bytes) -> list[str]:
    adapter = ExtractionAdapter()
    results = []
    for number, page in enumerate(PDFPage.get_pages(io.BytesIO(pdf)), start=1):
        results.append(classify_page(adapter.extract_page(page, number).metrics))
    return results


def numbered_page(index: int) -> PageSpec:
    return PageSpec(native_text_stream(f"page {index}", y=700 - (index % 50) * 12), helvetica=True)


def test_drawn_but_undecodable_text_is_never_extracted() -> None:
    assert classify_page(metrics(text_drawn=True, extracted_chars=0, unmapped_ratio=1.0)) == (
        "unreadable"
    )
    assert classify_page(metrics(text_drawn=True, extracted_chars=0)) == "unreadable"


def test_garble_dominated_text_is_unreadable() -> None:
    assert classify_page(metrics(extracted_chars=80, unmapped_ratio=0.1)) == "unreadable"
    assert classify_page(metrics(extracted_chars=80, unmapped_ratio=0.05)) == "extracted"


def test_image_dominant_page_is_unread_as_scanned_even_with_a_page_number() -> None:
    assert classify_page(metrics(extracted_chars=1, image_area_ratio=0.62)) == "image-only"
    assert classify_page(metrics(text_drawn=False, extracted_chars=0, image_area_ratio=0.62)) == (
        "image-only"
    )


def test_short_correctly_read_page_passes() -> None:
    assert classify_page(metrics(extracted_chars=7)) == "extracted"
    assert classify_page(metrics(extracted_chars=SHORT_PAGE_CHARS - 1)) == "extracted"


def test_short_page_with_scan_signals_needs_review() -> None:
    assert classify_page(metrics(extracted_chars=3, image_area_ratio=0.3)) == "suspect"
    assert classify_page(metrics(extracted_chars=3, unmapped_ratio=0.01)) == "suspect"


def test_blank_and_sparse_pages_are_tolerated() -> None:
    assert classify_page(metrics(text_drawn=False, extracted_chars=0)) == "blank"
    assert classify_page(metrics(text_drawn=False, extracted_chars=0, image_area_ratio=0.1)) == (
        "blank"
    )


def test_normal_text_is_extracted() -> None:
    assert classify_page(metrics()) == "extracted"
    assert classify_page(metrics(extracted_chars=SHORT_PAGE_CHARS)) == "extracted"


def test_fixtures_classify_through_the_real_adapter() -> None:
    assert classify_fixture(native_text_pdf("Appendix")) == ["extracted"]
    assert classify_fixture(undecodable_text_pdf()) == ["unreadable"]
    assert classify_fixture(scanned_page_number_pdf()) == ["image-only"]
    assert classify_fixture(image_only_pdf()) == ["image-only"]
    assert classify_fixture(blank_pdf()) == ["blank"]


def test_outcome_order_fails_no_usable_text_before_partial_or_complete() -> None:
    assert decide_document_outcome(["image-only"] * 3) == DocumentDecision("failed", NO_USABLE_TEXT)
    assert decide_document_outcome(["unreadable"]) == DocumentDecision("failed", NO_USABLE_TEXT)
    assert decide_document_outcome(["blank", "blank"]) == DocumentDecision("failed", NO_USABLE_TEXT)


def test_partial_needs_review_while_complete_does_not() -> None:
    assert decide_document_outcome(["extracted", "image-only"]) == DocumentDecision("partial", None)
    assert decide_document_outcome(["extracted", "suspect"]) == DocumentDecision("partial", None)
    assert decide_document_outcome(["extracted", "blank"]) == DocumentDecision("complete", None)


def test_suspect_only_offers_no_eligible_evidence() -> None:
    assert decide_document_outcome(["suspect"]) == DocumentDecision("failed", NO_USABLE_TEXT)
    assert decide_document_outcome(["suspect", "blank"]) == DocumentDecision(
        "failed", NO_USABLE_TEXT
    )


def test_unknown_page_classifications_are_rejected() -> None:
    try:
        decide_document_outcome(["mystery"])
    except ValueError as error:
        assert "known state" in str(error)
    else:
        raise AssertionError("unknown classification was accepted")


def test_page_limit_proves_pages_past_the_limit() -> None:
    at_limit = build_pdf([numbered_page(index) for index in range(MAX_EXTRACTION_PAGES)])
    over_limit = build_pdf([numbered_page(index) for index in range(MAX_EXTRACTION_PAGES + 1)])
    assert check_page_limit(count_pages(at_limit)) is None
    assert check_page_limit(count_pages(over_limit)) == PAGE_LIMIT_EXCEEDED
    assert count_pages(over_limit, stop_after=MAX_EXTRACTION_PAGES + 1) == (
        MAX_EXTRACTION_PAGES + 1
    )
    assert IMAGE_AREA_LIMIT == 0.4
