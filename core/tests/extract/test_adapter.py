"""Adapter metrics for M1.5: text_drawn, extracted_chars, unmapped_ratio, image_area_ratio."""

import io
import math

from pdf_fixtures import (
    IMAGE_RGB_2X2,
    PageSpec,
    blank_pdf,
    build_pdf,
    image_draw_stream,
    image_only_pdf,
    native_text_pdf,
    native_text_stream,
    scanned_page_number_pdf,
    undecodable_stream,
    undecodable_text_pdf,
)
from pdfminer.pdfpage import PDFPage

from core.extract import ExtractionAdapter, PageExtraction

STANDARD_IMAGE_RATIO = (500 * 600) / (612 * 792)


def extract_all(pdf: bytes) -> list[PageExtraction]:
    adapter = ExtractionAdapter()
    results: list[PageExtraction] = []
    for number, page in enumerate(PDFPage.get_pages(io.BytesIO(pdf)), start=1):
        results.append(adapter.extract_page(page, number))
    return results


def test_native_text_page_reports_mapped_text_and_no_unmapped_glyphs() -> None:
    pages = extract_all(native_text_pdf("Hello Corpi extraction 123"))
    metrics = pages[0].metrics
    assert metrics.text_drawn is True
    assert metrics.extracted_chars == 23
    assert metrics.unmapped_ratio == 0.0
    assert metrics.image_area_ratio == 0.0
    assert any("Hello Corpi extraction 123" in block.text for block in pages[0].blocks)


def test_undecodable_glyph_runs_never_count_as_extracted_text() -> None:
    metrics = extract_all(undecodable_text_pdf())[0].metrics
    assert metrics.text_drawn is True
    assert metrics.extracted_chars == 0
    assert metrics.unmapped_ratio == 1.0


def test_unmapped_tokens_count_at_the_parser_boundary() -> None:
    content = b"BT /F1 24 Tf 72 720 Td (Hi) Tj ET\nBT /F2 24 Tf 200 720 Td <00010002> Tj ET"
    pages = extract_all(build_pdf([PageSpec(content, helvetica=True, cid=True)]))
    metrics = pages[0].metrics
    assert metrics.text_drawn is True
    assert metrics.extracted_chars == 2
    assert metrics.unmapped_ratio == 0.5


def test_image_placement_area_is_measured_against_the_page() -> None:
    metrics = extract_all(image_only_pdf())[0].metrics
    assert metrics.text_drawn is False
    assert metrics.extracted_chars == 0
    assert math.isclose(metrics.image_area_ratio, STANDARD_IMAGE_RATIO, rel_tol=1e-6)
    assert metrics.image_area_ratio > 0.4


def test_scanned_page_number_records_drawn_text_and_image_area() -> None:
    metrics = extract_all(scanned_page_number_pdf())[0].metrics
    assert metrics.text_drawn is True
    assert metrics.extracted_chars == 1
    assert metrics.image_area_ratio > 0.4


def test_blank_page_has_no_drawn_text_no_characters_and_no_images() -> None:
    metrics = extract_all(blank_pdf())[0].metrics
    assert metrics.text_drawn is False
    assert metrics.extracted_chars == 0
    assert metrics.unmapped_ratio == 0.0
    assert metrics.image_area_ratio == 0.0


def test_multi_page_extraction_reports_each_page_separately() -> None:
    pages = extract_all(native_text_pdf("First page body", "Second page body", ""))
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert pages[0].metrics.extracted_chars == len("Firstpagebody")
    assert pages[1].metrics.extracted_chars == len("Secondpagebody")
    assert pages[2].metrics.text_drawn is False
    assert pages[0].blocks[0].reading_order == 0


def test_blocks_carry_layout_boxes_in_points() -> None:
    pages = extract_all(native_text_pdf("Boxed"))
    x0, y0, x1, y1 = pages[0].blocks[0].bbox
    assert x0 == 72.0
    assert 700 < y0 < 792
    assert x1 > x0 and y1 > y0


def test_fixture_bytes_are_deterministic() -> None:
    assert native_text_stream("Same") == native_text_stream("Same")
    assert undecodable_stream() == undecodable_stream()
    assert image_draw_stream() == image_draw_stream()
    assert IMAGE_RGB_2X2 == bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    assert native_text_pdf("Same") == native_text_pdf("Same")
