"""Location JSON v1: CropBox frame, rotation normalization, conservative block kinds."""

import io

import pytest
from pdf_fixtures import (
    cropped_text_pdf,
    rotated_text_pdf,
    two_column_pdf,
)
from pdfminer.pdfpage import PDFPage

from core.extract import ExtractionAdapter
from core.extract.locations import (
    LOCATION_FORMAT_VERSION,
    classify_block_kind,
    display_translation,
    layout_bbox_to_display,
    location_json,
)
from core.extract.types import PageExtraction

# Fixture font geometry for Helvetica 24pt (verified against the parser):
# "Anchor" spans x 0..76.03 and glyph rows sit 4.97 below / 19.03 above the baseline.
WORD_WIDTH = 76.03
GLYPH_HEIGHT = 24.0


def _rect(value: object) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = value  # type: ignore[misc]
    return (float(x0), float(y0), float(x1), float(y1))


def extract_first(pdf: bytes) -> tuple[PageExtraction, PDFPage]:
    adapter = ExtractionAdapter()
    pages = list(PDFPage.get_pages(io.BytesIO(pdf)))
    return adapter.extract_page(pages[0], 1), pages[0]


def display_bbox(pdf: bytes) -> tuple[float, float, float, float]:
    result, page = extract_first(pdf)
    block = result.blocks[0]
    return layout_bbox_to_display(
        block.bbox,
        cropbox=_rect(page.cropbox),
        mediabox=_rect(page.mediabox),
        rotation=page.rotate,
    )


def test_unrotated_uncropped_layout_is_already_the_displayed_frame() -> None:
    assert display_bbox(rotated_text_pdf("Anchor", 0)) == (72.0, 715.03, 148.03, 739.03)


def test_crop_translates_into_the_displayed_frame() -> None:
    bbox = display_bbox(cropped_text_pdf("Anchor", (100, 100, 500, 700), x=150, y=300))
    assert bbox == (50.0, 195.03, 126.03, 219.03)
    assert round(bbox[2] - bbox[0], 2) == WORD_WIDTH
    assert round(bbox[3] - bbox[1], 2) == GLYPH_HEIGHT


def test_rotation_90_normalizes_to_the_displayed_orientation() -> None:
    bbox = display_bbox(rotated_text_pdf("Anchor", 90))
    assert bbox == (715.03, 463.97, 739.03, 540.0)
    assert round(bbox[2] - bbox[0], 2) == GLYPH_HEIGHT
    assert round(bbox[3] - bbox[1], 2) == WORD_WIDTH


def test_rotation_180_and_270_land_in_the_displayed_corners() -> None:
    assert display_bbox(rotated_text_pdf("Anchor", 180)) == (463.97, 52.97, 540.0, 76.97)
    assert display_bbox(rotated_text_pdf("Anchor", 270)) == (52.97, 72.0, 76.97, 148.03)


def test_crop_and_rotation_compose() -> None:
    bbox = display_bbox(cropped_text_pdf("Anchor", (100, 100, 500, 700), rotate=90, x=150, y=300))
    # Block union in the rotated layout frame is (295.03, 385.97, 319.03, 462.0);
    # the displayed CropBox frame subtracts (cy0 - my0, mx1 - cx1) = (100, 112).
    assert bbox == (195.03, 273.97, 219.03, 350.0)


def test_two_columns_read_left_to_right() -> None:
    result, page = extract_first(two_column_pdf("Left column text", "Right column text"))
    assert len(result.blocks) == 2
    boxes = [
        layout_bbox_to_display(
            block.bbox,
            cropbox=_rect(page.cropbox),
            mediabox=_rect(page.mediabox),
            rotation=page.rotate,
        )
        for block in result.blocks
    ]
    assert result.blocks[0].reading_order == 0
    assert result.blocks[1].reading_order == 1
    assert boxes[0][0] < boxes[1][0]
    assert any("Left column text" in block.text for block in result.blocks)
    assert any("Right column text" in block.text for block in result.blocks)


def test_location_json_shape_and_precision() -> None:
    record = location_json(
        page_number=2, kind="text", bbox=(1.23456, 2.0, 3.999, 4.5), reading_order=7
    )
    assert record == {
        "page": 2,
        "block": "text",
        "bbox": [1.23, 2.0, 4.0, 4.5],
        "reading_order": 7,
    }
    assert LOCATION_FORMAT_VERSION == 1


def test_block_kinds_are_conservative() -> None:
    assert classify_block_kind("2. Item text") == "list_item"
    assert classify_block_kind("- bullet text") == "list_item"
    assert classify_block_kind("It ended here.") == "paragraph"
    assert classify_block_kind("First line\nsecond line") == "paragraph"
    assert classify_block_kind("Heading text") == "text"
    assert classify_block_kind("3") == "text"


def test_structure_is_classified_from_raw_layout_text() -> None:
    raw = "A paragraph that wraps\nacross two layout lines"
    assert classify_block_kind(raw) == "paragraph"


def test_rotation_must_be_a_quarter_turn() -> None:
    assert display_translation((0, 0, 10, 10), (0, 0, 10, 10), 450) == (0, 0)
    with pytest.raises(ValueError, match="multiple of 90"):
        display_translation((0, 0, 10, 10), (0, 0, 10, 10), 45)
