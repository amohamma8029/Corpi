"""Location JSON v1 in the CropBox frame the viewer displays (M1.5 decision 2).

pdfminer reports layout in its MediaBox-based frame with page rotation already applied
(verified empirically per rotation); the displayed frame is the same orientation with the
CropBox corner as origin, so the conversion is a pure translation. Units are PDF points at
2-decimal precision. Uncertain structure is recorded as `text`, never guessed.
"""

from __future__ import annotations

import re
from typing import Any, Literal

LOCATION_FORMAT_VERSION = 1

BlockKind = Literal["paragraph", "heading", "list_item", "table_row", "text"]

_LIST_MARKER = re.compile(r"^(?:[-*•–]|\(?\d{1,3}[.)])\s")
_SENTENCE_END = (".", "!", "?", ":")


def classify_block_kind(text: str) -> BlockKind:
    """Conservative block classification: anything uncertain stays `text`."""
    first_line = text.split("\n", 1)[0].strip()
    if _LIST_MARKER.match(first_line):
        return "list_item"
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= 2:
        return "paragraph"
    if len(lines) == 1 and lines[0].rstrip().endswith(_SENTENCE_END):
        return "paragraph"
    return "text"


def display_translation(
    cropbox: tuple[float, float, float, float],
    mediabox: tuple[float, float, float, float],
    rotation: int,
) -> tuple[float, float]:
    """Offset from pdfminer's rotated layout frame to the displayed CropBox frame."""
    rotation = rotation % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("page rotation must be a multiple of 90 degrees")
    mx0, my0, mx1, my1 = mediabox
    cx0, cy0, cx1, cy1 = cropbox
    if rotation == 90:
        return (cy0 - my0, mx1 - cx1)
    if rotation == 180:
        return (mx1 - cx1, my1 - cy1)
    if rotation == 270:
        return (my1 - cy1, cx0 - mx0)
    return (cx0 - mx0, cy0 - my0)


def layout_bbox_to_display(
    bbox: tuple[float, float, float, float],
    *,
    cropbox: tuple[float, float, float, float],
    mediabox: tuple[float, float, float, float],
    rotation: int,
) -> tuple[float, float, float, float]:
    """Convert one layout bbox to the displayed CropBox frame at 2-decimal precision."""
    tx, ty = display_translation(cropbox, mediabox, rotation)
    x0, y0, x1, y1 = bbox
    return (round(x0 - tx, 2), round(y0 - ty, 2), round(x1 - tx, 2), round(y1 - ty, 2))


def location_json(
    *,
    page_number: int,
    kind: BlockKind,
    bbox: tuple[float, float, float, float],
    reading_order: int,
) -> dict[str, Any]:
    """Build one location JSON v1 record."""
    return {
        "page": page_number,
        "block": kind,
        "bbox": [round(value, 2) for value in bbox],
        "reading_order": reading_order,
    }
