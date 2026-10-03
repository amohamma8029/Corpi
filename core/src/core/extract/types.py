"""Result types for M1.5 page extraction.

Metrics follow the ticket's definitions exactly:
- `text_drawn`: whether the page's content operations draw glyph runs, regardless of decode
  success (recorded separately from decoding).
- `extracted_chars`: mapped printable characters decoded successfully; parser-boundary
  unmapped-glyph tokens are excluded.
- `unmapped_ratio`: unmapped tokens divided by total glyph runs, counted at the parser boundary.
- `image_area_ratio`: fraction of the displayed page area covered by image/figure placements.

`PageFrame` carries the page geometry needed to convert layout boxes into the displayed
CropBox frame; `image_boxes` are the raw image/figure placements kept as visual metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    text: str
    bbox: tuple[float, float, float, float]
    reading_order: int


@dataclass(frozen=True, slots=True)
class PageMetrics:
    text_drawn: bool
    extracted_chars: int
    unmapped_ratio: float
    image_area_ratio: float


@dataclass(frozen=True, slots=True)
class PageFrame:
    cropbox: tuple[float, float, float, float]
    mediabox: tuple[float, float, float, float]
    rotation: int


@dataclass(frozen=True, slots=True)
class PageExtraction:
    page_number: int
    blocks: tuple[ExtractedBlock, ...]
    metrics: PageMetrics
    frame: PageFrame
    image_boxes: tuple[tuple[float, float, float, float], ...]
