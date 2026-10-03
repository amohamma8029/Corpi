"""M1.5 extraction adapter: page-by-page native text and metrics via pdfminer.six.

The adapter is intentionally narrow — `extract_page(page)` returns text blocks and the metrics
the quality rules consume. Normalization and location conventions build on these blocks in
later units; classification consumes the metrics. Parsing is driven page by page so the
supervisor can terminate work on one stuck page without losing the rest of the job state.
"""

from __future__ import annotations

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTFigure, LTImage, LTPage
from pdfminer.layout import LTTextContainer as _LTTextContainer
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from core.extract.types import ExtractedBlock, PageExtraction, PageFrame, PageMetrics

UNMAPPED_TOKEN_PREFIX = "(cid:"

PARSER_NAME = "pdfminer.six"


def _is_unmapped_token(text: str) -> bool:
    return text.startswith(UNMAPPED_TOKEN_PREFIX) and text.endswith(")")


def _is_mapped_printable(text: str) -> bool:
    return bool(text) and text.isprintable() and not text.isspace() and not _is_unmapped_token(text)


def _union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Exact area of the union of axis-aligned rectangles."""
    if not rectangles:
        return 0.0
    xs = sorted({x for rect in rectangles for x in (rect[0], rect[2])})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:], strict=False):
        if x1 <= x0:
            continue
        segments: list[tuple[float, float]] = []
        for rect in rectangles:
            rx0, _, rx1, _ = rect
            if rx0 <= x0 and x1 <= rx1:
                segments.append((rect[1], rect[3]))
        if not segments:
            continue
        segments.sort()
        covered = 0.0
        start, end = segments[0]
        for y0, y1 in segments[1:]:
            if y0 > end:
                covered += end - start
                start, end = y0, y1
            else:
                end = max(end, y1)
        covered += end - start
        total += (x1 - x0) * covered
    return total


def _collect(
    element: object,
    chars: list[str],
    image_boxes: list[tuple[float, float, float, float]],
) -> None:
    from pdfminer.layout import LTChar

    if isinstance(element, LTChar):
        chars.append(element.get_text())
        return
    if isinstance(element, (LTImage, LTFigure)):
        x0, y0, x1, y1 = element.bbox
        image_boxes.append((float(x0), float(y0), float(x1), float(y1)))
        return
    for child in getattr(element, "_objs", []) or []:
        _collect(child, chars, image_boxes)


class ExtractionAdapter:
    """Page-by-page native-text extraction with readability metrics."""

    def __init__(self) -> None:
        self._rsrcmgr = PDFResourceManager()
        self._device = PDFPageAggregator(self._rsrcmgr, laparams=LAParams())
        self._interpreter = PDFPageInterpreter(self._rsrcmgr, self._device)

    def extract_page(self, page: PDFPage, page_number: int) -> PageExtraction:
        self._interpreter.process_page(page)
        layout: LTPage = self._device.get_result()

        chars: list[str] = []
        image_boxes: list[tuple[float, float, float, float]] = []
        blocks: list[ExtractedBlock] = []
        for element in layout:
            if isinstance(element, _LTTextContainer):
                text = element.get_text()
                if text.strip():
                    x0, y0, x1, y1 = element.bbox
                    blocks.append(
                        ExtractedBlock(
                            text,
                            (float(x0), float(y0), float(x1), float(y1)),
                            len(blocks),
                        )
                    )
            _collect(element, chars, image_boxes)

        glyph_runs = len(chars)
        unmapped = sum(1 for text in chars if _is_unmapped_token(text) or not text)
        extracted_chars = sum(1 for text in chars if _is_mapped_printable(text))

        crop_x0, crop_y0, crop_x1, crop_y1 = (float(v) for v in page.cropbox)
        crop_width = max(0.0, crop_x1 - crop_x0)
        crop_height = max(0.0, crop_y1 - crop_y0)
        page_area = crop_width * crop_height
        clipped = []
        for x0, y0, x1, y1 in image_boxes:
            cx0 = max(x0, crop_x0)
            cy0 = max(y0, crop_y0)
            cx1 = min(x1, crop_x1)
            cy1 = min(y1, crop_y1)
            if cx1 > cx0 and cy1 > cy0:
                clipped.append((cx0, cy0, cx1, cy1))
        image_area_ratio = 0.0 if page_area <= 0 else _union_area(clipped) / page_area

        metrics = PageMetrics(
            text_drawn=glyph_runs > 0,
            extracted_chars=extracted_chars,
            unmapped_ratio=0.0 if glyph_runs == 0 else unmapped / glyph_runs,
            image_area_ratio=min(1.0, image_area_ratio),
        )
        media_x0, media_y0, media_x1, media_y1 = (float(v) for v in page.mediabox)
        frame = PageFrame(
            cropbox=(crop_x0, crop_y0, crop_x1, crop_y1),
            mediabox=(media_x0, media_y0, media_x1, media_y1),
            rotation=page.rotate % 360,
        )
        return PageExtraction(page_number, tuple(blocks), metrics, frame, tuple(image_boxes))
