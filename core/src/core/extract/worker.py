"""Child-process entrypoint for M1.5 extraction: streams page results as NDJSON on stdout.

One line per page result, then a completion record. The parent supervises this process and
kills it on timeout or cancellation; the child never touches the database.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

from pdfminer.pdfpage import PDFPage

from core.extract.adapter import PARSER_NAME, ExtractionAdapter


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        _emit({"type": "error", "message": "expected exactly one PDF path argument"})
        return 2
    try:
        adapter = ExtractionAdapter()
        count = 0
        with Path(args[0]).open("rb") as handle:
            for number, page in enumerate(PDFPage.get_pages(handle), start=1):
                result = adapter.extract_page(page, number)
                _emit(
                    {
                        "type": "page",
                        "page_number": result.page_number,
                        "frame": {
                            "cropbox": list(result.frame.cropbox),
                            "mediabox": list(result.frame.mediabox),
                            "rotation": result.frame.rotation,
                        },
                        "blocks": [
                            {
                                "text": block.text,
                                "bbox": list(block.bbox),
                                "reading_order": block.reading_order,
                            }
                            for block in result.blocks
                        ],
                        "image_boxes": [list(box) for box in result.image_boxes],
                        "metrics": {
                            "text_drawn": result.metrics.text_drawn,
                            "extracted_chars": result.metrics.extracted_chars,
                            "unmapped_ratio": result.metrics.unmapped_ratio,
                            "image_area_ratio": result.metrics.image_area_ratio,
                        },
                    }
                )
                count = number
        _emit(
            {
                "type": "done",
                "page_count": count,
                "parser_name": PARSER_NAME,
                "parser_version": version("pdfminer.six"),
            }
        )
        return 0
    except Exception as error:
        _emit({"type": "error", "message": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
