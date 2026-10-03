"""Parent-side supervision of the extraction child process (M1.5).

Timeout or cancellation stops and reaps the child before any retry can start; output arriving
after the stop decision is discarded. The parent owns every database write — the child only
parses. The child command is injectable so tests can exercise the stop paths against known
slow workers; the PDF path is always appended as the final argument.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.extract.types import ExtractedBlock, PageExtraction, PageMetrics

TERMINATE_GRACE_S = 5.0
POLL_SLICE_S = 0.1


class ExtractionTimeout(Exception):
    """The child exceeded its time budget and was stopped and reaped."""


class ExtractionCancelled(Exception):
    """Cancellation stopped and reaped the child before any retry."""


class ExtractionChildError(Exception):
    """The child failed and reported an error instead of a completion record."""


@dataclass(frozen=True, slots=True)
class ExtractionRun:
    pages: tuple[PageExtraction, ...]
    parser_name: str
    parser_version: str


def default_worker_command() -> list[str]:
    return [sys.executable, "-m", "core.extract.worker"]


def _stop_and_reap(process: subprocess.Popen[str], grace_s: float = TERMINATE_GRACE_S) -> None:
    process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _page_from_payload(payload: dict[str, Any]) -> PageExtraction:
    metrics: dict[str, Any] = payload["metrics"]
    blocks = []
    for raw in payload["blocks"]:
        x0, y0, x1, y1 = raw["bbox"]
        blocks.append(ExtractedBlock(raw["text"], (x0, y0, x1, y1), raw["reading_order"]))
    return PageExtraction(
        page_number=payload["page_number"],
        blocks=tuple(blocks),
        metrics=PageMetrics(
            metrics["text_drawn"],
            metrics["extracted_chars"],
            metrics["unmapped_ratio"],
            metrics["image_area_ratio"],
        ),
    )


def run_extraction(
    pdf_path: Path,
    *,
    timeout_s: float,
    cancel: threading.Event | None = None,
    on_page: Callable[[int], None] | None = None,
    worker_command: Sequence[str] | None = None,
) -> ExtractionRun:
    """Run one extraction in a supervisable child process.

    Raises ExtractionTimeout / ExtractionCancelled after stopping and reaping the child, or
    ExtractionChildError for a child-reported failure. Output produced after the stop decision
    is never returned.
    """
    command = list(worker_command) if worker_command is not None else default_worker_command()
    command.append(str(pdf_path))
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump() -> None:
        try:
            for line in process.stdout or ():
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=_pump, daemon=True).start()

    deadline = time.monotonic() + timeout_s
    pages: list[PageExtraction] = []
    result: ExtractionRun | None = None
    stop: Exception | None = None
    try:
        while True:
            if cancel is not None and cancel.is_set():
                stop = ExtractionCancelled("extraction cancelled; child stopped")
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop = ExtractionTimeout(f"extraction exceeded {timeout_s:g} s; child stopped")
                break
            try:
                line = lines.get(timeout=min(remaining, POLL_SLICE_S))
            except queue.Empty:
                continue
            if line is None:
                break
            payload = json.loads(line)
            kind = payload.get("type")
            if kind == "page":
                page = _page_from_payload(payload)
                pages.append(page)
                if on_page is not None:
                    on_page(page.page_number)
            elif kind == "done":
                result = ExtractionRun(
                    tuple(pages),
                    payload["parser_name"],
                    payload["parser_version"],
                )
            elif kind == "error":
                stop = ExtractionChildError(str(payload.get("message", "child failed")))
                break
    finally:
        if result is not None and stop is None:
            process.wait()
        else:
            _stop_and_reap(process)
        if process.stderr is not None:
            stderr_tail = process.stderr.read()[-2000:]
            process.stderr.close()
        else:
            stderr_tail = ""

    if stop is not None:
        if stderr_tail and isinstance(stop, ExtractionChildError):
            raise ExtractionChildError(f"{stop.args[0]}\nchild stderr:\n{stderr_tail}") from stop
        raise stop
    if result is None:
        detail = f"\nchild stderr:\n{stderr_tail}" if stderr_tail else ""
        raise ExtractionChildError("child ended without a completion record" + detail)
    return result
