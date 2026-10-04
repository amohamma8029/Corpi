"""Supervision of the extraction child: round-trip, timeout stop/reap, cancellation."""

import sys
import threading
import time
from pathlib import Path

import pytest
from pdf_fixtures import native_text_pdf, undecodable_text_pdf

from core.extract.supervisor import (
    ExtractionCancelled,
    ExtractionChildError,
    ExtractionTimeout,
    run_extraction,
)

SLOW_WORKER = [sys.executable, "-c", "import time; time.sleep(60)"]


def write_pdf(tmp_path: Path, name: str, data: bytes) -> Path:
    target = tmp_path / name
    target.write_bytes(data)
    return target


def test_child_round_trips_pages_blocks_and_metrics(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "native.pdf", native_text_pdf("Hello Corpi"))
    run = run_extraction(path, timeout_s=60)
    assert run.parser_name == "pdfminer.six"
    assert run.parser_version
    assert len(run.pages) == 1
    metrics = run.pages[0].metrics
    assert metrics.text_drawn is True
    assert metrics.extracted_chars == len("HelloCorpi")
    assert any("Hello Corpi" in block.text for block in run.pages[0].blocks)


def test_progress_reports_each_page_as_it_arrives(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "multi.pdf", native_text_pdf("One", "Two", "Three"))
    seen: list[int] = []
    run = run_extraction(path, timeout_s=60, on_page=seen.append)
    assert seen == [1, 2, 3]
    assert [page.page_number for page in run.pages] == [1, 2, 3]


def test_undecodable_metrics_survive_the_ipc_boundary(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "cid.pdf", undecodable_text_pdf())
    metrics = run_extraction(path, timeout_s=60).pages[0].metrics
    assert metrics.text_drawn is True
    assert metrics.extracted_chars == 0
    assert metrics.unmapped_ratio == 1.0


def test_missing_pdf_reports_the_child_error(tmp_path: Path) -> None:
    with pytest.raises(ExtractionChildError):
        run_extraction(tmp_path / "missing.pdf", timeout_s=60)


def test_timeout_stops_and_reaps_the_child(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(ExtractionTimeout):
        run_extraction(
            tmp_path / "ignored.pdf",
            timeout_s=0.5,
            worker_command=SLOW_WORKER,
        )
    assert time.monotonic() - started < 30


def test_cancel_stops_and_reaps_the_child(tmp_path: Path) -> None:
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(ExtractionCancelled):
        run_extraction(
            tmp_path / "ignored.pdf",
            timeout_s=60,
            cancel=cancel,
            worker_command=SLOW_WORKER,
        )
    assert time.monotonic() - started < 30


def test_slow_worker_cannot_leave_the_supervisor_hanging_past_the_budget(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(ExtractionTimeout):
        run_extraction(
            tmp_path / "ignored.pdf",
            timeout_s=0.2,
            worker_command=SLOW_WORKER,
        )
    assert time.monotonic() - started < 5


def test_a_retry_after_a_timeout_can_start_a_fresh_child(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "native.pdf", native_text_pdf("Retry me"))
    with pytest.raises(ExtractionTimeout):
        run_extraction(path, timeout_s=0.2, worker_command=SLOW_WORKER)
    run = run_extraction(path, timeout_s=60)
    assert run.pages[0].metrics.extracted_chars == len("Retryme")


def test_completed_result_survives_a_lingering_child(tmp_path: Path) -> None:
    """A `done` result must win even if the child holds stdout open past the deadline."""
    script = (
        "import json,sys,time;"
        "sys.stdout.write(json.dumps({'type':'done','page_count':1,"
        "'parser_name':'fake','parser_version':'9'})+'\\n');"
        "sys.stdout.flush();"
        "time.sleep(30)"
    )
    started = time.monotonic()
    run = run_extraction(
        tmp_path / "ignored.pdf",
        timeout_s=0.5,
        worker_command=[sys.executable, "-c", script],
    )
    assert run.parser_version == "9"
    assert time.monotonic() - started < 10
