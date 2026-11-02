"""Opt-in bounded private diagnostics. No text/environment data enters API errors."""

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class WorkerDiagnostics:
    def __init__(self, prefix: Path) -> None:
        self.record: dict[str, Any] = {
            "stderr_observed_bytes": 0,
            "stderr_retained_bytes": 0,
            "stderr_limit_bytes": 1024**2,
            "errors": [],
            "cleanup_complete": False,
        }
        self.failed = threading.Event()
        self.overflow = threading.Event()
        self.thread: threading.Thread | None = None
        self.closed = False
        self.process: subprocess.Popen[bytes] | None = None
        self.stderr = prefix.with_name(prefix.name + "-stderr.txt").open("xb")
        try:
            self.status = prefix.with_name(prefix.name + "-diagnostic.json").open(
                "x", encoding="utf-8"
            )
        except BaseException:
            self.stderr.close()
            raise

    def attach(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.record["redirector_pid"] = process.pid
        self.thread = threading.Thread(target=self.drain, daemon=True)
        self.thread.start()

    def drain(self) -> None:
        try:
            assert self.process is not None and self.process.stderr is not None
            with self.process.stderr as stream:
                while data := stream.read(65536):
                    self.record["stderr_observed_bytes"] += len(data)
                    available = (
                        self.record["stderr_limit_bytes"] - self.record["stderr_retained_bytes"]
                    )
                    retained = min(len(data), max(0, available))
                    self.stderr.write(data[:retained])
                    self.record["stderr_retained_bytes"] += retained
                    if self.record["stderr_observed_bytes"] > self.record["stderr_limit_bytes"]:
                        self.overflow.set()
        except BaseException as error:
            self.note_error("stderr-drain", error)
            self.failed.set()

    def note_error(self, phase: str, error: BaseException) -> None:
        if len(self.record["errors"]) < 16:
            row = {"phase": phase, "type": type(error).__name__}
            code = getattr(error, "code", None)
            if isinstance(code, str) and len(code) <= 64:
                row["safe_code"] = code
            self.record["errors"].append(row)

    def before_stop(self) -> None:
        assert self.process is not None
        errors = self.record["errors"]
        if errors and errors[-1]["type"] == "EmbeddingFailure" and not self.overflow.is_set():
            try:
                self.process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        self.record["redirector_exit_before_stop"] = self.process.poll()
        self.record["forced_stop_needed"] = self.process.poll() is None
        self.record["stop_requested_monotonic"] = time.monotonic()

    def finish(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.record["stderr_reader_did_not_stop"] = True
                self.failed.set()
        self.record["stderr_overflow"] = self.overflow.is_set()
        self.record["redirector_exit_after_stop"] = self.process.poll() if self.process else None
        self.record["cleanup_complete"] = not self.failed.is_set()
        if not self.thread or not self.thread.is_alive():
            self.stderr.close()
        try:
            json.dump(self.record, self.status, indent=2, allow_nan=False)
        finally:
            self.status.close()
        if self.failed.is_set():
            raise RuntimeError("embedding diagnostic cleanup failed")
