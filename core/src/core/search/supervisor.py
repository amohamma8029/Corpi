"""Bounded reusable inference session, terminated/reaped before retry or cleanup."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from core.search.memory import resident_bytes
from core.search.profiles import ModelProfile
from core.search.protocol import MAX_MESSAGE_BYTES, WORKER_ERROR_CODES
from core.search.provenance import validate_runtime_versions
from core.search.storage import validate_vector
from core.search.worker_diagnostics import WorkerDiagnostics


class OpenBlasOptions(TypedDict, total=False):
    openblas_threads: int


def openblas_options(settings: Mapping[str, object]) -> OpenBlasOptions:
    """Absent saved settings retain legacy workers; only the measured option is valid."""
    if "openblas_threads" not in settings:
        return {}
    value = settings["openblas_threads"]
    if type(value) is not int or value != 1:
        raise ValueError("unsupported_worker_settings")
    return {"openblas_threads": 1}


def worker_environment(*, openblas_threads: int | None = None) -> dict[str, str]:
    keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL", "TZ"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in keep}
    environment.update(
        {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"}
    )
    if openblas_threads is not None:
        openblas_options({"openblas_threads": openblas_threads})
        environment["OPENBLAS_NUM_THREADS"] = "1"
    return environment


def stop_owned_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is None:
        if sys.platform == "win32":
            subprocess.run(
                [
                    os.path.join(os.environ["SYSTEMROOT"], "System32", "taskkill.exe"),
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    else:
        process.wait()


class EmbeddingStopped(Exception):
    pass


class EmbeddingFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EmbeddingSession:
    def __init__(
        self,
        path: Path,
        profile: ModelProfile,
        *,
        should_stop: Callable[[], bool] = lambda: False,
        load_timeout: float = 120,
        memory_limit: int = 1024 * 1024 * 1024,
        command: Sequence[str] | None = None,
        diagnostic_prefix: Path | None = None,
        memory_arena: bool = True,
        stop_spinning: bool = True,
        openblas_threads: int | None = None,
    ) -> None:
        self.profile = profile
        self.should_stop = should_stop
        self.memory_limit = memory_limit
        self.peak_bytes = 0
        self._next_sample = 0.0
        self._request = 0
        self.worker_pid: int | None = None
        self._closed = False
        self._diagnostic = (
            WorkerDiagnostics(diagnostic_prefix) if diagnostic_prefix is not None else None
        )
        # Source tools get only OS/runtime essentials, never core/provider credentials.
        environment = {
            **worker_environment(openblas_threads=openblas_threads),
            "HF_HUB_OFFLINE": "1",
        }
        worker = (
            list(command) if command is not None else [sys.executable, "-m", "core.search.worker"]
        )
        try:
            self.process = subprocess.Popen(
                [*worker, profile.id, str(path), str(int(memory_arena)), str(int(stop_spinning))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if self._diagnostic else subprocess.DEVNULL,
                env=environment,
            )
        except BaseException as error:
            if self._diagnostic:
                self._diagnostic.note_error("spawn", error)
                try:
                    self._diagnostic.finish()
                except Exception as cleanup:
                    error.add_note("diagnostic cleanup failure: " + type(cleanup).__name__)
            raise
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=2)
        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()
        start = time.monotonic()
        try:
            if self._diagnostic:
                self._diagnostic.attach(self.process)
            deadline = start + load_timeout
            started = self._receive(deadline)
            if (
                started.get("type") != "started"
                or started.get("manifest") != profile.identity
                or not isinstance(started.get("pid"), int)
                or started["pid"] <= 0
            ):
                raise EmbeddingFailure("embedding_protocol_error")
            # Windows venv python.exe is a small redirector process. The actual
            # interpreter reports its PID before loading any model weights.
            self.worker_pid = started["pid"]
            self._next_sample = 0
            ready = self._receive(deadline)
            runtime = ready.pop("runtime", None)
            if ready != {"type": "ready", "manifest": profile.identity, "request": 0}:
                raise EmbeddingFailure("embedding_protocol_error")
            try:
                self.execution_runtime = validate_runtime_versions(runtime)
            except ValueError:
                raise EmbeddingFailure("runtime_mismatch") from None
        except BaseException as error:
            if self._diagnostic:
                self._diagnostic.note_error("load", error)
            try:
                self.close()
            except Exception as cleanup:
                if not self._diagnostic:
                    raise
                error.add_note("diagnostic cleanup failure: " + type(cleanup).__name__)
            raise
        self.load_seconds = time.monotonic() - start

    def _pump(self) -> None:
        try:
            assert self.process.stdout is not None
            while not self._closed:
                line = self.process.stdout.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_MESSAGE_BYTES:
                    payload = {"type": "error", "code": "embedding_message_limit"}
                else:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("invalid response")
                while not self._closed:
                    try:
                        self._responses.put(payload, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except (ValueError, OSError):
            if not self._closed:
                self._responses.put_nowait({"type": "error", "code": "embedding_protocol_error"})
        finally:
            try:
                self._responses.put_nowait(None)
            except queue.Full:
                pass

    def _receive(self, deadline: float) -> dict[str, Any]:
        while True:
            if self._diagnostic and (
                self._diagnostic.failed.is_set() or self._diagnostic.overflow.is_set()
            ):
                raise EmbeddingFailure("embedding_worker_failed")
            if self._closed or self.should_stop():
                raise EmbeddingStopped("embedding stopped")
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("embedding attempt deadline")
            if now >= self._next_sample and self.process.poll() is None:
                self._next_sample = now + 0.25
                try:
                    observed = resident_bytes(self.worker_pid or self.process.pid)
                except OSError as error:
                    if self.process.poll() is None:
                        raise EmbeddingFailure("embedding_memory_unavailable") from error
                else:
                    self.peak_bytes = max(self.peak_bytes, observed)
                    if observed > self.memory_limit:
                        raise EmbeddingFailure("embedding_memory_limit")
            try:
                payload = self._responses.get(timeout=min(0.05, deadline - now))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise EmbeddingFailure("embedding_worker_failed") from None
                continue
            if payload is None:
                raise EmbeddingFailure("embedding_worker_failed")
            if payload.get("type") == "error":
                code = payload.get("code")
                # The pump's own parse error is safe too. Never stringify an
                # unknown child value: it may contain source/query/native data.
                if not isinstance(code, str) or (
                    code not in WORKER_ERROR_CODES and code != "embedding_protocol_error"
                ):
                    code = "embedding_worker_failed"
                raise EmbeddingFailure(code)
            return payload

    def embed(
        self, texts: Sequence[str], *, query: bool = False, timeout: float = 120
    ) -> list[bytes]:
        if not 1 <= len(texts) <= 32:
            raise ValueError("invalid_embedding_batch")
        self._request += 1
        message = (
            json.dumps({"request": self._request, "texts": list(texts), "query": query}).encode()
            + b"\n"
        )
        if len(message) > MAX_MESSAGE_BYTES:
            raise EmbeddingFailure("embedding_message_limit")
        write_error = []

        def write() -> None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(message)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                write_error.append(error)

        sender = threading.Thread(target=write, daemon=True)
        sender.start()
        try:
            result = self._receive(time.monotonic() + timeout)
            if (
                write_error
                or result.get("type") != "result"
                or result.get("request") != self._request
            ):
                raise EmbeddingFailure("embedding_protocol_error")
            encoded = result.get("vectors")
            if not isinstance(encoded, list) or not all(
                isinstance(value, str) for value in encoded
            ):
                raise EmbeddingFailure("embedding_protocol_error")
            if len(encoded) != len(texts):
                raise EmbeddingFailure("invalid_vector_count")
            try:
                vectors = [bytes.fromhex(value) for value in encoded]
            except ValueError:
                raise EmbeddingFailure("embedding_protocol_error") from None
            for vector in vectors:
                try:
                    validate_vector(vector, self.profile.dimensions)
                except ValueError:
                    raise EmbeddingFailure("invalid_vector") from None
            return vectors
        except BaseException as error:
            if self._diagnostic:
                self._diagnostic.note_error("embed", error)
            try:
                self.close()
            except Exception as cleanup:
                if not self._diagnostic:
                    raise
                error.add_note("diagnostic cleanup failure: " + type(cleanup).__name__)
            raise
        finally:
            sender.join(timeout=1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._diagnostic:
            self._diagnostic.before_stop()
        try:
            stop_owned_process_tree(self.process)
            if self.process.stdin is not None:
                self.process.stdin.close()
            self._pump_thread.join(timeout=2)
            if self.process.stdout is not None:
                self.process.stdout.close()
        finally:
            if self._diagnostic:
                self._diagnostic.finish()

    def __enter__(self) -> EmbeddingSession:
        return self

    def __exit__(self, *_) -> None:
        self.close()
