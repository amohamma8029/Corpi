"""Local inference protocol: no database access, credentials, or implicit downloads."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from core.search.profiles import profiles
from core.search.protocol import MAX_MESSAGE_BYTES, WORKER_ERROR_CODES
from core.search.provenance import runtime_versions
from core.search.runtime import EmbeddingRuntime


def emit(payload: dict) -> None:
    sys.stdout.buffer.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    sys.stdout.buffer.flush()


def main() -> int:
    try:
        profile = profiles()[sys.argv[1]]
        emit({"type": "started", "manifest": profile.identity, "request": 0, "pid": os.getpid()})
        with contextlib.redirect_stdout(sys.stderr):
            options = sys.argv[3:5]
            if options and (len(options) != 2 or any(value not in {"0", "1"} for value in options)):
                raise ValueError("invalid_embedding_input")
            runtime = EmbeddingRuntime(
                Path(sys.argv[2]),
                profile,
                memory_arena=not options or options[0] == "1",
                stop_spinning=not options or options[1] == "1",
            )
        emit(
            {
                "type": "ready",
                "manifest": profile.identity,
                "request": 0,
                "runtime": runtime_versions(),
            }
        )
        while True:
            line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                return 0
            if len(line) > MAX_MESSAGE_BYTES:
                raise ValueError("embedding_message_limit")
            request = json.loads(line)
            texts = request["texts"]
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError("invalid_embedding_input")
            with contextlib.redirect_stdout(sys.stderr):
                vectors = runtime.embed(texts, query=request.get("query", False))
            emit(
                {
                    "type": "result",
                    "request": request["request"],
                    "vectors": [vector.hex() for vector in vectors],
                }
            )
    except Exception as error:
        # Never echo text, tokenizer/native error payloads, or environment data.
        safe = str(error) if isinstance(error, ValueError) else "embedding_worker_failed"
        emit(
            {
                "type": "error",
                "code": safe if safe in WORKER_ERROR_CODES else "embedding_worker_failed",
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
