"""Asset-only download child; no database or generation/core credentials."""

import json
import os
import sys
import threading
from pathlib import Path

from core.persistence.corelock import DataDirectoryLock
from core.search.download import download_profile
from core.search.profiles import profiles


def emit(payload: dict) -> None:
    sys.stdout.buffer.write(json.dumps(payload).encode() + b"\n")
    sys.stdout.buffer.flush()


def main() -> int:
    stop = threading.Event()

    def watch_parent() -> None:
        sys.stdin.buffer.read(1)
        stop.set()

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        profile = profiles()[sys.argv[1]]
        path = Path(sys.argv[2])
        with DataDirectoryLock.acquire(path):
            emit({"type": "started", "pid": os.getpid()})
            received = download_profile(
                path / "assets",
                profile,
                max_bytes=int(sys.argv[3]),
                timeout_seconds=float(sys.argv[4]),
                on_progress=lambda count: emit({"type": "progress", "bytes": count}),
                should_stop=stop.is_set,
            )
        emit({"type": "done", "bytes": received})
        return 0
    except Exception:
        emit({"type": "error", "code": "model_download_failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
