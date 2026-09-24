import os
import socket
import sys
import threading
import time

import uvicorn

from core.api.app import create_app

READY_MARKER = "CORPI_CORE_READY"


def pick_free_port() -> int:
    """Ask the OS for a free loopback port.

    The probe-then-bind sequence has a small TOCTOU race; acceptable for a local
    single-user dev tool in v1, documented in the Task.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def print_ready_when_listening(server: "uvicorn.Server", port: int) -> None:
    """Print READY only once the socket is accepting connections.

    uvicorn sets ``Server.started`` after binding and running the app's startup
    hooks, so the marker reaching stdout guarantees an immediate health check can
    succeed. Startup failure exits the process; this daemon thread dies with it
    and the shell's timeout handling takes over.
    """
    while not server.started:
        time.sleep(0.02)
    print(f"{READY_MARKER} port={port} pid={os.getpid()}", flush=True)


def main() -> None:
    token = os.environ.get("CORPI_CORE_TOKEN")
    if not token:
        print("CORPI_CORE_ERROR missing CORPI_CORE_TOKEN", file=sys.stderr, flush=True)
        raise SystemExit(2)

    allowed = os.environ.get("CORPI_CORE_ALLOWED_ORIGINS")
    origins = allowed.split(",") if allowed else None

    port = pick_free_port()
    app = create_app(token=token, allowed_origins=origins)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        server_header=False,
    )
    server = uvicorn.Server(config)

    threading.Thread(target=print_ready_when_listening, args=(server, port), daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
