import os
import socket
import sys
import threading
import time

import uvicorn

from core.api.app import create_app
from core.ingest.locks import FingerprintLockCoordinator
from core.ingest.service import PdfIngestService
from core.jobs.runner import JobRunner
from core.jobs.steps import register_pdf_extract_steps
from core.jobs.sweeper import TrashSweeper
from core.persistence.corelock import DataDirectoryLock
from core.persistence.database import Database, DatabaseConfig, verify_sqlite_runtime
from core.persistence.errors import PersistenceError
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore

READY_MARKER = "CORPI_CORE_READY"
PERSISTENCE_ERROR_MARKER = "CORPI_CORE_ERROR"
WATCH_STDIN_ENV = "CORPI_CORE_WATCH_STDIN"


def pick_free_port() -> int:
    """Ask the OS for a free loopback port.

    The probe-then-bind sequence has a small TOCTOU race; acceptable for a local
    single-user dev tool in v1, documented in the Task.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def exit_when_stdin_closes(server: "uvicorn.Server") -> None:
    """Graceful shutdown when the core's stdin reaches EOF.

    The shell holds the write end of the stdin pipe for as long as the app
    runs; app exit, crash, or an explicit stop closes it. That signal reaches
    this process directly even when process-group kills miss it (uv may regroup
    its children), making shutdown portable rather than Unix-kill-dependent.
    Opt-in via CORPI_CORE_WATCH_STDIN so terminal and test runs are unaffected.
    """
    try:
        sys.stdin.buffer.read()
    except (OSError, ValueError):
        pass
    server.should_exit = True


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

    # One core per data directory, taken BEFORE database initialization so it
    # also covers startup migrations; held until this process stops. The runtime
    # check runs first so an unsupported SQLite never creates the directory.
    try:
        database_config = DatabaseConfig.from_environment()
        verify_sqlite_runtime()
        data_lock = DataDirectoryLock.acquire(database_config.data_dir)
    except PersistenceError as error:
        print(f"{PERSISTENCE_ERROR_MARKER} code={error.code}", flush=True)
        raise SystemExit(2) from None

    try:
        database = Database(database_config)
        try:
            database.initialize()
        except PersistenceError as error:
            print(f"{PERSISTENCE_ERROR_MARKER} code={error.code}", flush=True)
            raise SystemExit(2) from None

        allowed = os.environ.get("CORPI_CORE_ALLOWED_ORIGINS")
        origins = allowed.split(",") if allowed else None

        store = PersistenceStore(database)
        owner_instance_id = f"core-{new_id()}"
        fingerprint_locks = FingerprintLockCoordinator()
        step_sets = register_pdf_extract_steps(store)
        runner = JobRunner(store, step_sets, owner_instance_id=owner_instance_id)
        pdf_ingest = PdfIngestService(
            store,
            owner_instance_id,
            fingerprint_locks=fingerprint_locks,
            notify_jobs=runner.notify,
        )
        sweeper = TrashSweeper(
            store,
            interval_seconds=pdf_ingest.limits.maintenance_interval_seconds,
            fingerprint_locks=fingerprint_locks,
            maintenance=pdf_ingest.maintenance,
        )

        port = pick_free_port()
        app = create_app(
            token=token,
            allowed_origins=origins,
            store=store,
            pdf_ingest=pdf_ingest,
            runner=runner,
            sweeper=sweeper,
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            server_header=False,
        )
        server = uvicorn.Server(config)

        threading.Thread(
            target=print_ready_when_listening, args=(server, port), daemon=True
        ).start()
        if os.environ.get(WATCH_STDIN_ENV) == "1":
            threading.Thread(target=exit_when_stdin_closes, args=(server,), daemon=True).start()
        server.run()
    finally:
        data_lock.release()


if __name__ == "__main__":
    main()
