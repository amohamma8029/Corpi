"""One supervised transfer per core, durable setup attempts and verified promotion."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from core.persistence.corelock import DataDirectoryLock
from core.persistence.errors import ConflictingWriteError, CoreAlreadyRunningError
from core.persistence.models import new_id, utc_timestamp, validate_id
from core.persistence.store import PersistenceStore
from core.search.handoff import SUPPORTED_PROFILE_IDS
from core.search.profiles import ModelProfile, installation_path, profiles, verify_installation
from core.search.supervisor import stop_owned_process_tree, worker_environment


class ModelSetupService:
    def __init__(
        self,
        store: PersistenceStore,
        *,
        notify=lambda: None,
        worker_command=None,
        timeout_seconds: float = 600,
        idle_seconds: float = 30,
        max_bytes: int = 1_000_000_000,
    ) -> None:
        self.store = store
        self.notify = notify
        self.worker_command = worker_command
        self.timeout_seconds = timeout_seconds
        self.idle_seconds = idle_seconds
        self.max_bytes = max_bytes
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def data_dir(self) -> Path:
        return self.store.database.config.data_dir

    def profile(self, profile_id: str) -> ModelProfile:
        if profile_id not in SUPPORTED_PROFILE_IDS:
            raise ValueError("unsupported_model")
        return profiles()[profile_id]

    def _path(self, relative: str) -> Path:
        parts = PurePosixPath(relative).parts
        if len(parts) != 3 or parts[:2] != ("models", ".setup") or "\\" in relative:
            raise ValueError("unsafe_model_staging")
        validate_id(parts[2])
        current = self.data_dir
        for part in parts:
            current /= part
            if current.is_symlink() or current.is_junction():
                raise ValueError("unsafe_model_staging")
        return current

    def _ensure(self, profile: ModelProfile) -> None:
        manifest = json.dumps(asdict(profile), sort_keys=True)
        with self.store._transaction() as connection:
            connection.execute(
                "INSERT INTO model_manifests VALUES(?,?,?) ON CONFLICT(id) DO NOTHING",
                (profile.identity, profile.id, manifest),
            )
            saved = connection.execute(
                "SELECT manifest_json FROM model_manifests WHERE id=?", (profile.identity,)
            ).fetchone()[0]
            if saved != manifest:
                raise ConflictingWriteError("model manifest identity mismatch")
            connection.execute(
                "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
                " VALUES(?,?,'missing',?) ON CONFLICT(manifest_id) DO NOTHING",
                (profile.identity, profile.id, utc_timestamp()),
            )

    async def start(self) -> None:
        await asyncio.to_thread(self.recover)

    async def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 20)
            if self._thread.is_alive():
                raise RuntimeError("model_setup_writer_did_not_stop")

    def state(self, profile_id: str) -> dict:
        profile = self.profile(profile_id)
        self._ensure(profile)
        with self.store.database.connect() as connection:
            return dict(
                connection.execute(
                    "SELECT * FROM model_installations WHERE manifest_id=?", (profile.identity,)
                ).fetchone()
            )

    def recover(self) -> None:
        for profile_id in SUPPORTED_PROFILE_IDS:
            profile = self.profile(profile_id)
            self._ensure(profile)
            state = self.state(profile_id)
            try:
                verify_installation(installation_path(self.data_dir, profile), profile)
                installed = True
            except (ValueError, OSError):
                installed = False
            with self.store._transaction() as connection:
                connection.execute(
                    "UPDATE model_setup_attempts SET state='interrupted',"
                    "failure_code='model_setup_interrupted',updated_at=?"
                    " WHERE manifest_id=? AND state='running'",
                    (utc_timestamp(), profile.identity),
                )
                if installed and state["state"] != "cancelled":
                    connection.execute(
                        "UPDATE model_installations SET state='installed',failure_code=NULL,"
                        "staging_path=NULL,updated_at=? WHERE manifest_id=?",
                        (utc_timestamp(), profile.identity),
                    )
                    connection.execute(
                        "UPDATE model_setup_attempts SET state='installed',failure_code=NULL"
                        " WHERE manifest_id=? AND generation=? AND state='interrupted'",
                        (profile.identity, state["generation"]),
                    )
                elif state["state"] in ("installed", "downloading"):
                    connection.execute(
                        "UPDATE model_installations SET state='failed',"
                        "failure_code='model_setup_interrupted',updated_at=? WHERE manifest_id=?",
                        (utc_timestamp(), profile.identity),
                    )
        self.cleanup()

    def cleanup(self) -> None:
        with self.store.database.connect() as connection:
            rows = connection.execute(
                "SELECT id,staging_path FROM model_setup_attempts"
                " WHERE cleanup_pending=1 AND state<>'running'"
            ).fetchall()
        for row in rows:
            try:
                path = self._path(row["staging_path"])
                if path.exists():
                    lock = DataDirectoryLock.acquire(path)
                    lock.release()
                    shutil.rmtree(path)
                with self.store._transaction() as connection:
                    connection.execute(
                        "UPDATE model_setup_attempts SET cleanup_pending=0"
                        " WHERE id=? AND state<>'running'",
                        (row["id"],),
                    )
            except (CoreAlreadyRunningError, OSError, ValueError):
                continue  # A surviving writer/failed removal keeps recorded cleanup pending.

    def begin(self, profile_id: str, request_key: str, expected_generation: int) -> dict:
        profile = self.profile(profile_id)
        self._ensure(profile)
        self.cleanup()
        if not request_key.strip() or len(request_key) > 200 or self._stop.is_set():
            raise ValueError("invalid_setup_request")
        with self._mutex:
            with self.store._transaction() as connection:
                previous = connection.execute(
                    "SELECT manifest_id FROM model_setup_attempts WHERE request_key=?",
                    (request_key,),
                ).fetchone()
                if previous is not None:
                    if previous[0] != profile.identity:
                        raise ConflictingWriteError("setup request belongs to a different model")
                    return dict(
                        connection.execute(
                            "SELECT * FROM model_installations WHERE manifest_id=?",
                            (profile.identity,),
                        ).fetchone()
                    )
                state = connection.execute(
                    "SELECT * FROM model_installations WHERE manifest_id=?", (profile.identity,)
                ).fetchone()
                if state["generation"] != expected_generation:
                    raise ConflictingWriteError("model setup generation changed")
                if state["state"] == "installed":
                    return dict(state)
                if self._thread is not None and self._thread.is_alive():
                    raise ConflictingWriteError("another model transfer is active")
                if connection.execute(
                    "SELECT 1 FROM model_setup_attempts"
                    " WHERE cleanup_pending=1 AND manifest_id=? LIMIT 1",
                    (profile.identity,),
                ).fetchone():
                    raise ConflictingWriteError("model setup cleanup is pending")
                generation = expected_generation + 1
                identity = new_id()
                staging = "models/.setup/" + identity
                self._path(staging)
                timestamp = utc_timestamp()
                connection.execute(
                    "INSERT INTO model_setup_attempts(id,request_key,manifest_id,generation,"
                    "state,staging_path,created_at,updated_at)"
                    " VALUES(?,?,?,?,'running',?,?,?)",
                    (
                        identity,
                        request_key,
                        profile.identity,
                        generation,
                        staging,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    "UPDATE model_installations SET state='downloading',generation=?,"
                    "received_bytes=0,spent_seconds=0,transfer_attempts=0,staging_path=?,"
                    "failure_code=NULL,updated_at=? WHERE manifest_id=?",
                    (generation, staging, timestamp, profile.identity),
                )
            self._thread = threading.Thread(
                target=self._run, args=(profile, generation, identity, staging), daemon=True
            )
            self._thread.start()
        return self.state(profile_id)

    def cancel(self, profile_id: str, generation: int) -> dict:
        profile = self.profile(profile_id)
        with self.store._transaction() as connection:
            cursor = connection.execute(
                "UPDATE model_installations SET state='cancelled',generation=generation+1,"
                "failure_code='model_setup_cancelled',updated_at=?"
                " WHERE manifest_id=? AND generation=? AND state='downloading'",
                (utc_timestamp(), profile.identity, generation),
            )
            if cursor.rowcount != 1:
                raise ConflictingWriteError("setup attempt is no longer cancellable")
            connection.execute(
                "UPDATE model_setup_attempts SET state='cancelled',"
                "failure_code='model_setup_cancelled',updated_at=?"
                " WHERE manifest_id=? AND generation=? AND state='running'",
                (utc_timestamp(), profile.identity, generation),
            )
        return self.state(profile_id)

    def _current(self, manifest: str, generation: int) -> bool:
        if self._stop.is_set():
            return False
        with self.store.database.connect() as connection:
            row = connection.execute(
                "SELECT state,generation FROM model_installations WHERE manifest_id=?", (manifest,)
            ).fetchone()
            return row is not None and row[0] == "downloading" and row[1] == generation

    def _run(self, profile: ModelProfile, generation: int, identity: str, relative: str) -> None:
        total_bytes = 0
        started = time.monotonic()
        failure = "model_download_failed"
        try:
            path = self._path(relative)
            for attempt in range(1, 4):
                if not self._current(profile.identity, generation):
                    break
                remaining = self.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0 or total_bytes >= self.max_bytes:
                    failure = "model_download_limit"
                    break
                with self.store._transaction() as connection:
                    connection.execute(
                        "UPDATE model_setup_attempts SET transfer_attempts=?,spent_seconds=?"
                        " WHERE id=? AND state='running'",
                        (attempt, self.timeout_seconds, identity),
                    )
                    connection.execute(
                        "UPDATE model_installations SET transfer_attempts=?,spent_seconds=?"
                        " WHERE manifest_id=? AND generation=? AND state='downloading'",
                        (attempt, self.timeout_seconds, profile.identity, generation),
                    )
                try:
                    base_bytes = total_bytes
                    received = self._transfer(
                        profile,
                        path,
                        generation,
                        remaining,
                        self.max_bytes - total_bytes,
                        lambda count, base_bytes=base_bytes: self._progress(
                            profile.identity, generation, identity, base_bytes + count
                        ),
                    )
                    total_bytes += received
                    verify_installation(path / "assets", profile)
                    final = installation_path(self.data_dir, profile)
                    if final.parent.is_symlink() or final.parent.is_junction():
                        raise ValueError("unsafe_model_directory")
                    final.parent.mkdir(parents=True, exist_ok=True)
                    if final.exists():
                        try:
                            verify_installation(final, profile)
                            final_valid = True
                        except (ValueError, OSError):
                            final_valid = False
                    else:
                        final_valid = False
                    with self.store._transaction() as connection:
                        current = connection.execute(
                            "SELECT state,generation FROM model_installations WHERE manifest_id=?",
                            (profile.identity,),
                        ).fetchone()
                        if (
                            current[0] != "downloading"
                            or current[1] != generation
                            or self._stop.is_set()
                        ):
                            break
                        # Hash verification and waiting for this transaction also
                        # consume the installation attempt's total deadline.
                        if time.monotonic() - started >= self.timeout_seconds:
                            failure = "model_download_limit"
                            break
                        if final.exists() and not final_valid:
                            owned = connection.execute(
                                "SELECT 1 FROM model_setup_attempts"
                                " WHERE manifest_id=? AND state='installed' LIMIT 1",
                                (profile.identity,),
                            ).fetchone()
                            if owned is None or final.is_symlink() or final.is_junction():
                                raise ValueError("model_installation_conflict")
                            final.rename(path / "previous")
                        if not final_valid:
                            (path / "assets").rename(final)
                        connection.execute(
                            "UPDATE model_installations SET state='installed',staging_path=NULL,"
                            "received_bytes=?,spent_seconds=?,failure_code=NULL,updated_at=?"
                            " WHERE manifest_id=?",
                            (
                                total_bytes,
                                time.monotonic() - started,
                                utc_timestamp(),
                                profile.identity,
                            ),
                        )
                        connection.execute(
                            "UPDATE model_setup_attempts SET state='installed',received_bytes=?,"
                            "spent_seconds=?,updated_at=? WHERE id=?",
                            (total_bytes, time.monotonic() - started, utc_timestamp(), identity),
                        )
                    self.notify()
                    return
                except (OSError, ValueError, TimeoutError, RuntimeError):
                    with self.store.database.connect() as connection:
                        saved = connection.execute(
                            "SELECT received_bytes FROM model_setup_attempts WHERE id=?",
                            (identity,),
                        ).fetchone()[0]
                    total_bytes = max(total_bytes, saved)
        finally:
            if self._stop.is_set():
                failure = "model_setup_interrupted"
            with self.store._transaction() as connection:
                connection.execute(
                    "UPDATE model_setup_attempts SET state=CASE WHEN state='running'"
                    " THEN ? ELSE state END,failure_code=COALESCE(failure_code,?),"
                    "spent_seconds=?,updated_at=? WHERE id=? AND state<>'installed'",
                    (
                        "interrupted" if self._stop.is_set() else "failed",
                        failure,
                        time.monotonic() - started,
                        utc_timestamp(),
                        identity,
                    ),
                )
                connection.execute(
                    "UPDATE model_installations SET state='failed',failure_code=?,"
                    "spent_seconds=?,updated_at=? WHERE manifest_id=? AND generation=?"
                    " AND state='downloading'",
                    (
                        failure,
                        time.monotonic() - started,
                        utc_timestamp(),
                        profile.identity,
                        generation,
                    ),
                )
            self.cleanup()
            self.notify()

    def _progress(self, manifest: str, generation: int, identity: str, count: int) -> None:
        if count > self.max_bytes:
            raise ValueError("model_download_limit")
        with self.store._transaction() as connection:
            cursor = connection.execute(
                "UPDATE model_installations SET received_bytes=?,updated_at=?"
                " WHERE manifest_id=? AND generation=? AND state='downloading'",
                (count, utc_timestamp(), manifest, generation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("setup attempt changed")
            connection.execute(
                "UPDATE model_setup_attempts SET received_bytes=?,updated_at=?"
                " WHERE id=? AND state='running'",
                (count, utc_timestamp(), identity),
            )

    def _transfer(self, profile, path, generation, timeout, max_bytes, progress) -> int:
        command = (
            list(self.worker_command)
            if self.worker_command
            else [sys.executable, "-m", "core.search.install_worker"]
        )
        process = subprocess.Popen(
            [*command, profile.id, str(path), str(max_bytes), str(timeout)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=worker_environment(),
        )
        messages: queue.Queue = queue.Queue(maxsize=4)
        done = threading.Event()

        def pump() -> None:
            try:
                assert process.stdout is not None
                while not done.is_set():
                    line = process.stdout.readline(4097)
                    if not line:
                        break
                    payload = json.loads(line) if len(line) <= 4096 else {"type": "error"}
                    while not done.is_set():
                        try:
                            messages.put(payload, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except (ValueError, OSError):
                pass

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        last_bytes = time.monotonic()
        count = 0
        try:
            while self._current(profile.identity, generation):
                now = time.monotonic()
                if now >= deadline or now - last_bytes >= self.idle_seconds:
                    raise TimeoutError("model download deadline")
                try:
                    payload = messages.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None:
                        raise RuntimeError("model download worker exited") from None
                    continue
                kind = payload.get("type")
                if kind in ("progress", "done"):
                    updated = payload["bytes"]
                    if (
                        not isinstance(updated, int)
                        or isinstance(updated, bool)
                        or updated < count
                        or updated > max_bytes
                    ):
                        raise ValueError("invalid download progress")
                    if updated > count:
                        last_bytes = now
                    count = updated
                    progress(count)
                    if kind == "done":
                        return count
                elif kind != "started":
                    raise RuntimeError("model download worker failed")
            raise RuntimeError("model setup stopped")
        finally:
            done.set()
            stop_owned_process_tree(process)
            if process.stdin:
                process.stdin.close()
            reader.join(timeout=2)
            if process.stdout:
                process.stdout.close()
