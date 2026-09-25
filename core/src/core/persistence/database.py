"""SQLite connection policy, database preflight, and startup initialization."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.persistence.errors import (
    DatabaseBusyError,
    InvalidDataDirectoryError,
    MigrationError,
    PersistenceError,
    UnsupportedSchemaError,
    UnsupportedSQLiteError,
    translate_sqlite_error,
)
from core.persistence.migrations import (
    CORPI_APPLICATION_ID,
    Migration,
    application_id,
    apply_migrations,
    is_bootstrap_database,
    load_migrations,
    validate_database_history,
    validate_migrations,
)

MINIMUM_SQLITE_VERSION = (3, 51, 3)
DATABASE_FILENAME = "corpi.sqlite3"
DATA_DIRECTORY_ENV = "CORPI_DATA_DIR"
LOCK_WAIT_ENV = "CORPI_SQLITE_BUSY_TIMEOUT_MS"
DEFAULT_LOCK_WAIT_MS = 5_000
MAX_LOCK_WAIT_MS = 2_147_483_647


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    data_dir: Path
    lock_wait_ms: int = DEFAULT_LOCK_WAIT_MS

    def __post_init__(self) -> None:
        if not self.data_dir.is_absolute():
            raise InvalidDataDirectoryError("CORPI_DATA_DIR must be an absolute path")
        if (
            not isinstance(self.lock_wait_ms, int)
            or isinstance(self.lock_wait_ms, bool)
            or not 1 <= self.lock_wait_ms <= MAX_LOCK_WAIT_MS
        ):
            raise PersistenceError("SQLite lock wait must be a positive 32-bit millisecond value")

    @classmethod
    def from_environment(cls) -> DatabaseConfig:
        raw_data_dir = os.environ.get(DATA_DIRECTORY_ENV)
        if not raw_data_dir:
            raise InvalidDataDirectoryError("CORPI_DATA_DIR is missing or empty")
        data_dir = Path(raw_data_dir)
        if not data_dir.is_absolute():
            raise InvalidDataDirectoryError("CORPI_DATA_DIR must be an absolute path")

        raw_lock_wait = os.environ.get(LOCK_WAIT_ENV)
        lock_wait_ms = DEFAULT_LOCK_WAIT_MS
        if raw_lock_wait is not None:
            try:
                lock_wait_ms = int(raw_lock_wait)
            except ValueError as error:
                raise PersistenceError("SQLite lock wait configuration is invalid") from error
        return cls(data_dir=data_dir, lock_wait_ms=lock_wait_ms)

    @property
    def database_path(self) -> Path:
        return self.data_dir / DATABASE_FILENAME

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "migration-snapshots"


class Database:
    """An initialized database and the connection factory used by its store."""

    def __init__(
        self,
        config: DatabaseConfig,
        migrations: tuple[Migration, ...] | None = None,
    ) -> None:
        self.config = config
        selected_migrations = migrations if migrations is not None else load_migrations()
        self.migrations = validate_migrations(selected_migrations)

    @property
    def path(self) -> Path:
        return self.config.database_path

    def initialize(self) -> None:
        self._verify_runtime()
        self._ensure_data_directory()
        target = self.path
        new_target = not target.exists() or target.stat().st_size == 0
        if target.exists() and not new_target:
            self._preflight_existing_database()
        elif not target.exists():
            self._create_empty_database_file()

        connection = self._connect(read_only=False)
        try:
            self._configure_connection(connection)
            self._prepare_application_identity(connection, allow_new=new_target)
            current_journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower()
            if current_journal_mode != "wal":
                if not is_bootstrap_database(connection):
                    raise UnsupportedSchemaError("existing Corpi database is not in WAL mode")
                mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
                if mode != "wal":
                    raise UnsupportedSchemaError("SQLite could not enable WAL mode")
            apply_migrations(connection, self.migrations, self.config.snapshot_dir)
            self._verify_connection_settings(connection)
        except (PersistenceError, UnsupportedSchemaError):
            raise
        except sqlite3.Error as error:
            translated = translate_sqlite_error(error)
            if isinstance(translated, DatabaseBusyError):
                raise translated from error
            raise MigrationError("database initialization failed") from error
        finally:
            connection.close()
        self._set_file_permissions(target)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured read/write connection without implicitly creating a file."""
        connection = self._connect(read_only=False)
        try:
            self._verify_connection_settings(connection)
            yield connection
        except sqlite3.Error as error:
            translated = translate_sqlite_error(error)
            raise translated from error
        finally:
            connection.close()

    def _verify_runtime(self) -> None:
        if sqlite3.sqlite_version_info < MINIMUM_SQLITE_VERSION:
            version = ".".join(str(part) for part in sqlite3.sqlite_version_info)
            required = ".".join(str(part) for part in MINIMUM_SQLITE_VERSION)
            raise UnsupportedSQLiteError(
                f"loaded SQLite {version} is older than required {required}"
            )

    def _ensure_data_directory(self) -> None:
        try:
            self.config.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not self.config.data_dir.is_dir():
                raise InvalidDataDirectoryError("CORPI_DATA_DIR is not a directory")
            if os.name != "nt":
                self.config.data_dir.chmod(0o700)
        except PersistenceError:
            raise
        except OSError as error:
            raise InvalidDataDirectoryError(
                "CORPI_DATA_DIR cannot be created or accessed"
            ) from error

    def _create_empty_database_file(self) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            return
        except OSError as error:
            raise MigrationError("database file cannot be created") from error
        else:
            os.close(descriptor)

    def _preflight_existing_database(self) -> None:
        connection = self._connect(read_only=True)
        try:
            mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            bootstrap = is_bootstrap_database(connection)
            if mode != "wal" and not bootstrap:
                raise UnsupportedSchemaError("existing Corpi database is not in WAL mode")
            validate_database_history(connection, self.migrations)
        except PersistenceError:
            raise
        except sqlite3.Error as error:
            raise UnsupportedSchemaError(
                "existing database is unreadable or has invalid history"
            ) from error
        finally:
            connection.close()

    def _prepare_application_identity(
        self,
        connection: sqlite3.Connection,
        allow_new: bool,
    ) -> None:
        identity = application_id(connection)
        if identity == CORPI_APPLICATION_ID:
            return
        if identity != 0 or not allow_new:
            raise UnsupportedSchemaError("database identity does not match Corpi")

        connection.execute("BEGIN IMMEDIATE")
        try:
            if application_id(connection) != 0:
                if application_id(connection) != CORPI_APPLICATION_ID:
                    raise UnsupportedSchemaError("database identity changed during initialization")
            else:
                user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                objects = connection.execute(
                    """SELECT 1 FROM sqlite_schema
                       WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' LIMIT 1"""
                ).fetchone()
                if user_version != 0 or objects is not None:
                    raise UnsupportedSchemaError(
                        "unidentified non-empty database cannot be initialized"
                    )
                connection.execute(f"PRAGMA application_id = {CORPI_APPLICATION_ID}")
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _connect(self, read_only: bool) -> sqlite3.Connection:
        try:
            if read_only:
                uri = f"{self.path.as_uri()}?mode=ro"
                connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=self.config.lock_wait_ms / 1000,
                    isolation_level=None,
                )
            else:
                uri = f"{self.path.as_uri()}?mode=rw"
                connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=self.config.lock_wait_ms / 1000,
                    isolation_level=None,
                )
        except sqlite3.Error as error:
            if read_only:
                raise UnsupportedSchemaError("existing database is unreadable") from error
            translated = translate_sqlite_error(error)
            if isinstance(translated, DatabaseBusyError):
                raise translated from error
            raise MigrationError("database could not be opened") from error
        connection.row_factory = sqlite3.Row
        return connection

    def _configure_connection(self, connection: sqlite3.Connection) -> None:
        connection.execute(f"PRAGMA busy_timeout = {self.config.lock_wait_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        if int(connection.execute("PRAGMA busy_timeout").fetchone()[0]) != self.config.lock_wait_ms:
            raise MigrationError("SQLite did not apply the configured lock wait")
        if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise MigrationError("SQLite foreign-key enforcement is unavailable")
        if int(connection.execute("PRAGMA synchronous").fetchone()[0]) != 2:
            raise MigrationError("SQLite full synchronization is unavailable")

    def _verify_connection_settings(self, connection: sqlite3.Connection) -> None:
        self._configure_connection(connection)
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            raise MigrationError("SQLite journal mode changed unexpectedly")

    @staticmethod
    def _set_file_permissions(path: Path) -> None:
        if os.name != "nt":
            try:
                path.chmod(0o600)
            except OSError as error:
                raise MigrationError("database file permissions could not be protected") from error
