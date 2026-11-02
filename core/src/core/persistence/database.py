"""SQLite connection policy, database preflight, and startup initialization."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

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


def verify_sqlite_runtime() -> None:
    """Fail before any filesystem effect when the loaded SQLite is too old.

    The server runs this before acquiring the data-directory lock, so an
    unsupported runtime never creates the data directory; `Database.initialize`
    also calls it to keep standalone use safe.
    """
    if sqlite3.sqlite_version_info < MINIMUM_SQLITE_VERSION:
        version = ".".join(str(part) for part in sqlite3.sqlite_version_info)
        required = ".".join(str(part) for part in MINIMUM_SQLITE_VERSION)
        raise UnsupportedSQLiteError(f"loaded SQLite {version} is older than required {required}")


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


class _ScopedConnection(sqlite3.Connection):
    """Attempt-local connection; borrowed cursors cannot keep locks between calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.borrowed = True
        self.owner = (os.getpid(), threading.get_ident())
        self.cursors: weakref.WeakSet[sqlite3.Cursor] = weakref.WeakSet()

    def _check_borrow(self) -> None:
        if not self.borrowed or self.owner != (os.getpid(), threading.get_ident()):
            raise PersistenceError("scoped connection is not borrowed by its owner")

    def cursor(self, factory: Any = sqlite3.Cursor):
        self._check_borrow()
        cursor = super().cursor(factory)
        self.cursors.add(cursor)
        return cursor

    def execute(self, sql, parameters=(), /):
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql, parameters, /):
        return self.cursor().executemany(sql, parameters)

    def executescript(self, sql, /):
        return self.cursor().executescript(sql)

    def commit(self) -> None:
        self._check_borrow()
        super().commit()

    def rollback(self) -> None:
        self._check_borrow()
        super().rollback()

    def blobopen(self, *args, **kwargs):
        raise PersistenceError("incremental blobs are not supported by a connection scope")

    def release_cursors(self) -> None:
        for cursor in tuple(self.cursors):
            cursor.close()
        self.cursors.clear()


@dataclass
class _ConnectionScope:
    owner: tuple[int, int]
    config: DatabaseConfig
    file_identity: tuple[int, int]
    resolved_path: str
    connection: _ScopedConnection | None = None
    borrowed: bool = False
    failed: bool = False


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
        self._connections = threading.local()

    @property
    def path(self) -> Path:
        return self.config.database_path

    def initialize(self) -> None:
        if getattr(self._connections, "scope", None) is not None:
            raise PersistenceError("cannot initialize inside a connection scope")
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
        scope = getattr(self._connections, "scope", None)
        if scope is not None:
            with self._borrow_connection(scope) as connection:
                yield connection
            return
        connection = self._connect(read_only=False)
        try:
            self._verify_connection_settings(connection)
            yield connection
        except sqlite3.Error as error:
            translated = translate_sqlite_error(error)
            raise translated from error
        finally:
            connection.close()

    @contextmanager
    def connection_scope(self) -> Iterator[None]:
        """Reuse only in this thread/attempt; callers still own separate transactions.

        Every borrow checks policy and file identity, and closes all its cursors.
        Policy failure invalidates the scope; application errors discard its connection.
        No lock or borrowed handle is allowed between calls, including inference.
        Other threads and callers outside this opt-in retain fresh connections.
        """
        if getattr(self._connections, "scope", None) is not None:
            raise PersistenceError("nested connection scopes are not supported")
        self._verify_runtime()
        scope = _ConnectionScope(
            (os.getpid(), threading.get_ident()),
            self.config,
            self._file_identity(),
            str(self.path.resolve()),
        )
        self._connections.scope = scope
        try:
            yield
        finally:
            if scope.connection is not None:
                scope.connection.close()
            del self._connections.scope

    def _file_identity(self) -> tuple[int, int]:
        try:
            info = self.path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise PersistenceError("scoped database file is missing or redirected")
            return info.st_dev, info.st_ino
        except OSError as error:
            raise PersistenceError("scoped database file is unavailable") from error

    @contextmanager
    def _borrow_connection(self, scope: _ConnectionScope) -> Iterator[sqlite3.Connection]:
        if scope.owner != (os.getpid(), threading.get_ident()) or scope.failed:
            raise PersistenceError("connection scope owner changed or scope is invalid")
        if scope.borrowed:
            raise PersistenceError("nested connection borrows are not supported")
        scope.borrowed = True
        inside_body = False
        try:
            if self.config != scope.config or self._file_identity() != scope.file_identity:
                raise PersistenceError("scoped database identity or configuration changed")
            if scope.connection is None:
                connection = self._connect(read_only=False)
                assert isinstance(connection, _ScopedConnection)
                scope.connection = connection
                self._verify_connection_settings(connection)
            connection = scope.connection
            connection.borrowed = True
            self._verify_scoped_settings(connection, scope)
            inside_body = True
            yield connection
            inside_body = False
            connection.release_cursors()
            self._verify_scoped_settings(connection, scope)
        except BaseException as error:
            scope.failed = not inside_body
            if scope.connection is not None:
                scope.connection.close()
                scope.connection = None
            if isinstance(error, sqlite3.Error):
                raise translate_sqlite_error(error) from error
            raise
        finally:
            if scope.connection is not None:
                scope.connection.release_cursors()
                scope.connection.borrowed = False
            scope.borrowed = False

    def _verify_scoped_settings(
        self, connection: _ScopedConnection, scope: _ConnectionScope
    ) -> None:
        self._verify_runtime()
        if self.config != scope.config or self._file_identity() != scope.file_identity:
            raise PersistenceError("scoped database identity or configuration changed")
        if (
            connection.in_transaction
            or connection.isolation_level is not None
            or connection.row_factory is not sqlite3.Row
            or connection.getconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION)
        ):
            raise PersistenceError("scoped connection has unsafe transaction or runtime state")
        row = connection.execute(
            "SELECT * FROM pragma_busy_timeout, pragma_foreign_keys, pragma_synchronous,"
            " pragma_journal_mode, pragma_application_id, pragma_user_version"
        ).fetchone()
        if (
            tuple(row)
            != (self.config.lock_wait_ms, 1, 2, "wal", CORPI_APPLICATION_ID, len(self.migrations))
            or connection.execute("SELECT vec_version()").fetchone()[0] != "v0.1.9"
        ):
            raise MigrationError("scoped connection policy or database identity changed")
        databases = connection.execute("PRAGMA database_list").fetchall()
        # The pathname is fixed for this scope. File identity and config are still
        # checked on every borrow; resolving the same path again adds no new proof.
        if [(row[1], row[2]) for row in databases] != [("main", scope.resolved_path)]:
            raise PersistenceError("scoped connection has unexpected attached databases")

    def _verify_runtime(self) -> None:
        verify_sqlite_runtime()

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
        factory = (
            _ScopedConnection
            if getattr(self._connections, "scope", None) is not None
            else sqlite3.Connection
        )
        try:
            if read_only:
                uri = f"{self.path.as_uri()}?mode=ro"
                connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=self.config.lock_wait_ms / 1000,
                    isolation_level=None,
                    factory=factory,
                )
            else:
                uri = f"{self.path.as_uri()}?mode=rw"
                connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=self.config.lock_wait_ms / 1000,
                    isolation_level=None,
                    factory=factory,
                )
        except sqlite3.Error as error:
            if read_only:
                raise UnsupportedSchemaError("existing database is unreadable") from error
            translated = translate_sqlite_error(error)
            if isinstance(translated, DatabaseBusyError):
                raise translated from error
            raise MigrationError("database could not be opened") from error
        connection.row_factory = sqlite3.Row
        try:
            try:
                connection.enable_load_extension(True)
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
        except (sqlite3.Error, AttributeError, OSError) as error:
            connection.close()
            raise MigrationError("required sqlite-vec runtime could not be loaded") from error
        return connection

    def _configure_connection(self, connection: sqlite3.Connection) -> None:
        if connection.execute("SELECT vec_version()").fetchone()[0] != "v0.1.9":
            raise MigrationError("required sqlite-vec runtime is unavailable")
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
