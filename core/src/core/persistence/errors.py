"""Sanitized persistence failures with stable startup error codes."""

from __future__ import annotations

import sqlite3


class PersistenceError(Exception):
    code = "migration_failed"


class InvalidDataDirectoryError(PersistenceError):
    code = "invalid_data_dir"


class UnsupportedSQLiteError(PersistenceError):
    code = "unsupported_sqlite"


class UnsupportedSchemaError(PersistenceError):
    code = "unsupported_schema"


class DatabaseBusyError(PersistenceError):
    code = "database_busy"


class MigrationError(PersistenceError):
    code = "migration_failed"


class IntegrityViolationError(PersistenceError):
    code = "migration_failed"


class ConflictingWriteError(PersistenceError):
    code = "migration_failed"


def translate_sqlite_error(error: sqlite3.Error) -> PersistenceError:
    """Return a stable error without copying SQLite's SQL-bearing message."""
    if isinstance(error, sqlite3.IntegrityError):
        return IntegrityViolationError("stored records violate a database constraint")
    if isinstance(error, sqlite3.OperationalError):
        message = str(error).lower()
        if "locked" in message or "busy" in message:
            return DatabaseBusyError("database remained busy until the lock wait expired")
    return PersistenceError("database operation failed")
