"""Checked-in SQL migrations and the transaction-safe migration runner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from core.persistence.errors import (
    DatabaseBusyError,
    MigrationError,
    UnsupportedSchemaError,
    translate_sqlite_error,
)
from core.persistence.models import utc_timestamp

CORPI_APPLICATION_ID = 0x434F5250
_MIGRATION_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
_HISTORY_TABLE = "schema_migrations"
_HISTORY_SQL = """CREATE TABLE schema_migrations (
    revision INTEGER PRIMARY KEY CHECK (revision > 0),
    name TEXT NOT NULL UNIQUE CHECK (length(name) > 0),
    checksum_sha256 TEXT NOT NULL CHECK (
        length(checksum_sha256) = 64 AND checksum_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    schema_sha256 TEXT NOT NULL CHECK (
        length(schema_sha256) = 64 AND schema_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    applied_at TEXT NOT NULL CHECK (
        length(applied_at) = 27 AND applied_at GLOB '????-??-??T??:??:??.??????Z'
    )
) STRICT"""


@dataclass(frozen=True, slots=True)
class Migration:
    revision: int
    name: str
    sql: str
    requires_backup: bool = False

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def load_migrations() -> tuple[Migration, ...]:
    package = files(__package__ or "core.persistence.migrations")
    migrations: list[Migration] = []
    for resource in package.iterdir():
        match = _MIGRATION_NAME.fullmatch(resource.name)
        if match:
            migrations.append(
                Migration(int(match.group(1)), resource.name, resource.read_text(encoding="utf-8"))
            )
    return validate_migrations(migrations)


def validate_migrations(migrations: Iterable[Migration]) -> tuple[Migration, ...]:
    ordered = tuple(sorted(migrations, key=lambda item: item.revision))
    if not ordered:
        raise MigrationError("no database migrations are available")
    revisions = [migration.revision for migration in ordered]
    if revisions != list(range(1, len(revisions) + 1)):
        raise MigrationError("database migration revisions must start at 1 and be contiguous")
    if len({migration.name for migration in ordered}) != len(ordered):
        raise MigrationError("database migration names must be unique")
    return ordered


def schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """SELECT type, name, tbl_name, sql
           FROM sqlite_schema
           WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
           ORDER BY type, name"""
    ).fetchall()
    canonical = json.dumps(
        [tuple(row) for row in rows],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def execute_sql_statements(connection: sqlite3.Connection, sql: str) -> None:
    """Execute complete SQLite statements without splitting trigger bodies."""
    buffer: list[str] = []
    for character in sql:
        buffer.append(character)
        if character == ";" and sqlite3.complete_statement("".join(buffer)):
            statement = "".join(buffer).strip()
            if statement:
                connection.execute(statement)
            buffer.clear()
    remainder = "".join(buffer)
    if remainder.strip() and any(
        line.strip() and not line.lstrip().startswith("--") for line in remainder.splitlines()
    ):
        raise MigrationError("migration contains an incomplete SQL statement")


def application_id(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA application_id").fetchone()[0])


def user_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def schema_objects(connection: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
    rows = connection.execute(
        """SELECT type, name, tbl_name, sql
           FROM sqlite_schema
           WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
           ORDER BY type, name"""
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def is_bootstrap_database(connection: sqlite3.Connection) -> bool:
    if application_id(connection) != CORPI_APPLICATION_ID or user_version(connection) != 0:
        return False
    objects = schema_objects(connection)
    if not objects:
        return True
    return (
        len(objects) == 1
        and objects[0][:3] == ("table", _HISTORY_TABLE, _HISTORY_TABLE)
        and _normalize_sql(objects[0][3]) == _normalize_sql(_HISTORY_SQL)
        and _empty_history_table(connection)
    )


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split()).lower()


def _empty_history_table(connection: sqlite3.Connection) -> bool:
    try:
        row = connection.execute(f"SELECT 1 FROM {_HISTORY_TABLE} LIMIT 1").fetchone()
    except sqlite3.Error:
        return False
    return row is None


def validate_database_history(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...],
) -> int:
    if application_id(connection) != CORPI_APPLICATION_ID:
        raise UnsupportedSchemaError("database identity does not match Corpi")
    revision = user_version(connection)
    if revision < 0 or revision > len(migrations):
        raise UnsupportedSchemaError("database schema revision is not supported")
    has_history = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (_HISTORY_TABLE,)
    ).fetchone()
    if revision == 0 and not has_history:
        if is_bootstrap_database(connection):
            return 0
        raise UnsupportedSchemaError("database migration history is missing")
    if not has_history:
        raise UnsupportedSchemaError("database migration history is missing")

    history = connection.execute(
        f"""SELECT revision, name, checksum_sha256, schema_sha256
            FROM {_HISTORY_TABLE} ORDER BY revision"""
    ).fetchall()
    if len(history) != revision:
        raise UnsupportedSchemaError("database migration history is incomplete")
    for expected, row in zip(migrations[:revision], history, strict=True):
        if row[0] != expected.revision or row[1] != expected.name or row[2] != expected.checksum:
            raise UnsupportedSchemaError(
                "database migration history does not match installed migrations"
            )
    if revision:
        if schema_fingerprint(connection) != history[-1][3]:
            raise UnsupportedSchemaError("database schema does not match its migration history")
    elif not _empty_history_table(connection):
        raise UnsupportedSchemaError("database migration history is inconsistent")
    return revision


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...],
    snapshot_dir: Path,
) -> None:
    """Apply pending revisions one transaction at a time under SQLite's writer lock."""
    try:
        _ensure_history_table(connection, migrations)
        while True:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = validate_database_history(connection, migrations)
                if current >= len(migrations):
                    connection.execute("COMMIT")
                    break
                migration = migrations[current]
                if migration.requires_backup:
                    connection.execute("ROLLBACK")
                    _make_recovery_snapshot(connection, snapshot_dir, current)
                    connection.execute("BEGIN IMMEDIATE")
                    after_snapshot = validate_database_history(connection, migrations)
                    if after_snapshot != current:
                        connection.execute("COMMIT")
                        _discard_snapshot(snapshot_dir, current)
                        continue
                if current != migration.revision - 1:
                    raise UnsupportedSchemaError("database revision changed unexpectedly")
                execute_sql_statements(connection, migration.sql)
                resulting_fingerprint = schema_fingerprint(connection)
                connection.execute(
                    f"""INSERT INTO {_HISTORY_TABLE}
                        (revision, name, checksum_sha256, schema_sha256, applied_at)
                        VALUES (?, ?, ?, ?, ?)""",
                    (
                        migration.revision,
                        migration.name,
                        migration.checksum,
                        resulting_fingerprint,
                        utc_timestamp(),
                    ),
                )
                connection.execute(f"PRAGMA user_version = {migration.revision}")
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        foreign_key_error = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_error is not None:
            raise MigrationError("database contains an invalid foreign-key reference")
    except (DatabaseBusyError, UnsupportedSchemaError, MigrationError):
        raise
    except sqlite3.Error as error:
        translated = translate_sqlite_error(error)
        if isinstance(translated, DatabaseBusyError):
            raise translated from error
        raise MigrationError("database migration failed") from error
    except Exception as error:
        raise MigrationError("database migration failed") from error


def _ensure_history_table(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...],
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        if application_id(connection) != CORPI_APPLICATION_ID or user_version(connection) != 0:
            validate_database_history(connection, migrations)
        exists = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (_HISTORY_TABLE,)
        ).fetchone()
        if exists is None:
            if not is_bootstrap_database(connection):
                raise UnsupportedSchemaError("database migration history is missing")
            connection.execute(_HISTORY_SQL)
        elif not is_bootstrap_database(connection) and user_version(connection) == 0:
            raise UnsupportedSchemaError("empty migration history is not a valid bootstrap state")
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _make_recovery_snapshot(
    connection: sqlite3.Connection,
    snapshot_dir: Path,
    revision: int,
) -> Path:
    snapshot_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        snapshot_dir.chmod(0o700)
    except OSError:
        pass
    destination_path = snapshot_dir / f"snapshot-r{revision:04d}.sqlite3"
    temporary_path = snapshot_dir / f"snapshot-r{revision:04d}.sqlite3.pending"
    temporary_path.unlink(missing_ok=True)
    descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(descriptor)
    destination_connection = sqlite3.connect(
        f"{temporary_path.as_uri()}?mode=rw",
        uri=True,
    )
    try:
        connection.backup(destination_connection)
    except sqlite3.Error as error:
        destination_connection.close()
        temporary_path.unlink(missing_ok=True)
        translated = translate_sqlite_error(error)
        if isinstance(translated, DatabaseBusyError):
            raise translated from error
        raise MigrationError("database recovery snapshot failed") from error
    else:
        destination_connection.close()
    try:
        temporary_path.chmod(0o600)
    except OSError:
        pass
    temporary_path.replace(destination_path)
    for existing in snapshot_dir.glob("snapshot-r*.sqlite3"):
        if existing != destination_path:
            existing.unlink(missing_ok=True)
    return destination_path


def _discard_snapshot(snapshot_dir: Path, revision: int) -> None:
    (snapshot_dir / f"snapshot-r{revision:04d}.sqlite3").unlink(missing_ok=True)
