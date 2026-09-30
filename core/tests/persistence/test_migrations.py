import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from core.persistence.database import Database
from core.persistence.errors import MigrationError, UnsupportedSchemaError
from core.persistence.migrations import (
    CORPI_APPLICATION_ID,
    Migration,
    load_migrations,
    schema_fingerprint,
)
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore

CORE_ROOT = Path(__file__).resolve().parents[2]


def notebook_exists(connection: sqlite3.Connection, notebook_id: str) -> bool:
    return (
        connection.execute("SELECT 1 FROM notebooks WHERE id=?", (notebook_id,)).fetchone()
        is not None
    )


def migration_two(*, requires_backup: bool = False) -> Migration:
    return Migration(
        5,
        "0005_add_notebook_color.sql",
        "ALTER TABLE notebooks ADD COLUMN color TEXT NOT NULL DEFAULT 'neutral';",
        requires_backup,
    )


def test_application_id_and_schema_fingerprint_are_stable(database: Database) -> None:
    with database.connect() as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x434F5250
        assert CORPI_APPLICATION_ID == 0x434F5250
        assert schema_fingerprint(connection) == (
            "029fdabd87f7192d0aad9bfc620f229e2d08b915c3f6760915ad9baa4d305555"
        )


def test_upgrade_preserves_existing_record_identity(database: Database) -> None:
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Kept across upgrade")
    upgraded = Database(
        database.config,
        migrations=load_migrations() + (migration_two(),),
    )

    upgraded.initialize()

    assert PersistenceStore(upgraded).load_notebook(notebook.id) == notebook
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        color = connection.execute(
            "SELECT color FROM notebooks WHERE id=?",
            (notebook.id,),
        ).fetchone()[0]
        assert color == "neutral"


def test_failed_risky_upgrade_keeps_rollback_and_recovery_snapshot(database: Database) -> None:
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Snapshot identity")
    migration_five = Migration(
        5,
        "0005_fail_for_test.sql",
        "ALTER TABLE table_that_does_not_exist ADD COLUMN value TEXT;",
        requires_backup=True,
    )
    migrations = load_migrations() + (migration_five,)
    target = Database(database.config, migrations=migrations)

    with pytest.raises(MigrationError):
        target.initialize()

    snapshot = database.config.snapshot_dir / "snapshot-r0004.sqlite3"
    assert snapshot.is_file()
    assert snapshot.is_relative_to(database.config.data_dir)
    assert list(database.config.snapshot_dir.glob("snapshot-r*.sqlite3")) == [snapshot]
    with sqlite3.connect(snapshot) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert notebook_exists(connection, notebook.id)
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 4
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert notebook_exists(connection, notebook.id)
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 4
    if os.name != "nt":
        assert snapshot.stat().st_mode & 0o777 == 0o600
        assert database.config.snapshot_dir.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("tamper", ["newer_revision", "drop_trigger", "wrong_checksum"])
def test_unsupported_or_tampered_database_remains_unchanged(
    database: Database,
    tamper: str,
) -> None:
    if tamper == "newer_revision":
        with sqlite3.connect(database.path) as connection:
            connection.execute("PRAGMA user_version=77")
    elif tamper == "drop_trigger":
        with sqlite3.connect(database.path) as connection:
            connection.execute("DROP TRIGGER passages_are_immutable")
    else:
        with sqlite3.connect(database.path) as connection:
            connection.execute("UPDATE schema_migrations SET checksum_sha256=?", ("0" * 64,))
    before = database.path.read_bytes()

    with pytest.raises(UnsupportedSchemaError):
        database.initialize()

    assert database.path.read_bytes() == before


def test_migration_checksum_mismatch_is_rejected_unchanged(database: Database) -> None:
    before = database.path.read_bytes()
    original = load_migrations()[0]
    tampered = Migration(original.revision, original.name, original.sql + "\n-- changed")

    with pytest.raises(UnsupportedSchemaError):
        Database(database.config, migrations=(tampered,)).initialize()

    assert database.path.read_bytes() == before


def test_two_processes_apply_a_revision_once(database: Database) -> None:
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Concurrent upgrade")
    command = """
from pathlib import Path
from core.persistence.database import Database, DatabaseConfig
from core.persistence.migrations import Migration, load_migrations
root = Path(__import__('sys').argv[1])
sql = "ALTER TABLE notebooks ADD COLUMN color TEXT NOT NULL DEFAULT 'neutral';"
migration = Migration(5, '0005_add_notebook_color.sql', sql)
Database(DatabaseConfig(root), migrations=load_migrations() + (migration,)).initialize()
"""
    env = os.environ.copy()
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", command, str(database.config.data_dir)],
            cwd=CORE_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=30) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], results
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 5
        assert notebook_exists(connection, notebook.id)
