import asyncio
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from core.ingest.service import PdfIngestService
from core.jobs.sweeper import TrashSweeper
from core.persistence.database import Database
from core.persistence.errors import (
    DeletedPdfAcquisitionError,
    MigrationError,
    UnsupportedSchemaError,
)
from core.persistence.migrations import (
    CORPI_APPLICATION_ID,
    Migration,
    load_migrations,
    schema_fingerprint,
)
from core.persistence.models import new_id, utc_timestamp
from core.persistence.store import PersistenceStore

CORE_ROOT = Path(__file__).resolve().parents[2]
HEAD_REVISION = len(load_migrations())


def notebook_exists(connection: sqlite3.Connection, notebook_id: str) -> bool:
    return (
        connection.execute("SELECT 1 FROM notebooks WHERE id=?", (notebook_id,)).fetchone()
        is not None
    )


def migration_two(*, requires_backup: bool = False) -> Migration:
    return Migration(
        HEAD_REVISION + 1,
        f"{HEAD_REVISION + 1:04d}_add_notebook_color.sql",
        "ALTER TABLE notebooks ADD COLUMN color TEXT NOT NULL DEFAULT 'neutral';",
        requires_backup,
    )


def test_application_id_and_schema_fingerprint_are_stable(database: Database) -> None:
    with database.connect() as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x434F5250
        assert CORPI_APPLICATION_ID == 0x434F5250
        assert schema_fingerprint(connection) == (
            "d05470bf0e809e09e954569b5917ef7a2d673597fb1da614f7e761109d549cb6"
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == HEAD_REVISION + 1
        color = connection.execute(
            "SELECT color FROM notebooks WHERE id=?",
            (notebook.id,),
        ).fetchone()[0]
        assert color == "neutral"


def test_failed_risky_upgrade_keeps_rollback_and_recovery_snapshot(database: Database) -> None:
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Snapshot identity")
    migration_five = Migration(
        HEAD_REVISION + 1,
        f"{HEAD_REVISION + 1:04d}_fail_for_test.sql",
        "ALTER TABLE table_that_does_not_exist ADD COLUMN value TEXT;",
        requires_backup=True,
    )
    migrations = load_migrations() + (migration_five,)
    target = Database(database.config, migrations=migrations)
    prior_snapshots = {
        path: path.read_bytes() for path in database.config.snapshot_dir.glob("snapshot-r*.sqlite3")
    }

    with pytest.raises(MigrationError):
        target.initialize()

    snapshot = database.config.snapshot_dir / f"snapshot-r{HEAD_REVISION:04d}.sqlite3"
    assert snapshot.is_file()
    assert snapshot.is_relative_to(database.config.data_dir)
    assert set(database.config.snapshot_dir.glob("snapshot-r*.sqlite3")) == {
        *prior_snapshots,
        snapshot,
    }
    assert all(path.read_bytes() == contents for path, contents in prior_snapshots.items())
    with sqlite3.connect(snapshot) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == HEAD_REVISION
        assert notebook_exists(connection, notebook.id)
        assert (
            connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
            == HEAD_REVISION
        )
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == HEAD_REVISION
        assert notebook_exists(connection, notebook.id)
        assert (
            connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
            == HEAD_REVISION
        )
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
migration = Migration(len(load_migrations()) + 1, 'next_add_notebook_color.sql', sql)
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == HEAD_REVISION + 1
        assert (
            connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
            == HEAD_REVISION + 1
        )
        assert notebook_exists(connection, notebook.id)


@pytest.mark.parametrize("failed_upgrade", [False, True])
def test_deletion_receipt_upgrade_preserves_populated_pdf_ownership_and_replay(
    data_dir: Path, failed_upgrade: bool
) -> None:
    from core.persistence.database import DatabaseConfig

    migrations = load_migrations()
    old = Database(DatabaseConfig(data_dir), migrations=migrations[:12])
    old.initialize()
    store = PersistenceStore(old)
    notebook = store.create_notebook(new_id(), "Populated revision12")
    source = store.create_source(new_id(), notebook.id, "replacement history")
    versions = [store.create_source_version(source.id, char * 64) for char in ("a", "b")]
    timestamp = utc_timestamp()
    actions = [new_id(), new_id()]
    imports = [new_id(), new_id()]
    with old.connect() as connection:
        for ordinal, (action, version, import_id) in enumerate(
            zip(actions, versions, imports, strict=True)
        ):
            connection.execute(
                """INSERT INTO pdf_acquisitions(
                       id,retry_key,notebook_id,intent,target_source_id,display_filename,state,
                       owner_instance_id,staging_relative_path,last_write_at,outcome_source_id,
                       outcome_import_id,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'completed','old-owner',?,?,?,?,?,?)""",
                (
                    action,
                    f"old-action-{ordinal}",
                    notebook.id,
                    "replace" if ordinal else "add",
                    source.id if ordinal else None,
                    "old.pdf",
                    f"files/staging/{action}.part",
                    timestamp,
                    source.id,
                    import_id,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """INSERT INTO pdf_imports(id,source_id,source_version_id,acquisition_id,
                       display_filename,state,created_at,updated_at)
                   VALUES(?,?,?,?,'old.pdf','ready',?,?)""",
                (import_id, source.id, version.id, action, timestamp, timestamp),
            )
        connection.execute(
            """INSERT INTO acquisition_file_cleanup(
                   id,acquisition_id,relative_path,file_kind,created_at)
               VALUES(?,?,?,'staging',?)""",
            (new_id(), actions[0], f"files/staging/{actions[0]}.part", timestamp),
        )
        tables = (
            "sources",
            "source_versions",
            "pdf_acquisitions",
            "pdf_imports",
            "acquisition_file_cleanup",
        )
        before = {
            table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in tables
        }
    if failed_upgrade:
        actual = migrations[12]
        broken = Migration(
            actual.revision,
            actual.name,
            actual.sql + "\nINSERT INTO missing_test_table VALUES(1);",
            True,
        )
        with pytest.raises(MigrationError):
            Database(old.config, migrations=migrations[:12] + (broken,)).initialize()
        # Failed addition restores the old fingerprint/history and all data;
        # the snapshot is a real pre-upgrade database, not a schema-only file.
        old.initialize()
        with old.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 12
            assert (
                connection.execute(
                    "SELECT 1 FROM sqlite_schema WHERE name='deleted_pdf_acquisitions'"
                ).fetchone()
                is None
            )
        with closing(
            sqlite3.connect(old.config.snapshot_dir / "snapshot-r0012.sqlite3")
        ) as snapshot:
            assert snapshot.execute("SELECT count(*) FROM pdf_acquisitions").fetchone()[0] == 2
    upgraded = Database(old.config)
    upgraded.initialize()
    with upgraded.connect() as connection:
        for table, expected in before.items():
            assert [
                tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")
            ] == expected
        assert (
            connection.execute("SELECT count(*) FROM deleted_pdf_acquisitions").fetchone()[0] == 0
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    migrated = PersistenceStore(upgraded)
    for ordinal, action in enumerate(actions):
        assert (
            migrated.create_pdf_acquisition(
                f"old-action-{ordinal}",
                notebook.id,
                "replace" if ordinal else "add",
                "old.pdf",
                "new-owner",
                source.id if ordinal else None,
            ).id
            == action
        )
    migrated.trash_source(source.id, "2000-01-01T00:00:00.000000Z")
    ingest = PdfIngestService(migrated, "upgraded-cleanup")
    try:
        ingest.maintenance()
        assert TrashSweeper(migrated).sweep_once() == (source.id,)
    finally:
        asyncio.run(ingest.stop())
    upgraded.initialize()
    restarted = PersistenceStore(upgraded)
    for ordinal, action in enumerate(actions):
        with pytest.raises(DeletedPdfAcquisitionError):
            restarted.load_pdf_acquisition(action)
        with pytest.raises(DeletedPdfAcquisitionError):
            restarted.create_pdf_acquisition(
                f"old-action-{ordinal}",
                notebook.id,
                "replace" if ordinal else "add",
                "old.pdf",
                "replay",
                source.id if ordinal else None,
            )
    with upgraded.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM deleted_pdf_acquisitions").fetchone()[0] == 2
        )
        assert connection.execute("SELECT count(*) FROM pdf_acquisitions").fetchone()[0] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
