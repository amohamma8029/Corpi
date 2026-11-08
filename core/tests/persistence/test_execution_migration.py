"""A real populated old database and recovery copy, never a user's library."""

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import MigrationError, UnsupportedSchemaError
from core.persistence.migrations import Migration, load_migrations
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore


def old_database(root):
    database = Database(DatabaseConfig(root), migrations=load_migrations()[:14])
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "pre-upgrade notebook")
    source = store.create_source(new_id(), notebook.id, "saved-original.pdf")
    store.create_source_version(source.id, "a" * 64)
    original = root / "files/objects/test-original.pdf"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"public synthetic original, not a personal document")
    return database, notebook, original


def records(database):
    with database.connect() as connection:
        return {
            name: [tuple(row) for row in connection.execute(f"SELECT * FROM {name} ORDER BY 1")]
            for name in (
                "notebooks",
                "sources",
                "source_versions",
                "jobs",
                "job_steps",
                "job_step_attempts",
                "search_generations",
                "search_batches",
            )
        }


def test_real_upgrade_keeps_old_records_and_restored_copy_opens_with_old_schema(tmp_path):
    old, notebook, original = old_database(tmp_path / "original")
    before = records(old)
    original_bytes = original.read_bytes()
    migrations = load_migrations()
    assert migrations[14].requires_backup and not any(m.requires_backup for m in migrations[:14])
    upgraded = Database(old.config)
    upgraded.initialize()
    assert records(upgraded) == before and original.read_bytes() == original_bytes
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 15
        assert connection.execute("SELECT count(*) FROM search_batch_executions").fetchone()[0] == 0
    database_bytes = upgraded.path.read_bytes()
    with pytest.raises(UnsupportedSchemaError):
        old.initialize()
    assert upgraded.path.read_bytes() == database_bytes
    recovery = old.config.snapshot_dir / "snapshot-r0014.sqlite3"
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    shutil.copyfile(recovery, restored_root / "corpi.sqlite3")
    restored_original = restored_root / original.relative_to(old.config.data_dir)
    restored_original.parent.mkdir(parents=True)
    shutil.copyfile(original, restored_original)
    restored = Database(DatabaseConfig(restored_root), migrations=migrations[:14])
    restored.initialize()
    assert records(restored) == before
    assert PersistenceStore(restored).load_notebook(notebook.id) == notebook
    assert restored_original.read_bytes() == original_bytes
    with restored.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("failure", ["snapshot", "sql", "timeout", "redirected"])
def test_failed_upgrade_leaves_old_database_readable_and_no_partial_receipt_table(
    tmp_path, monkeypatch, failure
):
    from core.persistence import migrations as module

    old, notebook, original = old_database(tmp_path / "original")
    before = records(old)
    if failure == "snapshot":

        def fail(*args):
            raise MigrationError("recovery copy unavailable")

        monkeypatch.setattr(module, "_make_recovery_snapshot", fail)
        migrations = load_migrations()
    elif failure == "timeout":
        times = iter([0.0, 301.0])
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(times)))
        migrations = load_migrations()
    elif failure == "redirected":
        real = Path.is_symlink
        monkeypatch.setattr(
            Path, "is_symlink", lambda path: path == old.config.snapshot_dir or real(path)
        )
        migrations = load_migrations()
    else:
        migrations = load_migrations()
        actual = migrations[14]
        broken = Migration(
            actual.revision,
            actual.name,
            actual.sql + "\nINSERT INTO missing_test_table VALUES(1);",
            True,
        )
        migrations = (*migrations[:14], broken)
    with pytest.raises(MigrationError):
        Database(old.config, migrations=migrations).initialize()
    old.initialize()
    assert records(old) == before
    assert PersistenceStore(old).load_notebook(notebook.id) == notebook
    assert original.is_file()
    with old.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 14
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='search_batch_executions'"
            ).fetchone()
            is None
        )
    if failure == "sql":
        with sqlite3.connect(old.config.snapshot_dir / "snapshot-r0014.sqlite3") as snapshot:
            assert snapshot.execute("SELECT id FROM notebooks").fetchone()[0] == notebook.id


def test_upgrade_preserves_existing_snapshots_and_unknown_pending_file(tmp_path):
    old, _, _ = old_database(tmp_path / "original")
    directory = old.config.snapshot_dir
    directory.mkdir()
    originals = {
        directory / "snapshot-r0012.sqlite3": b"older retained recovery",
        directory / "snapshot-r0014.sqlite3": b"existing retained recovery",
        directory / "snapshot-r0014.sqlite3.pending": b"unknown interrupted copy",
    }
    for path, contents in originals.items():
        path.write_bytes(contents)
    upgraded = Database(old.config)
    upgraded.initialize()
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    created = set(directory.iterdir()) - originals.keys()
    assert len(created) == 1
    with sqlite3.connect(created.pop()) as snapshot:
        assert snapshot.execute("PRAGMA user_version").fetchone()[0] == 14


def test_two_initializers_keep_a_real_pre_upgrade_copy_and_truthful_snapshot_names(tmp_path):
    from core.search.supervisor import stop_owned_process_tree, worker_environment

    old, _, _ = old_database(tmp_path / "original")
    script = """
import sys
from pathlib import Path
from core.persistence.database import Database, DatabaseConfig
Database(DatabaseConfig(Path(sys.argv[1]))).initialize()
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(old.config.data_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=worker_environment(),
        )
        for _ in range(2)
    ]
    try:
        results = [process.communicate(timeout=30) for process in processes]
        assert [process.returncode for process in processes] == [0, 0], results
    finally:
        for process in processes:
            stop_owned_process_tree(process)
    copies = list(old.config.snapshot_dir.glob("snapshot-r*.sqlite3"))
    revisions = []
    for path in copies:
        with sqlite3.connect(path) as snapshot:
            revision = snapshot.execute("PRAGMA user_version").fetchone()[0]
            revisions.append(revision)
            assert path.name.startswith(f"snapshot-r{revision:04d}")
    assert 14 in revisions
