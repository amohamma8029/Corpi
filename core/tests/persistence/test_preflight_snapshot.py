"""Read-only preflight cannot mix two valid committed schema revisions."""

import sqlite3

import pytest
from test_execution_migration import old_database, records

from core.persistence import migrations
from core.persistence.database import Database
from core.persistence.errors import UnsupportedSchemaError


def test_preflight_keeps_one_snapshot_when_another_initializer_commits(tmp_path, monkeypatch):
    old, _, original = old_database(tmp_path / "original")
    before, saved_original = records(old), original.read_bytes()
    fingerprint = migrations.schema_fingerprint
    observed = {}

    def commit_during_validation(connection):
        if not observed:
            # The validator has read revision/history already. Commit a real
            # upgrade on another connection before it reads the schema itself.
            observed["reader_before"] = connection.execute("PRAGMA user_version").fetchone()[0]
            observed["has_read_transaction"] = connection.in_transaction
            other = Database(old.config)
            other.initialize()
            with other.connect() as published:
                observed["published_revision"] = published.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
            observed["reader_after"] = connection.execute("PRAGMA user_version").fetchone()[0]
        return fingerprint(connection)

    monkeypatch.setattr(migrations, "schema_fingerprint", commit_during_validation)
    upgraded = Database(old.config)
    upgraded.initialize()
    assert observed == {
        "reader_before": 14,
        "has_read_transaction": True,
        "published_revision": len(migrations.load_migrations()),
        "reader_after": 14,
    }
    assert records(upgraded) == before and original.read_bytes() == saved_original
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == len(
            migrations.load_migrations()
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
    copies = list(old.config.snapshot_dir.glob("snapshot-r*.sqlite3"))
    assert len(copies) == 1
    with sqlite3.connect(copies[0]) as snapshot:
        assert snapshot.execute("PRAGMA user_version").fetchone()[0] == 14
        assert copies[0].name.startswith("snapshot-r0014")


def test_preflight_snapshot_still_rejects_actual_schema_tampering(tmp_path):
    database, _, _ = old_database(tmp_path / "original")
    with database.connect() as connection:
        connection.execute("CREATE TABLE unexpected_change (value TEXT)")
    before = database.path.read_bytes()
    with pytest.raises(UnsupportedSchemaError, match="schema does not match its migration history"):
        Database(database.config).initialize()
    assert database.path.read_bytes() == before
