from dataclasses import replace
from pathlib import Path

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import PersistenceError


def test_resolves_once_per_scope_but_still_checks_file_identity_for_each_borrow(
    tmp_path, monkeypatch
):
    database = Database(DatabaseConfig(tmp_path / "library"))
    database.initialize()
    resolves, identities = [], []
    resolve, identity = Path.resolve, database._file_identity

    def resolved(path, *args, **kwargs):
        if path == database.path:
            resolves.append(path)
        return resolve(path, *args, **kwargs)

    def identified():
        identities.append(None)
        return identity()

    monkeypatch.setattr(Path, "resolve", resolved)
    monkeypatch.setattr(database, "_file_identity", identified)
    with database.connection_scope():
        for _ in range(20):
            with database.connect() as connection:
                assert connection.execute("SELECT 1").fetchone()[0] == 1
    assert len(resolves) == 1
    assert len(identities) >= 40


def test_cached_path_does_not_admit_a_replaced_database(tmp_path):
    database = Database(DatabaseConfig(tmp_path / "library"))
    database.initialize()
    replacement = Database(DatabaseConfig(tmp_path / "replacement"))
    replacement.initialize()
    with database.connection_scope():
        database.path.rename(database.path.with_suffix(".preserved"))
        replacement.path.rename(database.path)
        with pytest.raises(PersistenceError, match="identity or configuration changed"):
            with database.connect():
                pytest.fail("a replaced file reused the scope")


def test_cached_path_does_not_admit_changed_configuration(tmp_path):
    database = Database(DatabaseConfig(tmp_path / "library"))
    database.initialize()
    with database.connection_scope():
        database.config = replace(database.config, lock_wait_ms=100)
        with pytest.raises(PersistenceError, match="identity or configuration changed"):
            with database.connect():
                pytest.fail("changed configuration reused the scope")
