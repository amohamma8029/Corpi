import sqlite3
import threading

import pytest

from core.persistence.errors import PersistenceError


def test_default_connections_remain_fresh_but_scoped_borrows_close_cursors_and_release_locks(
    database,
):
    with database.connect() as first:
        assert type(first) is sqlite3.Connection
    with database.connect() as second:
        assert second is not first
    with database.connection_scope():
        with database.connect() as scoped:
            cursor = scoped.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        with pytest.raises(sqlite3.ProgrammingError):
            cursor.fetchone()
        with pytest.raises(PersistenceError, match="not borrowed"):
            scoped.execute("SELECT 1")
        with sqlite3.connect(database.path, isolation_level=None, timeout=0.1) as independent:
            independent.execute("BEGIN IMMEDIATE")
            independent.execute("ROLLBACK")
        with database.connect() as again:
            assert again is scoped
    with pytest.raises((PersistenceError, sqlite3.ProgrammingError)):
        scoped.execute("SELECT 1")


@pytest.mark.parametrize(
    "mutation",
    [
        "PRAGMA foreign_keys=OFF",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA busy_timeout=1",
        "PRAGMA application_id=1",
        "PRAGMA user_version=1",
        "ATTACH ':memory:' AS extra",
        "CREATE TEMP TABLE unexpected(value)",
        "BEGIN IMMEDIATE",
    ],
)
def test_scoped_policy_changes_fail_closed_instead_of_being_silently_repaired(database, mutation):
    with database.connection_scope():
        with pytest.raises(PersistenceError):
            with database.connect() as connection:
                connection.execute(mutation).fetchall()
        with pytest.raises(PersistenceError, match="scope is invalid"):
            with database.connect():
                pytest.fail("invalidated scope reused")


def test_scoped_application_error_discards_connection_and_keeps_settlement_available(database):
    with database.connection_scope():
        failed = None
        with pytest.raises(ValueError):
            with database.connect() as failed:
                raise ValueError("controlled application error")
        assert failed is not None
        with database.connect() as settlement:
            assert settlement is not failed
            assert settlement.execute("SELECT 1").fetchone()[0] == 1


def test_scopes_refuse_nested_borrows_and_other_threads_have_independent_connections(database):
    outcomes = []
    with database.connection_scope():
        with database.connect() as connection:
            with pytest.raises(PersistenceError, match="nested connection borrows"):
                with database.connect():
                    pytest.fail("nested borrow entered")

        def other():
            with pytest.raises(PersistenceError, match="not borrowed"):
                connection.execute("SELECT 1")
            with database.connect() as independent:
                outcomes.append(type(independent) is sqlite3.Connection)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join(timeout=3)
        assert not thread.is_alive() and outcomes == [True]
