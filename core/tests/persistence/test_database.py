import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from core.persistence.database import (
    DEFAULT_LOCK_WAIT_MS,
    Database,
    DatabaseConfig,
)
from core.persistence.errors import (
    DatabaseBusyError,
    InvalidDataDirectoryError,
    UnsupportedSchemaError,
    UnsupportedSQLiteError,
)
from core.persistence.migrations import _HISTORY_SQL, CORPI_APPLICATION_ID


def test_initialization_creates_private_wal_database(data_dir: Path) -> None:
    database = Database(DatabaseConfig(data_dir))
    database.initialize()

    assert database.path.is_file()
    assert database.config.lock_wait_ms == DEFAULT_LOCK_WAIT_MS
    with database.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
        assert connection.execute("PRAGMA application_id").fetchone()[0] == CORPI_APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    if os.name != "nt":
        assert data_dir.stat().st_mode & 0o777 == 0o700
        assert database.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("bootstrap_stage", ["identity", "wal", "history"])
def test_initialization_recovers_each_bootstrap_stage(data_dir: Path, bootstrap_stage: str) -> None:
    data_dir.mkdir(parents=True)
    path = data_dir / "corpi.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA application_id = {CORPI_APPLICATION_ID}")
    if bootstrap_stage in {"wal", "history"}:
        connection.execute("PRAGMA journal_mode=WAL")
    if bootstrap_stage == "history":
        connection.execute(_HISTORY_SQL)
    connection.close()

    Database(DatabaseConfig(data_dir)).initialize()

    with sqlite3.connect(path) as reopened:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 1
        assert reopened.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_unsupported_database_is_not_rewritten(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    path = data_dir / "corpi.sqlite3"
    path.write_bytes(b"foreign data")
    before = path.read_bytes()

    with pytest.raises(UnsupportedSchemaError):
        Database(DatabaseConfig(data_dir)).initialize()

    assert path.read_bytes() == before


def test_foreign_sqlite_database_is_rejected_unchanged(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    path = data_dir / "corpi.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.execute("PRAGMA journal_mode=WAL")
    before = path.read_bytes()

    with pytest.raises(UnsupportedSchemaError):
        Database(DatabaseConfig(data_dir)).initialize()

    assert path.read_bytes() == before


def test_missing_valid_absolute_data_directory_is_created(data_dir: Path) -> None:
    assert not data_dir.exists()
    Database(DatabaseConfig(data_dir)).initialize()
    assert data_dir.is_dir()


def test_zero_byte_database_file_is_initialized(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    (data_dir / "corpi.sqlite3").write_bytes(b"")

    database = Database(DatabaseConfig(data_dir))
    database.initialize()

    with database.connect() as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == CORPI_APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


@pytest.mark.parametrize("value", [None, "", "relative/path"])
def test_environment_requires_absolute_data_directory(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("CORPI_DATA_DIR", raising=False)
    else:
        monkeypatch.setenv("CORPI_DATA_DIR", value)
    with pytest.raises(InvalidDataDirectoryError):
        DatabaseConfig.from_environment()


def test_lock_wait_can_be_configured(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    monkeypatch.setenv("CORPI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CORPI_SQLITE_BUSY_TIMEOUT_MS", "250")
    config = DatabaseConfig.from_environment()
    assert config.lock_wait_ms == 250


def test_unsupported_sqlite_runtime_fails_before_database_creation(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.persistence.database.sqlite3.sqlite_version_info", (3, 51, 2))
    database = Database(DatabaseConfig(data_dir))

    with pytest.raises(UnsupportedSQLiteError):
        database.initialize()

    assert not data_dir.exists()


def test_wrong_journal_mode_is_rejected_without_repair(data_dir: Path) -> None:
    database = Database(DatabaseConfig(data_dir))
    database.initialize()
    with sqlite3.connect(database.path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    before = database.path.read_bytes()

    with pytest.raises(UnsupportedSchemaError):
        database.initialize()

    assert database.path.read_bytes() == before
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_corrupted_live_schema_is_rejected_without_repair(data_dir: Path) -> None:
    database = Database(DatabaseConfig(data_dir))
    database.initialize()
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER passages_are_immutable")
    before = database.path.read_bytes()

    with pytest.raises(UnsupportedSchemaError):
        database.initialize()

    assert database.path.read_bytes() == before


def test_wal_companion_files_remain_owner_only(database: Database) -> None:
    if os.name == "nt":
        pytest.skip("Windows uses inherited per-user app-data ACLs")
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO notebooks(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            (
                "11111111-1111-4111-8111-111111111111",
                "Permission check",
                "2026-01-01T00:00:00.000000Z",
                "2026-01-01T00:00:00.000000Z",
            ),
        )
        wal_path = Path(f"{database.path}-wal")
        shm_path = Path(f"{database.path}-shm")
        assert wal_path.stat().st_mode & 0o777 == 0o600
        assert shm_path.stat().st_mode & 0o777 == 0o600
        connection.execute("ROLLBACK")


def test_schema_indexes_dependent_record_lookups(database: Database) -> None:
    with database.connect() as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        }

    assert {
        "sources_notebook_id_idx",
        "source_versions_source_id_idx",
        "source_versions_managed_file_id_idx",
        "extraction_versions_source_version_id_idx",
        "passages_source_version_lookup_idx",
        "passages_extraction_id_idx",
        "index_versions_extraction_id_idx",
        "retrieval_chunks_extraction_id_idx",
        "chunk_passages_passage_id_idx",
        "chunk_passages_index_version_id_idx",
        "jobs_source_id_idx",
        "jobs_source_version_id_idx",
        "jobs_extraction_version_id_idx",
        "jobs_index_version_id_idx",
        "job_steps_output_source_version_idx",
        "job_steps_output_extraction_version_idx",
        "job_steps_output_index_version_idx",
    } <= indexes


def test_initial_schema_has_no_provider_credential_fields(database: Database) -> None:
    with database.connect() as connection:
        declarations = "\n".join(
            row[0].lower()
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE type='table' AND sql IS NOT NULL"
            )
        )

    assert "provider_key" not in declarations
    assert "api_key" not in declarations
    assert "access_token" not in declarations


def test_competing_writer_succeeds_when_the_lock_is_released(database: Database) -> None:
    attempting = threading.Event()
    outcome: list[str] = []

    def wait_for_writer() -> None:
        attempting.set()
        with database.connect() as contender:
            contender.execute("BEGIN IMMEDIATE")
            contender.execute("COMMIT")
        outcome.append("acquired")

    with database.connect() as writer:
        writer.execute("BEGIN IMMEDIATE")
        thread = threading.Thread(target=wait_for_writer)
        thread.start()
        assert attempting.wait(timeout=1)
        time.sleep(0.05)
        writer.execute("COMMIT")
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert outcome == ["acquired"]


def test_wal_reader_remains_available_while_a_writer_has_uncommitted_rows(
    database: Database,
) -> None:
    with database.connect() as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO notebooks(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            (
                "22222222-2222-4222-8222-222222222222",
                "Pending write",
                "2026-01-01T00:00:00.000000Z",
                "2026-01-01T00:00:00.000000Z",
            ),
        )
        with database.connect() as reader:
            assert reader.execute("SELECT count(*) FROM notebooks").fetchone()[0] == 0
        writer.execute("COMMIT")
    with database.connect() as reader:
        assert reader.execute("SELECT count(*) FROM notebooks").fetchone()[0] == 1


def test_lock_timeout_is_a_busy_error_and_does_not_block_async_ticker(data_dir: Path) -> None:
    database = Database(DatabaseConfig(data_dir, lock_wait_ms=75))
    database.initialize()

    async def run_contention() -> int:
        ticks = 0
        with database.connect() as writer:
            writer.execute("BEGIN IMMEDIATE")

            def blocked_write() -> None:
                with database.connect() as contender:
                    contender.execute("BEGIN IMMEDIATE")

            task = asyncio.create_task(asyncio.to_thread(blocked_write))
            while not task.done():
                ticks += 1
                await asyncio.sleep(0.005)
            with pytest.raises(DatabaseBusyError):
                await task
            writer.execute("ROLLBACK")
        return ticks

    assert asyncio.run(run_contention()) > 2
