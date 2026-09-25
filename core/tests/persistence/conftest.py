from pathlib import Path

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "corpi-data"


@pytest.fixture()
def database(data_dir: Path) -> Database:
    database = Database(DatabaseConfig(data_dir))
    database.initialize()
    return database


@pytest.fixture()
def store(database: Database) -> PersistenceStore:
    return PersistenceStore(database)
