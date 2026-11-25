from pathlib import Path

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore


@pytest.fixture()
def store(tmp_path: Path) -> PersistenceStore:
    database = Database(DatabaseConfig(tmp_path / "provider-data"))
    database.initialize()
    return PersistenceStore(database)
