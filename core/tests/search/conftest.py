from pathlib import Path

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--models-directory", default=None, help="Installed diagnostic model directory"
    )
    parser.addoption(
        "--model-setup-export-directory",
        default=None,
        help="Explicit destination for verified live-setup assets reused by platform checks",
    )


@pytest.fixture
def store(tmp_path: Path) -> PersistenceStore:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    return PersistenceStore(database)
