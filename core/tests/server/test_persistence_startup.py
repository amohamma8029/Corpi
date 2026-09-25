from pathlib import Path
from secrets import token_hex

import pytest

from core.api.server import main
from core.persistence.errors import MigrationError


def run_main_expect_startup_error(capsys: pytest.CaptureFixture[str], code: str) -> str:
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    output = captured.out.strip()
    assert output == f"CORPI_CORE_ERROR code={code}"
    return output


def test_unsupported_sqlite_runtime_uses_stable_startup_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CORPI_CORE_TOKEN", "test-launch-token")
    monkeypatch.setenv("CORPI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("core.persistence.database.sqlite3.sqlite_version_info", (3, 51, 2))

    run_main_expect_startup_error(capsys, "unsupported_sqlite")
    assert not (tmp_path / "data").exists()


def test_migration_failure_uses_stable_startup_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_marker = token_hex(20)
    source_text_marker = token_hex(20)
    monkeypatch.setenv("CORPI_CORE_TOKEN", secret_marker)
    monkeypatch.setenv("CORPI_DATA_DIR", str(tmp_path / "data"))

    def fail_initialization(self: object) -> None:
        raise MigrationError(f"{source_text_marker} {secret_marker}")

    monkeypatch.setattr("core.api.server.Database.initialize", fail_initialization)

    output = run_main_expect_startup_error(capsys, "migration_failed")
    assert source_text_marker not in output
    assert secret_marker not in output
