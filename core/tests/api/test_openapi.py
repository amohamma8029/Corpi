import json
import subprocess
import sys
from pathlib import Path


def test_exported_openapi_contains_core_routes(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"
    result = subprocess.run(
        [sys.executable, "scripts/export_openapi.py", str(output)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert result.returncode == 0, result.stderr

    schema = json.loads(output.read_text(encoding="utf-8"))
    assert "/health" in schema["paths"]
    assert "/demo/stream" in schema["paths"]
