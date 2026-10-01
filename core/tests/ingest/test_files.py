from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from core.ingest.files import ManagedFileIntegrityError, ManagedFilePathError, ManagedPdfFiles


def test_managed_pdf_read_checks_actual_size_and_checksum(tmp_path: Path) -> None:
    files = ManagedPdfFiles(tmp_path)
    relative = "files/objects/report.pdf"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    content = b"%PDF-1.4\nverified bytes\n"
    path.write_bytes(content)

    assert (
        files.read_verified(relative, hashlib.sha256(content).hexdigest(), len(content)) == content
    )
    with pytest.raises(ManagedFileIntegrityError):
        files.read_verified(relative, "0" * 64, len(content))
    with pytest.raises(ManagedFileIntegrityError):
        files.read_verified(relative, hashlib.sha256(content).hexdigest(), len(content) + 1)


def test_managed_pdf_read_rejects_escaped_paths(tmp_path: Path) -> None:
    files = ManagedPdfFiles(tmp_path)
    with pytest.raises(ManagedFilePathError):
        files.read_verified("../outside.pdf", "0" * 64, 0)
    with pytest.raises(ManagedFilePathError):
        files.read_verified("files/../outside.pdf", "0" * 64, 0)


def test_staging_directories_and_files_use_owner_only_posix_modes(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows uses inherited per-user app-data ACLs")
    files = ManagedPdfFiles(tmp_path)
    relative_path = "files/staging/owned.part"
    with files.open_staging(relative_path):
        pass
    assert (tmp_path / "files").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "files/staging").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / relative_path).stat().st_mode & 0o777 == 0o600


def test_managed_pdf_read_rejects_redirected_path_components(tmp_path: Path) -> None:
    files = ManagedPdfFiles(tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-outside"
    external.mkdir()
    (external / "report.pdf").write_bytes(b"%PDF-1.4\nprivate")
    managed = tmp_path / "files" / "objects"
    managed.parent.mkdir(parents=True)
    try:
        managed.symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"this platform can't create the redirect control: {type(error).__name__}")

    with pytest.raises(ManagedFilePathError):
        files.read_verified("files/objects/report.pdf", "0" * 64, 16)
