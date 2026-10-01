"""Safe app-data paths and content-verified managed PDF operations."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

from core.persistence.errors import PersistenceError


class ManagedFileIntegrityError(PersistenceError):
    code = "managed_file_integrity"


class ManagedFilePathError(PersistenceError):
    code = "managed_file_path"


class ManagedPdfFiles:
    """Owns managed file reads, staging, placement, and recorded cleanup."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.objects_dir = self.data_dir / "files" / "objects"
        self.staging_dir = self.data_dir / "files" / "staging"

    def open_staging(self, relative_path: str):
        path = self._path(relative_path, create_parents=True, allow_missing=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "wb")

    def path_for(
        self,
        relative_path: str,
        *,
        create_parents: bool = False,
        allow_missing: bool = False,
    ) -> Path:
        """Return a checked managed path for operations that need an atomic OS call."""
        return self._path(
            relative_path,
            create_parents=create_parents,
            allow_missing=allow_missing,
        )

    def place_staging(self, staging_relative_path: str, final_relative_path: str) -> None:
        staging = self._path(staging_relative_path)
        destination = self._path(final_relative_path, create_parents=True, allow_missing=True)
        try:
            os.link(staging, destination, follow_symlinks=False)
        except FileExistsError:
            raise
        self._sync_directory(destination.parent)

    def read_verified(self, relative_path: str, checksum: str, size_bytes: int) -> bytes:
        path = self._path(relative_path)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as error:
            raise ManagedFileIntegrityError("managed PDF is missing from its saved path") from error
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or _is_reparse_point(file_stat):
                raise ManagedFilePathError("managed PDF is not a regular app-owned file")
            if file_stat.st_size != size_bytes:
                raise ManagedFileIntegrityError(
                    "managed PDF size does not match its saved identity"
                )
            digest = hashlib.sha256()
            content = bytearray()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    content.extend(chunk)
            if digest.hexdigest() != checksum:
                raise ManagedFileIntegrityError(
                    "managed PDF checksum does not match its saved identity"
                )
            return bytes(content)
        finally:
            os.close(descriptor)

    def stat_size(self, relative_path: str) -> int | None:
        path = self._path(relative_path, allow_missing=True)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_point(metadata):
            raise ManagedFilePathError("managed file is not a regular app-owned file")
        return int(metadata.st_size)

    def remove_recorded(
        self,
        relative_path: str,
        checksum: str | None = None,
        size_bytes: int | None = None,
    ) -> None:
        path = self._path(relative_path, allow_missing=True)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_point(metadata):
            raise ManagedFilePathError("recorded cleanup target is not a regular file")
        if checksum is not None and size_bytes is not None:
            self.read_verified(relative_path, checksum, size_bytes)
        path.unlink()
        self._sync_directory(path.parent)

    def _path(
        self,
        relative_path: str,
        *,
        create_parents: bool = False,
        allow_missing: bool = False,
    ) -> Path:
        try:
            _validate_relative_path(relative_path)
        except ValueError as error:
            raise ManagedFilePathError("managed file path is invalid") from error
        pure = PurePosixPath(relative_path)
        windows = PureWindowsPath(relative_path)
        if pure.parts[0] != "files" or windows.drive:
            raise ManagedFilePathError("managed file path is outside Corpi's file directory")
        candidate = self.data_dir.joinpath(*pure.parts)
        current = self.data_dir
        for component in pure.parts:
            current = current / component
            if current == candidate and not current.exists() and allow_missing:
                break
            if not current.exists():
                if create_parents and current != candidate:
                    current.mkdir(mode=0o700)
                    continue
                if allow_missing:
                    break
                raise FileNotFoundError(current)
            try:
                metadata = current.lstat()
            except OSError as error:
                raise ManagedFilePathError("managed file path cannot be inspected") from error
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
                raise ManagedFilePathError("managed file path contains a redirected component")
        resolved_parent = candidate.parent.resolve()
        if not resolved_parent.is_relative_to(self.data_dir):
            raise ManagedFilePathError("managed file path escapes the app-data directory")
        if candidate.exists() and not candidate.resolve().is_relative_to(self.data_dir):
            raise ManagedFilePathError("managed file path escapes the app-data directory")
        return candidate

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _validate_relative_path(path: str) -> None:
    pure_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        not path
        or "\\" in path
        or pure_path.is_absolute()
        or windows_path.drive
        or ".." in pure_path.parts
        or not pure_path.parts
        or pure_path.as_posix() != path
    ):
        raise ValueError("path must be a normalized app-data-relative path")
