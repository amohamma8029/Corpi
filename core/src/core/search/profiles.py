"""Pinned embedding profiles; installation identity includes all effective settings."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ModelFile:
    path: str
    bytes: int
    sha256: str
    url: str


@dataclass(frozen=True)
class ModelProfile:
    id: str
    model: str
    repository: str
    revision: str
    license: str
    pooling: str
    dimensions: int
    input_tokens: int
    chunk_tokens: int
    overlap_tokens: int
    query_prefix: str
    passage_prefix: str
    model_file: str
    runtime: str
    vector_format: str
    separator: str
    files: tuple[ModelFile, ...]

    @property
    def identity(self) -> str:
        from dataclasses import asdict

        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def download_bytes(self) -> int:
        return sum(item.bytes for item in self.files)


def profiles() -> dict[str, ModelProfile]:
    raw = json.loads(files(__package__).joinpath("model_manifests.json").read_text("utf-8"))
    result = {}
    for item in raw:
        item["files"] = tuple(ModelFile(**entry) for entry in item["files"])
        profile = ModelProfile(**item)
        if not re.fullmatch(r"[a-z0-9-]+", profile.id):
            raise ValueError("invalid profile identifier")
        for asset in profile.files:
            path = PurePosixPath(asset.path)
            if path.is_absolute() or ".." in path.parts or "\\" in asset.path:
                raise ValueError("unsafe model asset path")
            if asset.bytes < 1 or not re.fullmatch(r"[a-f0-9]{64}", asset.sha256):
                raise ValueError("invalid model file identity")
            expected = f"https://huggingface.co/{profile.repository}/resolve/{profile.revision}/"
            if asset.url != expected + asset.path:
                raise ValueError("model URL is not pinned to its manifest")
        if profile.id in result:
            raise ValueError("duplicate model profile")
        result[profile.id] = profile
    return result


def installation_path(data_dir: Path, profile: ModelProfile) -> Path:
    return data_dir / "models" / profile.identity


def verify_installation(path: Path, profile: ModelProfile) -> None:
    """A complete local hash check, never a cache probe or network request."""
    if path.is_symlink() or path.is_junction() or not path.is_dir():
        raise ValueError("model_not_installed")
    root = path.resolve()
    for asset in profile.files:
        candidate = path / asset.path
        current = path
        for part in PurePosixPath(asset.path).parts:
            current /= part
            if current.is_symlink() or current.is_junction():
                raise ValueError("model_corrupt")
        if not candidate.resolve().is_relative_to(root) or not candidate.is_file():
            raise ValueError("model_corrupt")
        if candidate.stat().st_size != asset.bytes:
            raise ValueError("model_corrupt")
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != asset.sha256:
            raise ValueError("model_corrupt")
