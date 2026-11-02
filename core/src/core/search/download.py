"""Bounded transfers for fixed manifest assets, with no inference-side downloads."""

from __future__ import annotations

import hashlib
import os
import stat
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from core.search.profiles import ModelProfile, verify_installation

_DOWNLOAD_HOSTS = frozenset({"huggingface.co", "us.aws.cdn.hf.co"})
_USER_AGENT = "Corpi-model-setup/1"


def _validate_url(url: str, *, allow_query: bool = True) -> None:
    try:
        address = urllib.parse.urlsplit(url)
        safe = (
            address.scheme == "https"
            and address.hostname in _DOWNLOAD_HOSTS
            and address.port in (None, 443)
            and address.username is None
            and address.password is None
            and not address.fragment
            and (allow_query or not address.query)
            and not any(ord(character) <= 32 or character == "\\" for character in url)
        )
    except ValueError:
        safe = False
    if not safe:
        raise ValueError("model_download_destination")


class _AssetRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, deadline: float, should_stop: Callable[[], bool]) -> None:
        self.deadline = deadline
        self.should_stop = should_stop
        self.redirects = 0

    def http_error_302(self, req, fp, code, msg, headers):
        # urllib's default redirect handler drains arbitrary redirect bodies and
        # copies request headers. Neither belongs in an anonymous asset transfer.
        try:
            location = headers.get("Location")
            if (
                not location
                or self.redirects >= 10
                or any(ord(character) <= 32 or character == "\\" for character in location)
            ):
                raise ValueError("model_download_redirect")
            target = urllib.parse.urljoin(req.full_url, location)
            _validate_url(target)
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.should_stop():
                raise TimeoutError("model_download_stopped")
            self.redirects += 1
            redirected = urllib.request.Request(target, headers={"User-Agent": _USER_AGENT})
        finally:
            fp.close()  # Do not download an unbounded, unaccounted redirect body.
        timeout = min(remaining, req.timeout) if req.timeout is not None else remaining
        return self.parent.open(redirected, timeout=timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _asset_path(staging: Path, relative: str) -> Path:
    if staging.is_symlink() or staging.is_junction():
        raise ValueError("unsafe_model_staging")
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or PurePosixPath(relative).is_absolute()
        or ".." in parts
        or PurePosixPath(relative).as_posix() != relative
        or "\\" in relative
        or ":" in relative
        or "\0" in relative
    ):
        raise ValueError("unsafe_model_staging")
    current = staging
    for part in parts:
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ValueError("unsafe_model_staging")
        if current != staging / relative:
            current.mkdir(mode=0o700, exist_ok=True)
    if not current.resolve().is_relative_to(staging.resolve()):
        raise ValueError("unsafe_model_staging")
    if current.exists():
        metadata = current.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("unsafe_model_staging")
    return current


def _open_asset(candidate: Path):
    flags = os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("unsafe_model_staging")
        os.ftruncate(descriptor, 0)  # Validate the opened file before truncating a prior partial.
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def download_profile(
    staging: Path,
    profile: ModelProfile,
    *,
    on_progress: Callable[[int], None] = lambda _: None,
    should_stop: Callable[[], bool] = lambda: False,
    max_bytes: int = 1_000_000_000,
    timeout_seconds: float = 600,
    idle_seconds: float = 30,
) -> int:
    """The caller owns staging, promotion, retry accounting, and writer supervision.

    This routine does one transfer attempt. An outer supervisor provides hard
    cancellation/deadline enforcement even during a blocking socket operation.
    """
    received = 0
    deadline = time.monotonic() + timeout_seconds
    if staging.is_symlink() or staging.is_junction():
        raise ValueError("unsafe_model_staging")
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    for asset in profile.files:
        _asset_path(staging, asset.path)
        expected = (
            f"https://huggingface.co/{profile.repository}/resolve/{profile.revision}/{asset.path}"
        )
        if asset.url != expected:
            raise ValueError("model_download_destination")
        _validate_url(asset.url, allow_query=False)
        digest = hashlib.sha256()
        file_bytes = 0
        remaining = deadline - time.monotonic()
        if remaining <= 0 or should_stop():
            raise TimeoutError("model_download_stopped")
        request = urllib.request.Request(asset.url, headers={"User-Agent": _USER_AGENT})
        # An owned opener has no global cookie/auth handlers or provider headers.
        opener = urllib.request.build_opener(_AssetRedirects(deadline, should_stop))
        with opener.open(request, timeout=min(remaining, idle_seconds)) as response:
            _validate_url(response.url)
            if response.status != 200:
                raise ValueError("model_download_response")
            # A request may block for seconds; recheck ownership after that
            # wait, before opening/truncating any destination.
            with _open_asset(_asset_path(staging, asset.path)) as output:
                while True:
                    if should_stop() or time.monotonic() >= deadline:
                        raise TimeoutError("model_download_stopped")
                    block = response.read1(64 * 1024)
                    if not block:
                        break
                    received += len(block)
                    file_bytes += len(block)
                    if received > max_bytes or file_bytes > asset.bytes:
                        raise ValueError("model_download_byte_limit")
                    output.write(block)
                    digest.update(block)
                    on_progress(received)
        if file_bytes != asset.bytes or digest.hexdigest() != asset.sha256:
            raise ValueError("model_download_integrity")
    verify_installation(staging, profile)
    return received
