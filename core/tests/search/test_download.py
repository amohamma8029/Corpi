"""Real urllib routing with an in-memory HTTP transport; no external requests."""

import hashlib
import io
import os
import subprocess
import urllib.request
import urllib.response
from dataclasses import replace
from email.message import Message
from pathlib import Path

import pytest

from core.search.download import download_profile
from core.search.profiles import ModelFile, profiles

ORIGIN = "https://huggingface.co/test/model/resolve/" + "a" * 40 + "/asset.bin"
CDN = "https://us.aws.cdn.hf.co/model/asset.bin?asset-signature=public-test-grant"
GOOD = b"good"


@pytest.fixture()
def transport(monkeypatch):
    routes = {}
    requests = []
    redirect_reads = []

    class Body(io.BytesIO):
        def read(self, size: int | None = -1):
            redirect_reads.append(size)
            return super().read(size)

    class Response(urllib.response.addinfourl):
        msg = "controlled transport"

    def receive(_handler, request):
        requests.append(request)
        code, location, content = routes[request.full_url]
        if callable(content):
            content = content()
        assert isinstance(content, bytes)
        headers = Message()
        if location is not None:
            headers["Location"] = location
        stream = Body(content) if code in (301, 302, 303, 307, 308) else io.BytesIO(content)
        return Response(stream, headers, request.full_url, code)

    monkeypatch.setattr(urllib.request, "_opener", None)
    for handler, method in (
        (urllib.request.HTTPSHandler, "https_open"),
        (urllib.request.HTTPHandler, "http_open"),
        (urllib.request.FTPHandler, "ftp_open"),
    ):
        monkeypatch.setattr(handler, method, receive)
    return routes, requests, redirect_reads


def profile(path="asset.bin", url=ORIGIN):
    return replace(
        profiles()["minilm-en"],
        repository="test/model",
        revision="a" * 40,
        files=(ModelFile(path, 4, hashlib.sha256(GOOD).hexdigest(), url),),
    )


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_approved_redirects_use_anonymous_get_and_do_not_read_redirect_bodies(
    tmp_path, transport, status
):
    routes, requests, reads = transport
    routes[ORIGIN] = (status, CDN, b"redirect body must not be downloaded")
    routes[CDN] = (200, None, GOOD)
    progress = []
    assert download_profile(tmp_path / "assets", profile(), on_progress=progress.append) == 4
    assert (tmp_path / "assets/asset.bin").read_bytes() == GOOD
    assert progress[-1] == 4
    assert reads == []
    assert [request.full_url for request in requests] == [ORIGIN, CDN]
    for request in requests:
        assert request.get_method() == "GET" and request.data is None
        assert {key.lower() for key, _ in request.header_items()} <= {
            "user-agent",
            "host",
            "connection",
        }


@pytest.mark.parametrize(
    "target",
    [
        "http://huggingface.co/public.bin",
        "http://127.0.0.1/private",
        "ftp://huggingface.co/a",
        "https://unapproved.example/a",
        "https://huggingface.co.unapproved.example/a",
        "https://127.0.0.1/a",
        "https://[::1]/a",
        "https://huggingface.co:444/a",
        "https://user:secret@us.aws.cdn.hf.co/a",
        "https://us.aws.cdn.hf.co/a#fragment",
        "https://huggingface.co/a\n",
        "https://huggingface.co/a b",
        "https://huggingface.co/a\\b",
    ],
)
def test_bad_redirect_is_rejected_before_any_destination_contact(tmp_path, transport, target):
    routes, requests, _ = transport
    routes[ORIGIN] = (302, target, b"")
    routes[target] = (200, None, GOOD)
    with pytest.raises(ValueError):
        download_profile(tmp_path / "assets", profile())
    assert [request.full_url for request in requests] == [ORIGIN]


def test_relative_redirect_and_final_hash_checks_remain_required(tmp_path, transport):
    routes, requests, _ = transport
    target = ORIGIN + "?download=true"
    routes[ORIGIN] = (307, "?download=true", b"")
    routes[target] = (200, None, b"evil")
    with pytest.raises(ValueError, match="model_download_integrity"):
        download_profile(tmp_path / "assets", profile())
    assert [request.full_url for request in requests] == [ORIGIN, target]


def test_redirect_loop_is_finite_and_never_drains_bodies(tmp_path, transport):
    routes, requests, reads = transport
    routes[ORIGIN] = (302, ORIGIN, b"unused redirect data")
    with pytest.raises(ValueError, match="model_download_redirect"):
        download_profile(tmp_path / "assets", profile())
    assert len(requests) == 11  # Initial GET plus at most ten followed redirects.
    assert reads == []


def test_redirect_rechecks_cancellation_before_following(tmp_path, transport):
    routes, requests, reads = transport
    routes[ORIGIN] = (302, CDN, b"unused redirect data")
    with pytest.raises(TimeoutError, match="model_download_stopped"):
        download_profile(tmp_path / "assets", profile(), should_stop=lambda: bool(requests))
    assert [request.full_url for request in requests] == [ORIGIN] and reads == []


def test_prior_partial_can_be_retried_without_changing_other_assets(tmp_path, transport):
    routes, _, _ = transport
    routes[ORIGIN] = (200, None, GOOD)
    staging = tmp_path / "assets"
    staging.mkdir()
    (staging / "asset.bin").write_bytes(b"partial bytes longer than the complete asset")
    sentinel = staging / "other"
    sentinel.write_bytes(b"retain unrelated data")
    assert download_profile(staging, profile()) == 4
    assert (staging / "asset.bin").read_bytes() == GOOD
    assert sentinel.read_bytes() == b"retain unrelated data"


def test_download_does_not_use_global_cookie_or_auth_opener(tmp_path, transport):
    routes, requests, _ = transport
    routes[ORIGIN] = (302, CDN, b"")
    routes[CDN] = (200, None, GOOD)

    class Credentialed(urllib.request.BaseHandler):
        def http_request(self, request):
            request.add_header("Authorization", "Bearer public-private-marker")
            request.add_header("Cookie", "public-cookie-marker")
            request.add_header("Referer", "public-research-marker")
            return request

        https_request = http_request

    urllib.request.install_opener(urllib.request.build_opener(Credentialed()))
    assert download_profile(tmp_path / "assets", profile()) == 4
    assert all(
        not any(
            marker in str(request.header_items())
            for marker in (
                "public-private-marker",
                "public-cookie-marker",
                "public-research-marker",
            )
        )
        for request in requests
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://unapproved.example/asset.bin",
        "http://huggingface.co/asset.bin",
        ORIGIN + "?research=public-marker",
        ORIGIN + "#fragment",
    ],
)
def test_initial_request_must_be_exact_pinned_manifest_url(tmp_path, transport, url):
    routes, requests, _ = transport
    routes[url] = (200, None, GOOD)
    with pytest.raises(ValueError):
        download_profile(tmp_path / "assets", profile(url=url))
    assert requests == []


def redirect_directory(link: Path, target: Path):
    if os.name == "nt":
        command = Path(os.environ["SYSTEMROOT"]) / "System32/cmd.exe"
        result = subprocess.run(
            [str(command), "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert link.is_junction()
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("where", ["root", "parent"])
def test_redirected_staging_cannot_create_directories_or_write_outside_owned_root(
    tmp_path, transport, where
):
    routes, requests, _ = transport
    routes[ORIGIN] = (200, None, GOOD)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_bytes(b"untouched")
    staging = tmp_path / "assets"
    if where == "root":
        redirect_directory(staging, external)
        asset = profile()
    else:
        staging.mkdir()
        redirect_directory(staging / "redirect", external)
        asset = profile(path="redirect/new/asset.bin")
        asset = replace(
            asset,
            files=(
                replace(asset.files[0], url=ORIGIN.replace("asset.bin", "redirect/new/asset.bin")),
            ),
        )
        routes[asset.files[0].url] = (200, None, GOOD)
    with pytest.raises(ValueError, match="unsafe_model_staging"):
        download_profile(staging, asset)
    assert sentinel.read_bytes() == b"untouched"
    assert sorted(path.name for path in external.iterdir()) == ["sentinel"]
    assert requests == []


def test_asset_hardlink_cannot_truncate_another_owned_file(tmp_path, transport):
    routes, requests, _ = transport
    routes[ORIGIN] = (200, None, GOOD)
    external = tmp_path / "protected.bin"
    external.write_bytes(b"protected original")
    staging = tmp_path / "assets"
    staging.mkdir()
    os.link(external, staging / "asset.bin")
    with pytest.raises(ValueError, match="unsafe_model_staging"):
        download_profile(staging, profile())
    assert external.read_bytes() == b"protected original"


def test_staging_is_rechecked_after_waiting_for_network_response(tmp_path, transport):
    routes, _, _ = transport
    staging = tmp_path / "assets"
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "asset.bin"
    protected.write_bytes(b"protected original")

    def move_during_request():
        staging.rename(tmp_path / "previous-owned-assets")
        redirect_directory(staging, external)
        return GOOD

    routes[ORIGIN] = (200, None, move_during_request)
    with pytest.raises(ValueError) as failure:
        download_profile(staging, profile())
    assert protected.read_bytes() == b"protected original"
    assert str(failure.value) == "unsafe_model_staging"


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "C:/drive", "asset.bin:stream", "a\\b"]
)
def test_unsafe_asset_paths_fail_before_network_or_directory_creation(tmp_path, transport, path):
    routes, requests, _ = transport
    routes[ORIGIN] = (200, None, GOOD)
    with pytest.raises(ValueError):
        download_profile(tmp_path / "assets", profile(path=path))
    assert requests == []
