"""Supervised download credentials and cross-process staging ownership controls."""

import asyncio
import subprocess
import sys
import time

from test_download import redirect_directory
from test_model_setup import SmallSetup, wait

from core.persistence.models import new_id, utc_timestamp
from core.search.supervisor import stop_owned_process_tree, worker_environment


def test_download_child_has_no_core_provider_proxy_or_research_environment(store, monkeypatch):
    names = (
        "CORPI_CORE_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "CORPI_SOURCE_TEXT",
        "CORPI_QUERY_TEXT",
    )
    for name in names:
        monkeypatch.setenv(name, "public-sensitive-marker")
    command = """
import json
import os
import sys
import time
from pathlib import Path
assert len(sys.argv) == 5 and sys.argv[1] == 'minilm-en'
assert all(name not in os.environ for name in NAMES)
assert 'public-sensitive-marker' not in repr(sys.argv)
assert os.environ.get('PYTHONIOENCODING') == 'utf-8'
path = Path(sys.argv[2]) / 'assets'
path.mkdir(parents=True)
(path / 'asset.bin').write_bytes(b'good')
print(json.dumps({'type': 'done', 'bytes': 4}), flush=True)
time.sleep(30)
""".replace("NAMES", repr(names))
    original = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        # The inline test program contains its own public marker assertion.
        # Inspect the four actual worker arguments, not the test program text.
        assert "public-sensitive-marker" not in repr(args[0][-4:])
        assert "public-sensitive-marker" not in repr(kwargs.get("env"))
        child = original(*args, **kwargs)
        if args[0][0] == sys.executable:
            children.append(child)
        return child

    monkeypatch.setattr("core.search.install.subprocess.Popen", spawn)
    service = SmallSetup(store, worker_command=[sys.executable, "-c", command])
    try:
        service.begin("minilm-en", "anonymous-download", 0)
        assert wait(service)["state"] == "installed"
    finally:
        asyncio.run(service.stop())
    assert len(children) == 1 and children[0].poll() is not None


def interrupted_attempt(service, *, relative=None):
    profile = service.profile("minilm-en")
    service._ensure(profile)
    identity = new_id()
    relative = relative or "models/.setup/" + identity
    with service.store._transaction() as connection:
        timestamp = utc_timestamp()
        connection.execute(
            "INSERT INTO model_setup_attempts(id,request_key,manifest_id,generation,state,"
            "staging_path,created_at,updated_at) VALUES(?,?,?,1,'running',?,?,?)",
            (identity, "staging-owner", profile.identity, relative, timestamp, timestamp),
        )
        connection.execute(
            "UPDATE model_installations SET state='downloading',generation=1,staging_path=? "
            "WHERE manifest_id=?",
            (relative, profile.identity),
        )
    return identity, relative


def test_recovery_waits_for_actual_child_lock_and_cleans_after_process_exit(store):
    service = SmallSetup(store)
    identity, relative = interrupted_attempt(service)
    path = service._path(relative)
    script = """
import sys
import time
from pathlib import Path
from core.persistence.corelock import DataDirectoryLock
path = Path(sys.argv[1])
with DataDirectoryLock.acquire(path):
    (path / 'assets').mkdir()
    (path / 'assets' / 'asset.bin').write_bytes(b'good')
    (path / 'writer-ready').write_bytes(b'locked')
    time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=worker_environment(),
    )
    try:
        deadline = time.monotonic() + 10
        while not (path / "writer-ready").exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        restarted = SmallSetup(store)
        restarted.recover()
        assert restarted.state("minilm-en")["state"] == "failed"
        assert (path / "assets/asset.bin").read_bytes() == b"good"
        with store.database.connect() as connection:
            assert connection.execute(
                "SELECT state,cleanup_pending FROM model_setup_attempts WHERE id=?", (identity,)
            ).fetchone()[:] == ("interrupted", 1)
    finally:
        stop_owned_process_tree(child)
    assert child.poll() is not None
    restarted.cleanup()
    assert not path.exists()
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT cleanup_pending FROM model_setup_attempts WHERE id=?", (identity,)
            ).fetchone()[0]
            == 0
        )


def test_cleanup_refuses_redirected_owner_path_and_preserves_unrecorded_files(store, tmp_path):
    service = SmallSetup(store)
    identity, relative = interrupted_attempt(service)
    path = service.data_dir / relative
    path.parent.mkdir(parents=True)
    unrelated = path.parent / "unrecorded"
    unrelated.mkdir()
    (unrelated / "sentinel").write_bytes(b"retain unowned staging")
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"retain outside owner")
    redirect_directory(path, external)
    service.recover()
    assert (external / "sentinel").read_bytes() == b"retain outside owner"
    assert (unrelated / "sentinel").read_bytes() == b"retain unowned staging"
    assert path.exists()
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT cleanup_pending FROM model_setup_attempts WHERE id=?", (identity,)
            ).fetchone()[0]
            == 1
        )
