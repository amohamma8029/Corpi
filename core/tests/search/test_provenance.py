"""Version lookup is process-local; returned metadata never owns the expectation."""

import importlib
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest

from core.search import provenance


@pytest.fixture
def metadata(monkeypatch):
    importlib.reload(provenance)
    calls = []

    def installed(name):
        calls.append(name)
        return "installed-" + name

    with monkeypatch.context() as scoped:
        scoped.setattr(provenance, "version", installed)
        try:
            yield calls
        finally:
            # Other modules may retain references to public functions. Reload
            # their shared globals too, so fake metadata cannot escape this test.
            importlib.reload(provenance)


def expected():
    return {
        "contract": provenance.EMBEDDING_CONTRACT,
        **{name: "installed-" + name for name in provenance.RUNTIME_PACKAGES},
    }


def test_runtime_metadata_is_read_once_per_interpreter(metadata):
    for _ in range(3):
        assert provenance.runtime_versions() == expected()
        assert provenance.validate_runtime_versions(expected()) == expected()
    assert metadata == list(provenance.RUNTIME_PACKAGES)


def test_mutating_returned_metadata_cannot_change_the_validation_expectation(metadata):
    altered = provenance.runtime_versions()
    altered["fastembed"] = "untrusted-version"
    assert provenance.runtime_versions() == expected()
    with pytest.raises(ValueError, match="^runtime_mismatch$"):
        provenance.validate_runtime_versions(altered)
    validated = provenance.validate_runtime_versions(expected())
    validated["contract"] = "untrusted-contract"
    assert provenance.runtime_versions() == expected()
    with pytest.raises(ValueError, match="^runtime_mismatch$"):
        provenance.validate_runtime_versions(validated)


@pytest.mark.parametrize("bad", [None, {}, "versions", {"fastembed": "untrusted-version"}])
def test_malformed_runtime_metadata_still_fails_closed(metadata, bad):
    with pytest.raises(ValueError, match="^runtime_mismatch$"):
        provenance.validate_runtime_versions(bad)


def test_missing_package_does_not_cache_a_partial_or_failed_snapshot(metadata, monkeypatch):
    lookup = provenance.version
    failed = False

    def temporarily_missing(name):
        nonlocal failed
        if name == "onnxruntime" and not failed:
            failed = True
            raise PackageNotFoundError(name)
        return lookup(name)

    monkeypatch.setattr(provenance, "version", temporarily_missing)
    with pytest.raises(PackageNotFoundError):
        provenance.runtime_versions()
    assert provenance.runtime_versions() == expected()
    assert metadata == ["fastembed", *provenance.RUNTIME_PACKAGES]


def test_new_interpreters_capture_independent_runtime_metadata(metadata):
    assert provenance.runtime_versions() == expected()
    for prefix in ("first-worker-", "replacement-worker-"):
        code = """
import json, os, sys
from core.search import provenance
calls = []
def installed(name):
    calls.append(name)
    return sys.argv[1] + name
provenance.version = installed
first = provenance.runtime_versions()
assert provenance.validate_runtime_versions(first) == first
assert calls == list(provenance.RUNTIME_PACKAGES)
print(json.dumps({'pid': os.getpid(), 'versions': first}))
"""
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, prefix],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        child = json.loads(process.stdout)
        assert child["pid"] != os.getpid()
        assert child["versions"] == {
            "contract": provenance.EMBEDDING_CONTRACT,
            **{name: prefix + name for name in provenance.RUNTIME_PACKAGES},
        }
    assert provenance.runtime_versions() == expected()
