"""Import-light execution metadata; immutable model IDs describe a separate contract."""

from functools import cache
from importlib.metadata import version

EMBEDDING_CONTRACT = "fastembed-0.8.1-v1"
RUNTIME_PACKAGES = ("fastembed", "onnxruntime", "tokenizers", "numpy", "pillow")


@cache
def _runtime_items() -> tuple[tuple[str, str], ...]:
    # Installed libraries are fixed for this interpreter's lifetime. Each new
    # worker captures its own metadata; never share a caller-mutable snapshot.
    return (("contract", EMBEDDING_CONTRACT), *((name, version(name)) for name in RUNTIME_PACKAGES))


def runtime_versions() -> dict[str, str]:
    return dict(_runtime_items())


def validate_runtime_versions(value: object) -> dict[str, str]:
    """Only accept the exact owned interpreter's metadata, never arbitrary child text."""
    expected = runtime_versions()
    if not isinstance(value, dict) or value != expected:
        raise ValueError("runtime_mismatch")
    return expected
