"""Benchmark-only pinned Potion and amgix static adapters.

The fixed single-F32-table safetensors export is memory-mapped read-only. Arithmetic
follows inspected Model2Vec StaticModel: no special tokens, exclude tokenizer UNK,
mean token rows, float32 L2 normalization. Reject overflow instead of truncating.
Reference:
https://github.com/MinishLab/model2vec/blob/066f91ab70768a64205f0f8aef128510056776af/model2vec/model.py
Amgix follows ST 5.7.0 StaticEmbedding mean, including UNK, without special tokens.
Its explicit 512-token diagnostic ceiling rejects overflow; the original tokenizer's
32768-token truncation is disabled. No Model2Vec/ST package is installed or executed.
"""

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

# Set before importing NumPy in the isolated static inference child.
if "--worker" in sys.argv:
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
from tokenizers import Tokenizer

from core.search.download import download_profile
from core.search.profiles import ModelFile, ModelProfile, verify_installation
from core.search.runtime import vector_bytes
from core.search.supervisor import worker_environment

MODEL_REPO = "minishlab/potion-multilingual-128M"
MODEL_REVISION = "73908c3438cf03b6a01bcb9611d62b23d0726f08"
MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/static_multilingual_model.json"
RETRIEVAL_MANIFEST = MANIFEST.with_name("static_retrieval_model.json")


def load_profile(manifest: Path = MANIFEST) -> ModelProfile:
    raw = json.loads(manifest.read_text("utf-8"))
    raw["files"] = tuple(ModelFile(**item) for item in raw["files"])
    return ModelProfile(**raw)


def pin(manifest: Path, *, retrieval: bool = False) -> None:
    from pin_search_models import fetch

    repo = "amgix/static-retrieval-multilingual-69m-v1" if retrieval else MODEL_REPO
    revision = "f777a93cd958b280b6c7fad5aaac3cd59d676822" if retrieval else MODEL_REVISION
    info = json.loads(
        fetch(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    )
    required = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    if retrieval:
        required = {"model.safetensors", "config.json", "modules.json", "tokenizer.json"}
    assets = []
    for item in info["siblings"]:
        name = item["rfilename"]
        if name not in required:
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        if "lfs" in item:
            size, digest = item["lfs"]["size"], item["lfs"]["sha256"]
        else:
            data = fetch(url)
            size, digest = len(data), hashlib.sha256(data).hexdigest()
        assets.append({"path": name, "bytes": size, "sha256": digest, "url": url})
    if {item["path"] for item in assets} != required or info["sha"] != revision:
        raise ValueError("incomplete pinned candidate")
    raw = dict(
        id="amgix-static-retrieval" if retrieval else "potion-multilingual-static",
        model=repo,
        repository=repo,
        revision=revision,
        license="apache-2.0" if retrieval else "mit",
        pooling="static-mean-include-unk" if retrieval else "static-mean",
        dimensions=384 if retrieval else 256,
        input_tokens=512,
        chunk_tokens=384,
        overlap_tokens=64,
        query_prefix="",
        passage_prefix="",
        model_file="model.safetensors",
        runtime="benchmark-st-5.7-static-f32-mmap-v1"
        if retrieval
        else "benchmark-static-f32-mmap-v1",
        vector_format="normalized-float32-le",
        separator="\n\n",
        files=sorted(assets, key=lambda item: item["path"]),
    )
    if manifest.exists():
        raise ValueError("manifest already exists")
    manifest.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "profile": raw["id"],
                "revision": revision,
                "download_bytes": sum(item["bytes"] for item in assets),
            },
            indent=2,
        )
    )


def map_embeddings(path: Path, dimensions: int, *, include_unk: bool = False):
    """Read only the verified export's one contiguous little-endian F32 table."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("invalid_static_table")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 1 <= header_size <= 65536:
            raise ValueError("invalid_static_table")
        raw = json.loads(stream.read(header_size))
    keys = set(raw) - {"__metadata__"} if include_unk else set(raw)
    key = "embedding.weight" if include_unk and "embedding.weight" in keys else "embeddings"
    if keys != {key}:
        raise ValueError("invalid_static_table")
    item = raw[key]
    shape = item["shape"]
    if item["dtype"] != "F32" or len(shape) != 2 or shape[1] != dimensions:
        raise ValueError("invalid_static_table")
    if not isinstance(shape[0], int) or not 1 <= shape[0] <= 1_000_000:
        raise ValueError("invalid_static_table")
    data_bytes = shape[0] * dimensions * 4
    if (
        item["data_offsets"] != [0, data_bytes]
        or path.stat().st_size != 8 + header_size + data_bytes
    ):
        raise ValueError("invalid_static_table")
    return np.memmap(path, dtype="<f4", mode="r", offset=8 + header_size, shape=tuple(shape))


class StaticTokenizer:
    def __init__(self, path: Path, profile: ModelProfile):
        self.profile = profile
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        model = json.loads(self.tokenizer.to_str())["model"]
        if model["type"] == "Unigram":
            self.unknown_id = model.get("unk_id")
        else:
            token = model.get("unk_token")
            self.unknown_id = self.tokenizer.token_to_id(token) if token is not None else None

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def prepare(self, text: str, *, query: bool = False) -> str:
        if self.count(text) > self.profile.input_tokens:
            raise ValueError("model_input_too_long")
        return text


class StaticRuntime:
    def __init__(self, path: Path, profile: ModelProfile):
        verify_installation(path, profile)
        self.profile = profile
        self.tokenizer = StaticTokenizer(path, profile)
        self.include_unk = profile.pooling == "static-mean-include-unk"
        self.embeddings = map_embeddings(
            path / profile.model_file, profile.dimensions, include_unk=self.include_unk
        )
        # Some published special-token aliases exceed the table; encode without
        # added special tokens and reject any actually encountered out-of-range ID.

    def embed(self, texts, *, query=False):
        if not 1 <= len(texts) <= 32:
            raise ValueError("invalid_embedding_batch")
        vectors = []
        for text in texts:
            self.tokenizer.prepare(text, query=query)
            ids = self.tokenizer.tokenizer.encode(text, add_special_tokens=False).ids
            if not self.include_unk:
                ids = [index for index in ids if index != self.tokenizer.unknown_id]
            if not ids or min(ids) < 0 or max(ids) >= len(self.embeddings):
                raise ValueError("invalid_embedding_input")
            mean = self.embeddings[ids].mean(axis=0)
            normalized = mean / (np.linalg.norm(mean) + np.float32(1e-32))
            vectors.append(vector_bytes(normalized.tolist(), self.profile.dimensions))
        return vectors


def deny_network(*args, **kwargs):
    raise OSError("benchmark inference network denied")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--retrieval", action="store_true", help="Select pinned amgix static model")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-child", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("worker_args", nargs="*")
    args = parser.parse_args()
    if args.retrieval:
        args.manifest = RETRIEVAL_MANIFEST
    if args.pin:
        pin(args.manifest, retrieval=args.retrieval)
        return
    profile = load_profile(args.manifest)
    if args.worker:
        from core.search import worker

        socket.socket.connect = deny_network
        socket.socket.connect_ex = deny_network
        worker.profiles = lambda: {profile.id: profile}
        # Replace the factory in this child only, retaining the bounded protocol.
        worker.__dict__["EmbeddingRuntime"] = StaticRuntime
        sys.argv = [sys.argv[0], *args.worker_args]
        raise SystemExit(worker.main())
    if args.directory is None:
        parser.error("directory is required for explicit setup")
    target = args.directory / profile.identity
    if args.download_child:
        started = time.monotonic()
        received = download_profile(target, profile)
        print(json.dumps({"received_bytes": received, "seconds": time.monotonic() - started}))
        return
    if args.download:
        if target.exists():
            verify_installation(target, profile)
            print("Existing pinned candidate verified")
            return
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--download-child",
            "--manifest",
            str(args.manifest.resolve()),
            "--directory",
            str(args.directory),
        ]
        started = time.monotonic()
        process = subprocess.Popen(command, env=worker_environment())
        from core.search.supervisor import stop_owned_process_tree

        try:
            if process.wait(timeout=600) != 0:
                raise RuntimeError("candidate setup failed; incomplete staging retained")
        finally:
            stop_owned_process_tree(process)
        result = {
            "profile": profile.id,
            "manifest": profile.identity,
            "installed_bytes": profile.download_bytes,
            "setup_wall_seconds": time.monotonic() - started,
            "launcher_reaped": process.poll() is not None,
        }
        (args.directory / f"{profile.id}-setup.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
