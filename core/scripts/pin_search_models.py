"""Developer-only manifest generation; the app never resolves mutable model revisions.

Run from core with an explicit output path. Large weights are not downloaded here:
their SHA-256/size are supplied by the pinned repository's LFS metadata. Small
configuration files are downloaded and hashed. Review the generated manifest diff.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

CANDIDATES = (
    (
        "bge-small-en",
        "BAAI/bge-small-en-v1.5",
        "Qdrant/bge-small-en-v1.5-onnx-Q",
        "model_optimized.onnx",
        "mit",
        "cls",
        512,
        384,
        64,
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    (
        "multilingual-minilm",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        "model_optimized.onnx",
        "apache-2.0",
        "mean",
        128,
        96,
        16,
        "",
        "",
    ),
    (
        "multilingual-e5",
        "intfloat/multilingual-e5-small",
        "Xenova/multilingual-e5-small",
        "onnx/model_quantized.onnx",
        "mit",
        "mean",
        512,
        384,
        64,
        "query: ",
        "passage: ",
    ),
    (
        "minilm-en",
        "sentence-transformers/all-MiniLM-L6-v2",
        "Xenova/all-MiniLM-L6-v2",
        "onnx/model_quantized.onnx",
        "apache-2.0",
        "mean",
        256,
        192,
        32,
        "",
        "",
    ),
)


def fetch(url: str, limit: int = 20_000_000) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata/configuration exceeds limit")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=[candidate[0] for candidate in CANDIDATES])
    args = parser.parse_args()
    manifests = json.loads(args.output.read_text("utf-8")) if args.output.exists() else []
    for (
        key,
        name,
        repo,
        weights,
        license_name,
        pooling,
        limit,
        target,
        overlap,
        query,
        passage,
    ) in CANDIDATES:
        if args.profile and args.profile != key:
            continue
        info = json.loads(fetch(f"https://huggingface.co/api/models/{repo}?blobs=true"))
        revision = info["sha"]
        required = {
            weights,
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        }
        files = []
        for item in info["siblings"]:
            path = item["rfilename"]
            if path not in required:
                continue
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
            if path == weights:
                size = item["lfs"]["size"]
                digest = item["lfs"]["sha256"]
            else:
                data = fetch(url)
                size = len(data)
                digest = hashlib.sha256(data).hexdigest()
            files.append({"path": path, "bytes": size, "sha256": digest, "url": url})
        if {item["path"] for item in files} != required:
            raise ValueError(f"missing required files in {repo}")
        manifests = [manifest for manifest in manifests if manifest["id"] != key]
        manifests.append(
            {
                "id": key,
                "model": name,
                "repository": repo,
                "revision": revision,
                "license": license_name,
                "pooling": pooling,
                "dimensions": 384,
                "input_tokens": limit,
                "chunk_tokens": target,
                "overlap_tokens": overlap,
                "query_prefix": query,
                "passage_prefix": passage,
                "model_file": weights,
                "runtime": "fastembed==0.8.1",
                "vector_format": "normalized-float32-le",
                "separator": "\n\n",
                "files": sorted(files, key=lambda item: item["path"]),
            }
        )
        print(key, revision, sum(item["bytes"] for item in files))
    args.output.write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
