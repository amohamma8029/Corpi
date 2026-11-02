"""Pinned contextual sentence-output candidates, isolated from the app catalog.

Use the trained pooler or exported final sentence embedding, not an arbitrary
CLS/mean substitute. All inference remains CPU-only, local, bounded and supervised.
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
from collections import OrderedDict
from pathlib import Path

if "--worker" in sys.argv:
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from core.search.download import download_profile
from core.search.profiles import ModelFile, ModelProfile, verify_installation
from core.search.runtime import InputTokenizer, vector_bytes
from core.search.supervisor import stop_owned_process_tree, worker_environment

MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/contextual_multilingual_models.json"
POOLER_ASSET = dict(
    path="reference/model.safetensors",
    bytes=276003000,
    sha256="653b81e09f9243357533edf511c3f1361efad08df77b94db0b4e51a2b70fe4ec",
    url="https://huggingface.co/setu4993/LEALLA-small/resolve/8fadf81fe3979f373ba9922ab616468a4184b266/model.safetensors",
)
SPECS = (
    (
        "bekko-a8m",
        "hotchpotch/bekko-embedding-v1-a8m",
        "hotchpotch/bekko-embedding-v1-a8m",
        "c721113d59a1d91b447450324f51c4b3332c924a",
        384,
        "masked-mean",
    ),
    (
        "lealla-small",
        "setu4993/LEALLA-small",
        "sunzhuo/LEALLA-small",
        "0277927f474f1b1c5c8eccb6f372b18403035cdd",
        128,
        "pooler_output",
    ),
    (
        "mmbert-student",
        "Horizon-Labs/multilingual-embedding-small",
        "Horizon-Labs/multilingual-embedding-small",
        "6418071ac0a526fea6f72230390298d5e1a5c7b7",
        1024,
        "sentence_embedding",
    ),
    (
        "distiluse-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "bfe45d0732ca50787611c0fe107ba278c7f3f889",
        512,
        "mean-dense-tanh",
    ),
    (
        "granite-107m",
        "ibm-granite/granite-embedding-107m-multilingual",
        "gety-ai/granite-embedding-107m-multilingual-onnx",
        "4713bc4053d578f45e263e5857262b69f1f4296f",
        384,
        "cls",
    ),
    (
        "granite-97m-r2",
        "ibm-granite/granite-embedding-97m-multilingual-r2",
        "ibm-granite/granite-embedding-97m-multilingual-r2",
        "835ad14087e140460703cf0fae09f97d469d65c2",
        384,
        "cls",
    ),
)
CPU_FILES = {
    "bekko-a8m": "onnx/model.onnx",
    "distiluse-v2": "onnx/model_quint8_avx2.onnx",
    "granite-107m": "onnx/model_quint8_avx2.onnx",
    "granite-97m-r2": "onnx/model_quint8_avx2.onnx",
}


def profiles():
    if not MANIFEST.exists():
        return {}
    result = {}
    for raw in json.loads(MANIFEST.read_text("utf-8")):
        raw["files"] = tuple(ModelFile(**item) for item in raw["files"])
        model = ModelProfile(**raw)
        result[model.id] = model
    return result


def pin(add=False):
    from pin_search_models import fetch

    if MANIFEST.exists() and not add:
        raise ValueError("contextual manifest already exists")
    models = json.loads(MANIFEST.read_text("utf-8")) if add else []
    existing = {model["id"] for model in models}
    for key, model, repo, revision, dimensions, output in SPECS:
        if key in existing:
            continue
        info = json.loads(
            fetch(
                f"https://huggingface.co/api/models/{repo}/revision/{revision}"
                "?blobs=true&expand[]=sha&expand[]=siblings"
            )
        )
        graph_file = CPU_FILES.get(key, "onnx/model_quantized.onnx")
        required = {
            graph_file,
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        }
        if key == "distiluse-v2":
            required |= {
                "2_Dense/model.safetensors",
                "2_Dense/config.json",
                "1_Pooling/config.json",
                "sentence_bert_config.json",
            }
        elif key.startswith("granite") or key == "bekko-a8m":
            required |= {"1_Pooling/config.json"}
        if key == "bekko-a8m":
            required |= {"modules.json", "sentence_bert_config.json"}
        assets = []
        for item in info["siblings"]:
            path = item["rfilename"]
            if path not in required:
                continue
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
            if "lfs" in item:
                size, digest = item["lfs"]["size"], item["lfs"]["sha256"]
            else:
                raw = fetch(url)
                size, digest = len(raw), hashlib.sha256(raw).hexdigest()
            assets.append(dict(path=path, bytes=size, sha256=digest, url=url))
        if {item["path"] for item in assets} != required or info["sha"] != revision:
            raise ValueError("pinned candidate assets incomplete")
        if key == "lealla-small":
            assets.append(POOLER_ASSET)
        models.append(
            dict(
                id=key,
                model=model,
                repository=repo,
                revision=revision,
                license="mit" if key == "bekko-a8m" else "apache-2.0",
                pooling=output,
                dimensions=dimensions,
                input_tokens=8192 if key == "bekko-a8m" else 128 if key == "distiluse-v2" else 512,
                chunk_tokens=96 if key == "distiluse-v2" else 384,
                overlap_tokens=16 if key == "distiluse-v2" else 64,
                query_prefix="",
                passage_prefix="",
                model_file=graph_file,
                runtime="benchmark-ort-trained-pooler-f32-v2"
                if key == "lealla-small"
                else "benchmark-ort-sentence-output-v1",
                vector_format="normalized-float32-le",
                separator="\n\n",
                files=sorted(assets, key=lambda item: item["path"]),
            )
        )
    MANIFEST.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {model["id"]: sum(item["bytes"] for item in model["files"]) for model in models},
            indent=2,
        )
    )


def dense_tensors(path, names, shapes):
    """Map only named trained F32 dense tensors from a verified checkpoint."""
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 1 <= size <= 1_000_000:
            raise ValueError("invalid_pooler_header")
        header = json.loads(stream.read(size))
    result = []
    for name, shape in zip(names, shapes, strict=True):
        item = header[name]
        start, end = item["data_offsets"]
        if (
            item["dtype"] != "F32"
            or item["shape"] != shape
            or start < 0
            or end - start != int(np.prod(shape)) * 4
            or 8 + size + end > path.stat().st_size
        ):
            raise ValueError("invalid_pooler_tensor")
        result.append(
            np.memmap(path, dtype="<f4", mode="r", offset=8 + size + start, shape=tuple(shape))
        )
    return tuple(result)


def pooler_tensors(path):
    return dense_tensors(path, ("pooler.dense.weight", "pooler.dense.bias"), ([128, 128], [128]))


def masked_mean(values, attention_mask):
    """SentenceTransformer mean pooling includes real special tokens, excludes pads."""
    if values.ndim != 3 or attention_mask.shape != values.shape[:2]:
        raise ValueError("invalid_candidate_token_states")
    mask = np.asarray(attention_mask, dtype=np.float32)
    sums = np.einsum("bsh,bs->bh", values, mask, dtype=np.float32, optimize=False)
    return sums / np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)


def failure_metadata(stage, error, path):
    if METADATA is not None:
        METADATA.write_text(
            json.dumps(
                {
                    "scope": "public pinned asset loading before any query",
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error).replace(str(path), "<verified-assets>"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class UTF8InputTokenizer(InputTokenizer):
    """Use the complete compiled tokenizer, without platform-default config decoding."""

    def __init__(self, path, profile):
        self.profile = profile
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self._counts = OrderedDict()


class ContextualRuntime:
    def __init__(self, path, profile):
        verify_installation(path, profile)
        self.profile = profile
        config = json.loads((path / "config.json").read_text("utf-8"))
        self.pad_id = config["pad_token_id"]
        try:
            self.tokenizer = UTF8InputTokenizer(path, profile)
        except Exception as error:
            failure_metadata("tokenizer_load", error, path)
            raise
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        options.enable_cpu_mem_arena = False
        try:
            self.model = ort.InferenceSession(
                str(path / profile.model_file),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )
        except Exception as error:
            failure_metadata("onnx_load", error, path)
            raise
        self.inputs = {item.name for item in self.model.get_inputs()}
        outputs = self.model.get_outputs()
        # Metadata belongs to the explicit benchmark result directory, not assets.
        if METADATA is not None:
            METADATA.write_text(
                json.dumps(
                    {
                        "inputs": [
                            {"name": item.name, "shape": item.shape, "type": item.type}
                            for item in self.model.get_inputs()
                        ],
                        "outputs": [
                            {"name": item.name, "shape": item.shape, "type": item.type}
                            for item in outputs
                        ],
                        "required_sentence_output": profile.pooling,
                        "providers": self.model.get_providers(),
                        "intra_op_threads": options.intra_op_num_threads,
                        "inter_op_threads": options.inter_op_num_threads,
                        "padding_token_id": self.pad_id,
                        "tokenizer_loader": "compiled JSON UTF-8; no truncation",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.output = profile.pooling
        self.pooler = None
        self.dense = None
        if profile.pooling in ("cls", "mean-dense-tanh", "masked-mean"):
            self.output = "last_hidden_state"
        if profile.pooling == "mean-dense-tanh":
            self.dense = dense_tensors(
                path / "2_Dense/model.safetensors",
                ("linear.weight", "linear.bias"),
                ([512, 768], [512]),
            )
        if profile.id == "lealla-small" and any(
            item.path == POOLER_ASSET["path"] for item in profile.files
        ):
            self.output = "last_hidden_state"
            self.pooler = pooler_tensors(path / POOLER_ASSET["path"])
        if self.output not in {item.name for item in outputs}:
            raise ValueError("required_trained_sentence_output_missing")
        if not self.inputs <= {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("unsupported_candidate_inputs")

    def embed(self, texts, *, query=False):
        if not 1 <= len(texts) <= 32:
            raise ValueError("invalid_embedding_batch")
        prepared = [self.tokenizer.prepare(text, query=query) for text in texts]
        tokenizer = self.tokenizer.tokenizer
        pad_token = tokenizer.id_to_token(self.pad_id)
        if pad_token is None:
            raise ValueError("invalid_padding_token")
        tokenizer.enable_padding(pad_id=self.pad_id, pad_token=pad_token)
        try:
            encodings = tokenizer.encode_batch(prepared)
        finally:
            tokenizer.no_padding()
        inputs = {
            "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.asarray(
                [item.attention_mask for item in encodings], dtype=np.int64
            ),
            "token_type_ids": np.asarray([item.type_ids for item in encodings], dtype=np.int64),
        }
        result = self.model.run(
            [self.output],
            {key: value for key, value in inputs.items() if key in self.inputs},
        )[0]
        if not isinstance(result, np.ndarray):
            raise ValueError("invalid_vector")
        if self.pooler is not None:
            weight, bias = self.pooler
            result = np.tanh(result[:, 0, :] @ weight.T + bias)
        elif self.dense is not None:
            weight, bias = self.dense
            result = np.tanh(masked_mean(result, inputs["attention_mask"]) @ weight.T + bias)
        elif self.profile.pooling == "cls":
            result = result[:, 0, :]
        elif self.profile.pooling == "masked-mean":
            result = masked_mean(result, inputs["attention_mask"])
        if result.shape != (len(texts), self.profile.dimensions):
            raise ValueError("invalid_vector_count")
        return [vector_bytes(row.tolist(), self.profile.dimensions) for row in result]


METADATA = None


def denied(*args, **kwargs):
    raise OSError("benchmark inference network denied")


def main():
    global METADATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--add-candidates", action="store_true")
    parser.add_argument("--restore-pooler", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-child", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--profile", choices=[item[0] for item in SPECS])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("worker_args", nargs="*")
    args = parser.parse_args()
    if args.pin or args.add_candidates:
        pin(add=args.add_candidates)
        return
    if args.restore_pooler:
        raw = json.loads(MANIFEST.read_text("utf-8"))
        backup = MANIFEST.with_name("contextual_multilingual_models.initial.json")
        if backup.exists():
            raise ValueError("initial manifest already retained")
        backup.write_bytes(MANIFEST.read_bytes())
        model = next(item for item in raw if item["id"] == "lealla-small")
        model["files"].append(POOLER_ASSET)
        model["files"].sort(key=lambda item: item["path"])
        model["runtime"] = "benchmark-ort-trained-pooler-f32-v2"
        MANIFEST.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print("Trained pooler asset pinned; initial failed manifest retained")
        return
    catalog = profiles()
    if args.worker:
        from core.search import worker

        METADATA = args.metadata
        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        worker.profiles = lambda: catalog
        worker.__dict__["EmbeddingRuntime"] = ContextualRuntime
        sys.argv = [sys.argv[0], *args.worker_args]
        raise SystemExit(worker.main())
    if args.profile is None or args.directory is None:
        parser.error("explicit setup requires profile and directory")
    profile = catalog[args.profile]
    target = args.directory / profile.identity
    if args.download_child:
        print(json.dumps({"received_bytes": download_profile(target, profile)}))
    elif args.download:
        if target.exists():
            verify_installation(target, profile)
            print("Existing pinned candidate verified")
            return
        started = time.monotonic()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--download-child",
            "--profile",
            args.profile,
            "--directory",
            str(args.directory),
        ]
        process = subprocess.Popen(command, env=worker_environment())
        try:
            if process.wait(timeout=600) != 0:
                raise RuntimeError("candidate setup failed; staging retained")
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
