"""Inspect verified local shortlist assets without loading a contextual encoder."""

import argparse
import hashlib
import json
import struct
from dataclasses import asdict
from pathlib import Path

from benchmark_shared_search import candidates, static_profile
from benchmark_static_multilingual import StaticTokenizer, map_embeddings
from tokenizers import Tokenizer

from core.search.profiles import verify_installation


def inspect(models_directory):
    results = []
    for name in ("amgix-static-retrieval", "bekko-a8m"):
        profile = candidates()[name]
        path = models_directory / profile.identity
        verify_installation(path, profile)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        raw = json.loads(tokenizer.to_str())
        config = json.loads((path / "config.json").read_text("utf-8"))
        modules = json.loads((path / "modules.json").read_text("utf-8"))
        item = {
            "profile": asdict(profile),
            "manifest": profile.identity,
            "hashes_verified": True,
            "config": config,
            "modules": modules,
            "tokenizer": {
                key: raw.get(key)
                for key in (
                    "truncation",
                    "padding",
                    "normalizer",
                    "pre_tokenizer",
                    "post_processor",
                )
            },
            "vocabulary_size": tokenizer.get_vocab_size(),
            "tokenizer_model_type": raw["model"]["type"],
        }
        if static_profile(profile):
            table = map_embeddings(path / profile.model_file, profile.dimensions, include_unk=True)
            with (path / profile.model_file).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                item["table_header"] = json.loads(stream.read(size))
            counter = StaticTokenizer(path, profile)
            item.update(table_shape=list(table.shape), unknown_id=counter.unknown_id)
            if list(table.shape) != [179936, 384] or config["normalize"]:
                raise ValueError("unexpected static reference settings")
            if [module["type"] for module in modules] != [
                "sentence_transformers.models.StaticEmbedding"
            ]:
                raise ValueError("unexpected static reference modules")
        else:
            pooling = json.loads((path / "1_Pooling/config.json").read_text("utf-8"))
            item["pooling"] = pooling
            if not pooling["pooling_mode_mean_tokens"] or any(
                value
                for key, value in pooling.items()
                if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens"
            ):
                raise ValueError("unexpected contextual pooling")
            item["sentence_config"] = json.loads(
                (path / "sentence_bert_config.json").read_text("utf-8")
            )
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("inspection output already exists")
    data = {
        "scope": "verified pinned local assets; no contextual inference",
        "models": inspect(args.models_directory),
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "models": [item["profile"]["id"] for item in data["models"]],
                "receipt_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
