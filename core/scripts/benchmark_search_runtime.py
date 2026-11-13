"""Early real-inference diagnostic, not the S14 PDF/quality acceptance benchmark.

Explicit --download permits asset setup in the supplied isolated directory.
No download occurs in the normal embedding adapter. Writes only measured JSON.
"""

import argparse
import json
import os
import platform
import threading
import time
from importlib.metadata import version
from pathlib import Path

from core.search.download import download_profile
from core.search.memory import resident_bytes
from core.search.profiles import profiles
from core.search.runtime import EmbeddingRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--profile", choices=list(profiles()), required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--content-tokens", type=int)
    parser.add_argument("--disable-arena", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batches <= 313:
        parser.error("batches must be 1..313")
    profile = profiles()[args.profile]
    target = args.content_tokens or profile.chunk_tokens
    if not 1 <= args.batch_size <= 32 or not 1 <= target <= profile.chunk_tokens:
        parser.error("batch size must be 1..32 and content tokens within the profile target")
    path = args.directory / profile.identity
    if args.download:
        download_profile(path, profile)
    stop = threading.Event()
    samples = []

    def sample() -> None:
        while not stop.is_set():
            samples.append(resident_bytes(os.getpid()))
            stop.wait(0.25)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        start = time.perf_counter()
        runtime = EmbeddingRuntime(path, profile, memory_arena=not args.disable_arena)
        loaded = time.perf_counter() - start
        sentence = "The experiment compared rainfall and soil moisture across the study sites. "
        text = sentence
        while runtime.tokenizer.count(text + sentence) <= target:
            text += sentence
        times = []
        for _ in range(args.batches):
            start = time.perf_counter()
            vectors = runtime.embed([text] * args.batch_size)
            times.append(time.perf_counter() - start)
            assert len(vectors) == args.batch_size
        start = time.perf_counter()
        runtime.embed(["Which environmental measurements did the experiment compare?"], query=True)
        query_seconds = time.perf_counter() - start
    finally:
        stop.set()
        sampler.join()
    result = {
        "diagnostic_only": True,
        "profile": profile.id,
        "manifest": profile.identity,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "fastembed": version("fastembed"),
        "onnxruntime": version("onnxruntime"),
        "threads": 2,
        "batch_size": args.batch_size,
        "content_tokens": runtime.tokenizer.count(text),
        "cpu_memory_arena": not args.disable_arena,
        "load_seconds": loaded,
        "batch_seconds": times,
        "query_seconds": query_seconds,
        "observed_process_peak_bytes": max(samples),
        "installed_bytes": profile.download_bytes,
        "warning": (
            "Repeated controlled text; no PDF, quality, publication, or capacity acceptance claim."
        ),
    }
    print(json.dumps(result, indent=2))
    label = f"{profile.id}-b{args.batch_size}-t{target}-arena{not args.disable_arena}"
    (args.directory / f"{label}-diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
