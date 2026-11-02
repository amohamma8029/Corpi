"""Full bounded search preparation from a synthetic, valid saved extraction.

This boundary diagnostic exercises planning, real CPU inference, guarded native
index completion and readiness for near-10k chunks. It does not claim PDF parsing,
desktop elapsed-time or retrieval-quality evidence. Original input text is kept
in a checksummed fixture file; no text is dropped to satisfy the budget.
"""

import argparse
import hashlib
import json
import platform
import threading
import time
from dataclasses import replace
from pathlib import Path

from benchmark_shared_search import session_command
from benchmark_static_multilingual import RETRIEVAL_MANIFEST, StaticTokenizer, load_profile
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.jobs.runner import JobRunner
from core.persistence.database import Database, DatabaseConfig
from core.persistence.models import (
    ExtractionOutcome,
    ExtractionPage,
    ExtractionVersion,
    Passage,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore
from core.search.memory import resident_bytes
from core.search.preparation import SearchLimits, build_search_steps, create_search_job
from core.search.profiles import profiles, verify_installation
from core.search.runtime import InputTokenizer
from core.search.storage import SearchStorage
from core.search.supervisor import EmbeddingSession


def seed(store, notebook, texts, name):
    timestamp = utc_timestamp()
    source = store.create_source(new_id(), notebook, name)
    raw = "\n\f\n".join(texts).encode()
    version = store.create_source_version(source.id, hashlib.sha256(raw).hexdigest())
    extraction = ExtractionVersion(
        new_id(),
        source.id,
        version.id,
        "synthetic-saved-extraction",
        "1",
        "1",
        1,
        {},
        timestamp,
    )
    passages, pages = [], []
    for ordinal, text in enumerate(texts):
        page = ordinal + 1
        passages.append(
            Passage(
                new_id(),
                source.id,
                version.id,
                extraction.id,
                ordinal,
                text,
                1,
                {"page": page, "reading_order": 0},
                timestamp,
            )
        )
        pages.append(
            ExtractionPage(
                new_id(),
                source.id,
                version.id,
                extraction.id,
                page,
                True,
                len(text),
                0,
                0,
                "extracted",
                timestamp,
            )
        )
    store.save_extraction_result(
        extraction,
        passages,
        pages,
        [],
        ExtractionOutcome(extraction.id, source.id, version.id, "complete", timestamp),
        None,
    )
    return source.id, version.id, extraction.id, raw


def wait(store, job, deadline, profile_stop=None):
    while time.monotonic() < deadline:
        if profile_stop is not None and profile_stop.is_set():
            store.cancel_job(job, "bounded stage profile complete")
            status = store.load_job_status(job)
            assert status is not None
            return status
        status = store.load_job_status(job)
        if status is not None and status.status in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.1)
    store.cancel_job(job, "capacity harness deadline")
    raise TimeoutError("capacity harness deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("minilm-en", "multilingual-e5", "amgix-static-retrieval"),
        required=True,
    )
    parser.add_argument("--input-run", type=Path, help="Reuse exact checksummed boundary text")
    parser.add_argument("--chunks", type=int, default=9950)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--varied",
        action="store_true",
        help="Unique observation labels; retain the original dense paragraphs",
    )
    parser.add_argument(
        "--profile-batches",
        type=int,
        help="Cancel after N inference batches; profiling only, not acceptance",
    )
    args = parser.parse_args()
    if not 9800 <= args.chunks <= 10_000 or not 1 <= args.batch_size <= 32:
        parser.error("chunks must be 9800..10000 and batch size 1..32")
    if args.profile_batches is not None and not 1 <= args.profile_batches <= 32:
        parser.error("profile batches must be 1..32")
    catalog = profiles()
    experimental = args.profile == "amgix-static-retrieval"
    if experimental:
        catalog[args.profile] = load_profile(RETRIEVAL_MANIFEST)
        # This process and its owned database only; deployed app catalog is unchanged.
        from core.search import preparation

        preparation.profiles = lambda: catalog
        preparation.__dict__["InputTokenizer"] = StaticTokenizer
    profile = catalog[args.profile]
    installed = args.models_directory / profile.identity
    verify_installation(installed, profile)
    tokenizer = (
        StaticTokenizer(installed, profile) if experimental else InputTokenizer(installed, profile)
    )
    sentence = (
        "The field study measured seasonal rainfall, soil moisture and temperature. "
        "Researchers compared observations across regions and checked instrument calibration. "
    )
    original_input_hash = None
    if args.input_run:
        reference = json.loads((args.input_run / "capacity-results.json").read_text("utf-8"))
        original = (args.input_run / "input.txt").read_bytes()
        original_input_hash = hashlib.sha256(original).hexdigest()
        if original_input_hash != reference["input_sha256"]:
            raise ValueError("original boundary input changed")
        texts = original.decode("utf-8").split("\n\f\n")
        if len(texts) != args.chunks:
            raise ValueError("original boundary passage count differs")
        text = texts[0]
    else:
        text = sentence
        while tokenizer.count(text + sentence) <= profile.chunk_tokens - 8:
            text += sentence
        texts = [text] * args.chunks
    if args.varied:
        texts = [f"Observation {number:05d}. " + value for number, value in enumerate(texts)]
    token_counts = [tokenizer.count(value) for value in texts]
    if max(token_counts) > profile.chunk_tokens:
        raise ValueError("varied fixture no longer fits one complete chunk per passage")
    for value in texts:
        tokenizer.prepare(value)
    text_bytes = sum(len(value.encode()) for value in texts)
    if text_bytes > 20_000_000:
        raise ValueError("valid boundary fixture exceeds the unchanged text ceiling")
    root = args.directory / f"capacity-{profile.id}-{new_id()}"
    root.mkdir(parents=True)
    database = Database(DatabaseConfig(root))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Synthetic full-preparation boundary")
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
            " VALUES(?,?,?)",
            (notebook.id, profile.identity, utc_timestamp()),
        )
    source, version, extraction, raw = seed(store, notebook.id, texts, "synthetic-capacity-input")
    (root / "input.txt").write_bytes(raw)
    limits = SearchLimits(batch_size=args.batch_size)
    job = create_search_job(store, source, extraction, profile.id, limits=limits)
    operation_times, step_times = [], []
    profile_stop = threading.Event()
    capacity_finished = False
    profiled_batches = 0

    def observed_step(step):
        def measured(context):
            began = time.perf_counter()
            try:
                return step.run(context)
            finally:
                step_times.append(
                    {
                        "role": "capacity" if context.job_id == job else "small",
                        "step": step.step_key,
                        "seconds": time.perf_counter() - began,
                    }
                )

        return replace(step, run=measured)

    steps = tuple(observed_step(s) for s in build_search_steps(store, args.models_directory))
    runner = JobRunner(store, {"pdf-search": steps})
    app = create_app("capacity-benchmark-token", store=store, runner=runner)
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.is_set():
            samples.append(resident_bytes(__import__("os").getpid()))
            stop.wait(0.25)

    sampler = threading.Thread(target=sample)
    sampler.start()
    result = {
        "scope": "synthetic saved-extraction boundary; full search preparation, not PDF/UI timing",
        "profile": profile.id,
        "manifest": profile.identity,
        "intended_chunks": args.chunks,
        "batch_size": args.batch_size,
        "threads": 1 if experimental else 2,
        "total_budget_seconds": 600,
        "content_tokens_per_passage": tokenizer.count(text),
        "content_token_range": [min(token_counts), max(token_counts)],
        "unique_passage_texts": len(set(texts)),
        "varied_input": args.varied,
        "text_bytes": text_bytes,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "original_input_sha256": original_input_hash,
        "experimental_adapter_only": experimental,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "profiling_only": args.profile_batches is not None,
    }
    sessions = []

    class ObservedSession(EmbeddingSession):
        def __init__(self, *parameters, **keywords):
            self.role = "small" if capacity_finished else "capacity"
            began = time.perf_counter()
            try:
                if experimental:
                    keywords["command"] = session_command(profile, root)
                super().__init__(*parameters, **keywords)
            finally:
                operation_times.append(
                    {
                        "role": self.role,
                        "operation": "model_load",
                        "seconds": time.perf_counter() - began,
                    }
                )

        def embed(self, texts, *, query=False, timeout=120):
            nonlocal profiled_batches
            began = time.perf_counter()
            completed = False
            try:
                vectors = super().embed(texts, query=query, timeout=timeout)
                completed = True
                return vectors
            finally:
                operation_times.append(
                    {
                        "role": self.role,
                        "operation": "inference_and_ipc",
                        "documents": len(texts),
                        "completed": completed,
                        "seconds": time.perf_counter() - began,
                    }
                )
                if completed and self.role == "capacity":
                    profiled_batches += 1
                    if args.profile_batches and profiled_batches >= args.profile_batches:
                        profile_stop.set()

        def close(self):
            closed = self._closed
            super().close()
            if not closed:
                sessions.append(
                    {
                        "peak_bytes": self.peak_bytes,
                        "launcher_reaped": self.process.poll() is not None,
                    }
                )

    # Observe the real supervisor without changing its limits, protocol or writes.
    from core.search import preparation

    preparation.EmbeddingSession = ObservedSession
    original_chunks = preparation.make_chunks

    def measured_chunks(*parameters, **keywords):
        began = time.perf_counter()
        try:
            return original_chunks(*parameters, **keywords)
        finally:
            operation_times.append(
                {
                    "role": "small" if capacity_finished else "capacity",
                    "operation": "token_safe_chunking",
                    "seconds": time.perf_counter() - began,
                }
            )

    preparation.make_chunks = measured_chunks
    original_generation = SearchStorage.create_generation

    def measured_generation(self, *parameters, **keywords):
        began = time.perf_counter()
        try:
            return original_generation(self, *parameters, **keywords)
        finally:
            operation_times.append(
                {
                    "role": "capacity" if parameters[0] == job else "small",
                    "operation": "guarded_generation_plan_storage",
                    "seconds": time.perf_counter() - began,
                }
            )

    SearchStorage.create_generation = measured_generation
    original_commit = SearchStorage.commit_batch

    def measured_commit(self, *parameters, **keywords):
        began = time.perf_counter()
        try:
            return original_commit(self, *parameters, **keywords)
        finally:
            operation_times.append(
                {
                    "role": "capacity" if parameters[0] == job else "small",
                    "operation": "guarded_vector_commit",
                    "seconds": time.perf_counter() - began,
                }
            )

    SearchStorage.commit_batch = measured_commit
    try:
        started = time.monotonic()
        with TestClient(app):
            runner.notify()
            status = wait(
                store,
                job,
                started + 650,
                profile_stop if args.profile_batches is not None else None,
            )
            capacity_finished = True
            result.update(
                status=status.status,
                failure_code=status.failure_code,
                failure_reason=status.failure_reason,
                preparation_wall_seconds=time.monotonic() - started,
                readiness_reached=store.load_readiness(source) is not None,
            )
            with store.database.connect() as connection:
                generation = connection.execute(
                    "SELECT g.*,i.settings_json FROM search_generations g "
                    "JOIN index_versions i ON i.id=g.index_version_id WHERE i.source_id=?",
                    (source,),
                ).fetchone()
                if generation is not None:
                    result.update(
                        planned_chunks=generation["expected_chunks"],
                        budget_charged_seconds=generation["spent_seconds"],
                        effective_settings=json.loads(generation["settings_json"]),
                    )
                    result["committed_chunks"] = connection.execute(
                        "SELECT count(*) FROM search_rows r JOIN search_vectors v "
                        "ON v.rowid=r.row_id WHERE r.index_version_id=?",
                        (generation["index_version_id"],),
                    ).fetchone()[0]
            result["visible_word_hits"] = len(
                SearchStorage(store).search(
                    notebook.id,
                    [version],
                    query="rainfall",
                    limit=10,
                )
            )
            # Demonstrate capacity release: a following small source must progress.
            small, _, small_extraction, _ = seed(
                store,
                notebook.id,
                [sentence],
                "following-small-control",
            )
            next_job = create_search_job(store, small, small_extraction, profile.id, limits=limits)
            runner.notify()
            next_status = wait(store, next_job, time.monotonic() + 150)
            result["following_small_job_status"] = next_status.status
            result["following_small_ready"] = store.load_readiness(small) is not None
    finally:
        stop.set()
        sampler.join()
        result["parent_observed_peak_bytes"] = max(samples)
        result["observed_sessions"] = sessions
        result["operation_times"] = operation_times
        result["step_times"] = step_times
        (root / "capacity-results.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        totals = {}
        for entry in operation_times:
            key = entry["role"] + ":" + entry["operation"]
            totals[key] = totals.get(key, 0.0) + entry["seconds"]
        summary = {key: value for key, value in result.items() if key != "operation_times"}
        summary["operation_seconds"] = totals
        print(json.dumps(summary, indent=2))
    if result.get("status") != "completed" and args.profile_batches is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
