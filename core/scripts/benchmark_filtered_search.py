"""Real native 10k/50k filtering workloads with controlled vectors, not inference.

Canonical on-disk chunks, mappings, batches, components and publication use core
operations. Stronger ineligible decoys make filter-after-top-K return no target.
This does not measure PDF import, model quality or full source preparation.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.persistence.database import Database, DatabaseConfig
from core.persistence.models import (
    AttemptGuard,
    ExtractionOutcome,
    ExtractionPage,
    ExtractionVersion,
    Passage,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore
from core.search.chunking import Chunk, PassageSlice
from core.search.handoff import ensure_notebook_profile
from core.search.profiles import profiles
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage


def seed(store, notebook, count, case, *, source=None, publish=True, profile_id="minilm-en"):
    if source is None:
        source = new_id()
        store.create_source(source, notebook, f"controlled-{case}.pdf")
    timestamp = utc_timestamp()
    texts = [
        f"Controlled public storage fixture {case} record {number:05d}. "
        + (
            "Rainfall and soil moisture were measured."
            if case == "target"
            else "Rainfall rainfall rainfall rainfall rainfall rainfall decoy observations."
        )
        for number in range(count)
    ]
    version = store.create_source_version(
        source, hashlib.sha256("\n".join(texts).encode()).hexdigest()
    )
    extraction = ExtractionVersion(
        new_id(), source, version.id, "controlled", "1", "1", 1, {}, timestamp
    )
    passages = [
        Passage(
            new_id(),
            source,
            version.id,
            extraction.id,
            number,
            text,
            1,
            {"page": 1, "reading_order": number},
            timestamp,
        )
        for number, text in enumerate(texts)
    ]
    page = ExtractionPage(
        new_id(),
        source,
        version.id,
        extraction.id,
        1,
        True,
        sum(map(len, texts)),
        0,
        0,
        "extracted",
        timestamp,
    )
    store.save_extraction_result(
        extraction,
        passages,
        [page],
        [],
        ExtractionOutcome(extraction.id, source, version.id, "complete", timestamp),
        None,
    )
    job = store.create_job(
        "controlled-filter:" + version.id, "pdf-search", source, version.id, extraction.id
    )
    store.create_job_status(job.id)
    store.claim_job(job.id, "filter-benchmark")
    attempt = store.start_step_attempt(job.id, "controlled-index")
    guard = AttemptGuard("filter-benchmark", "controlled-index", attempt.attempt_number)
    profile = profiles()[profile_id]
    storage = SearchStorage(store)
    chunks = tuple(Chunk(p.text, (PassageSlice(p, 0, len(p.text)),)) for p in passages)
    index = storage.create_generation(
        job.id, guard, profile, chunks, {"batch_size": 32, "separator": profile.separator}
    )
    graph = store.load_index(index)
    for offset in range(0, count, 32):
        batch = graph.chunks[offset : offset + 32]
        vectors = []
        for number in range(offset, offset + len(batch)):
            cosine = (0.25 if case == "target" else 0.9) + 0.05 * number / count
            vectors.append(vector_bytes([cosine, math.sqrt(1 - cosine**2)] + [0.0] * 382, 384))
        storage.commit_batch(job.id, guard, index, offset // 32, [c.id for c in batch], vectors)
    storage.finish_components(job.id, guard, index)
    store.commit_index_step(
        job.id, "controlled-index", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    if publish:
        store.publish_readiness(job.id, source, version.id, [index], guard=guard)
    store.finish_job(job.id, "completed")
    return source, version.id, index


def run(
    root: Path,
    total: int,
    repetitions: int,
    *,
    reuse=False,
    result_name="filter-results.json",
    probe=False,
):
    database = Database(DatabaseConfig(root))
    database.initialize()
    store = PersistenceStore(database)
    if reuse:
        previous = json.loads((root / "filter-results.json").read_text("utf-8"))
        if (
            previous["stored_chunks"] != total
            or not previous["word_and_vector_exact_controls_passed"]
        ):
            raise ValueError("not a matching completed filtering corpus")
        with store.database.connect() as connection:

            def lookup(case):
                row = connection.execute(
                    "SELECT s.notebook_id,i.source_version_id,i.id FROM sources s "
                    "JOIN index_versions i ON i.source_id=s.id WHERE s.display_name=? "
                    "ORDER BY i.created_at,i.rowid LIMIT 1",
                    (f"controlled-{case}.pdf",),
                ).fetchone()
                if row is None:
                    raise ValueError("controlled fixture identity is missing")
                return row

            notebook, target_version, target_index = lookup("target")
            selected = [
                lookup(case)[1]
                for case in (
                    "target",
                    "foreign",
                    "trashed",
                    "unpublished",
                    "history",
                    "inactive",
                )
            ]
        build_seconds = None
    else:
        notebook, target_version, target_index, selected, build_seconds = build_corpus(store, total)
    measure(
        store,
        root,
        notebook,
        target_version,
        target_index,
        selected,
        total,
        repetitions,
        build_seconds,
        result_name,
        probe,
    )


def build_corpus(store, total):
    notebook = store.create_notebook(new_id(), "Controlled filtering benchmark").id
    other = store.create_notebook(new_id(), "Ineligible notebook").id
    with store._transaction() as connection:
        ensure_notebook_profile(connection, notebook)
        ensure_notebook_profile(connection, other)
    count = total // 10
    began = time.perf_counter()
    _, target_version, target_index = seed(store, notebook, count, "target")
    seed(store, notebook, count, "unselected")
    _, foreign_version, _ = seed(store, other, count, "foreign")
    source, trashed_version, _ = seed(store, notebook, count, "trashed")
    deadline = (datetime.now(UTC) + timedelta(days=30)).isoformat(timespec="microseconds")
    store.trash_source(source, deadline.replace("+00:00", "Z"))
    _, unfinished_version, _ = seed(store, notebook, count, "unpublished", publish=False)
    source, historical_version, _ = seed(store, notebook, count, "history")
    seed(store, notebook, count, "history-replacement", source=source)
    _, inactive_version, _ = seed(store, notebook, count, "inactive", profile_id="multilingual-e5")
    seed(store, other, count, "foreign-two")
    seed(store, notebook, count, "unselected-two")
    build_seconds = time.perf_counter() - began
    selected = [
        target_version,
        foreign_version,
        trashed_version,
        unfinished_version,
        historical_version,
        inactive_version,
    ]
    return notebook, target_version, target_index, selected, build_seconds


def measure(
    store,
    root,
    notebook,
    target_version,
    target_index,
    selected,
    total,
    repetitions,
    build_seconds,
    result_name,
    probe,
):
    count = total // 10
    query = vector_bytes([1.0] + [0.0] * 383, 384)
    probes = {}
    with store.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM search_rows").fetchone()[0] == total
        assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == total
        control_started = time.perf_counter()
        expected = connection.execute(
            "SELECT r.chunk_id,vec_distance_cosine(v.embedding,?) AS distance "
            "FROM search_rows r JOIN search_vectors v ON v.rowid=r.row_id "
            "WHERE r.index_version_id=? ORDER BY distance,r.row_id LIMIT 50",
            (query, target_index),
        ).fetchall()
        scalar_seconds = time.perf_counter() - control_started
        global_rows = connection.execute(
            "SELECT rowid FROM search_vectors WHERE embedding MATCH ? AND k=50", (query,)
        ).fetchall()
        target_rows = {
            r[0]
            for r in connection.execute(
                "SELECT row_id FROM search_rows WHERE index_version_id=?", (target_index,)
            )
        }
        assert not target_rows.intersection(r[0] for r in global_rows)
        expected_words = connection.execute(
            "SELECT r.chunk_id FROM search_words w JOIN search_rows r ON r.row_id=w.rowid "
            "WHERE search_words MATCH 'rainfall' AND r.index_version_id=? "
            "ORDER BY bm25(search_words),w.rowid LIMIT 50",
            (target_index,),
        ).fetchall()
        if probe:
            variants = {
                "rowid-in": (
                    "SELECT rowid,bm25(search_words) AS score FROM search_words "
                    "WHERE search_words MATCH 'rainfall' AND rowid IN "
                    "(SELECT row_id FROM search_rows WHERE index_version_id=?) "
                    "ORDER BY score,rowid LIMIT 50"
                ),
                "fts-first": (
                    "WITH eligible AS MATERIALIZED (SELECT row_id FROM search_rows "
                    "WHERE index_version_id=?) SELECT search_words.rowid,"
                    "bm25(search_words) AS score FROM search_words CROSS JOIN eligible "
                    "WHERE search_words MATCH 'rainfall' AND eligible.row_id=search_words.rowid "
                    "ORDER BY score,search_words.rowid LIMIT 50"
                ),
                "fts-match-membership": (
                    "SELECT rowid,bm25(search_words) AS score FROM search_words "
                    "WHERE search_words MATCH 'rainfall' AND +rowid IN "
                    "(SELECT row_id FROM search_rows WHERE index_version_id=?) "
                    "ORDER BY score,rowid LIMIT 50"
                ),
            }
            reference_rows = connection.execute(
                "SELECT row_id FROM search_rows WHERE chunk_id IN ("
                + ",".join("?" for _ in expected_words)
                + ")",
                tuple(r[0] for r in expected_words),
            ).fetchall()
            reference = sorted(r[0] for r in reference_rows)
            for name, sql in variants.items():
                plans = connection.execute("EXPLAIN QUERY PLAN " + sql, (target_index,)).fetchall()
                started = time.perf_counter()
                rows = connection.execute(sql, (target_index,)).fetchall()
                elapsed = time.perf_counter() - started
                assert [r[0] for r in rows] == reference
                probes[name] = {
                    "seconds": elapsed,
                    "plan": [r[3] for r in plans],
                    "exact_target_rows_matched": True,
                }
    storage = SearchStorage(store)
    timings = {"word": [], "vector": []}
    for _ in range(repetitions):
        for method in timings:
            started = time.perf_counter()
            if method == "word":
                hits = storage.search(notebook, selected, limit=50, query="rainfall")
            else:
                hits = storage.search(
                    notebook,
                    selected,
                    limit=50,
                    vector=query,
                    vector_manifest_id=profiles()["minilm-en"].identity,
                )
            timings[method].append(time.perf_counter() - started)
            reference = expected_words if method == "word" else expected
            assert [h.chunk_id for h in hits] == [r[0] for r in reference]
            assert all(h.source_version_id == target_version and h.mappings for h in hits)
    result = {
        "scope": "native filtering, controlled vectors; no inference or PDF preparation",
        "stored_chunks": total,
        "eligible_chunks": count,
        "limit": 50,
        "repetitions": repetitions,
        "fixture_build_seconds": build_seconds,
        "exact_eligible_scalar_seconds": scalar_seconds,
        "filter_after_global_top_k_target_hits": 0,
        "sql_probes": probes,
        "word_and_vector_exact_controls_passed": True,
        "timings": {
            k: {"seconds": v, "median": statistics.median(v), "slowest": max(v)}
            for k, v in timings.items()
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sqlite": __import__("sqlite3").sqlite_version,
    }
    (root / result_name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--chunks", type=int, choices=(10_000, 50_000), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--reuse", type=Path, help="Use a completed owned corpus without rebuilding"
    )
    parser.add_argument("--result-name", default="filter-results.json")
    parser.add_argument("--probe-word-plan", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        parser.error("repetitions must be 1..20")
    if Path(args.result_name).name != args.result_name or not args.result_name.endswith(".json"):
        parser.error("result name must be a JSON filename")
    if args.reuse and args.result_name == "filter-results.json":
        parser.error("retain the baseline by choosing a different result name for reuse")
    root = args.reuse or args.directory / f"filter-{args.chunks}-{new_id()}"
    if not args.reuse:
        root.mkdir(parents=True)
    run(
        root,
        args.chunks,
        args.repetitions,
        reuse=bool(args.reuse),
        result_name=args.result_name,
        probe=args.probe_word_plan,
    )


if __name__ == "__main__":
    main()
