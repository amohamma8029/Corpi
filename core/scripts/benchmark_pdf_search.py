"""Real public PDF -> authenticated import -> extraction -> native searchable readiness.

Core-only measurements: the desktop's Import-to-UI-ready wait remains separate.
Fixed expected text spans resolve to saved passage IDs before model/chunk tuning.
Fresh runs have separate databases and evidence identities. Never reuse extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import threading
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient

from core.api.app import create_app
from core.ingest.service import PdfIngestService
from core.jobs.runner import JobRunner
from core.jobs.steps import build_pdf_extract_steps
from core.persistence.database import Database, DatabaseConfig
from core.persistence.models import new_id, utc_timestamp
from core.persistence.store import PersistenceStore
from core.search.handoff import SearchCoordinator
from core.search.memory import resident_bytes
from core.search.preparation import SearchLimits, build_search_steps, create_search_job
from core.search.profiles import profiles
from core.search.storage import SearchHit, SearchStorage
from core.search.supervisor import EmbeddingSession

DEFINITIONS = Path(__file__).resolve().parents[1] / "benchmarks" / "search_questions.json"


def normalized(text: str) -> tuple[str, list[int]]:
    characters = []
    positions = []
    for word in re.finditer(r"\S+", text):
        if characters:
            characters.append(" ")
            positions.append(word.start())
        for offset, character in enumerate(word.group()):
            folded = unicodedata.normalize("NFKC", character).casefold()
            characters.extend(folded)
            positions.extend([word.start() + offset] * len(folded))
    return "".join(characters), positions


def expected_spans(report, questions: list[dict]) -> dict[str, list[list[dict]]]:
    result = {}
    for question in questions:
        groups = []
        quotes = question.get("quotes", [question.get("quote")])
        for original in quotes:
            quote = normalized(original)[0]
            spans = []
            for passage in report.passages:
                if question.get("page") and passage.location.get("page") != question["page"]:
                    continue
                text, offsets = normalized(passage.text)
                for match in re.finditer(re.escape(quote), text):
                    spans.append(
                        {
                            "passage_id": passage.id,
                            "start": offsets[match.start()],
                            "end": offsets[match.end() - 1] + 1,
                            "page": passage.location.get("page"),
                        }
                    )
            if not spans:
                raise ValueError(f"expected evidence not found: {question['id']}")
            groups.append(spans)
        result[question["id"]] = groups
    return result


def coverage(
    hits: tuple[SearchHit, ...] | list[SearchHit], groups: list[list[dict]], report
) -> tuple[bool, bool]:
    outcomes = [_span_coverage(hits, spans, report) for spans in groups]
    return all(item[0] for item in outcomes), all(item[1] for item in outcomes)


def _span_coverage(hits, spans, report) -> tuple[bool, bool]:
    passages = {passage.id: passage for passage in report.passages}
    found = False
    for span in spans:
        ranges = []
        for hit in hits:
            for link in hit.mappings:
                if link.passage_id == span["passage_id"]:
                    found = True
                    ranges.append(
                        (
                            link.start_offset or 0,
                            len(passages[link.passage_id].text)
                            if link.end_offset is None
                            else link.end_offset,
                        )
                    )
        cursor = span["start"]
        for start, end in sorted(ranges):
            if start <= cursor < end:
                cursor = end
        if cursor >= span["end"]:
            return True, True
    return found, False


def hybrid(word: tuple[SearchHit, ...], vector: tuple[SearchHit, ...]) -> list[SearchHit]:
    scores = {}
    records = {}
    for ranking in (word, vector):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (60 + rank)
            records[hit.chunk_id] = hit
    return [records[key] for key in sorted(scores, key=lambda key: (-scores[key], key))[:10]]


def wait_job(store: PersistenceStore, job_id: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        status = store.load_job_status(job_id)
        if status is not None and status.status == "completed":
            return
        if status is not None and status.status in ("failed", "cancelled"):
            raise RuntimeError(f"{status.failure_code or status.status}: {status.failure_reason}")
        time.sleep(0.025)
    store.cancel_job(job_id, "benchmark deadline")
    raise TimeoutError("benchmark deadline")


def run(args, number: int) -> dict:
    definitions = json.loads(DEFINITIONS.read_text("utf-8"))
    expected = definitions["pdfs"][args.pdf]
    data = (args.pdf_directory / f"{args.pdf}.pdf").read_bytes()
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("PDF is not the fixed benchmark fixture")
    profile = profiles()[args.profile]
    root = args.output_directory / f"{args.pdf}-{args.profile}-{new_id()}"
    root.mkdir(parents=True)
    database = Database(DatabaseConfig(root))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Public PDF benchmark")
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
            " VALUES(?,?,?)",
            (notebook.id, profile.identity, utc_timestamp()),
        )
    coordinator = SearchCoordinator(store, args.models_directory)

    def benchmark_claim_blocker(job):
        # Experiments explicitly freeze evidence and choose settings after extraction.
        # Keep the ordinary automatic handoff waiting rather than running a second
        # unlabelled preparation pass with application-default settings.
        with store.database.connect() as connection:
            automatic = connection.execute(
                "SELECT 1 FROM search_handoffs WHERE job_id=?",
                (job.id,),
            ).fetchone()
        return coordinator.claim_blocking_reason(job) if automatic is not None else None

    runner = JobRunner(
        store,
        {
            "pdf-extract": build_pdf_extract_steps(store),
            "pdf-search": build_search_steps(store, args.models_directory),
        },
        owner_instance_id="public-benchmark",
        claim_blocking_reason=benchmark_claim_blocker,
    )
    ingest = PdfIngestService(store, "public-benchmark", notify_jobs=runner.notify)
    app = create_app("public-benchmark-token", store=store, pdf_ingest=ingest, runner=runner)
    headers = {"Authorization": "Bearer public-benchmark-token"}
    questions = [q for q in definitions["questions"] if q["source"] == args.pdf]
    stop = threading.Event()
    parent_samples = []

    def sample() -> None:
        while not stop.is_set():
            parent_samples.append(resident_bytes(os.getpid()))
            stop.wait(0.25)

    sampler = threading.Thread(target=sample)
    sampler.start()
    result = {
        "pdf": args.pdf,
        "pdf_sha256": expected["sha256"],
        "expected_pages": expected["pages"],
        "profile": profile.id,
        "manifest": profile.identity,
        "run": number,
        "overlap_override": args.overlap,
        "batch_size": args.batch_size,
        "threads": 2,
        "fresh_extraction_and_indexes": not args.extract_only,
        "model_reloaded": False,
        "measurement_scope": "extraction inspection only; no inference or readiness measurement"
        if args.extract_only
        else "core API import-to-ready, not desktop UI elapsed time",
        "pipeline_version": "supervised-search-v2",
        "fixture_definition_sha256": hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest(),
    }
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            started = time.monotonic()
            acquired = client.post(
                f"/notebooks/{notebook.id}/pdf-acquisitions",
                headers=headers,
                json={"retry_key": new_id(), "filename": f"{args.pdf}.pdf"},
            )
            acquired.raise_for_status()
            uploaded = client.put(
                f"/pdf-acquisitions/{acquired.json()['id']}/content",
                headers={**headers, "Content-Type": "application/pdf"},
                content=data,
            )
            uploaded.raise_for_status()
            source_id = uploaded.json()["source_id"]
            with store.database.connect() as connection:
                extract_job = connection.execute(
                    "SELECT id FROM jobs WHERE source_id=? AND kind='pdf-extract'", (source_id,)
                ).fetchone()[0]
            wait_job(store, extract_job, started + 650)
            extraction_finished = time.monotonic()
            report = store.load_extraction_report_for_source(source_id)
            assert report is not None and report.outcome is not None
            result.update(
                parser_name=report.extraction.parser_name,
                parser_version=report.extraction.parser_version,
                normalization_version=report.extraction.normalization_version,
            )
            if len(report.pages) != expected["pages"]:
                raise ValueError("PDF page count differs from the fixed fixture")
            if report.outcome.outcome == "partial":
                if not args.confirm_partial:
                    raise ValueError("partial fixture requires explicit --confirm-partial")
                assert report.review is not None
                confirmed = client.post(
                    f"/extractions/{report.extraction.id}/review",
                    headers=headers,
                    json={"decision": "confirmed", "generation": report.review.generation},
                )
                confirmed.raise_for_status()
                report = store.load_extraction_report(report.extraction.id)
                assert report.outcome is not None
            labels_started = time.monotonic()
            # Preserve the inspected source even if a fixed quotation needs correction.
            (root / "extracted-passages.json").write_text(
                json.dumps(
                    [
                        {
                            "id": p.id,
                            "page": p.location.get("page")
                            if isinstance(p.location, dict)
                            else None,
                            "text": p.text,
                        }
                        for p in report.passages
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "extraction-pages.json").write_text(
                json.dumps(
                    [
                        {"page": p.page_number, "classification": p.classification}
                        for p in report.pages
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if args.extract_only:
                result.update(
                    status="extracted",
                    pages=len(report.pages),
                    extraction_outcome=report.outcome.outcome,
                    eligible_pages=sum(p.classification == "extracted" for p in report.pages),
                    passages=len(report.passages),
                )
                return save_result(result, root, parent_samples)
            expected_evidence = expected_spans(report, questions)
            # Resolve and persist labels before any search construction or scoring.
            (root / "expected-evidence.json").write_text(
                json.dumps(expected_evidence, indent=2) + "\n", encoding="utf-8"
            )
            label_seconds = time.monotonic() - labels_started
            search_started = time.monotonic()
            search_job = create_search_job(
                store,
                source_id,
                report.extraction.id,
                profile.id,
                limits=SearchLimits(batch_size=args.batch_size),
                overlap=args.overlap,
            )
            runner.notify()
            wait_job(store, search_job, search_started + 650)
            ready_at = time.monotonic()
            readiness = store.load_readiness(source_id)
            assert readiness is not None and len(readiness.index_version_ids) == 1
            graph = store.load_index(readiness.index_version_ids[0])
            steps = store.load_job_steps(search_job)
            embedding = next(step.checkpoint for step in steps if step.step_key == "embed-search")
            assert isinstance(embedding, dict)
            with store.database.connect() as connection:
                charged = connection.execute(
                    "SELECT spent_seconds FROM search_generations WHERE index_version_id=?",
                    (graph.index.id,),
                ).fetchone()[0]
            result.update(
                {
                    "status": "ready",
                    "model_reloaded": True,
                    "readiness_reached": True,
                    "pages": len(report.pages),
                    "extraction_outcome": report.outcome.outcome,
                    "confirmed_partial": args.confirm_partial
                    and report.outcome.outcome == "partial",
                    "eligible_pages": sum(p.classification == "extracted" for p in report.pages),
                    "passages": len(report.passages),
                    "chunks": len(graph.chunks),
                    "text_bytes": sum(len(p.text.encode()) for p in report.passages),
                    "ingestion_extraction_seconds": extraction_finished - started,
                    "search_preparation_seconds": ready_at - search_started,
                    "core_import_to_ready_seconds": ready_at - started - label_seconds,
                    "harness_wall_to_ready_seconds": ready_at - started,
                    "label_resolution_seconds": label_seconds,
                    "budget_charged_seconds": charged,
                    "embedding": embedding,
                    "effective_settings": graph.index.settings,
                }
            )
            search = SearchStorage(store)
            cases = []
            # One explicit local session for query evaluation; preparation session was reaped.
            with EmbeddingSession(args.models_directory / profile.identity, profile) as session:
                for question in questions:
                    variants = [
                        (
                            question.get("language", expected.get("language", "english")),
                            question["query"],
                        )
                    ]
                    # Exactly five French and five Arabic cross-language paper questions.
                    if "french" in question:
                        language = "french" if question["id"].startswith("a") else "arabic"
                        variants.append((language, question[language]))
                    for language, query in variants:
                        began = time.monotonic()
                        vector = session.embed([query], query=True)[0]
                        words = search.search(
                            notebook.id, [graph.index.source_version_id], query=query
                        )
                        vectors = search.search(
                            notebook.id,
                            [graph.index.source_version_id],
                            vector=vector,
                            vector_manifest_id=profile.identity,
                        )
                        rankings = {
                            "word": words[:10],
                            "vector": vectors[:10],
                            "hybrid": hybrid(words, vectors),
                        }
                        cases.append(
                            {
                                "id": question["id"],
                                "language": language,
                                "source_language": expected.get("language", "english"),
                                "question_group": "non_english"
                                if expected.get("language", "english") != "english"
                                else "english"
                                if language == "english"
                                else "cross_language",
                                "query_and_search_seconds": time.monotonic() - began,
                                "scores": {
                                    method: dict(
                                        zip(
                                            ("passage_hit", "span_covered"),
                                            coverage(
                                                hits, expected_evidence[question["id"]], report
                                            ),
                                            strict=True,
                                        )
                                    )
                                    for method, hits in rankings.items()
                                },
                            }
                        )
                result["query_child_peak_bytes"] = session.peak_bytes
            result["quality_cases"] = cases
            result["database_bytes"] = sum(
                path.stat().st_size for path in root.glob("corpi.sqlite3*")
            )
    except Exception as error:
        result.update(
            {"status": "failed", "failure": str(error), "failure_type": type(error).__name__}
        )
    finally:
        stop.set()
        sampler.join()
    return save_result(result, root, parent_samples)


def save_result(result, root, parent_samples):
    result.update(
        {
            "parent_observed_peak_bytes": max(parent_samples),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "sqlite": __import__("sqlite3").sqlite_version,
            "fastembed": version("fastembed"),
            "onnxruntime": version("onnxruntime"),
        }
    )
    output = root / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **result,
                "quality_cases": len(result.get("quality_cases", [])),
                "result_file": str(output),
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-directory", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--pdf", choices=list(json.loads(DEFINITIONS.read_text("utf-8"))["pdfs"]), required=True
    )
    parser.add_argument("--profile", choices=list(profiles()), required=True)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--confirm-partial", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Inspect saved text before freezing new labels; no model inference",
    )
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 5:
        parser.error("repetitions must be 1..5")
    outcomes = [run(args, number) for number in range(1, args.repetitions + 1)]
    if any(
        result["status"] != ("extracted" if args.extract_only else "ready") for result in outcomes
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
