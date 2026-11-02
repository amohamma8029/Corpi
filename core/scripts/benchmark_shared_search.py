"""Frozen shared-passage quality and standalone candidate-inference diagnostics.

All profiles score identical citation-preserving chunks and the existing questions.
Native FTS5 and exact eligible sqlite-vec cosine distances are used in a separate
benchmark database, not production readiness/publication or profile activation.
"""

import argparse
import hashlib
import json
import platform
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import sqlite_vec
from benchmark_contextual_multilingual import UTF8InputTokenizer
from benchmark_contextual_multilingual import profiles as contextual_profiles
from benchmark_pdf_search import DEFINITIONS, coverage, expected_spans, hybrid
from benchmark_static_multilingual import (
    MANIFEST,
    RETRIEVAL_MANIFEST,
    StaticTokenizer,
    load_profile,
)

from core.persistence.models import ChunkPassage, Passage, new_id
from core.search.chunking import make_chunks
from core.search.profiles import profiles, verify_installation
from core.search.runtime import InputTokenizer
from core.search.storage import SearchHit
from core.search.supervisor import EmbeddingSession

SOURCES = {
    "attention": "attention-minilm-en-20a0b09f-0e93-479b-8ff5-eff38235661e",
    "bert": "bert-minilm-en-f02f7562-758f-45a9-961d-60eee3ed5cd8",
    "http-semantics": "http-semantics-minilm-en-dc68e080-893b-434e-a092-a9cbaf366cbe",
    "udhr-french": "udhr-french-minilm-en-f78be3f1-e481-4dc7-9a9c-51afeab2400f",
    "udhr-arabic": "udhr-arabic-multilingual-e5-2b112b8c-28f7-4347-83cc-ecf4ac7568ff",
}
OFFLINE_CHILD = """
import socket, runpy
def denied(*args, **kwargs):
    raise OSError('benchmark inference network denied')
socket.socket.connect = denied
socket.socket.connect_ex = denied
runpy.run_module('core.search.worker', run_name='__main__')
"""


def candidates():
    catalog = {**profiles(), "potion-multilingual-static": load_profile(), **contextual_profiles()}
    if RETRIEVAL_MANIFEST.exists():
        catalog["amgix-static-retrieval"] = load_profile(RETRIEVAL_MANIFEST)
    return catalog


def static_profile(profile):
    return profile.pooling in ("static-mean", "static-mean-include-unk")


def session_command(profile, directory=None):
    if profile.id in contextual_profiles():
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_contextual_multilingual.py")),
            "--worker",
        ]
        if directory is not None:
            command += ["--metadata", str(directory / f"{profile.id}-runtime-metadata.json")]
        return command
    if static_profile(profile):
        return [
            sys.executable,
            str(Path(__file__).with_name("benchmark_static_multilingual.py")),
            "--worker",
            "--manifest",
            str(RETRIEVAL_MANIFEST if profile.id == "amgix-static-retrieval" else MANIFEST),
        ]
    return [sys.executable, "-c", OFFLINE_CHILD]


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def freeze(args):
    if args.directory.exists():
        raise ValueError("shared benchmark directory already exists")
    definitions = json.loads(DEFINITIONS.read_text("utf-8"))
    catalog = candidates()
    reference = None
    if args.passage_preserving_reference:
        reference = json.loads(args.passage_preserving_reference.read_text("utf-8"))
        if reference["definition_sha256"] != hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest():
            raise ValueError("reference question definitions changed")
        catalog = {key: catalog[key] for key in reference["candidate_manifests"]}
        if {key: value.identity for key, value in catalog.items()} != reference[
            "candidate_manifests"
        ]:
            raise ValueError("reference token-counting model identities changed")
    if args.native_chunks:
        catalog = {"potion-multilingual-static": catalog["potion-multilingual-static"]}
    if args.chunk_profile:
        catalog = {args.chunk_profile: catalog[args.chunk_profile]}
    tokenizers = []
    for profile in catalog.values():
        installed = args.models_directory / profile.identity
        verify_installation(installed, profile)
        cls = (
            StaticTokenizer
            if static_profile(profile)
            else UTF8InputTokenizer
            if profile.id in contextual_profiles()
            else InputTokenizer
        )
        tokenizers.append(cls(installed, profile))
    common = (
        next(iter(catalog.values()))
        if args.native_chunks or args.chunk_profile
        else replace(profiles()["minilm-en"], chunk_tokens=64, overlap_tokens=8)
    )

    def count(text):
        return max(tokenizer.count(text) for tokenizer in tokenizers)

    def validate(text):
        for tokenizer in tokenizers:
            tokenizer.prepare(text)
        return text

    documents = {}
    selected_sources = (
        json.loads(args.source_runs.read_text("utf-8")) if args.source_runs else SOURCES
    )
    if set(selected_sources) != set(SOURCES):
        raise ValueError("all five fixed PDF sources are required")
    for source, directory in selected_sources.items():
        relative = Path(directory)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source-run paths must be relative to runs-directory")
        root = args.runs_directory / directory
        result = json.loads((root / "result.json").read_text("utf-8"))
        spec = definitions["pdfs"][source]
        if result["status"] != "ready" or result["pdf_sha256"] != spec["sha256"]:
            raise ValueError("source does not match a completed fixed PDF run")
        if (
            hashlib.sha256((args.pdf_directory / f"{source}.pdf").read_bytes()).hexdigest()
            != spec["sha256"]
        ):
            raise ValueError("fixed PDF bytes changed")
        if result.get("extraction_outcome") == "partial" and not result.get("confirmed_partial"):
            raise ValueError("partial extraction was not confirmed")
        if (
            source == "udhr-arabic"
            and result.get("normalization_version") != "faithfully-normalized-text-v2"
        ):
            raise ValueError("Arabic shared source requires the corrected saved extraction")
        raw = json.loads((root / "extracted-passages.json").read_text("utf-8"))
        page_file = root / "extraction-pages.json"
        if page_file.exists():
            page_bytes = page_file.read_bytes()
            pages = json.loads(page_bytes)
            page_evidence = "saved extraction-pages.json"
        else:
            # Older retained runs did not export page JSON. Read actual persisted
            # classifications without initializing, migrating or modifying them.
            with sqlite3.connect((root / "corpi.sqlite3").as_uri() + "?mode=ro", uri=True) as old:
                if (
                    old.execute(
                        "SELECT count(DISTINCT extraction_version_id) FROM extraction_pages"
                    ).fetchone()[0]
                    != 1
                ):
                    raise ValueError("historical page ownership is ambiguous")
                pages = [
                    {"page": number, "classification": classification}
                    for number, classification in old.execute(
                        "SELECT page_number,classification FROM extraction_pages "
                        "ORDER BY page_number"
                    )
                ]
            page_bytes = json.dumps(pages, sort_keys=True).encode()
            page_evidence = "read-only historical database classifications"
        if len(pages) != spec["pages"]:
            raise ValueError("saved page count differs from the fixed PDF")
        eligible = {page["page"] for page in pages if page["classification"] == "extracted"}
        # These wrapper ownership fields are unused by chunking/scoring. Real
        # saved passage IDs, Unicode text, order, pages and slices remain exact.
        passages = tuple(
            Passage(item["id"], "", "", "", number, item["text"], 1, {"page": item["page"]}, "")
            for number, item in enumerate(raw)
            if item["page"] in eligible
        )
        report = SimpleNamespace(passages=passages)
        questions = [q for q in definitions["questions"] if q["source"] == source]
        labels = expected_spans(report, questions)
        if reference and (
            labels != reference["documents"][source]["expected_evidence"]
            or hashlib.sha256((root / "extracted-passages.json").read_bytes()).hexdigest()
            != reference["documents"][source]["saved_passages_sha256"]
        ):
            raise ValueError("reference passages or frozen labels changed")
        chunks = make_chunks(
            passages,
            common,
            count,
            validate,
            preserve_fitting_passages=reference is not None,
        )
        saved_chunks = []
        for number, chunk in enumerate(chunks, start=1):
            if common.separator.join(mapping.text for mapping in chunk.mappings) != chunk.text:
                raise ValueError("shared chunk reconstruction failed")
            saved_chunks.append(
                {
                    "id": number,
                    "text": chunk.text,
                    "mappings": [
                        {
                            "passage_id": part.passage.id,
                            "start_offset": part.start,
                            "end_offset": part.end,
                        }
                        for part in chunk.mappings
                    ],
                }
            )
        documents[source] = {
            "pdf_sha256": spec["sha256"],
            "source_run": directory,
            "saved_passages_sha256": hashlib.sha256(
                (root / "extracted-passages.json").read_bytes()
            ).hexdigest(),
            "page_evidence": page_evidence,
            "page_evidence_sha256": hashlib.sha256(page_bytes).hexdigest(),
            "normalization_version": result.get(
                "normalization_version", "not recorded in historical result"
            ),
            "passages": [
                {
                    "id": p.id,
                    "text": p.text,
                    "page": p.location["page"] if isinstance(p.location, dict) else None,
                }
                for p in passages
            ],
            "chunks": saved_chunks,
            "expected_evidence": labels,
        }
    corpus = {
        "scope": "frozen profile-specific diagnostic inputs"
        if args.native_chunks or args.chunk_profile
        else "frozen shared inputs, not production generations",
        "chunk_tokens": common.chunk_tokens,
        "overlap_tokens": common.overlap_tokens,
        "counting": "candidate tokenizer only"
        if args.native_chunks or args.chunk_profile
        else "maximum content-token count across declared tokenizers",
        "definition_sha256": hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest(),
        "candidate_manifests": {key: value.identity for key, value in catalog.items()},
        "documents": documents,
    }
    if reference is not None:
        corpus.update(
            scope="separate passage-preserving diagnostic; original frozen scores unchanged",
            passage_boundary_policy="preserve-fitting-passages-v1",
            original_shared_corpus_sha256=hashlib.sha256(
                args.passage_preserving_reference.read_bytes()
            ).hexdigest(),
        )
    args.directory.mkdir()
    save(args.directory / "shared-corpus.json", corpus)
    print(
        json.dumps(
            {
                "corpus_sha256": hashlib.sha256(
                    (args.directory / "shared-corpus.json").read_bytes()
                ).hexdigest(),
                "sources": {key: len(value["chunks"]) for key, value in documents.items()},
            },
            indent=2,
        )
    )


def hit(chunk, score, manifest):
    identity = f"{chunk['id']:08d}"
    return SearchHit(
        "",
        "",
        "",
        identity,
        manifest,
        chunk["text"],
        score,
        tuple(
            ChunkPassage(
                identity,
                part["passage_id"],
                "",
                "",
                "",
                "",
                number,
                part["start_offset"],
                part["end_offset"],
            )
            for number, part in enumerate(chunk["mappings"])
        ),
    )


def quality(args):
    corpus_file = args.directory / "shared-corpus.json"
    corpus = json.loads(corpus_file.read_text("utf-8"))
    definition_hash = hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest()
    if definition_hash != corpus["definition_sha256"]:
        raise ValueError("frozen question definitions changed")
    profile = candidates()[args.profile]
    frozen_identity = corpus["candidate_manifests"].get(profile.id)
    if (frozen_identity is not None and frozen_identity != profile.identity) or (
        frozen_identity is None
        and profile.id not in contextual_profiles()
        and profile.id != "amgix-static-retrieval"
    ):
        raise ValueError("frozen model identity changed")
    output = args.directory / f"{profile.id}-quality.json"
    if output.exists():
        raise ValueError("quality result already exists")
    definitions = json.loads(DEFINITIONS.read_text("utf-8"))
    database = args.directory / f"{profile.id}-quality-{new_id()}.sqlite3"
    if database.exists():
        raise ValueError("benchmark database already exists")
    connection = sqlite3.connect(database)
    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)
    connection.execute("CREATE TABLE vectors(id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)")
    connection.execute(
        "CREATE VIRTUAL TABLE words USING fts5(text, tokenize='unicode61 remove_diacritics 2')"
    )
    result = {
        "scope": "retrieval diagnostic, not application preparation/readiness",
        "input_scope": corpus["scope"],
        "chunk_tokens": corpus["chunk_tokens"],
        "overlap_tokens": corpus["overlap_tokens"],
        "profile": profile.id,
        "database_file": database.name,
        "manifest": profile.identity,
        "dimensions": profile.dimensions,
        "corpus_sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
        "definition_sha256": definition_hash,
        "threads": 1 if static_profile(profile) else 2,
        "memory_limit_bytes": 1073741824,
        "batch_size": args.batch_size or (8 if profile.id == "multilingual-e5" else 32),
        "network_denied": True,
        "sources": {},
        "quality_cases": [],
    }
    started = time.monotonic()
    session = None
    try:
        with EmbeddingSession(
            args.models_directory / profile.identity,
            profile,
            command=session_command(profile, args.directory),
        ) as session:
            result["load_seconds"] = session.load_seconds
            for source, document in corpus["documents"].items():
                source_started = time.monotonic()
                connection.execute("DELETE FROM vectors")
                connection.execute("DELETE FROM words")
                chunks = document["chunks"]
                batch_size = result["batch_size"]
                embedding_started = time.monotonic()
                for offset in range(0, len(chunks), batch_size):
                    if time.monotonic() - source_started >= 600:
                        raise TimeoutError("source diagnostic budget exceeded")
                    batch = chunks[offset : offset + batch_size]
                    vectors = session.embed(
                        [chunk["text"] for chunk in batch],
                        timeout=min(120, 600 - (time.monotonic() - source_started)),
                    )
                    connection.executemany(
                        "INSERT INTO vectors VALUES(?,?)",
                        [
                            (chunk["id"], vector)
                            for chunk, vector in zip(batch, vectors, strict=True)
                        ],
                    )
                    connection.executemany(
                        "INSERT INTO words(rowid,text) VALUES(?,?)",
                        [(chunk["id"], chunk["text"]) for chunk in batch],
                    )
                connection.commit()
                embedding_seconds = time.monotonic() - embedding_started
                report = SimpleNamespace(
                    passages=[
                        SimpleNamespace(id=p["id"], text=p["text"]) for p in document["passages"]
                    ]
                )
                lookup = {chunk["id"]: chunk for chunk in chunks}
                for question in (q for q in definitions["questions"] if q["source"] == source):
                    variants = [
                        (
                            question.get(
                                "language", definitions["pdfs"][source].get("language", "english")
                            ),
                            question["query"],
                        )
                    ]
                    if "french" in question:
                        language = "french" if question["id"].startswith("a") else "arabic"
                        variants.append((language, question[language]))
                    for language, query in variants:
                        vector = session.embed([query], query=True)[0]
                        tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
                        literal = " OR ".join(
                            '"' + token.replace('"', '""') + '"' for token in tokens
                        )
                        words = (
                            tuple(
                                hit(lookup[index], score, profile.identity)
                                for index, score in connection.execute(
                                    "SELECT rowid,bm25(words) FROM words WHERE words MATCH ? "
                                    "ORDER BY bm25(words),rowid LIMIT 50",
                                    (literal,),
                                )
                            )
                            if tokens
                            else ()
                        )
                        vectors = tuple(
                            hit(lookup[index], score, profile.identity)
                            for index, score in connection.execute(
                                "SELECT id,vec_distance_cosine(embedding,?) AS distance "
                                "FROM vectors "
                                "ORDER BY distance,id LIMIT 50",
                                (vector,),
                            )
                        )
                        # This database contains exactly one eligible source per phase.
                        rankings = {
                            "word": words[:10],
                            "vector": vectors[:10],
                            "hybrid": hybrid(words, vectors),
                        }
                        source_language = definitions["pdfs"][source].get("language", "english")
                        result["quality_cases"].append(
                            {
                                "id": question["id"],
                                "source": source,
                                "language": language,
                                "question_group": "non_english"
                                if source_language != "english"
                                else "english"
                                if language == "english"
                                else "cross_language",
                                "scores": {
                                    method: dict(
                                        zip(
                                            ("passage_hit", "span_covered"),
                                            coverage(
                                                ranking,
                                                document["expected_evidence"][question["id"]],
                                                report,
                                            ),
                                            strict=True,
                                        )
                                    )
                                    for method, ranking in rankings.items()
                                },
                                "top_chunks": {
                                    method: [item.chunk_id for item in ranking]
                                    for method, ranking in rankings.items()
                                },
                            }
                        )
                        if args.retain_rankings:
                            result["quality_cases"][-1]["candidate_top_chunks"] = {
                                "word": [item.chunk_id for item in words],
                                "vector": [item.chunk_id for item in vectors],
                            }
                result["sources"][source] = {
                    "chunks": len(chunks),
                    "embedding_and_storage_seconds": embedding_seconds,
                    "source_elapsed_seconds": time.monotonic() - source_started,
                }
            if len(result["quality_cases"]) != 33:
                raise ValueError("frozen quality-case count differs")
            result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", failure_type=type(error).__name__, failure=str(error))
    finally:
        connection.close()
        result.update(
            elapsed_seconds=time.monotonic() - started,
            child_peak_bytes=session.peak_bytes if session else None,
            launcher_reaped=session.process.poll() is not None if session else True,
            platform=platform.platform(),
            python=platform.python_version(),
            versions={
                name: version(name)
                for name in ("fastembed", "onnxruntime", "tokenizers", "sqlite-vec")
            },
        )
        save(output, result)
    print(json.dumps({**result, "quality_cases": len(result["quality_cases"])}, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


def capacity(args):
    profile = candidates()[args.profile] if args.profile else load_profile()
    output = args.directory / (
        "potion-capacity-varied-encoding.json" if args.varied else "potion-capacity-encoding.json"
    )
    if profile.id != "potion-multilingual-static":
        variant = "varied-encoding" if args.varied else "encoding"
        output = args.directory / f"{profile.id}-capacity-{variant}.json"
    if args.screen_batches:
        output = args.directory / f"{profile.id}-speed-screen-batch{args.batch_size or 32}.json"
    if output.exists():
        raise ValueError("capacity result already exists")
    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("capacity input changed")
    paragraphs = raw.decode("utf-8").split("\n\f\n")
    if len(paragraphs) != 9950:
        raise ValueError("expected original near-cap input")
    if args.varied:
        paragraphs = [
            f"Unique observation {number:05d}: " + text for number, text in enumerate(paragraphs)
        ]
    input_bytes = "\n\f\n".join(paragraphs).encode()
    if len(input_bytes) > 20_000_000:
        raise ValueError("candidate text byte limit exceeded")
    result = {
        "scope": "original near-cap text encoded only; "
        "no chunk storage/publication or application readiness",
        "profile": profile.id,
        "manifest": profile.identity,
        "original_input_sha256": source["input_sha256"],
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "input_variant": "varied labels, original dense text retained"
        if args.varied
        else "original repeated passages",
        "unique_passages": len(set(paragraphs)),
        "input_bytes": len(input_bytes),
        "intended_chunks": len(paragraphs),
        "batch_size": args.batch_size or 32,
        "threads": 1 if static_profile(profile) else 2,
        "memory_limit_bytes": 1073741824,
        "total_seconds": 600,
        "network_denied": True,
        "vector_reuse": False,
        "readiness_reached": False,
        "encoded": 0,
        "screen_batches": args.screen_batches,
        "batch_seconds": [],
    }
    started = time.monotonic()
    session = None
    try:
        cls = (
            StaticTokenizer
            if static_profile(profile)
            else UTF8InputTokenizer
            if profile.id in contextual_profiles()
            else InputTokenizer
        )
        tokenizer = cls(args.models_directory / profile.identity, profile)
        counts = [tokenizer.count(text) for text in set(paragraphs)]
        result["content_token_range"] = [min(counts), max(counts)]
        result["fits_profile_chunk_budget"] = max(counts) <= profile.chunk_tokens
        vector_digest = hashlib.sha256()
        remaining = 600 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("candidate source inference budget exceeded")
        with EmbeddingSession(
            args.models_directory / profile.identity,
            profile,
            command=session_command(profile, args.directory),
            load_timeout=min(120, remaining),
        ) as session:
            result["load_seconds"] = session.load_seconds
            batch_size = result["batch_size"]
            for offset in range(0, len(paragraphs), batch_size):
                if args.screen_batches and offset >= args.screen_batches * batch_size:
                    break
                remaining = 600 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("candidate source inference budget exceeded")
                batch_started = time.monotonic()
                vectors = session.embed(
                    paragraphs[offset : offset + batch_size], timeout=min(120, remaining)
                )
                result["batch_seconds"].append(time.monotonic() - batch_started)
                for vector in vectors:
                    vector_digest.update(vector)
                result["encoded"] += len(vectors)
            result["vectors_sha256"] = vector_digest.hexdigest()
            if args.screen_batches:
                result["scope"] = "bounded original-input speed screen; not capacity or readiness"
                result["warm_passages_per_second"] = batch_size / statistics.median(
                    result["batch_seconds"][1:]
                )
            result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", failure_type=type(error).__name__, failure=str(error))
    finally:
        result.update(
            elapsed_seconds=time.monotonic() - started,
            child_peak_bytes=session.peak_bytes if session else None,
            launcher_reaped=session.process.poll() is not None if session else True,
        )
        save(output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


def summarize(args):
    rows = []
    failures = []
    reference = None
    corpus = json.loads((args.directory / "shared-corpus.json").read_text("utf-8"))
    for profile in candidates().values():
        if args.summary_profiles and profile.id not in args.summary_profiles:
            continue
        if (
            profile.id not in corpus["candidate_manifests"]
            and not (args.directory / f"{profile.id}-quality.json").exists()
        ):
            continue
        data = json.loads((args.directory / f"{profile.id}-quality.json").read_text("utf-8"))
        if data["status"] != "completed":
            failures.append(
                {
                    "profile": profile.id,
                    "status": data["status"],
                    "failure": data.get("failure"),
                    "failure_type": data.get("failure_type"),
                }
            )
            continue
        cases = data["quality_cases"]
        word_control = [
            (item["source"], item["id"], item["language"], item["top_chunks"]["word"])
            for item in cases
        ]
        if reference is None:
            reference = (data["corpus_sha256"], data["definition_sha256"], word_control)
        if reference != (data["corpus_sha256"], data["definition_sha256"], word_control):
            raise ValueError("shared inputs, questions or native word rankings differ")
        groups = {}
        for name in ("english", "cross_language", "non_english"):
            group = [item for item in cases if item["question_group"] == name]
            expected_count = 13 if name == "english" else 10
            if len(group) != expected_count:
                raise ValueError("question denominator differs")
            groups[name] = {
                "questions": len(group),
                "methods": {
                    method: {
                        score: sum(item["scores"][method][score] for item in group)
                        for score in ("passage_hit", "span_covered")
                    }
                    for method in ("word", "vector", "hybrid")
                },
            }
        languages = {}
        for name in ("french", "arabic"):
            languages[name] = {}
            for kind in ("cross_language", "non_english"):
                group = [
                    item
                    for item in cases
                    if item["language"] == name and item["question_group"] == kind
                ]
                languages[name][kind] = {
                    "questions": len(group),
                    "methods": {
                        method: sum(item["scores"][method]["span_covered"] for item in group)
                        for method in ("word", "vector", "hybrid")
                    },
                }
        rows.append(
            {
                "profile": profile.id,
                "manifest": profile.identity,
                "dimensions": profile.dimensions,
                "groups": groups,
                "languages": languages,
                "embedding_and_storage_seconds": sum(
                    item["embedding_and_storage_seconds"] for item in data["sources"].values()
                ),
                "load_seconds": data["load_seconds"],
                "child_peak_bytes": data["child_peak_bytes"],
                "elapsed_seconds": data["elapsed_seconds"],
            }
        )
    if reference is None:
        raise ValueError("no declared quality profiles")
    result = {
        "scope": "one frozen-input diagnostic per profile; not readiness or answer accuracy",
        "input_scope": corpus["scope"],
        "corpus_sha256": reference[0],
        "definition_sha256": reference[1],
        "identical_word_rankings_control": len(rows) > 1,
        "profiles": rows,
        "failed_profiles": failures,
    }
    save(args.directory / "shared-summary.json", result)
    if args.compact:
        print(
            json.dumps(
                {
                    "corpus_sha256": result["corpus_sha256"],
                    "identical_word_rankings_control": result["identical_word_rankings_control"],
                    "profiles": [
                        {
                            "profile": row["profile"],
                            "vector_coverage": {
                                key: value["methods"]["vector"]["span_covered"]
                                for key, value in row["groups"].items()
                            },
                            "hybrid_coverage": {
                                key: value["methods"]["hybrid"]["span_covered"]
                                for key, value in row["groups"].items()
                            },
                            "embedding_and_storage_seconds": row["embedding_and_storage_seconds"],
                        }
                        for row in rows
                    ],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--passage-preserving-reference",
        type=Path,
        help="Separate diagnostic using the original declared tokenizers and unchanged labels",
    )
    parser.add_argument(
        "--chunk-profile",
        choices=list(candidates()),
        help="Freeze one candidate's intended chunk settings",
    )
    parser.add_argument(
        "--native-chunks",
        action="store_true",
        help="Freeze Potion's profile-specific chunks instead of shared short inputs",
    )
    parser.add_argument("--runs-directory", type=Path)
    parser.add_argument(
        "--source-runs",
        type=Path,
        help="JSON mapping the five PDF keys to relative completed run folders",
    )
    parser.add_argument("--pdf-directory", type=Path)
    parser.add_argument("--profile", choices=list(candidates()))
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(8, 16, 32),
        help="Explicit diagnostic batch size within the existing ceiling",
    )
    parser.add_argument("--capacity-run", type=Path)
    parser.add_argument("--screen-batches", type=int, choices=(4, 8), default=0)
    parser.add_argument(
        "--varied",
        action="store_true",
        help="Add unique labels while retaining the original near-cap dense text",
    )
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-profiles", nargs="+", choices=list(candidates()))
    parser.add_argument(
        "--retain-rankings",
        action="store_true",
        help="Retain up to fifty word/vector candidate IDs",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Print compact summary; preserve full saved results"
    )
    args = parser.parse_args()
    if args.summary_profiles and not args.summarize:
        parser.error("summary-profiles requires summarize")
    if args.passage_preserving_reference and (
        not args.freeze or args.native_chunks or args.chunk_profile
    ):
        parser.error(
            "passage-preserving-reference requires shared freeze without native/profile inputs"
        )
    if args.native_chunks and not args.freeze:
        parser.error("native-chunks only selects inputs during freeze")
    if args.chunk_profile and (not args.freeze or args.native_chunks):
        parser.error("chunk-profile requires freeze and is separate from native-chunks")
    if args.varied and not args.capacity_run:
        parser.error("varied requires a capacity-run")
    if args.screen_batches and (not args.capacity_run or args.varied):
        parser.error("screen-batches requires original capacity-run without varied labels")
    if args.freeze:
        if not args.runs_directory or not args.pdf_directory:
            parser.error("freeze requires runs-directory and pdf-directory")
        freeze(args)
    elif args.summarize:
        summarize(args)
    elif args.capacity_run:
        capacity(args)
    elif args.profile:
        quality(args)
    else:
        parser.error("select freeze, profile or capacity-run")


if __name__ == "__main__":
    main()
