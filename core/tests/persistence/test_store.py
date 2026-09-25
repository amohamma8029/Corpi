from dataclasses import replace
from pathlib import Path

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import (
    ConflictingWriteError,
    IntegrityViolationError,
    PersistenceError,
)
from core.persistence.models import (
    ChunkPassage,
    ExtractionVersion,
    IndexGraph,
    IndexVersion,
    ManagedFile,
    Passage,
    RetrievalChunk,
    SourceVersion,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore

HASH_A = "a" * 64
HASH_B = "b" * 64


def seed_source(store: PersistenceStore, name: str = "report.pdf") -> tuple[str, SourceVersion]:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Notebook {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    managed_file = ManagedFile(new_id(), f"originals/{name}", 17, HASH_A, utc_timestamp())
    store.save_managed_file(managed_file)
    source_version = store.create_source_version(source_id, HASH_A, managed_file.id)
    return source_id, source_version


def make_extraction(
    source_id: str, source_version: SourceVersion, *, parser: str = "pdf"
) -> tuple[ExtractionVersion, tuple[Passage, ...]]:
    extraction_id = new_id()
    extraction = ExtractionVersion(
        extraction_id,
        source_id,
        source_version.id,
        parser,
        "1.0",
        "unicode-nfc-v1",
        1,
        {"keep_hyphens": True, "pages": [1]},
        utc_timestamp(),
    )
    passages = (
        Passage(
            new_id(),
            source_id,
            source_version.id,
            extraction_id,
            0,
            "The saved evidence remains stable.",
            1,
            {"page": 1, "start": 0, "end": 34},
            utc_timestamp(),
        ),
        Passage(
            new_id(),
            source_id,
            source_version.id,
            extraction_id,
            1,
            "A second exact passage.",
            1,
            {"page": 2},
            utc_timestamp(),
        ),
    )
    return extraction, passages


def make_index(
    extraction: ExtractionVersion,
    passages: tuple[Passage, ...],
    *,
    index_id: str | None = None,
) -> tuple[IndexVersion, tuple[RetrievalChunk, ...], tuple[ChunkPassage, ...]]:
    index_id = index_id or new_id()
    chunks = (
        RetrievalChunk(
            new_id(),
            index_id,
            extraction.source_id,
            extraction.source_version_id,
            extraction.id,
            0,
            passages[0].text,
        ),
    )
    mappings = (
        ChunkPassage(
            chunks[0].id,
            passages[0].id,
            index_id,
            extraction.source_id,
            extraction.source_version_id,
            extraction.id,
            0,
            0,
            len(passages[0].text),
        ),
    )
    index = IndexVersion(
        index_id,
        extraction.source_id,
        extraction.source_version_id,
        extraction.id,
        "keyword",
        "chunker-v1",
        1,
        {"maximum_characters": 900},
        None,
        None,
        utc_timestamp(),
    )
    return index, chunks, mappings


def test_restart_round_trip_preserves_the_complete_record_graph(
    database: Database,
    data_dir: Path,
) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction, passages = make_extraction(source_id, source_version)
    extraction_job = store.create_job(
        "extract:source-version-one",
        "extract",
        source_id,
        source_version_id=source_version.id,
    )
    store.commit_extraction_step(
        extraction_job.id,
        "extract-passages",
        extraction,
        passages,
        1,
        {"complete": True, "passages": len(passages)},
    )
    index, chunks, mappings = make_index(extraction, passages)
    index_job = store.create_job(
        "index:extraction-one",
        "index",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=extraction.id,
    )
    store.commit_index_step(
        index_job.id,
        "build-chunks",
        index,
        chunks,
        mappings,
        1,
        {"complete": True, "chunks": len(chunks)},
    )
    before = store.load_snapshot()

    reopened_database = Database(DatabaseConfig(data_dir))
    reopened_database.initialize()
    after = PersistenceStore(reopened_database).load_snapshot()

    assert after == before


def test_exact_job_and_step_retry_reuses_one_committed_output(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction, passages = make_extraction(source_id, source_version)
    job = store.create_job(
        "extract:retry",
        "extract",
        source_id,
        source_version_id=source_version.id,
    )
    step = store.commit_extraction_step(job.id, "extract", extraction, passages, 1, {"done": True})
    retried_job = store.create_job(
        "extract:retry",
        "extract",
        source_id,
        source_version_id=source_version.id,
    )
    assert retried_job == job
    assert (
        store.commit_extraction_step(job.id, "extract", extraction, passages, 1, {"done": True})
        == step
    )

    with pytest.raises(ConflictingWriteError):
        store.commit_extraction_step(job.id, "extract", extraction, passages, 1, {"done": False})

    snapshot = store.load_snapshot()
    assert len(snapshot.extractions) == 1
    assert len(snapshot.job_steps) == 1


def test_job_output_and_checkpoint_roll_back_together_on_target_mismatch(
    database: Database,
) -> None:
    store = PersistenceStore(database)
    source_a, version_a = seed_source(store, "a.pdf")
    source_b, version_b = seed_source(store, "b.pdf")
    extraction, passages = make_extraction(source_a, version_a)
    job = store.create_job(
        "extract:wrong-source",
        "extract",
        source_b,
        source_version_id=version_b.id,
    )

    with pytest.raises(IntegrityViolationError):
        store.commit_extraction_step(job.id, "extract", extraction, passages, 1, {"done": True})

    with pytest.raises(PersistenceError):
        store.load_extraction(extraction.id)
    assert store.load_snapshot().job_steps == ()


def test_source_version_job_output_must_match_its_source(database: Database) -> None:
    store = PersistenceStore(database)
    source_a, _ = seed_source(store, "a.pdf")
    source_b, version_b = seed_source(store, "b.pdf")
    job = store.create_job("import:wrong-source", "import", source_a)

    with pytest.raises(IntegrityViolationError):
        store.commit_source_version_step(job.id, "save-version", version_b, 1, {"saved": True})

    assert store.load_snapshot().job_steps == ()


def test_index_job_output_must_match_its_extraction_target(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction_one, passages_one = make_extraction(source_id, source_version)
    extraction_two, passages_two = make_extraction(source_id, source_version, parser="next")
    store.save_extraction(extraction_one, passages_one)
    store.save_extraction(extraction_two, passages_two)
    index, chunks, mappings = make_index(extraction_one, passages_one)
    job = store.create_job(
        "index:wrong-extraction",
        "index",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=extraction_two.id,
    )

    with pytest.raises(IntegrityViolationError):
        store.commit_index_step(job.id, "build-index", index, chunks, mappings, 1, {"done": True})

    with pytest.raises(PersistenceError):
        store.load_index(index.id)
    assert store.load_snapshot().job_steps == ()


def test_database_trigger_rejects_passage_update(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction, passages = make_extraction(source_id, source_version)
    store.save_extraction(extraction, passages)

    with pytest.raises(IntegrityViolationError):
        with database.connect() as connection:
            connection.execute("UPDATE passages SET text='rewritten' WHERE id=?", (passages[0].id,))

    assert store.load_extraction(extraction.id).passages == passages


def test_database_rejects_malformed_record_timestamps(database: Database) -> None:
    with pytest.raises(IntegrityViolationError):
        with database.connect() as connection:
            connection.execute(
                """INSERT INTO notebooks(id,title,created_at,updated_at)
                   VALUES(?,?,?,?)""",
                (new_id(), "Invalid timestamp", "not-a-timestamp", "not-a-timestamp"),
            )


def test_database_foreign_keys_reject_mapping_to_another_extraction(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction_one, passages_one = make_extraction(source_id, source_version)
    extraction_two, passages_two = make_extraction(source_id, source_version, parser="next")
    store.save_extraction(extraction_one, passages_one)
    store.save_extraction(extraction_two, passages_two)
    index, chunks, mappings = make_index(extraction_one, passages_one)
    store.save_index(index, chunks, mappings)

    with pytest.raises(IntegrityViolationError):
        with database.connect() as connection:
            connection.execute(
                """INSERT INTO chunk_passages(
                       retrieval_chunk_id,passage_id,index_version_id,source_id,
                       source_version_id,extraction_version_id,ordinal,start_offset,end_offset
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    chunks[0].id,
                    passages_two[0].id,
                    index.id,
                    source_id,
                    source_version.id,
                    extraction_one.id,
                    1,
                    None,
                    None,
                ),
            )

    assert store.load_index(index.id).mappings == mappings


def test_reindex_and_reextract_keep_old_generations_unchanged(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction_one, passages_one = make_extraction(source_id, source_version)
    store.save_extraction(extraction_one, passages_one)
    index_one, chunks_one, mappings_one = make_index(extraction_one, passages_one)
    store.save_index(index_one, chunks_one, mappings_one)

    index_two, chunks_two, mappings_two = make_index(extraction_one, passages_one)
    store.save_index(index_two, chunks_two, mappings_two)
    assert store.load_index(index_one.id) == IndexGraph(index_one, chunks_one, mappings_one)

    extraction_two, passages_two = make_extraction(source_id, source_version, parser="pdf-next")
    passages_two = tuple(
        replace(item, id=new_id(), extraction_version_id=extraction_two.id) for item in passages_two
    )
    store.save_extraction(extraction_two, passages_two)
    index_three, chunks_three, mappings_three = make_index(extraction_two, passages_two)
    store.save_index(index_three, chunks_three, mappings_three)

    assert store.load_extraction(extraction_one.id).passages == passages_one
    assert store.load_index(index_one.id).chunks == chunks_one
    assert store.load_index(index_two.id).mappings == mappings_two
    assert all(item.id not in {passage.id for passage in passages_one} for item in passages_two)
    assert all(
        item.extraction_version_id == extraction_two.id
        for item in store.load_index(index_three.id).chunks
    )


def test_index_mapping_rejects_offsets_beyond_saved_passage(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction, passages = make_extraction(source_id, source_version)
    store.save_extraction(extraction, passages)
    index, chunks, mappings = make_index(extraction, passages)
    invalid = replace(mappings[0], end_offset=len(passages[0].text) + 1)

    with pytest.raises(ValueError, match="exceed"):
        store.save_index(index, chunks, (invalid,))


def test_non_finite_json_values_are_rejected(database: Database) -> None:
    store = PersistenceStore(database)
    source_id, source_version = seed_source(store)
    extraction, passages = make_extraction(source_id, source_version)
    malformed = replace(extraction, settings={"not_a_number": float("nan")})

    with pytest.raises(ValueError, match="valid JSON"):
        store.save_extraction(malformed, passages)
