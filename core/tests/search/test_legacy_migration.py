"""A populated, real revision-13 canonical index survives the additive input-proof migration."""

import hashlib

import pytest

from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import MigrationError
from core.persistence.migrations import Migration, load_migrations
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
from core.search.chunking import make_chunks
from core.search.preparation import SearchLimits, effective_settings
from core.search.profiles import profiles
from core.search.query import QueryService
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage, chunk_set_hash, verify_generation


@pytest.mark.parametrize("failed_first", [False, True])
def test_revision_fourteen_keeps_canonical_search_identity_choices_and_charges(
    tmp_path, failed_first
):
    migrations = load_migrations()
    old = Database(DatabaseConfig(tmp_path / "legacy"), migrations=migrations[:13])
    old.initialize()
    store = PersistenceStore(old)
    notebook = store.create_notebook(new_id(), "Legacy choice")
    source = store.create_source(new_id(), notebook.id, "old.pdf")
    version = store.create_source_version(source.id, "a" * 64)
    stamp = utc_timestamp()
    extraction = ExtractionVersion(
        new_id(), source.id, version.id, "legacy-test", "1", "1", 1, {}, stamp
    )
    text = "ﻣﻄـﺮ canonical rainfall"
    passage = Passage(
        new_id(), source.id, version.id, extraction.id, 0, text, 1, {"page": 1}, stamp
    )
    page = ExtractionPage(
        new_id(), source.id, version.id, extraction.id, 1, True, len(text), 0, 0, "extracted", stamp
    )
    store.save_extraction_result(
        extraction,
        [passage],
        [page],
        [],
        ExtractionOutcome(extraction.id, source.id, version.id, "complete", stamp),
        None,
    )
    job = store.create_job("legacy-search", "pdf-search", source.id, version.id, extraction.id)
    store.create_job_status(job.id)
    store.claim_job(job.id, "legacy")
    attempt = store.start_step_attempt(job.id, "prepare")
    guard = AttemptGuard("legacy", "prepare", attempt.attempt_number)
    profile = profiles()["minilm-en"]
    settings = effective_settings(profile, SearchLimits(total_seconds=600), None)
    settings["cpu_memory_arena"] = False
    settings.pop("idle_worker_policy")
    storage = SearchStorage(store)
    index = storage.create_generation(
        job.id, guard, profile, make_chunks([passage], profile, lambda _: 1, lambda t: t), settings
    )
    graph = store.load_index(index)
    vector = vector_bytes([1.0] + [0.0] * 383, 384)
    # Write the pre-14 native format: no derived-input table or batch proof column exists.
    with old.connect() as connection:
        row = connection.execute(
            "SELECT row_id FROM search_rows WHERE index_version_id=?", (index,)
        ).fetchone()[0]
        connection.execute("INSERT INTO search_vectors(rowid,embedding) VALUES(?,?)", (row, vector))
        connection.execute("INSERT INTO search_words(rowid,text) VALUES(?,?)", (row, text))
        connection.execute(
            "UPDATE search_batches SET complete=1,attempts=1,vectors_sha256=? "
            "WHERE index_version_id=?",
            (hashlib.sha256(vector).hexdigest(), index),
        )
        for component in ("word", "vector"):
            connection.execute(
                "INSERT INTO search_components VALUES(?,?,?)",
                (index, component, chunk_set_hash([graph.chunks[0].id])),
            )
    store.commit_index_step(
        job.id, "prepare", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    store.finish_step_attempt(job.id, "prepare", attempt.attempt_number, None)
    store.finish_job(job.id, "completed")
    with old.connect() as connection:
        connection.execute(
            "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at) "
            "VALUES(?,?,?)",
            (notebook.id, profile.identity, stamp),
        )
        connection.execute(
            "INSERT INTO source_readiness(source_id,source_version_id,published_by_job_id,"
            "published_at,updated_at) VALUES(?,?,?,?,?)",
            (source.id, version.id, job.id, stamp, stamp),
        )
        connection.execute("INSERT INTO source_readiness_indexes VALUES(?,?)", (source.id, index))
        connection.execute(
            "UPDATE search_generations SET spent_seconds=123.5,published_by_job_id=? "
            "WHERE index_version_id=?",
            (job.id, index),
        )
        tables = (
            "notebooks",
            "sources",
            "passages",
            "chunk_passages",
            "index_versions",
            "notebook_profiles",
            "search_generations",
            "search_batches",
            "source_readiness",
        )
        columns = {
            table: [r[1] for r in connection.execute(f"PRAGMA table_info({table})")]
            for table in tables
        }
        before = {
            table: [tuple(r) for r in connection.execute(f"SELECT * FROM {table}")]
            for table in tables
        }
    old.initialize()
    if failed_first:
        actual = migrations[13]
        broken = Migration(
            actual.revision,
            actual.name,
            actual.sql + "\nINSERT INTO missing_test_table VALUES(1);",
            requires_backup=True,
        )
        with pytest.raises(MigrationError):
            Database(old.config, migrations=migrations[:13] + (broken,)).initialize()
        old.initialize()
    upgraded = Database(old.config)
    upgraded.initialize()
    current = PersistenceStore(upgraded)
    with upgraded.connect() as connection:
        for table in tables:
            assert [
                tuple(r)
                for r in connection.execute(f"SELECT {','.join(columns[table])} FROM {table}")
            ] == before[table]
        assert connection.execute("SELECT count(*) FROM search_text_inputs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT search_input_sha256 FROM search_batches").fetchone()[0]
            is None
        )
        verify_generation(connection, index)
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
    result = QueryService(current, tmp_path / "unused-models").search(
        notebook.id, "rainfall", "words"
    )
    assert result["results"][0]["text"] == text
    assert current.load_index(index) == graph
    assert current.load_index(index).index.settings == settings
    assert settings["total_seconds"] == 600 and settings["cpu_memory_arena"] is False
