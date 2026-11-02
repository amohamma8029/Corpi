import struct
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import (
    AttemptGuard,
    ExtractionOutcome,
    ExtractionPage,
    ExtractionVersion,
    JSONValue,
    Passage,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore
from core.search.chunking import make_chunks
from core.search.profiles import profiles
from core.search.representation import CANONICAL, settings_for
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage, purge_search, verify_generation


def seed(
    store: PersistenceStore,
    text: str,
    vector: list[float],
    *,
    notebook_id: str | None = None,
    profile_id: str = "bge-small-en",
    representation: str = CANONICAL,
    runtime_settings: dict[str, JSONValue] | None = None,
) -> tuple[str, str, str, str, str, AttemptGuard]:
    notebook_id = notebook_id or new_id()
    store.create_notebook(notebook_id, notebook_id)
    source_id = new_id()
    store.create_source(source_id, notebook_id, "paper.pdf")
    version = store.create_source_version(source_id, "a" * 64)
    extraction = ExtractionVersion(
        new_id(), source_id, version.id, "parser", "1", "1", 1, {}, utc_timestamp()
    )
    passage = Passage(
        new_id(),
        source_id,
        version.id,
        extraction.id,
        0,
        text,
        1,
        {"page": 1, "reading_order": 0},
        utc_timestamp(),
    )
    page = ExtractionPage(
        new_id(),
        source_id,
        version.id,
        extraction.id,
        1,
        True,
        len(text),
        0,
        0,
        "extracted",
        utc_timestamp(),
    )
    outcome = ExtractionOutcome(extraction.id, source_id, version.id, "complete", utc_timestamp())
    store.save_extraction_result(extraction, [passage], [page], [], outcome, None)
    job = store.create_job(
        "search:" + source_id, "pdf-search", source_id, version.id, extraction.id
    )
    store.create_job_status(job.id)
    store.claim_job(job.id, "test-worker")
    attempt = store.start_step_attempt(job.id, "prepare")
    guard = AttemptGuard("test-worker", "prepare", attempt.attempt_number)
    profile = profiles()[profile_id]
    settings: dict[str, JSONValue] = {
        "batch_size": 32 if representation == CANONICAL else 1,
        "separator": profile.separator,
        **settings_for(representation),
        **(runtime_settings or {}),
    }
    chunks = make_chunks([passage], profile, lambda _: 1, lambda text: text)
    search = SearchStorage(store)
    index_id = search.create_generation(job.id, guard, profile, chunks, settings)
    graph = store.load_index(index_id)
    with store.database.connect() as connection:
        _, input_hash = search.batch_inputs(connection, index_id, [c.id for c in graph.chunks])
    search.commit_batch(
        job.id,
        guard,
        index_id,
        0,
        [c.id for c in graph.chunks],
        [vector_bytes(vector, 384)],
        search_input_sha256=input_hash,
    )
    search.finish_components(job.id, guard, index_id)
    store.commit_index_step(
        job.id, "prepare", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    with store._transaction() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
            " VALUES(?,?,?)",
            (notebook_id, profile.identity, utc_timestamp()),
        )
    store.publish_readiness(job.id, source_id, version.id, [index_id], guard=guard)
    return notebook_id, source_id, version.id, index_id, job.id, guard


def test_filters_precede_vector_top_k_and_word_limit(store: PersistenceStore) -> None:
    target = [1.0] + [0.0] * 383
    weak = [0.0, 1.0] + [0.0] * 382
    notebook, source, version, index, _, _ = seed(store, "selected rainfall", weak)
    # Stronger decoys in another notebook and an unselected source in this one.
    seed(store, "rainfall rainfall rainfall", target)
    seed(store, "rainfall rainfall rainfall", target, notebook_id=notebook)
    search = SearchStorage(store)
    hits = search.search(
        notebook,
        [version],
        vector=vector_bytes(target, 384),
        vector_manifest_id=profiles()["bge-small-en"].identity,
        limit=1,
    )
    assert len(hits) == 1 and hits[0].source_id == source and hits[0].index_version_id == index
    assert hits[0].score == pytest.approx(1.0)  # Exact eligible cosine control.
    word = search.search(notebook, [version], query='rainfall OR " ) NEAR(', limit=1)
    assert len(word) == 1 and word[0].source_id == source
    assert word[0].chunk_id == hits[0].chunk_id
    assert word[0].mappings[0].source_version_id == version
    assert search.search(notebook, [version], query='"():') == ()


def test_missing_native_rows_cannot_pass_completion_proof(store: PersistenceStore) -> None:
    _, _, _, index, _, _ = seed(store, "measurements", [1.0] + [0.0] * 383)
    with store._transaction() as connection:
        row = connection.execute(
            "SELECT row_id FROM search_rows WHERE index_version_id=?", (index,)
        ).fetchone()
        connection.execute("DELETE FROM search_vectors WHERE rowid=?", (row[0],))
        with pytest.raises(ConflictingWriteError, match="incomplete"):
            verify_generation(connection, index)


def test_publication_checks_native_rows_inside_its_write_transaction(
    store: PersistenceStore,
) -> None:
    _, source, version, index, job, guard = seed(store, "measurements", [1.0] + [0.0] * 383)
    store.remove_readiness(source)
    with store._transaction() as connection:
        row = connection.execute(
            "SELECT row_id FROM search_rows WHERE index_version_id=?", (index,)
        ).fetchone()
        connection.execute("DELETE FROM search_vectors WHERE rowid=?", (row[0],))
    with pytest.raises(ConflictingWriteError, match="incomplete"):
        store.publish_readiness(job, source, version, [index], guard=guard)
    assert store.load_readiness(source) is None


def test_publication_deadline_rolls_back_readiness(store: PersistenceStore) -> None:
    _, source, version, index, job, guard = seed(store, "measurements", [1.0] + [0.0] * 383)
    store.remove_readiness(source)
    with pytest.raises(ConflictingWriteError, match="deadline expired"):
        store.publish_readiness(
            job, source, version, [index], guard=guard, search_deadline=time.monotonic() - 1
        )
    assert store.load_readiness(source) is None


def test_valid_but_changed_vectors_do_not_pass_the_saved_batch_proof(
    store: PersistenceStore,
) -> None:
    _, _, _, index, _, _ = seed(store, "measurements", [1.0] + [0.0] * 383)
    with store._transaction() as connection:
        row = connection.execute(
            "SELECT row_id FROM search_rows WHERE index_version_id=?", (index,)
        ).fetchone()
        connection.execute("DELETE FROM search_vectors WHERE rowid=?", (row[0],))
        connection.execute(
            "INSERT INTO search_vectors(rowid,embedding) VALUES(?,?)",
            (
                row[0],
                vector_bytes([0.0, 1.0] + [0.0] * 382, 384),
            ),
        )
        with pytest.raises(ConflictingWriteError, match="differ from committed output"):
            verify_generation(connection, index)


def test_cancelled_guard_rejects_batch_commit_and_hides_search(store: PersistenceStore) -> None:
    notebook, _, version, index, job, guard = seed(store, "measurements", [1.0] + [0.0] * 383)
    store.cancel_job(job)
    graph = store.load_index(index)
    with pytest.raises(ConflictingWriteError):
        SearchStorage(store).commit_batch(
            job,
            guard,
            index,
            0,
            [c.id for c in graph.chunks],
            [vector_bytes([1.0] + [0.0] * 383, 384)],
        )
    assert SearchStorage(store).search(notebook, [version], query="measurements") == ()


def test_batch_idempotence_rejects_different_vectors(store: PersistenceStore) -> None:
    _, _, _, index, job, guard = seed(store, "measurements", [1.0] + [0.0] * 383)
    graph = store.load_index(index)
    with pytest.raises(ConflictingWriteError, match="different vectors"):
        SearchStorage(store).commit_batch(
            job,
            guard,
            index,
            0,
            [c.id for c in graph.chunks],
            [vector_bytes([0.0, 1.0] + [0.0] * 382, 384)],
        )


def test_same_dimensions_do_not_authorize_wrong_query_profile(store: PersistenceStore) -> None:
    notebook, _, version, _, _, _ = seed(store, "measurements", [1.0] + [0.0] * 383)
    with pytest.raises(ConflictingWriteError, match="active notebook model"):
        SearchStorage(store).search(
            notebook,
            [version],
            vector=vector_bytes([1.0] + [0.0] * 383, 384),
            vector_manifest_id=profiles()["multilingual-minilm"].identity,
        )


def test_native_purge_removes_one_sources_rows_without_touching_others(
    store: PersistenceStore,
) -> None:
    _, source, _, index, _, _ = seed(store, "measurements", [1.0] + [0.0] * 383)
    notebook, _, version, _, _, _ = seed(store, "other measurements", [1.0] + [0.0] * 383)
    with store._transaction() as connection:
        purge_search(connection, source_id=source)
        assert (
            connection.execute(
                "SELECT 1 FROM search_rows WHERE index_version_id=?", (index,)
            ).fetchone()
            is None
        )
        assert connection.execute("SELECT count(*) FROM search_words").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == 1
    assert SearchStorage(store).search(notebook, [version], query="measurements")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0])
def test_vector_validation_rejects_invalid_outputs(bad: float) -> None:
    with pytest.raises(ValueError, match="invalid_vector"):
        vector_bytes([bad] * 384, 384)
    with pytest.raises(ValueError, match="invalid_vector"):
        from core.search.storage import validate_vector

        validate_vector(struct.pack("<384f", *([bad] * 384)))


def test_changed_profile_settings_have_distinct_manifest_identity() -> None:
    profile = profiles()["bge-small-en"]
    assert replace(profile, query_prefix="different instruction").identity != profile.identity


def test_word_search_does_not_restart_fts_ranking_for_each_eligible_row(
    store: PersistenceStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook, _, version, _, _, _ = seed(store, "selected rainfall", [1.0] + [0.0] * 383)
    connect = store.database.connect
    statements = []

    @contextmanager
    def traced_connection():
        with connect() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(store.database, "connect", traced_connection)
    assert SearchStorage(store).search(notebook, [version], query="rainfall")
    sql = next(statement for statement in statements if "bm25(search_words)" in statement)
    with connect() as connection:
        plan = connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    fts = [row[3] for row in plan if "SCAN search_words VIRTUAL TABLE INDEX" in row[3]]
    assert len(fts) == 1
    # '=' identifies FTS5's rowid lookup plan. That restarts MATCH/ranking
    # per eligible row and grew to twelve seconds in the 50k corpus control.
    assert "=" not in fts[0].split("VIRTUAL TABLE INDEX", 1)[1]


def test_generation_proof_uses_the_owned_page_lookup_instead_of_scanning_all_pages(
    store: PersistenceStore,
) -> None:
    _, _, _, index, _, _ = seed(store, "rainfall", [1.0] + [0.0] * 383)
    statements = []
    with store.database.connect() as connection:
        connection.set_trace_callback(statements.append)
        verify_generation(connection, index)
        sql = next(s for s in statements if "LEFT JOIN extraction_pages q" in s)
        plan = connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    pages = [row[3] for row in plan if "SEARCH q " in row[3]]
    assert len(pages) == 1
    assert "source_id=?" in pages[0] and "page_number=?" in pages[0]
