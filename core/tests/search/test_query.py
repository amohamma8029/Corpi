"""Real native search rows with controlled inference; no models or downloads required."""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from test_storage import seed

from core.api.app import create_app
from core.persistence.models import new_id, utc_timestamp
from core.search.profiles import profiles
from core.search.query import QueryService, SearchFailure
from core.search.representation import ARABIC, CANONICAL, search_text
from core.search.runtime import vector_bytes
from core.search.supervisor import EmbeddingFailure

VECTOR = [1.0] + [0.0] * 383


def install(store, profile_id):
    profile = profiles()[profile_id]
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
            " VALUES(?,?,'installed',?)",
            (profile.identity, profile_id, utc_timestamp()),
        )


def test_words_scope_is_all_ready_languages_not_other_notebooks_or_pending_versions(store):
    notebook, source, version, index, _, _ = seed(store, "rainfall", VECTOR)
    _, french, _, _, _, _ = seed(store, "pluie rainfall", VECTOR, notebook_id=notebook)
    _, arabic, _, _, _, _ = seed(store, "المطر rainfall", VECTOR, notebook_id=notebook)
    seed(store, "rainfall rainfall", VECTOR)
    store.create_source_version(source, "b" * 64)
    service = QueryService(store, store.database.config.data_dir / "models")
    response = service.search(notebook, "rainfall", "words")
    assert response["ready_sources"] == 3
    assert {r["source_id"] for r in response["results"]} == {source, french, arabic}
    hit = next(r for r in response["results"] if r["source_id"] == source)
    assert (hit["source_version_id"], hit["index_version_id"]) == (version, index)
    assert hit["text"] == "rainfall" and hit["citations"][0]["page_number"] == 1
    assert hit["import_id"] is None


@pytest.mark.parametrize("representation", [CANONICAL, ARABIC])
def test_query_matches_saved_representation_and_runtime_without_a_transaction(
    store,
    representation,
    monkeypatch,
):
    text = "ﻣﻄـﺮ rainfall"
    notebook, _, _, _, _, _ = seed(
        store,
        text,
        VECTOR,
        profile_id="minilm-en",
        representation=representation,
    )
    install(store, "minilm-en")
    transactions = []
    transaction = store._transaction

    @contextmanager
    def tracked(*args, **kwargs):
        with transaction(*args, **kwargs) as connection:
            transactions.append(connection)
            try:
                yield connection
            finally:
                transactions.pop()

    monkeypatch.setattr(store, "_transaction", tracked)
    calls = []

    class Session:
        def __init__(self, path, profile, **options):
            assert not transactions
            assert path.name == profile.identity
            assert options["memory_arena"] is False and options["stop_spinning"] is False
            self.profile = profile

        def __enter__(self):
            return self

        def embed(self, texts, **options):
            assert not transactions
            calls.append((texts, options))
            return [vector_bytes(VECTOR, 384)]

        def __exit__(self, *_):
            calls.append("closed")

    result = QueryService(store, store.database.config.data_dir, session_factory=Session).search(
        notebook,
        text,
        "meaning",
    )
    assert result["results"][0]["text"] == text
    assert calls == [
        ([search_text(text, representation)], {"query": True, "timeout": 120}),
        "closed",
    ]


def test_final_snapshot_discards_cancelled_hits_and_exact_pdf_version_is_used(store, monkeypatch):
    notebook, source, version, _, job, _ = seed(store, "rainfall", VECTOR)
    import_id = new_id()
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO pdf_imports(id,source_id,source_version_id,display_filename,state,"
            "created_at,updated_at) VALUES(?,?,?,'original.pdf','ready',?,?)",
            (import_id, source, version, utc_timestamp(), utc_timestamp()),
        )
    service = QueryService(store, store.database.config.data_dir)
    hit = service.search(notebook, "rainfall", "words")["results"][0]
    assert hit["import_id"] == import_id and hit["filename"] == "original.pdf"
    from core.search.storage import SearchStorage

    search = SearchStorage.search

    def cancelled(*args, **kwargs):
        hits = search(*args, **kwargs)
        store.cancel_job(job)
        return hits

    monkeypatch.setattr(SearchStorage, "search", cancelled)
    assert service.search(notebook, "rainfall", "words")["results"] == []


def test_missing_model_busy_and_failed_worker_are_explicit_and_release_admission(store):
    notebook, _, _, _, _, _ = seed(store, "rainfall", VECTOR)

    def failure(*args, **kwargs):
        raise EmbeddingFailure("embedding_worker_failed")

    service = QueryService(store, store.database.config.data_dir, session_factory=failure)
    with pytest.raises(SearchFailure) as error:
        service.search(notebook, "rainfall", "meaning")
    assert error.value.code == "model_not_installed"
    install(store, "bge-small-en")
    assert service._meaning_query.acquire(blocking=False)
    try:
        with pytest.raises(SearchFailure) as error:
            service.search(notebook, "rainfall", "meaning")
        assert error.value.code == "search_busy"
        assert service.search(notebook, "rainfall", "words")["results"]
    finally:
        service._meaning_query.release()
    for _ in range(2):
        with pytest.raises(SearchFailure) as error:
            service.search(notebook, "rainfall", "meaning")
        assert error.value.code == "embedding_worker_failed"


def test_search_api_auth_validation_empty_and_canonical_response(store):
    notebook, _, _, _, _, _ = seed(store, "المطر rainfall", VECTOR)
    app = create_app("test-search-only", store=store)
    headers = {"Authorization": "Bearer test-search-only"}
    path = f"/notebooks/{notebook}/search"
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post(path, json={"query": "rainfall"}).status_code == 401
        assert (
            client.post(
                path,
                json={"query": "rainfall"},
                headers={**headers, "Origin": "https://other.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                path, json={"query": "rainfall"}, headers={**headers, "Host": "other.example"}
            ).status_code
            == 403
        )
        assert client.post(path, json={"query": " "}, headers=headers).status_code == 400
        assert (
            client.post(path, json={"query": "rainfall", "limit": 51}, headers=headers).status_code
            == 422
        )
        response = client.post(path, json={"query": "rainfall", "mode": "words"}, headers=headers)
        assert (
            response.status_code == 200
            and response.json()["results"][0]["text"] == "المطر rainfall"
        )
        assert (
            client.post(
                f"/notebooks/{new_id()}/search", json={"query": "rainfall"}, headers=headers
            ).status_code
            == 404
        )
        empty = store.create_notebook(new_id(), "Empty")
        response = client.post(
            f"/notebooks/{empty.id}/search", json={"query": "rainfall"}, headers=headers
        )
        assert response.json()["ready_sources"] == 0 and response.json()["results"] == []
