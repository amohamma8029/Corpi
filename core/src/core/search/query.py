"""Notebook-scoped local queries; inference never owns a database transaction."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal

from core.persistence.errors import ConflictingWriteError
from core.persistence.store import PersistenceStore
from core.search.profiles import profiles
from core.search.representation import encode_query, selected
from core.search.storage import SearchHit, SearchStorage
from core.search.supervisor import EmbeddingFailure, EmbeddingSession, openblas_options

SearchMode = Literal["meaning", "words"]


class SearchFailure(Exception):
    def __init__(self, code: str, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.code, self.status = code, status


class QueryService:
    def __init__(
        self,
        store: PersistenceStore,
        models_directory: Path,
        *,
        session_factory: Callable[..., AbstractContextManager[Any]] = EmbeddingSession,
    ) -> None:
        self.store = store
        self.models_directory = models_directory
        self.session_factory = session_factory
        # Admission only, not a data lock. Concurrent word searches stay available.
        self._meaning_query = threading.BoundedSemaphore(1)

    def _scope(
        self, notebook_id: str, connection: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        if connection is None:
            with self.store.database.connect() as local:
                return self._scope(notebook_id, local)
        return [
            dict(r)
            for r in connection.execute(
                """SELECT i.id AS index_id,i.source_version_id,g.manifest_id,g.settings_json
                   FROM source_readiness ready JOIN sources s ON s.id=ready.source_id
                   JOIN source_readiness_indexes ri ON ri.source_id=s.id
                   JOIN index_versions i ON i.id=ri.index_version_id
                     AND i.source_version_id=ready.source_version_id
                   JOIN search_generations g ON g.index_version_id=i.id
                   JOIN notebook_profiles n ON n.notebook_id=s.notebook_id
                   JOIN extraction_outcomes o ON o.extraction_version_id=i.extraction_version_id
                   LEFT JOIN extraction_reviews r ON r.extraction_version_id=i.extraction_version_id
                   WHERE s.notebook_id=? AND g.manifest_id=n.active_manifest_id
                      AND (o.outcome='complete' OR (o.outcome='partial' AND r.state='confirmed'))
                      AND i.extraction_version_id=(SELECT e.id FROM extraction_versions e
                         WHERE e.source_version_id=i.source_version_id
                         ORDER BY e.created_at DESC,e.rowid DESC LIMIT 1)
                     AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)
                     AND NOT EXISTS(SELECT 1 FROM job_status j
                        WHERE j.job_id=ready.published_by_job_id
                        AND (j.status IN ('failed','cancelled')
                             OR j.superseded_by_job_id IS NOT NULL))
                   ORDER BY i.id""",
                (notebook_id,),
            )
        ]

    def search(
        self, notebook_id: str, query: str, mode: SearchMode, *, limit: int = 10
    ) -> dict[str, Any]:
        self.store.load_notebook(notebook_id)
        if (
            not query.strip()
            or len(query) > 8192
            or mode not in {"meaning", "words"}
            or not 1 <= limit <= 50
        ):
            raise ValueError("invalid_search_query")
        scope = self._scope(notebook_id)
        if not scope:
            return {"notebook_id": notebook_id, "mode": mode, "ready_sources": 0, "results": []}
        groups: dict[str, list[str]] = defaultdict(list)
        runtime_groups: dict[tuple[bool, bool, int | None], dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        try:
            for row in scope:
                settings = json.loads(row["settings_json"])
                representation = selected(settings)
                groups[representation].append(row["source_version_id"])
                if mode == "meaning":
                    runtime = (
                        bool(settings.get("cpu_memory_arena", False)),
                        settings.get("idle_worker_policy") == "stop-spinning-after-run-v1",
                        openblas_options(settings).get("openblas_threads"),
                    )
                    runtime_groups[runtime][representation].append(row["source_version_id"])
        except ValueError:
            raise SearchFailure(
                "search_input_changed", "Search data uses an unsupported input version."
            ) from None
        storage = SearchStorage(self.store)
        hits: list[SearchHit] = []
        if mode == "words":
            for representation, versions in groups.items():
                hits.extend(
                    storage.search(
                        notebook_id,
                        versions,
                        query=query,
                        representation=representation,
                        limit=limit,
                    )
                )
        else:
            manifest = scope[0]["manifest_id"]
            profile = next((p for p in profiles().values() if p.identity == manifest), None)
            if profile is None:
                raise SearchFailure("unsupported_model", "The notebook model is unavailable.")
            with self.store.database.connect() as connection:
                installed = connection.execute(
                    "SELECT state FROM model_installations WHERE manifest_id=?", (manifest,)
                ).fetchone()
            if installed is None or installed[0] != "installed":
                raise SearchFailure(
                    "model_not_installed", "Download the notebook model in Local search first."
                )
            if not self._meaning_query.acquire(blocking=False):
                raise SearchFailure(
                    "search_busy", "Another meaning search is running. Please retry shortly.", 503
                )
            try:
                # One owned session at a time. Old and new generations retain
                # their saved runtime choices, even in the same notebook.
                for (arena, spinning, threads), representations in runtime_groups.items():
                    options = {} if threads is None else {"openblas_threads": threads}
                    with self.session_factory(
                        self.models_directory / profile.identity,
                        profile,
                        load_timeout=120,
                        memory_limit=1024**3,
                        memory_arena=arena,
                        stop_spinning=spinning,
                        **options,
                    ) as session:
                        for representation, versions in representations.items():
                            encoded = encode_query(session, query, representation, timeout=120)
                            hits.extend(
                                storage.search_encoded(notebook_id, versions, encoded, limit=limit)
                            )
            except EmbeddingFailure as error:
                if error.code in {"model_input_too_long", "invalid_embedding_input"}:
                    raise ValueError("invalid_search_query") from None
                raise SearchFailure(
                    error.code, "The local search worker could not finish. Please retry.", 503
                ) from None
            except TimeoutError:
                raise SearchFailure(
                    "embedding_timeout", "The local search worker reached its time limit.", 503
                ) from None
            finally:
                self._meaning_query.release()
        return {
            "notebook_id": notebook_id,
            "mode": mode,
            **self._decorate(notebook_id, scope, hits, limit),
        }

    def _decorate(
        self, notebook_id: str, scope: list[dict[str, Any]], hits: list[SearchHit], limit: int
    ) -> dict[str, Any]:
        result = []
        # Eligibility and citation/PDF decoration share one post-inference snapshot.
        with self.store._transaction(write=False) as connection:
            current = {r["index_id"]: r for r in self._scope(notebook_id, connection)}
            allowed = {r["index_id"] for r in scope} & current.keys()
            ordered = sorted(
                (h for h in hits if h.index_version_id in allowed),
                key=lambda h: (h.score, h.source_version_id, h.chunk_id),
            )[:limit]
            for hit in ordered:
                source = connection.execute(
                    "SELECT display_name FROM sources WHERE id=?", (hit.source_id,)
                ).fetchone()
                if source is None:
                    continue
                pdf = connection.execute(
                    "SELECT p.id,p.display_filename FROM pdf_imports p "
                    "WHERE p.source_id=? AND p.source_version_id=? AND p.state='ready' "
                    "AND NOT EXISTS(SELECT 1 FROM pdf_import_trash t WHERE t.import_id=p.id)",
                    (hit.source_id, hit.source_version_id),
                ).fetchone()
                citations = []
                for mapping in hit.mappings:
                    passage = connection.execute(
                        "SELECT location_json FROM passages WHERE id=? AND source_version_id=?",
                        (mapping.passage_id, hit.source_version_id),
                    ).fetchone()
                    if passage is None:
                        raise ConflictingWriteError("search citation changed")
                    page = json.loads(passage[0]).get("page")
                    citations.append(
                        {
                            "passage_id": mapping.passage_id,
                            "ordinal": mapping.ordinal,
                            "start_offset": mapping.start_offset,
                            "end_offset": mapping.end_offset,
                            "page_number": page if isinstance(page, int) and page >= 1 else None,
                        }
                    )
                result.append(
                    {
                        "source_id": hit.source_id,
                        "source_version_id": hit.source_version_id,
                        "index_version_id": hit.index_version_id,
                        "chunk_id": hit.chunk_id,
                        "manifest_id": hit.manifest_id,
                        "text": hit.text,
                        "score": hit.score,
                        "source_name": source[0],
                        "import_id": pdf[0] if pdf else None,
                        "filename": pdf[1] if pdf else None,
                        "citations": citations,
                    }
                )
        return {
            "ready_sources": len({current[i]["source_version_id"] for i in allowed}),
            "results": result,
        }
