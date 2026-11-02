"""Search ownership, atomic batch commits, and eligible-before-top-K operations.

The store supplies the transaction boundary. Native rows never independently
establish ownership or readiness: canonical chunk identities are authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import (
    AttemptGuard,
    ChunkPassage,
    IndexVersion,
    JSONValue,
    RetrievalChunk,
    new_id,
    utc_timestamp,
)
from core.search.chunking import CHUNKER_VERSION, Chunk
from core.search.profiles import ModelProfile
from core.search.representation import (
    CANONICAL,
    EncodedQuery,
    batch_input_hash,
    search_text,
    selected,
    settings_for,
    text_hash,
)

if TYPE_CHECKING:
    from core.persistence.store import PersistenceStore


def chunk_set_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def generation_identity(job, profile: ModelProfile, settings: dict[str, JSONValue]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "source": job.source_id,
                "version": job.source_version_id,
                "extraction": job.extraction_version_id,
                "manifest": profile.identity,
                "chunker": CHUNKER_VERSION,
                "settings": settings,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_vector(blob: bytes, dimensions: int = 384) -> None:
    if len(blob) != dimensions * 4:
        raise ValueError("invalid_vector")
    values = struct.unpack(f"<{dimensions}f", blob)
    if not all(math.isfinite(v) for v in values) or not math.isclose(
        math.hypot(*values), 1, abs_tol=1e-5
    ):
        raise ValueError("invalid_vector")


def verify_generation(connection: sqlite3.Connection, index_id: str) -> None:
    """Prove actual rows, mappings, and completion for exactly one generation."""
    generation = connection.execute(
        "SELECT * FROM search_generations WHERE index_version_id=?", (index_id,)
    ).fetchone()
    if generation is None:
        raise ConflictingWriteError("search generation is incomplete")
    settings = json.loads(generation["settings_json"])
    try:
        representation = selected(settings)
    except ValueError as error:
        raise ConflictingWriteError("search representation is unsupported") from error
    index = connection.execute("SELECT * FROM index_versions WHERE id=?", (index_id,)).fetchone()
    if index is None or json.loads(index["settings_json"]) != settings:
        raise ConflictingWriteError("search generation settings differ from canonical index")
    expected_identity = hashlib.sha256(
        json.dumps(
            {
                "source": index["source_id"],
                "version": index["source_version_id"],
                "extraction": index["extraction_version_id"],
                "manifest": generation["manifest_id"],
                "chunker": CHUNKER_VERSION,
                "settings": settings,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if (
        generation["identity_sha256"] != expected_identity
        or index["embedding_revision"] != generation["manifest_id"]
    ):
        raise ConflictingWriteError("search generation identity differs from canonical index")
    rows = connection.execute(
        """SELECT r.row_id,r.chunk_id,c.text,c.extraction_version_id
           FROM search_rows r JOIN retrieval_chunks c ON c.id=r.chunk_id
           WHERE r.index_version_id=? ORDER BY c.ordinal""",
        (index_id,),
    ).fetchall()
    expected = connection.execute(
        "SELECT id FROM retrieval_chunks WHERE index_version_id=? ORDER BY ordinal", (index_id,)
    ).fetchall()
    ids = [row["chunk_id"] for row in rows]
    if not ids or len(ids) != generation["expected_chunks"] or ids != [row[0] for row in expected]:
        raise ConflictingWriteError("search chunk membership is incomplete")
    separator = settings["separator"]
    canonical = {}
    for row in rows:
        links = connection.execute(
            """SELECT m.*,p.text AS passage_text,p.location_json,q.classification
               FROM chunk_passages m JOIN passages p ON p.id=m.passage_id
                LEFT JOIN extraction_pages q ON q.extraction_version_id=p.extraction_version_id
                  AND q.source_id=p.source_id AND q.source_version_id=p.source_version_id
                  AND q.page_number=json_extract(p.location_json,'$.page')
               WHERE m.retrieval_chunk_id=? ORDER BY m.ordinal""",
            (row["chunk_id"],),
        ).fetchall()
        if not links or any(link["classification"] != "extracted" for link in links):
            raise ConflictingWriteError("search mapping is not eligible")
        fragments = []
        for link in links:
            start = link["start_offset"]
            end = link["end_offset"]
            fragments.append(
                link["passage_text"] if start is None else link["passage_text"][start:end]
            )
        if separator.join(fragments) != row["text"]:
            raise ConflictingWriteError("search text cannot be reconstructed")
        word = connection.execute(
            "SELECT text FROM search_words WHERE rowid=?", (row["row_id"],)
        ).fetchone()
        vector = connection.execute(
            "SELECT embedding FROM search_vectors WHERE rowid=?", (row["row_id"],)
        ).fetchone()
        expected_text = search_text(row["text"], representation)
        proof = connection.execute(
            "SELECT * FROM search_text_inputs WHERE row_id=?", (row["row_id"],)
        ).fetchone()
        if representation == CANONICAL:
            if proof is not None:
                raise ConflictingWriteError("canonical search has unexpected derived input")
        elif proof is None or (
            proof["representation"],
            proof["canonical_sha256"],
            proof["search_sha256"],
            proof["search_text"],
        ) != (representation, text_hash(row["text"]), text_hash(expected_text), expected_text):
            raise ConflictingWriteError("search derived input proof differs from canonical text")
        canonical[row["chunk_id"]] = row["text"]
        if word is None or word[0] != expected_text or vector is None:
            raise ConflictingWriteError("search component rows are incomplete")
        validate_vector(vector[0])
    components = connection.execute(
        "SELECT component,chunk_ids_sha256 FROM search_components WHERE index_version_id=?",
        (index_id,),
    ).fetchall()
    if dict(components) != {"word": chunk_set_hash(ids), "vector": chunk_set_hash(ids)}:
        raise ConflictingWriteError("search component proof is incomplete")
    batches = connection.execute(
        "SELECT * FROM search_batches WHERE index_version_id=? ORDER BY ordinal", (index_id,)
    ).fetchall()
    batched = [identity for batch in batches for identity in json.loads(batch["chunk_ids_json"])]
    if batched != ids or any(
        not batch["complete"] or batch["vectors_sha256"] is None for batch in batches
    ):
        raise ConflictingWriteError("search batch proof is incomplete")
    for batch in batches:
        batch_ids = json.loads(batch["chunk_ids_json"])
        if representation != CANONICAL and (settings.get("batch_size") != 1 or len(batch_ids) != 1):
            raise ConflictingWriteError("derived search batch is not singleton")
        expected_input = (
            None
            if representation == CANONICAL
            else batch_input_hash(batch_ids, [canonical[key] for key in batch_ids], settings)
        )
        if batch["search_input_sha256"] != expected_input:
            raise ConflictingWriteError(
                "search vector batch input proof differs from canonical text"
            )
        digest = hashlib.sha256()
        for identity in json.loads(batch["chunk_ids_json"]):
            vector = connection.execute(
                """SELECT v.embedding FROM search_rows r
                   JOIN search_vectors v ON v.rowid=r.row_id
                   WHERE r.chunk_id=? AND r.index_version_id=?""",
                (identity, index_id),
            ).fetchone()
            if vector is None:
                raise ConflictingWriteError("search batch vectors are incomplete")
            digest.update(vector[0])
        if digest.hexdigest() != batch["vectors_sha256"]:
            raise ConflictingWriteError("search batch vectors differ from committed output")


@dataclass(frozen=True)
class SearchHit:
    source_id: str
    source_version_id: str
    index_version_id: str
    chunk_id: str
    manifest_id: str
    text: str
    score: float
    mappings: tuple[ChunkPassage, ...]


class SearchStorage:
    def __init__(self, store: PersistenceStore) -> None:
        self.store = store

    def search_encoded(
        self, notebook_id: str, source_versions: list[str], query: EncodedQuery, *, limit: int = 50
    ) -> tuple[SearchHit, ...]:
        rules = settings_for(query.representation)
        if query.rule_sha256 != rules.get("search_rule_sha256"):
            raise ConflictingWriteError(
                "query representation rule differs from current search input"
            )
        return self.search(
            notebook_id,
            source_versions,
            vector=query.vector,
            vector_manifest_id=query.manifest_id,
            representation=query.representation,
            vector_representation=query.representation,
            vector_rule_sha256=query.rule_sha256,
            limit=limit,
        )

    def lookup_generation(
        self, job_id: str, profile: ModelProfile, settings: dict[str, JSONValue]
    ) -> str | None:
        selected(settings)
        job = self.store.load_job_by_id(job_id)
        with self.store.database.connect() as connection:
            row = connection.execute(
                "SELECT index_version_id FROM search_generations WHERE identity_sha256=?",
                (generation_identity(job, profile, settings),),
            ).fetchone()
            return None if row is None else row[0]

    def create_generation(
        self,
        job_id: str,
        guard: AttemptGuard,
        profile: ModelProfile,
        chunks: tuple[Chunk, ...],
        settings: dict[str, JSONValue],
    ) -> str:
        from core.persistence.store import _save_index, _verify_attempt_current

        representation = selected(settings)
        if representation != CANONICAL and settings["batch_size"] != 1:
            raise ValueError("derived_search_requires_singleton")

        job = self.store.load_job_by_id(job_id)
        if not job.source_version_id or not job.extraction_version_id or not chunks:
            raise ValueError("invalid_search_target")
        identity = generation_identity(job, profile, settings)
        with self.store._transaction() as connection:
            _verify_attempt_current(connection, job_id, guard.step_key, guard)
            existing = connection.execute(
                "SELECT index_version_id FROM search_generations WHERE identity_sha256=?",
                (identity,),
            ).fetchone()
            if existing is not None:
                return existing[0]
            index_id = new_id()
            index = IndexVersion(
                index_id,
                job.source_id,
                job.source_version_id,
                job.extraction_version_id,
                "hybrid-search",
                CHUNKER_VERSION,
                1,
                settings,
                profile.model,
                profile.identity,
                utc_timestamp(),
            )
            records = []
            mappings = []
            for ordinal, chunk in enumerate(chunks):
                record = RetrievalChunk(
                    new_id(),
                    index_id,
                    job.source_id,
                    job.source_version_id,
                    job.extraction_version_id,
                    ordinal,
                    chunk.text,
                )
                records.append(record)
                for position, part in enumerate(chunk.mappings):
                    whole = part.start == 0 and part.end == len(part.passage.text)
                    mappings.append(
                        ChunkPassage(
                            record.id,
                            part.passage.id,
                            index_id,
                            job.source_id,
                            job.source_version_id,
                            job.extraction_version_id,
                            position,
                            None if whole else part.start,
                            None if whole else part.end,
                        )
                    )
            _save_index(connection, index, records, mappings)
            connection.execute(
                """INSERT INTO search_generations(index_version_id,manifest_id,identity_sha256,
                   settings_json,expected_chunks) VALUES(?,?,?,?,?)""",
                (
                    index_id,
                    profile.identity,
                    identity,
                    json.dumps(settings, sort_keys=True),
                    len(records),
                ),
            )
            connection.executemany(
                "INSERT INTO search_rows(chunk_id,index_version_id) VALUES(?,?)",
                [(record.id, index_id) for record in records],
            )
            batch_size = int(settings["batch_size"])  # type: ignore[arg-type]
            if representation != CANONICAL:
                for record in records:
                    row_id = connection.execute(
                        "SELECT row_id FROM search_rows WHERE chunk_id=?", (record.id,)
                    ).fetchone()[0]
                    derived = search_text(record.text, representation)
                    connection.execute(
                        "INSERT INTO search_text_inputs VALUES(?,?,?,?,?)",
                        (
                            row_id,
                            representation,
                            text_hash(record.text),
                            text_hash(derived),
                            derived,
                        ),
                    )
            for offset in range(0, len(records), batch_size):
                connection.execute(
                    "INSERT INTO search_batches(index_version_id,ordinal,chunk_ids_json)"
                    " VALUES(?,?,?)",
                    (
                        index_id,
                        offset // batch_size,
                        json.dumps([r.id for r in records[offset : offset + batch_size]]),
                    ),
                )
            return index_id

    def commit_batch(
        self,
        job_id: str,
        guard: AttemptGuard,
        index_id: str,
        ordinal: int,
        chunk_ids: list[str],
        vectors: list[bytes],
        *,
        search_input_sha256: str | None = None,
        deadline: float | None = None,
    ) -> None:
        from core.persistence.store import _verify_attempt_current

        if len(chunk_ids) != len(vectors) or not chunk_ids:
            raise ValueError("invalid_vector_count")
        for blob in vectors:
            validate_vector(blob)
        digest = hashlib.sha256(b"".join(vectors)).hexdigest()
        with self.store._transaction() as connection:
            _verify_attempt_current(connection, job_id, guard.step_key, guard)
            job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            index = connection.execute(
                "SELECT * FROM index_versions WHERE id=?", (index_id,)
            ).fetchone()
            if (
                index is None
                or job is None
                or any(
                    index[key] != job[key]
                    for key in ("source_id", "source_version_id", "extraction_version_id")
                )
            ):
                raise ConflictingWriteError("search batch ownership mismatch")
            batch = connection.execute(
                "SELECT * FROM search_batches WHERE index_version_id=? AND ordinal=?",
                (index_id, ordinal),
            ).fetchone()
            if batch is None or json.loads(batch["chunk_ids_json"]) != chunk_ids:
                raise ConflictingWriteError("search batch membership mismatch")
            _, expected_input = self.batch_inputs(connection, index_id, chunk_ids)
            if search_input_sha256 != expected_input:
                raise ConflictingWriteError("search vector batch input differs from generation")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("search batch deadline expired")
            if batch["complete"]:
                if (
                    batch["vectors_sha256"] != digest
                    or batch["search_input_sha256"] != expected_input
                ):
                    raise ConflictingWriteError("search batch already has different vectors")
                return
            for chunk_id, blob in zip(chunk_ids, vectors, strict=True):
                row = connection.execute(
                    "SELECT row_id FROM search_rows WHERE chunk_id=? AND index_version_id=?",
                    (chunk_id, index_id),
                ).fetchone()
                if row is None:
                    raise ConflictingWriteError("search batch chunk missing")
                connection.execute(
                    "INSERT INTO search_vectors(rowid,embedding) VALUES(?,?)", (row[0], blob)
                )
            connection.execute(
                """UPDATE search_batches SET complete=1,vectors_sha256=?,search_input_sha256=?
                    WHERE index_version_id=? AND ordinal=?""",
                (digest, expected_input, index_id, ordinal),
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("search batch deadline expired")

    @staticmethod
    def batch_inputs(
        connection: sqlite3.Connection, index_id: str, chunk_ids: list[str]
    ) -> tuple[list[str], str | None]:
        generation = connection.execute(
            "SELECT settings_json FROM search_generations WHERE index_version_id=?", (index_id,)
        ).fetchone()
        if generation is None:
            raise ConflictingWriteError("search generation is incomplete")
        settings = json.loads(generation[0])
        name = selected(settings)
        if name != CANONICAL and len(chunk_ids) != 1:
            raise ValueError("derived_search_requires_singleton")
        texts = []
        for key in chunk_ids:
            row = connection.execute(
                "SELECT text FROM retrieval_chunks WHERE id=? AND index_version_id=?",
                (key, index_id),
            ).fetchone()
            if row is None:
                raise ConflictingWriteError("search batch input chunk missing")
            texts.append(row[0])
        digest = None if name == CANONICAL else batch_input_hash(chunk_ids, texts, settings)
        return [search_text(text, name) for text in texts], digest

    def finish_components(self, job_id: str, guard: AttemptGuard, index_id: str) -> None:
        from core.persistence.store import _verify_attempt_current

        with self.store._transaction() as connection:
            _verify_attempt_current(connection, job_id, guard.step_key, guard)
            job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            index = connection.execute(
                "SELECT * FROM index_versions WHERE id=?", (index_id,)
            ).fetchone()
            if (
                index is None
                or job is None
                or any(
                    index[key] != job[key]
                    for key in ("source_id", "source_version_id", "extraction_version_id")
                )
            ):
                raise ConflictingWriteError("search component ownership mismatch")
            rows = connection.execute(
                """SELECT r.row_id,r.chunk_id,c.text FROM search_rows r
                   JOIN retrieval_chunks c ON c.id=r.chunk_id
                   WHERE r.index_version_id=? ORDER BY c.ordinal""",
                (index_id,),
            ).fetchall()
            digest = chunk_set_hash([row["chunk_id"] for row in rows])
            settings = json.loads(
                connection.execute(
                    "SELECT settings_json FROM search_generations WHERE index_version_id=?",
                    (index_id,),
                ).fetchone()[0]
            )
            representation = selected(settings)
            for row in rows:
                connection.execute("DELETE FROM search_words WHERE rowid=?", (row["row_id"],))
                connection.execute(
                    "INSERT INTO search_words(rowid,text) VALUES(?,?)",
                    (row["row_id"], search_text(row["text"], representation)),
                )
            for component in ("word", "vector"):
                connection.execute(
                    "INSERT OR REPLACE INTO search_components VALUES(?,?,?)",
                    (index_id, component, digest),
                )
            verify_generation(connection, index_id)

    def search(
        self,
        notebook_id: str,
        source_versions: list[str],
        *,
        query: str | None = None,
        vector: bytes | None = None,
        vector_manifest_id: str | None = None,
        representation: str = CANONICAL,
        vector_representation: str | None = None,
        vector_rule_sha256: str | None = None,
        limit: int = 50,
    ) -> tuple[SearchHit, ...]:
        if not 1 <= limit <= 50:
            raise ValueError("invalid_search_limit")
        if not source_versions:
            return ()
        if (query is None) == (vector is None):
            raise ValueError("supply exactly one search input")
        rules = settings_for(representation)
        if vector is not None:
            if vector_manifest_id is None:
                raise ValueError("query_vector_manifest_required")
            validate_vector(vector)
            if vector_representation != representation and not (
                representation == CANONICAL and vector_representation is None
            ):
                raise ConflictingWriteError("query vector representation differs from search input")
            if vector_rule_sha256 != rules.get("search_rule_sha256"):
                raise ConflictingWriteError("query vector rule differs from current search input")
        placeholders = ",".join("?" for _ in source_versions)
        eligible = f"""SELECT r.row_id FROM search_rows r
            JOIN index_versions i ON i.id=r.index_version_id
            JOIN search_generations g ON g.index_version_id=i.id
            JOIN sources s ON s.id=i.source_id
            JOIN notebook_profiles n ON n.notebook_id=s.notebook_id
            JOIN source_readiness ready ON ready.source_id=s.id
                AND ready.source_version_id=i.source_version_id
            JOIN source_readiness_indexes ri ON ri.source_id=s.id AND ri.index_version_id=i.id
            JOIN extraction_outcomes o ON o.extraction_version_id=i.extraction_version_id
            LEFT JOIN extraction_reviews review
                ON review.extraction_version_id=i.extraction_version_id
            WHERE s.notebook_id=? AND i.source_version_id IN ({placeholders})
              AND g.manifest_id=n.active_manifest_id
              AND COALESCE(json_extract(g.settings_json,'$.search_representation'),
                           'canonical-v1')=?
              AND ((?='canonical-v1'
                  AND json_type(g.settings_json,'$.search_unicode_version') IS NULL
                  AND json_type(g.settings_json,'$.search_rule_sha256') IS NULL) OR (
                 json_extract(g.settings_json,'$.search_unicode_version')=?
                 AND json_extract(g.settings_json,'$.search_rule_sha256')=?))
              AND (o.outcome='complete' OR (o.outcome='partial' AND review.state='confirmed'))
              AND i.extraction_version_id=(SELECT e.id FROM extraction_versions e
                 WHERE e.source_version_id=i.source_version_id
                 ORDER BY e.created_at DESC,e.rowid DESC LIMIT 1)
              AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)
              AND NOT EXISTS(SELECT 1 FROM job_status j WHERE j.job_id=ready.published_by_job_id
                             AND (j.status IN ('failed','cancelled')
                                  OR j.superseded_by_job_id IS NOT NULL))"""
        parameters = (
            notebook_id,
            *source_versions,
            representation,
            representation,
            rules.get("search_unicode_version"),
            rules.get("search_rule_sha256"),
        )
        with self.store._transaction(write=False) as connection:
            if vector is not None:
                active = connection.execute(
                    "SELECT active_manifest_id FROM notebook_profiles WHERE notebook_id=?",
                    (notebook_id,),
                ).fetchone()
                if active is None or active[0] != vector_manifest_id:
                    raise ConflictingWriteError(
                        "query vector model is not the active notebook model"
                    )
            # Both native tables receive the eligible-row restriction before K/limit.
            if query is not None:
                tokens = re.findall(
                    r"[^\W_]+", search_text(query, representation), flags=re.UNICODE
                )
                if not tokens:
                    return ()
                literal = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
                # Unary '+' keeps INTEGER rowid membership as a SQL filter,
                # avoiding one FTS5 MATCH/ranking restart per eligible row.
                # Eligibility still applies before ORDER BY and LIMIT.
                matches = connection.execute(
                    f"""SELECT rowid,bm25(search_words) AS score FROM search_words
                         WHERE search_words MATCH ? AND +rowid IN ({eligible})
                        ORDER BY score,rowid LIMIT ?""",
                    (literal, *parameters, limit),
                ).fetchall()
            else:
                matches = connection.execute(
                    f"""WITH nearest AS MATERIALIZED (
                        SELECT rowid,distance FROM search_vectors
                        WHERE embedding MATCH ? AND k=? AND rowid IN ({eligible})
                        ) SELECT rowid,distance AS score FROM nearest ORDER BY distance,rowid""",
                    (vector, limit, *parameters),
                ).fetchall()
            hits = []
            for match in matches:
                row = connection.execute(
                    """SELECT c.*,g.manifest_id FROM search_rows r
                       JOIN retrieval_chunks c ON c.id=r.chunk_id
                       JOIN search_generations g ON g.index_version_id=r.index_version_id
                       WHERE r.row_id=?""",
                    (match["rowid"],),
                ).fetchone()
                links = connection.execute(
                    "SELECT * FROM chunk_passages WHERE retrieval_chunk_id=? ORDER BY ordinal",
                    (row["id"],),
                ).fetchall()
                hits.append(
                    SearchHit(
                        row["source_id"],
                        row["source_version_id"],
                        row["index_version_id"],
                        row["id"],
                        row["manifest_id"],
                        row["text"],
                        match["score"],
                        tuple(ChunkPassage(**dict(link)) for link in links),
                    )
                )
            return tuple(hits)


def purge_search(
    connection: sqlite3.Connection, *, source_id: str, version_id: str | None = None
) -> None:
    clause = "source_id=?" + (" AND source_version_id=?" if version_id else "")
    parameters = (source_id, version_id) if version_id else (source_id,)
    connection.execute(
        f"DELETE FROM search_job_inputs WHERE job_id IN (SELECT id FROM jobs WHERE {clause})",
        parameters,
    )
    index_ids = [
        row[0]
        for row in connection.execute(f"SELECT id FROM index_versions WHERE {clause}", parameters)
    ]
    for index_id in index_ids:
        for row in connection.execute(
            "SELECT row_id FROM search_rows WHERE index_version_id=?", (index_id,)
        ).fetchall():
            connection.execute("DELETE FROM search_words WHERE rowid=?", (row[0],))
            connection.execute("DELETE FROM search_vectors WHERE rowid=?", (row[0],))
            connection.execute("DELETE FROM search_text_inputs WHERE row_id=?", (row[0],))
        for table in (
            "profile_search_publications",
            "search_components",
            "search_batches",
            "search_rows",
            "search_generations",
        ):
            connection.execute(f"DELETE FROM {table} WHERE index_version_id=?", (index_id,))
    connection.execute(
        "DELETE FROM search_handoffs WHERE extraction_version_id IN "
        f"(SELECT id FROM extraction_versions WHERE {clause})",
        parameters,
    )
