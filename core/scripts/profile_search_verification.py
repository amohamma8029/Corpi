"""Read-only verification query diagnosis on an owned capacity database."""

import argparse
import json
import time
from pathlib import Path

from core.persistence.database import Database, DatabaseConfig
from core.search.storage import verify_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    database = Database(DatabaseConfig(args.directory))
    with database.connect() as connection:
        generation = connection.execute(
            "SELECT index_version_id FROM search_generations WHERE expected_chunks=9950"
        ).fetchone()
        if generation is None:
            raise ValueError("not the retained near-cap corpus")
        index = generation[0]
        rows = connection.execute(
            "SELECT row_id,chunk_id FROM search_rows WHERE index_version_id=? ORDER BY row_id",
            (index,),
        ).fetchall()
        queries = {
            "joined_batch_vector": (
                "SELECT v.embedding FROM search_rows r JOIN search_vectors v ON v.rowid=r.row_id "
                "WHERE r.chunk_id=? AND r.index_version_id=?"
            ),
            "direct_vector": "SELECT embedding FROM search_vectors WHERE rowid=?",
        }
        result = {"chunks": len(rows), "probes": {}}
        references = []
        for name, sql in queries.items():
            params = (rows[0][1], index) if name == "joined_batch_vector" else (rows[0][0],)
            plan = connection.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
            started = time.perf_counter()
            vectors = []
            for row in rows[:64]:
                params = (row[1], index) if name == "joined_batch_vector" else (row[0],)
                vectors.append(connection.execute(sql, params).fetchone()[0])
            elapsed = time.perf_counter() - started
            if not references:
                references = vectors
            assert vectors == references
            result["probes"][name] = {
                "lookups": 64,
                "seconds": elapsed,
                "plan": [r[3] for r in plan],
                "exact_bytes_matched": True,
            }
        sql = (
            "SELECT rowid,embedding FROM search_vectors WHERE rowid IN "
            "(SELECT row_id FROM search_rows WHERE index_version_id=?)"
        )
        started = time.perf_counter()
        bulk = dict(connection.execute(sql, (index,)).fetchall())
        elapsed = time.perf_counter() - started
        assert len(bulk) == len(rows)
        assert [bulk[row[0]] for row in rows[:64]] == references
        result["probes"]["bulk_vectors"] = {
            "rows": len(bulk),
            "seconds": elapsed,
            "exact_bytes_matched": True,
        }
        prefix = (
            "SELECT m.*,p.text,q.classification FROM chunk_passages m "
            "JOIN passages p ON p.id=m.passage_id LEFT JOIN extraction_pages q "
            "ON q.extraction_version_id=p.extraction_version_id "
        )
        mapping_queries = {
            "page-original": prefix + "AND q.page_number=json_extract(p.location_json,'$.page') "
            "WHERE m.retrieval_chunk_id=? ORDER BY m.ordinal",
            "page-owner-bound": prefix
            + "AND q.source_id=p.source_id AND q.source_version_id=p.source_version_id "
            "AND q.page_number=json_extract(p.location_json,'$.page') "
            "WHERE m.retrieval_chunk_id=? ORDER BY m.ordinal",
        }
        reference_links = None
        for name, sql in mapping_queries.items():
            plan = connection.execute("EXPLAIN QUERY PLAN " + sql, (rows[0][1],)).fetchall()
            started = time.perf_counter()
            links = [connection.execute(sql, (r[1],)).fetchall() for r in rows[:64]]
            elapsed = time.perf_counter() - started
            if reference_links is None:
                reference_links = links
            assert links == reference_links
            result["probes"][name] = {
                "lookups": 64,
                "seconds": elapsed,
                "plan": [r[3] for r in plan],
                "exact_links_matched": True,
            }
        started = time.perf_counter()
        verify_generation(connection, index)
        result["full_generation_proof_seconds"] = time.perf_counter() - started
    (args.directory / "verification-query-profile.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
