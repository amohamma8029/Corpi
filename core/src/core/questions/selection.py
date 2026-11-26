"""Resolve the exact ready PDFs a question may record at submission."""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

SelectionFailure = Literal[
    "no_sources_selected",
    "no_ready_sources",
    "duplicate_source",
    "selected_source_unavailable",
    "selected_version_changed",
]


class QuestionSelectionError(ValueError):
    def __init__(self, code: SelectionFailure, source_id: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.source_id = source_id


@dataclass(frozen=True, slots=True)
class SelectedPdf:
    source_id: str
    source_version_id: str


@dataclass(frozen=True, slots=True)
class ReadyQuestionPdf:
    source_id: str
    source_version_id: str
    import_id: str
    extraction_version_id: str
    review_generation: int | None
    manifest_id: str
    index_version_ids: tuple[str, ...]


# Match M1 search eligibility, with the additional requirement that the ready
# version is the current usable PDF. Pending imports and pending model choices
# have separate pointers and must not replace any of these identities.
_READY_PDFS = """
    SELECT s.id AS source_id, ready.source_version_id, pdf.id AS import_id,
           i.extraction_version_id, r.generation AS review_generation,
           g.manifest_id, i.id AS index_version_id
    FROM sources s
    JOIN source_pdf_state p ON p.source_id=s.id
    JOIN pdf_imports pdf ON pdf.id=p.current_import_id AND pdf.source_id=s.id
    JOIN source_readiness ready ON ready.source_id=s.id
        AND ready.source_version_id=pdf.source_version_id
    JOIN source_readiness_indexes ri ON ri.source_id=s.id
    JOIN index_versions i ON i.id=ri.index_version_id
        AND i.source_version_id=ready.source_version_id
    JOIN search_generations g ON g.index_version_id=i.id
    JOIN notebook_profiles n ON n.notebook_id=s.notebook_id AND n.active_manifest_id=g.manifest_id
    JOIN extraction_outcomes o ON o.extraction_version_id=i.extraction_version_id
    LEFT JOIN extraction_reviews r ON r.extraction_version_id=i.extraction_version_id
    WHERE s.notebook_id=? AND pdf.state='ready'
      AND (o.outcome='complete' OR (o.outcome='partial' AND r.state='confirmed'))
      AND i.extraction_version_id=(SELECT e.id FROM extraction_versions e
          WHERE e.source_version_id=i.source_version_id
          ORDER BY e.created_at DESC,e.rowid DESC LIMIT 1)
      AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)
      AND NOT EXISTS(SELECT 1 FROM pdf_import_trash t WHERE t.import_id=pdf.id)
      AND NOT EXISTS(SELECT 1 FROM job_status j WHERE j.job_id=ready.published_by_job_id
          AND (j.status IN ('failed','cancelled') OR j.superseded_by_job_id IS NOT NULL))
"""


def resolve_question_selection(
    connection: sqlite3.Connection,
    notebook_id: str,
    selected: Sequence[SelectedPdf],
) -> tuple[ReadyQuestionPdf, ...]:
    """Resolve all selections or reject the question without substituting a PDF.

    Use the caller's Corpi database transaction so resolution and acceptance can
    commit together. The order matches the caller's selection, not query order.
    """
    source_ids: set[str] = set()
    for pdf in selected:
        if pdf.source_id in source_ids:
            raise QuestionSelectionError("duplicate_source", pdf.source_id)
        source_ids.add(pdf.source_id)

    by_source: dict[str, list[sqlite3.Row]] = {}
    if source_ids:
        rows = connection.execute(
            _READY_PDFS + " AND s.id IN (SELECT value FROM json_each(?)) ORDER BY i.id",
            (notebook_id, json.dumps(list(source_ids))),
        )
        for row in rows:
            by_source.setdefault(row["source_id"], []).append(row)

    if not by_source:
        has_ready_pdf = connection.execute(
            f"SELECT EXISTS({_READY_PDFS})", (notebook_id,)
        ).fetchone()[0]
        if not has_ready_pdf:
            raise QuestionSelectionError("no_ready_sources")
        if not selected:
            raise QuestionSelectionError("no_sources_selected")

    result: list[ReadyQuestionPdf] = []
    for pdf in selected:
        matches = by_source.get(pdf.source_id)
        if not matches:
            raise QuestionSelectionError("selected_source_unavailable", pdf.source_id)
        current = matches[0]
        if current["source_version_id"] != pdf.source_version_id:
            raise QuestionSelectionError("selected_version_changed", pdf.source_id)
        result.append(
            ReadyQuestionPdf(
                source_id=pdf.source_id,
                source_version_id=pdf.source_version_id,
                import_id=current["import_id"],
                extraction_version_id=current["extraction_version_id"],
                review_generation=current["review_generation"],
                manifest_id=current["manifest_id"],
                index_version_ids=tuple(row["index_version_id"] for row in matches),
            )
        )
    return tuple(result)
