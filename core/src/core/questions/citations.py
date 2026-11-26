"""Find a saved PDF passage independently of current search readiness."""

import json
import sqlite3
from dataclasses import dataclass

from core.persistence.models import JSONValue


class SavedPassageUnavailable(LookupError):
    def __init__(self) -> None:
        super().__init__("The saved PDF passage is unavailable.")


@dataclass(frozen=True, slots=True)
class SavedPassageRef:
    source_id: str
    source_version_id: str
    passage_id: str
    text_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class SavedPdfPassage:
    reference: SavedPassageRef
    import_id: str
    extraction_version_id: str
    page_number: int
    quote: str
    location_format_version: int
    location: dict[str, JSONValue]


def lookup_saved_pdf_passage(
    connection: sqlite3.Connection,
    notebook_id: str,
    reference: SavedPassageRef,
) -> SavedPdfPassage:
    """Resolve a canonical reference from the caller's notebook read snapshot.

    The caller must first authorize the saved answer or search result that
    supplied the reference. This checks only passage and original availability;
    it neither verifies PDF bytes nor guesses whether a box can be highlighted.
    """
    row = connection.execute(
        """SELECT p.extraction_version_id, p.text, p.location_format_version,
                  p.location_json, pdf.id AS import_id
           FROM passages p
           JOIN sources s ON s.id=p.source_id
           JOIN source_versions v ON v.id=p.source_version_id AND v.source_id=s.id
           JOIN pdf_imports pdf ON pdf.source_version_id=v.id AND pdf.source_id=s.id
           JOIN managed_files f ON f.id=v.managed_file_id
           WHERE s.notebook_id=? AND p.source_id=? AND p.source_version_id=? AND p.id=?
             AND pdf.state='ready'
             AND NOT EXISTS(SELECT 1 FROM source_trash t WHERE t.source_id=s.id)
             AND NOT EXISTS(SELECT 1 FROM pdf_import_trash t WHERE t.import_id=pdf.id)
             AND NOT EXISTS(SELECT 1 FROM managed_file_removals r WHERE r.managed_file_id=f.id)""",
        (notebook_id, reference.source_id, reference.source_version_id, reference.passage_id),
    ).fetchone()
    if row is None:
        raise SavedPassageUnavailable()

    location = json.loads(row["location_json"])
    page = location.get("page") if isinstance(location, dict) else None
    if type(page) is not int or page < 1:
        raise SavedPassageUnavailable()

    quote = row["text"]
    if reference.text_range is not None:
        start, end = reference.text_range
        if not (0 <= start < end <= len(quote)):
            raise SavedPassageUnavailable()
        quote = quote[start:end]
    return SavedPdfPassage(
        reference=reference,
        import_id=row["import_id"],
        extraction_version_id=row["extraction_version_id"],
        page_number=page,
        quote=quote,
        location_format_version=row["location_format_version"],
        location=location,
    )
