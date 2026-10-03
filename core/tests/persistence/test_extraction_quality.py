"""M1.5 extraction records: page quality, visual metadata, outcome, and the review gate."""

import sqlite3

import pytest

from core.persistence.database import Database
from core.persistence.errors import ConflictingWriteError, PersistenceError
from core.persistence.models import (
    ExtractionOutcome,
    ExtractionPage,
    ExtractionReview,
    ExtractionVersion,
    JSONValue,
    ManagedFile,
    Passage,
    SourceVersion,
    VisualLocation,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore

HASH_A = "a" * 64
HASH_B = "b" * 64
SETTINGS: JSONValue = {"rules_version": "corpi-rules-1", "min_chars": 24}
SETTINGS_V2: JSONValue = {"rules_version": "corpi-rules-1", "min_chars": 30}


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
    source_id: str, source_version: SourceVersion, settings: JSONValue = SETTINGS
) -> ExtractionVersion:
    return ExtractionVersion(
        new_id(),
        source_id,
        source_version.id,
        "pdfminer.six",
        "20260107",
        "faithfully-normalized-text-v1",
        1,
        settings,
        utc_timestamp(),
    )


def make_passage(
    extraction: ExtractionVersion, *, ordinal: int = 0, text: str = "Hello world"
) -> Passage:
    return Passage(
        new_id(),
        extraction.source_id,
        extraction.source_version_id,
        extraction.id,
        ordinal,
        text,
        1,
        {"page": 1, "block": "paragraph", "bbox": [0, 0, 10, 10], "reading_order": ordinal},
        utc_timestamp(),
    )


def make_page(
    extraction: ExtractionVersion,
    *,
    page_number: int = 1,
    text_drawn: bool = True,
    extracted_chars: int = 100,
    unmapped_ratio: float = 0.0,
    image_area_ratio: float = 0.0,
    classification: str = "extracted",
) -> ExtractionPage:
    return ExtractionPage(
        new_id(),
        extraction.source_id,
        extraction.source_version_id,
        extraction.id,
        page_number,
        text_drawn,
        extracted_chars,
        unmapped_ratio,
        image_area_ratio,
        classification,
        utc_timestamp(),
    )


def make_visual(
    extraction: ExtractionVersion, *, page_number: int = 1, ordinal: int = 0
) -> VisualLocation:
    return VisualLocation(
        new_id(),
        extraction.source_id,
        extraction.source_version_id,
        extraction.id,
        page_number,
        ordinal,
        "image",
        [0.0, 0.0, 100.0, 100.0],
        utc_timestamp(),
    )


def make_outcome(extraction: ExtractionVersion, outcome: str) -> ExtractionOutcome:
    return ExtractionOutcome(
        extraction.id,
        extraction.source_id,
        extraction.source_version_id,
        outcome,
        utc_timestamp(),
    )


def make_review(extraction: ExtractionVersion) -> ExtractionReview:
    return ExtractionReview(
        extraction.id,
        extraction.source_id,
        extraction.source_version_id,
        "pending",
        1,
        None,
        utc_timestamp(),
    )


def test_result_commits_pages_visuals_outcome_and_review_together(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    passage = make_passage(extraction)
    scanned_page = make_page(
        extraction,
        page_number=2,
        text_drawn=True,
        extracted_chars=2,
        image_area_ratio=0.9,
        classification="image-only",
    )
    visual = make_visual(extraction, page_number=2)
    report = store.save_extraction_result(
        extraction,
        (passage,),
        (make_page(extraction), scanned_page),
        (visual,),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    assert report.extraction == extraction
    assert report.passages == (passage,)
    assert [page.classification for page in report.pages] == ["extracted", "image-only"]
    assert report.visuals == (visual,)
    assert report.outcome is not None and report.outcome.outcome == "partial"
    assert report.review is not None
    assert (report.review.state, report.review.generation, report.review.decided_at) == (
        "pending",
        1,
        None,
    )
    reloaded = store.load_extraction_report(extraction.id)
    assert reloaded == report


def test_invalid_page_rejects_the_whole_result(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    impossible = make_page(extraction, text_drawn=False, extracted_chars=50)
    with pytest.raises(ValueError, match="drawn text"):
        store.save_extraction_result(
            extraction,
            (make_passage(extraction),),
            (impossible,),
            (),
            make_outcome(extraction, "complete"),
        )
    with pytest.raises(PersistenceError, match="does not exist"):
        store.load_extraction_report(extraction.id)


def test_page_and_visual_records_are_immutable(store: PersistenceStore, database: Database) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    page = make_page(extraction, classification="suspect", extracted_chars=5)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (page,),
        (make_visual(extraction),),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    with database.connect() as connection:
        with pytest.raises(sqlite3.Error, match="immutable"):
            connection.execute(
                "UPDATE extraction_pages SET classification='extracted' WHERE id=?",
                (page.id,),
            )
        with pytest.raises(sqlite3.Error, match="immutable"):
            connection.execute(
                "UPDATE extraction_visuals SET kind='figure' WHERE extraction_version_id=?",
                (extraction.id,),
            )
        with pytest.raises(sqlite3.Error, match="immutable"):
            connection.execute(
                "UPDATE extraction_outcomes SET outcome='complete' WHERE extraction_version_id=?",
                (extraction.id,),
            )


def test_passage_nonempty_guarantee_still_enforced(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    with pytest.raises(ValueError):
        store.save_extraction_result(
            extraction,
            (make_passage(extraction, text=""),),
            (make_page(extraction),),
            (),
            make_outcome(extraction, "complete"),
        )


def test_outcome_and_review_must_agree(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    with pytest.raises(ValueError, match="review state"):
        store.save_extraction_result(
            extraction,
            (make_passage(extraction),),
            (make_page(extraction),),
            (),
            make_outcome(extraction, "complete"),
            make_review(extraction),
        )
    with pytest.raises(ValueError, match="review state"):
        store.save_extraction_result(
            extraction,
            (make_passage(extraction),),
            (make_page(extraction),),
            (),
            make_outcome(extraction, "partial"),
        )
    report = store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "complete"),
    )
    assert report.outcome is not None and report.outcome.outcome == "complete"
    assert report.review is None


def test_confirm_is_idempotent(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    first = store.record_review_decision(extraction.id, "confirmed", 1)
    again = store.record_review_decision(extraction.id, "confirmed", 1)
    assert first == again
    assert first.state == "confirmed"
    assert first.decided_at is not None


def test_stale_or_unknown_generation_is_rejected(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    reopened = store.reopen_review(extraction.id)
    assert (reopened.state, reopened.generation) == ("pending", 2)
    with pytest.raises(ConflictingWriteError, match="stale"):
        store.record_review_decision(extraction.id, "confirmed", 1)
    with pytest.raises(ValueError, match="generation"):
        store.record_review_decision(extraction.id, "confirmed", 3)
    decided = store.record_review_decision(extraction.id, "confirmed", 2)
    assert decided.state == "confirmed"


def test_conflicting_decision_at_same_generation_is_rejected(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    store.record_review_decision(extraction.id, "declined", 1)
    with pytest.raises(ConflictingWriteError, match="decided differently"):
        store.record_review_decision(extraction.id, "confirmed", 1)
    reopened = store.reopen_review(extraction.id)
    assert reopened.state == "pending"
    assert store.record_review_decision(extraction.id, "confirmed", 2).state == "confirmed"


def test_confirm_rejects_a_superseded_source_version(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    managed_file = ManagedFile(new_id(), "originals/newer.pdf", 17, HASH_B, utc_timestamp())
    store.save_managed_file(managed_file)
    store.create_source_version(source_id, HASH_B, managed_file.id)
    with pytest.raises(ConflictingWriteError, match="not current"):
        store.record_review_decision(extraction.id, "confirmed", 1)
    assert store.record_review_decision(extraction.id, "declined", 1).state == "declined"


def test_confirm_rejects_a_trashed_source(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    store.trash_source(source_id, utc_timestamp())
    with pytest.raises(ConflictingWriteError, match="trashed"):
        store.record_review_decision(extraction.id, "confirmed", 1)


def test_reuse_lookup_matches_full_identity_only(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "complete"),
    )
    match = store.lookup_reusable_extraction(
        source_id,
        source_version.id,
        "pdfminer.six",
        "20260107",
        "faithfully-normalized-text-v1",
        1,
        SETTINGS,
    )
    assert match is not None and match.extraction.id == extraction.id
    assert (
        store.lookup_reusable_extraction(
            source_id,
            source_version.id,
            "pdfminer.six",
            "20260107",
            "faithfully-normalized-text-v1",
            1,
            SETTINGS_V2,
        )
        is None
    )
    other_source, _ = seed_source(store, name="shared-bytes.pdf")
    assert (
        store.lookup_reusable_extraction(
            other_source,
            source_version.id,
            "pdfminer.six",
            "20260107",
            "faithfully-normalized-text-v1",
            1,
            SETTINGS,
        )
        is None
    )


def test_conflicting_retry_rejected_identical_retry_reuses(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    passage = make_passage(extraction)
    page = make_page(extraction)
    outcome = make_outcome(extraction, "complete")
    first = store.save_extraction_result(extraction, (passage,), (page,), (), outcome)
    again = store.save_extraction_result(extraction, (passage,), (page,), (), outcome)
    assert again == first
    with pytest.raises(ConflictingWriteError, match="page set"):
        store.save_extraction_result(
            extraction,
            (passage,),
            (make_page(extraction, page_number=2),),
            (),
            outcome,
        )
