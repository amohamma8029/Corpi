"""M1.5 extraction records: page quality, visual metadata, outcome, and the review gate."""

import sqlite3

import pytest

from core.persistence.database import Database
from core.persistence.errors import ConflictingWriteError, PersistenceError
from core.persistence.models import (
    AttemptGuard,
    ExtractionOutcome,
    ExtractionPage,
    ExtractionReview,
    ExtractionVersion,
    IndexVersion,
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


def make_index(extraction: ExtractionVersion) -> IndexVersion:
    return IndexVersion(
        new_id(),
        extraction.source_id,
        extraction.source_version_id,
        extraction.id,
        "keyword-v1",
        "chunker-1",
        1,
        {"k": 60},
        None,
        None,
        utc_timestamp(),
    )


def commit_index(store: PersistenceStore, job_id: str, index: IndexVersion) -> AttemptGuard:
    if store.load_job_status(job_id) is None:
        store.create_job_status(job_id)
    store.claim_job(job_id, "guard-tests")
    attempt = store.start_step_attempt(job_id, "build-index")
    guard = AttemptGuard("guard-tests", "build-index", attempt.attempt_number)
    store.commit_index_step(job_id, "build-index", index, (), (), 1, {"index": 1}, guard=guard)
    return guard


def attach_import(
    store: PersistenceStore, source_id: str, version_id: str, state: str = "ready"
) -> str:
    """Register the import row and source state a real processing flow would leave behind."""
    import_id = new_id()
    now = utc_timestamp()
    with store.database.connect() as connection:
        connection.execute(
            """INSERT INTO pdf_imports(id, source_id, source_version_id, acquisition_id,
                   display_filename, state, created_at, updated_at)
               VALUES(?,?,?,NULL,?,?,?,?)""",
            (import_id, source_id, version_id, "import.pdf", state, now, now),
        )
        connection.execute(
            """INSERT INTO source_pdf_state(
                   source_id,current_import_id,pending_import_id,revision,updated_at
               ) VALUES(?,?,NULL,1,?)
               ON CONFLICT(source_id) DO UPDATE SET
                   current_import_id=excluded.current_import_id,
                   revision=source_pdf_state.revision+1,
                   updated_at=excluded.updated_at""",
            (source_id, import_id, now),
        )
    return import_id


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


def test_blank_page_may_carry_decorative_image(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    sparse = make_page(
        extraction,
        text_drawn=False,
        extracted_chars=0,
        image_area_ratio=0.1,
        classification="blank",
    )
    report = store.save_extraction_result(
        extraction,
        (),
        (sparse,),
        (),
        make_outcome(extraction, "complete"),
    )
    assert report.pages[0].classification == "blank"


def test_publication_rejects_an_unconfirmed_indexed_extraction(store: PersistenceStore) -> None:
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
    job = store.create_job(
        "guard:publish",
        "index-build",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=extraction.id,
    )
    index = make_index(extraction)
    guard = commit_index(store, job.id, index)

    with pytest.raises(ConflictingWriteError, match="extraction decision"):
        store.publish_readiness(job.id, source_id, source_version.id, [index.id], guard=guard)

    store.record_review_decision(extraction.id, "confirmed", 1)
    ready = store.publish_readiness(job.id, source_id, source_version.id, [index.id], guard=guard)
    assert ready.published_by_job_id == job.id


def test_publication_rejects_an_extraction_without_an_outcome(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction(extraction, (make_passage(extraction),))
    job = store.create_job(
        "guard:missing",
        "index-build",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=extraction.id,
    )
    index = make_index(extraction)
    guard = commit_index(store, job.id, index)
    with pytest.raises(ConflictingWriteError, match="extraction decision is missing"):
        store.publish_readiness(job.id, source_id, source_version.id, [index.id], guard=guard)


def test_publication_keys_to_the_indexed_extraction_not_an_older_complete_one(
    store: PersistenceStore,
) -> None:
    source_id, source_version = seed_source(store)
    older = make_extraction(source_id, source_version)
    store.save_extraction_result(
        older,
        (make_passage(older),),
        (make_page(older),),
        (),
        make_outcome(older, "complete"),
    )
    newer = make_extraction(source_id, source_version)
    store.save_extraction_result(
        newer,
        (make_passage(newer),),
        (
            make_page(
                newer,
                text_drawn=True,
                extracted_chars=1,
                image_area_ratio=0.9,
                classification="image-only",
            ),
        ),
        (),
        make_outcome(newer, "partial"),
        make_review(newer),
    )
    job = store.create_job(
        "guard:keyed",
        "index-build",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=newer.id,
    )
    index = make_index(newer)
    guard = commit_index(store, job.id, index)

    with pytest.raises(ConflictingWriteError, match="extraction decision"):
        store.publish_readiness(job.id, source_id, source_version.id, [index.id], guard=guard)


def test_publication_allows_a_complete_indexed_extraction(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (),
        make_outcome(extraction, "complete"),
    )
    job = store.create_job(
        "guard:complete",
        "index-build",
        source_id,
        source_version_id=source_version.id,
        extraction_version_id=extraction.id,
    )
    index = make_index(extraction)
    guard = commit_index(store, job.id, index)
    ready = store.publish_readiness(job.id, source_id, source_version.id, [index.id], guard=guard)
    assert ready.published_by_job_id == job.id


def test_reuse_preserves_decided_review_states(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    passage = make_passage(extraction)
    page = make_page(extraction)
    outcome = make_outcome(extraction, "partial")
    review = make_review(extraction)
    store.save_extraction_result(extraction, (passage,), (page,), (), outcome, review)

    confirmed = store.record_review_decision(extraction.id, "confirmed", 1)
    again = store.save_extraction_result(extraction, (passage,), (page,), (), outcome, confirmed)
    assert again.review is not None and again.review.state == "confirmed"

    reopened = store.reopen_review(extraction.id)
    assert (reopened.state, reopened.generation) == ("pending", 2)
    after_reopen = store.save_extraction_result(
        extraction, (passage,), (page,), (), outcome, reopened
    )
    assert after_reopen.review is not None and after_reopen.review.generation == 2

    declined = store.record_review_decision(extraction.id, "declined", 2)
    after_decline = store.save_extraction_result(
        extraction, (passage,), (page,), (), outcome, declined
    )
    assert after_decline.review is not None and after_decline.review.state == "declined"


def test_confirm_rejects_a_cancelled_candidate(store: PersistenceStore) -> None:
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
    job = store.create_job(
        "guard:cancelled", "pdf-extract", source_id, source_version_id=source_version.id
    )
    store.create_job_status(job.id)
    store.finish_job(job.id, "cancelled", "cancelled by the user")
    with pytest.raises(ConflictingWriteError, match="cancelled"):
        store.record_review_decision(extraction.id, "confirmed", 1)
    assert store.record_review_decision(extraction.id, "declined", 1).state == "declined"


def test_confirm_rejects_a_failed_candidate_import(store: PersistenceStore) -> None:
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
    attach_import(store, source_id, source_version.id, state="failed")
    with pytest.raises(ConflictingWriteError, match="candidate import failed"):
        store.record_review_decision(extraction.id, "confirmed", 1)


def test_purge_removes_extraction_records_for_an_expired_source(store: PersistenceStore) -> None:
    source_id, source_version = seed_source(store)
    extraction = make_extraction(source_id, source_version)
    store.save_extraction_result(
        extraction,
        (make_passage(extraction),),
        (make_page(extraction),),
        (make_visual(extraction),),
        make_outcome(extraction, "partial"),
        make_review(extraction),
    )
    store.trash_source(source_id, utc_timestamp())
    store.remove_trashed_records(source_id)
    with pytest.raises(PersistenceError, match="does not exist"):
        store.load_extraction_report(extraction.id)


def test_purge_removes_extraction_records_for_an_older_import(store: PersistenceStore) -> None:
    source_id, old_version = seed_source(store)
    older = make_extraction(source_id, old_version)
    store.save_extraction_result(
        older,
        (make_passage(older),),
        (make_page(older),),
        (make_visual(older),),
        make_outcome(older, "partial"),
        make_review(older),
    )
    old_import_id = attach_import(store, source_id, old_version.id)

    managed = ManagedFile(new_id(), "originals/newer.pdf", 17, HASH_B, utc_timestamp())
    store.save_managed_file(managed)
    new_version = store.create_source_version(source_id, HASH_B, managed.id)
    attach_import(store, source_id, new_version.id)

    store.trash_pdf_import(old_import_id, utc_timestamp())
    with store.database.connect() as connection:
        connection.execute(
            """UPDATE pdf_import_trash SET file_removed_at=?, snapshots_removed_at=?
               WHERE import_id=?""",
            (utc_timestamp(), utc_timestamp(), old_import_id),
        )
    store.remove_trashed_pdf_import_records(old_import_id)
    with pytest.raises(PersistenceError, match="does not exist"):
        store.load_extraction_report(older.id)


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
