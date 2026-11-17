"""Disk/process recovery must traverse the real runner's coordinator claim gate.

On 4139df1 the ordinary crash leaves the accepted publisher running; restarting
cancels it because activation already cleared the pending choice. The legacy
variant explicitly retains that older split state to discriminate the claim fix.
No worker is allowed to tokenize/embed again during either recovery.
"""

import json
import subprocess
import sys
from hashlib import sha256

import pytest
from pdf_fixtures import native_text_pdf
from test_publication_finalization import (
    assert_saved_search,
    generation_charge,
    prepare_publication,
    protected_output,
    run_pending,
)
from test_publication_finalization import publication_runtime as publication_runtime
from test_replacement_safety import original as original
from test_replacement_safety import upload
from test_search_expiry import reopen

from core.jobs.runner import StepContext, _canonical_json
from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard
from core.search.handoff import DEFAULT_PROFILE_ID, SearchCoordinator
from core.search.profiles import installation_path, profiles
from core.search.supervisor import stop_owned_process_tree, worker_environment
from core.search.switching import NotebookModelService, publish_replacement

CRASH_AFTER_ACTIVATION = """
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from core.jobs.runner import JobRunner
from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore
from core.search import preparation, switching
from core.search.handoff import SearchCoordinator

database = Database(DatabaseConfig(Path(sys.argv[1])))
database.initialize()
store = PersistenceStore(database)
job = sys.argv[2]
if sys.argv[3] == 'legacy':
    # Model an existing pre-fix checkpoint, not migration/reset of failed work.
    # Setting this unused name on 4139df1 is harmless; its old publication already
    # has the split state. The baseline fails on job/eligibility, not missing API.
    switching._complete_search_publication = lambda *args: None
def forbidden(*args, **kwargs):
    raise AssertionError('committed inputs must not be tokenized or embedded again')
preparation.InputTokenizer = forbidden
preparation.EmbeddingSession = forbidden
transaction = store._transaction
@contextmanager
def crash_after_commit(write=True):
    committed = False
    with transaction(write=write) as connection:
        yield connection
        if write:
            committed = connection.execute(
                "SELECT 1 FROM job_steps WHERE job_id=? AND step_key='publish-search'",
                (job,),
            ).fetchone() is not None
    if committed:
        print(json.dumps({
            'committed': True, 'status': store.load_job_status(job).status,
        }), flush=True)
        os._exit(73)
store._transaction = crash_after_commit
async def exercise():
    models = database.config.data_dir / 'models'
    coordinator = SearchCoordinator(store, models)
    runner = JobRunner(
        store, {'pdf-search': preparation.build_search_steps(store, models)},
        reconcile=coordinator.reconcile, claim_blocking_reason=coordinator.claim_blocking_reason,
    )
    await runner.start()
    try:
        await asyncio.sleep(15)
        raise AssertionError('post-activation COMMIT seam was not reached')
    finally:
        await runner.stop()
asyncio.run(exercise())
"""


@pytest.mark.parametrize("legacy", [False, True])
def test_activation_commit_process_exit_recovers_exact_output_through_runner_claim(
    store, original, publication_runtime, monkeypatch, legacy
):
    case = prepare_publication(store, original, "model-switch")
    before = protected_output(store, case)
    documents = list(publication_runtime.document_calls)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CRASH_AFTER_ACTIVATION,
            str(store.database.config.data_dir),
            case.job,
            "legacy" if legacy else "atomic",
        ],
        env=worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=25)
    finally:
        stop_owned_process_tree(process)
    assert process.returncode == 73, (stdout, stderr)
    assert stderr == b""
    assert json.loads(stdout) == {
        "committed": True,
        "status": "running" if legacy else "completed",
    }
    restarted = reopen(store)
    active = NotebookModelService(restarted).state(case.notebook)
    assert active["pending_manifest_id"] is None
    assert active["active_manifest_id"] == restarted.load_index(case.index).index.embedding_revision
    before_charge = generation_charge(restarted, case)
    before_attempts = restarted.load_step_attempts(case.job, "publish-search")
    assert len(before_attempts) == 1
    assert_saved_search(restarted, case, publication_runtime)

    def forbidden(*args, **kwargs):
        pytest.fail("recovery started tokenization or document inference")

    monkeypatch.setattr("core.search.preparation.InputTokenizer", forbidden)
    monkeypatch.setattr("core.search.preparation.EmbeddingSession", forbidden)
    status = run_pending(restarted, case)
    assert status.status == "completed", (
        "the claim gate cancelled an already-committed current activation",
        status,
    )
    after_attempts = restarted.load_step_attempts(case.job, "publish-search")
    assert len(after_attempts) == 1 and after_attempts[0].outcome == "ok"
    assert after_attempts[0].finished_at is not None
    assert (
        after_attempts[0].job_id,
        after_attempts[0].step_key,
        after_attempts[0].attempt_number,
        after_attempts[0].started_at,
    ) == (
        before_attempts[0].job_id,
        before_attempts[0].step_key,
        before_attempts[0].attempt_number,
        before_attempts[0].started_at,
    )
    if not legacy:
        assert after_attempts == before_attempts
    assert generation_charge(restarted, case) == before_charge
    assert protected_output(restarted, case) == before
    assert publication_runtime.document_calls == documents
    assert_saved_search(restarted, case, publication_runtime)


def legacy_activation(store, case):
    """Create the real pre-fix split records using existing store interfaces."""
    store.claim_job(case.job, "legacy-activation")
    attempt = store.start_step_attempt(case.job, "publish-search")
    guard = AttemptGuard("legacy-activation", "publish-search", attempt.attempt_number)
    context = StepContext(
        case.job, "publish-search", attempt.attempt_number, lambda: False, lambda: False, guard
    )
    final = case.steps[-1]
    assert final.inputs is not None
    fingerprint = sha256(_canonical_json(final.inputs(context)).encode()).hexdigest()
    outcome = final.run(context)
    publish_replacement(store, case.job, guard, case.index)
    store.commit_step(
        case.job,
        "publish-search",
        outcome.checkpoint_format_version,
        outcome.checkpoint,
        guard=guard,
        inputs_sha256=fingerprint,
    )
    store.release_running_jobs()


def activation_receipts(store, case):
    """Accepted output and selection history, excluding mutable waiting text/owner."""
    with store.database.connect() as connection:
        queries = {
            "profile": (
                "SELECT * FROM notebook_profiles WHERE notebook_id=?",
                (case.notebook,),
            ),
            "publication": (
                "SELECT * FROM profile_search_publications WHERE published_by_job_id=?",
                (case.job,),
            ),
            "readiness": ("SELECT * FROM source_readiness WHERE source_id=?", (case.source,)),
            "indexes": (
                "SELECT * FROM source_readiness_indexes WHERE source_id=?"
                " ORDER BY index_version_id",
                (case.source,),
            ),
            "handoff": (
                "SELECT extraction_version_id,review_generation,manifest_id,selection_generation,"
                "purpose,state,job_id,origin_job_id FROM search_handoffs WHERE job_id=?",
                (case.job,),
            ),
            "requests": (
                "SELECT * FROM notebook_model_requests WHERE notebook_id=? ORDER BY generation",
                (case.notebook,),
            ),
            "jobs": (
                "SELECT * FROM jobs WHERE source_id=? ORDER BY created_at,id",
                (case.source,),
            ),
            "checkpoints": (
                "SELECT * FROM job_steps WHERE job_id=? ORDER BY step_key",
                (case.job,),
            ),
            "generations": (
                "SELECT g.* FROM search_generations g"
                " JOIN index_versions i ON i.id=g.index_version_id"
                " WHERE i.source_id=? ORDER BY i.id",
                (case.source,),
            ),
            "inputs": (
                "SELECT inputs.* FROM search_job_inputs inputs JOIN jobs j ON j.id=inputs.job_id"
                " WHERE j.source_id=? ORDER BY j.id",
                (case.source,),
            ),
            "installations": ("SELECT * FROM model_installations ORDER BY manifest_id", ()),
            "setup": ("SELECT * FROM model_setup_attempts ORDER BY id", ()),
        }
        return {
            name: [tuple(row) for row in connection.execute(sql, parameters)]
            for name, (sql, parameters) in queries.items()
        }


@pytest.mark.parametrize("later_choice", ["pending", "cancel-pending", "choose-active"])
@pytest.mark.parametrize("finish_attempt", [False, True], ids=["open-attempt", "finished-attempt"])
def test_committed_activation_survives_legitimate_later_model_choices_on_restart(
    store, original, publication_runtime, monkeypatch, later_choice, finish_attempt
):
    """R1v1 cancels accepted E5 at claim, despite the later choice keeping it active.

    Exercise both legacy exit windows: before attempt bookkeeping, and after that
    bookkeeping but before job completion. All interfaces exist on 4139df1/R1v1;
    the expected negative is lost eligibility/cancelled state, not a missing helper.
    """
    case = prepare_publication(store, original, "model-switch")
    legacy_activation(store, case)
    if finish_attempt:
        store.finish_step_attempt(case.job, "publish-search", 1, None)
    models = NotebookModelService(store)
    accepted = models.state(case.notebook)
    e5 = profiles()["multilingual-e5"]
    bekko = profiles()[DEFAULT_PROFILE_ID]
    assert accepted["active_manifest_id"] == e5.identity
    assert accepted["pending_manifest_id"] is None
    assert accepted["selection_generation"] == 1
    with store.database.connect() as connection:
        installed = connection.execute(
            "SELECT state FROM model_installations WHERE manifest_id=?", (bekko.identity,)
        ).fetchone()
    assert installed is None or installed[0] != "installed"
    assert not installation_path(store.database.config.data_dir, bekko).exists()

    pending = models.choose(case.notebook, DEFAULT_PROFILE_ID, 1, "missing-bekko-after-accepted-e5")
    assert pending["active_manifest_id"] == e5.identity
    assert pending["pending_manifest_id"] == bekko.identity
    assert pending["selection_generation"] == 2
    if later_choice == "cancel-pending":
        chosen = models.cancel(case.notebook, 2)
    elif later_choice == "choose-active":
        chosen = models.choose(case.notebook, e5.id, 2, "keep-already-active-e5")
    else:
        chosen = pending
    expected_pending = bekko.identity if later_choice == "pending" else None
    assert chosen["active_manifest_id"] == e5.identity
    assert chosen["pending_manifest_id"] == expected_pending
    assert chosen["selection_generation"] == (2 if later_choice == "pending" else 3)
    with store.database.connect() as connection:
        history = [
            tuple(row)
            for row in connection.execute(
                "SELECT generation,manifest_id,state FROM notebook_model_requests"
                " WHERE notebook_id=? ORDER BY generation",
                (case.notebook,),
            )
        ]
        bekko_job = connection.execute(
            "SELECT h.job_id FROM search_handoffs h JOIN jobs j ON j.id=h.job_id"
            " WHERE j.source_id=? AND h.manifest_id=? AND h.selection_generation=2"
            " AND h.purpose='model-switch'",
            (case.source, bekko.identity),
        ).fetchone()[0]
    expected_history = [
        (1, e5.identity, "active"),
        (2, bekko.identity, "pending" if later_choice == "pending" else "cancelled"),
    ]
    if later_choice == "choose-active":
        expected_history.append((3, e5.identity, "active"))
    assert history == expected_history

    before = protected_output(store, case)
    receipts = activation_receipts(store, case)
    charge = generation_charge(store, case)
    attempts = store.load_job_attempts(case.job)
    previous_status = store.load_job_status(case.job)
    assert previous_status is not None and previous_status.status == "queued"
    bekko_status = store.load_job_status(bekko_job)
    expected_bekko_status = "queued" if later_choice == "pending" else "cancelled"
    assert bekko_status is not None and bekko_status.status == expected_bekko_status
    assert bekko_status.attempt_count == 0 and bekko_status.policy is None
    assert store.load_job_attempts(bekko_job) == ()
    assert store.load_job_steps(bekko_job) == ()
    documents = list(publication_runtime.document_calls)

    def forbidden(*args, **kwargs):
        pytest.fail("finishing accepted E5 or waiting for missing Bekko must not start inference")

    def e5_query_only(path, profile, **options):
        assert profile.identity == e5.identity
        return publication_runtime(path, profile, **options)

    monkeypatch.setattr("core.search.preparation.InputTokenizer", forbidden)
    monkeypatch.setattr("core.search.preparation.EmbeddingSession", forbidden)
    restarted = reopen(store)
    assert_saved_search(restarted, case, e5_query_only)
    status = run_pending(restarted, case)
    # Check actual eligibility first: R1v1 cancels this still-current publisher and
    # both modes lose its accepted saved output. No model installation is needed.
    assert_saved_search(restarted, case, e5_query_only)
    assert status.status == "completed", status
    assert status.attempt_count == previous_status.attempt_count
    after = NotebookModelService(restarted).state(case.notebook)
    assert after == chosen
    assert protected_output(restarted, case) == before
    assert activation_receipts(restarted, case) == receipts
    assert generation_charge(restarted, case) == charge
    assert publication_runtime.document_calls == documents
    assert not installation_path(restarted.database.config.data_dir, bekko).exists()
    after_bekko = restarted.load_job_status(bekko_job)
    assert after_bekko is not None and after_bekko.status == expected_bekko_status
    assert after_bekko.attempt_count == 0 and after_bekko.policy == bekko_status.policy
    assert after_bekko.owner_instance_id is None
    assert restarted.load_job_attempts(bekko_job) == ()
    assert restarted.load_job_steps(bekko_job) == ()
    recovered_attempts = restarted.load_job_attempts(case.job)
    assert len(recovered_attempts) == len(attempts)
    for previous, recovered in zip(attempts, recovered_attempts, strict=True):
        assert (
            recovered.job_id,
            recovered.step_key,
            recovered.attempt_number,
            recovered.started_at,
        ) == (
            previous.job_id,
            previous.step_key,
            previous.attempt_number,
            previous.started_at,
        )
        if previous.finished_at is not None:
            assert recovered == previous
        else:
            assert recovered.step_key == "publish-search"
            assert recovered.finished_at is not None and recovered.outcome == "ok"
            assert recovered.failure_reason is None


@pytest.mark.parametrize(
    "change",
    [
        "original-missing",
        "original-pending",
        "original-cancelled",
        "latest-missing",
        "latest-active",
        "latest-cancelled",
        "pending-manifest",
        "pending-generation",
        "cancel-generation",
        "active-generation",
        "source",
        "index",
        "publisher",
        "model",
        "review",
        "failed-attempt",
        "unknown-attempt",
        "failed-job",
        "cancelled-job",
        "superseded-job",
    ],
)
def test_later_choice_recovery_still_requires_receipts_and_exact_accepted_ownership(
    store, original, publication_runtime, change
):
    """Corrupt only the named fence; never repair history to make recovery fit."""
    case = prepare_publication(store, original, "model-switch")
    legacy_activation(store, case)
    models = NotebookModelService(store)
    models.choose(case.notebook, DEFAULT_PROFILE_ID, 1, "fenced-later-bekko")
    ownership_changes = {
        "source",
        "index",
        "publisher",
        "model",
        "review",
        "failed-attempt",
        "unknown-attempt",
        "failed-job",
        "cancelled-job",
        "superseded-job",
    }
    if change == "cancel-generation" or change in ownership_changes:
        # Exercise the cancellation-only generation gap, including every original
        # ownership/terminal guard. The valid later receipt cannot bypass them.
        models.cancel(case.notebook, 2)
    elif change == "active-generation":
        models.choose(case.notebook, "multilingual-e5", 2, "fenced-keep-active-e5")
    replacement_version = None
    if change == "source":
        _, client, _ = original
        replacement = upload(
            client,
            case.notebook,
            native_text_pdf("A different saved rainfall source version is not accepted E5 output."),
            case.source,
        )
        replacement_version = store.load_pdf_import(replacement).source_version_id
    if change in {"failed-attempt", "unknown-attempt"}:
        store.finish_step_attempt(
            case.job,
            "publish-search",
            1,
            "controlled original failure" if change == "failed-attempt" else None,
            outcome="failed" if change == "failed-attempt" else "unknown",
        )
    elif change in {"failed-job", "cancelled-job"}:
        store.finish_job(case.job, "failed" if change == "failed-job" else "cancelled")
    else:
        with store._transaction() as connection:
            if change.startswith(("original-", "latest-")):
                generation = 1 if change.startswith("original-") else 2
                state = change.split("-", 1)[1]
                if state == "missing":
                    connection.execute(
                        "DELETE FROM notebook_model_requests WHERE notebook_id=? AND generation=?",
                        (case.notebook, generation),
                    )
                else:
                    connection.execute(
                        "UPDATE notebook_model_requests SET state=?"
                        " WHERE notebook_id=? AND generation=?",
                        (state, case.notebook, generation),
                    )
            elif change == "pending-manifest":
                connection.execute(
                    "UPDATE notebook_profiles SET pending_manifest_id=active_manifest_id"
                    " WHERE notebook_id=?",
                    (case.notebook,),
                )
            elif change.endswith("-generation"):
                connection.execute(
                    "UPDATE notebook_profiles SET selection_generation=selection_generation+1"
                    " WHERE notebook_id=?",
                    (case.notebook,),
                )
            elif change == "source":
                assert replacement_version is not None
                connection.execute(
                    "UPDATE source_readiness SET source_version_id=? WHERE source_id=?",
                    (replacement_version, case.source),
                )
            elif change == "index":
                connection.execute(
                    "UPDATE source_readiness_indexes SET index_version_id=? WHERE source_id=?",
                    (case.old.index, case.source),
                )
            elif change in {"publisher", "superseded-job"}:
                old_publisher = connection.execute(
                    "SELECT job_id FROM job_steps WHERE output_index_version_id=?",
                    (case.old.index,),
                ).fetchone()[0]
                if change == "publisher":
                    connection.execute(
                        "UPDATE source_readiness SET published_by_job_id=? WHERE source_id=?",
                        (old_publisher, case.source),
                    )
                else:
                    connection.execute(
                        "UPDATE job_status SET superseded_by_job_id=? WHERE job_id=?",
                        (old_publisher, case.job),
                    )
            elif change == "model":
                connection.execute(
                    "UPDATE notebook_profiles SET active_manifest_id=? WHERE notebook_id=?",
                    (profiles()["minilm-en"].identity, case.notebook),
                )
            elif change == "review":
                connection.execute(
                    "UPDATE search_handoffs SET review_generation=review_generation+1"
                    " WHERE job_id=?",
                    (case.job,),
                )
            else:
                raise AssertionError(f"unhandled recovery fence: {change}")

    before = protected_output(store, case)
    receipts = activation_receipts(store, case)
    attempts = store.load_job_attempts(case.job)
    previous_status = store.load_job_status(case.job)
    assert previous_status is not None
    documents = list(publication_runtime.document_calls)
    restarted = reopen(store)
    coordinator = SearchCoordinator(restarted, restarted.database.config.data_dir / "models")
    assert coordinator.claim_blocking_reason(restarted.load_job_by_id(case.job)) == (
        "search_input_changed"
    )
    status = restarted.load_job_status(case.job)
    assert status is not None
    if previous_status.status in {"failed", "cancelled"}:
        assert status == previous_status
    else:
        assert status.status == "cancelled"
        assert status.attempt_count == previous_status.attempt_count
    assert protected_output(restarted, case) == before
    assert restarted.load_job_attempts(case.job) == attempts
    after = activation_receipts(restarted, case)
    for name in receipts.keys() - {"handoff"}:
        assert after[name] == receipts[name], name
    assert publication_runtime.document_calls == documents


@pytest.mark.parametrize("later_choice", ["pending", "cancel-pending", "choose-active"])
def test_accepted_activation_recovery_does_not_grant_new_publication_authority(
    store, original, publication_runtime, later_choice
):
    case = prepare_publication(store, original, "model-switch")
    legacy_activation(store, case)
    models = NotebookModelService(store)
    models.choose(case.notebook, DEFAULT_PROFILE_ID, 1, "no-republication-bekko")
    if later_choice == "cancel-pending":
        models.cancel(case.notebook, 2)
    elif later_choice == "choose-active":
        models.choose(case.notebook, "multilingual-e5", 2, "no-republication-keep-e5")
    restarted = reopen(store)
    restarted.claim_job(case.job, "late-publication-control")
    attempts = restarted.load_step_attempts(case.job, "publish-search")
    assert len(attempts) == 1 and attempts[0].finished_at is None
    guard = AttemptGuard("late-publication-control", "publish-search", attempts[0].attempt_number)
    before = protected_output(restarted, case)
    receipts = activation_receipts(restarted, case)
    with pytest.raises(ConflictingWriteError, match="model selection or ready source changed"):
        publish_replacement(restarted, case.job, guard, case.index)
    assert protected_output(restarted, case) == before
    assert activation_receipts(restarted, case) == receipts
    assert restarted.load_step_attempts(case.job, "publish-search") == attempts
    assert_saved_search(restarted, case, publication_runtime)


def test_committed_activation_recovery_keeps_ready_pdf_during_pending_replacement(
    store, original, publication_runtime
):
    case = prepare_publication(store, original, "model-switch")
    legacy_activation(store, case)
    before = protected_output(store, case)
    documents = list(publication_runtime.document_calls)
    _, client, _ = original
    candidate = upload(
        client,
        case.notebook,
        native_text_pdf("A new pending rainfall PDF must not hide the active saved version."),
        case.source,
    )
    assert run_pending(store, case).status == "completed"
    state = store.load_source_pdf_state(case.source)
    assert state is not None and state.current_import_id == case.import_id
    assert state.pending_import_id == candidate
    assert protected_output(store, case) == before
    assert publication_runtime.document_calls == documents
    assert len(store.load_step_attempts(case.job, "publish-search")) == 1
    assert_saved_search(store, case, publication_runtime)


@pytest.mark.parametrize(
    "change",
    ["selection", "model", "review", "publisher", "failed-attempt", "failed-job", "cancelled-job"],
)
def test_committed_activation_recovery_keeps_current_input_and_terminal_fences(
    store, original, publication_runtime, change
):
    case = prepare_publication(store, original, "model-switch")
    legacy_activation(store, case)
    before = protected_output(store, case)
    documents = list(publication_runtime.document_calls)
    if change == "failed-attempt":
        store.finish_step_attempt(case.job, "publish-search", 1, "controlled original failure")
    elif change in {"failed-job", "cancelled-job"}:
        store.finish_job(case.job, "failed" if change == "failed-job" else "cancelled")
    else:
        with store._transaction() as connection:
            if change == "selection":
                # This is an unrecorded generation mutation, unlike legitimate
                # choose/cancel history above. It must not grant publication rights.
                connection.execute(
                    "UPDATE notebook_profiles SET selection_generation=selection_generation+1"
                    " WHERE notebook_id=?",
                    (case.notebook,),
                )
                assert (
                    connection.execute(
                        "SELECT 1 FROM notebook_model_requests WHERE notebook_id=? AND generation=("
                        "SELECT selection_generation FROM notebook_profiles WHERE notebook_id=?)",
                        (case.notebook, case.notebook),
                    ).fetchone()
                    is None
                )
            elif change == "model":
                connection.execute(
                    "UPDATE notebook_profiles SET active_manifest_id=? WHERE notebook_id=?",
                    (store.load_index(case.old.index).index.embedding_revision, case.notebook),
                )
            elif change == "review":
                connection.execute(
                    "UPDATE search_handoffs SET review_generation=review_generation+1"
                    " WHERE job_id=?",
                    (case.job,),
                )
            else:
                old_publisher = connection.execute(
                    "SELECT job_id FROM job_steps WHERE output_index_version_id=?",
                    (case.old.index,),
                ).fetchone()[0]
                connection.execute(
                    "UPDATE source_readiness SET published_by_job_id=? WHERE source_id=?",
                    (old_publisher, case.source),
                )
    coordinator = SearchCoordinator(store, store.database.config.data_dir / "models")
    if change in {"failed-job", "cancelled-job"}:
        before_status = store.load_job_status(case.job)
        assert run_pending(store, case) == before_status
    else:
        assert coordinator.claim_blocking_reason(store.load_job_by_id(case.job)) is not None
        assert store.load_job_status(case.job).status == "cancelled"
    assert protected_output(store, case) == before
    assert publication_runtime.document_calls == documents
    assert len(store.load_step_attempts(case.job, "publish-search")) == 1
