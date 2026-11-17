"""Real SQLite contention must not strand a stopped setup writer."""

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_model_setup import SmallSetup, wait

from core.persistence.errors import ConflictingWriteError, DatabaseBusyError
from core.search import install
from core.search.profiles import installation_path, verify_installation


def attempts(service):
    with service.store.database.connect() as connection:
        return [
            dict(row)
            for row in connection.execute("SELECT * FROM model_setup_attempts ORDER BY rowid")
        ]


@pytest.fixture
def contended_setup(store, monkeypatch):
    # Keep the real connection/transaction/error-translation path. Only shorten its
    # existing lock wait; neither failed write is replaced by an injected exception.
    store.database.config = replace(store.database.config, lock_wait_ms=75)
    control = SimpleNamespace(
        service=SmallSetup(store),
        next_attempt=threading.Event(),
        allow_attempt=threading.Event(),
        terminal_busy=threading.Event(),
        maintenance_busy=threading.Event(),
        writer_returned=threading.Event(),
        allow_exit=threading.Event(),
        live_cleanup=threading.Event(),
        notified=threading.Event(),
        transfers=[],
        busy_errors=[],
        writer_errors=[],
    )
    service = control.service
    service.notify = control.notified.set
    current = service._current
    run = service._run
    cleanup = service.cleanup
    transaction = store._transaction
    current_calls = 0

    def transfer(profile, path, generation, timeout, max_bytes, progress):
        control.transfers.append((generation, threading.current_thread()))
        assets = path / "assets"
        assets.mkdir(parents=True)
        # Retain nonzero bytes/attempts before the next operational write fails.
        (assets / "asset.bin").write_bytes(b"evil" if generation == 1 else b"good")
        progress(4)
        return 4

    def gated_current(*args):
        nonlocal current_calls
        result = current(*args)
        current_calls += 1
        if current_calls == 2:
            control.next_attempt.set()
            assert control.allow_attempt.wait(10), "test did not release the next setup write"
        return result

    @contextmanager
    def observed_transaction(*args, **kwargs):
        try:
            with transaction(*args, **kwargs) as connection:
                yield connection
        except DatabaseBusyError as error:
            if threading.current_thread() is service._thread:
                control.busy_errors.append(error)
                if len(control.busy_errors) == 2:
                    control.terminal_busy.set()
            elif control.writer_returned.is_set() and not service._thread.is_alive():
                control.maintenance_busy.set()
            raise

    def paused_writer_exit(*args):
        try:
            run(*args)
        except Exception as error:
            # Also works on the old implementation: retain its abandoned-writer
            # exception, then fail on the actual phantom state, not a thread warning.
            control.writer_errors.append(error)
        finally:
            if args[1] == 1:
                control.writer_returned.set()
                assert control.allow_exit.wait(10), "test did not release the old setup writer"

    def observed_cleanup():
        cleanup()
        if control.writer_returned.is_set() and service._thread.is_alive():
            control.live_cleanup.set()

    monkeypatch.setattr(service, "_transfer", transfer)
    monkeypatch.setattr(service, "_current", gated_current)
    monkeypatch.setattr(service, "_run", paused_writer_exit)
    monkeypatch.setattr(service, "cleanup", observed_cleanup)
    monkeypatch.setattr(store, "_transaction", observed_transaction)
    try:
        yield control
    finally:
        control.allow_attempt.set()
        control.allow_exit.set()
        asyncio.run(service.stop())


def abandon_terminal_write(control):
    service = control.service
    service.begin("minilm-en", "contended-terminal", 0)
    assert control.next_attempt.wait(5), "first transfer did not reach its next reservation"
    with service.store._transaction() as writer:
        assert writer.in_transaction
        control.allow_attempt.set()
        assert control.terminal_busy.wait(5), "operational and terminal writes did not both fail"
        assert control.writer_returned.wait(5), "writer did not hand off its terminal outcome"
        assert service._thread.is_alive()
        saved = attempts(service)[0]
        assert saved["state"] == "running" and saved["cleanup_pending"] == 1
        assert saved["received_bytes"] == 4 and saved["transfer_attempts"] == 1
        assert saved["spent_seconds"] == service.timeout_seconds
        assert len(control.busy_errors) == 2 and len(control.transfers) == 1
    return saved


async def release_writer(control):
    writer = control.service._thread
    control.allow_exit.set()
    await asyncio.to_thread(writer.join, 5)
    assert not writer.is_alive()


async def stop_service(control):
    control.allow_attempt.set()
    control.allow_exit.set()
    await control.service.stop()


def test_initial_terminal_charge_includes_successful_transaction_acquisition(store, monkeypatch):
    service = SmallSetup(store, max_bytes=4)
    clock = SimpleNamespace(now=100.0)
    # Replace only install.py's clock, not asyncio/threading's actual bounded waits.
    monkeypatch.setattr(install, "time", SimpleNamespace(monotonic=lambda: clock.now))
    transaction = store._transaction
    writer_transactions = 0

    def transfer(profile, path, generation, timeout, max_bytes, progress):
        assets = path / "assets"
        assets.mkdir(parents=True)
        (assets / "asset.bin").write_bytes(b"evil")
        progress(4)
        clock.now = 110.0
        return 4

    @contextmanager
    def clocked_transaction(*args, **kwargs):
        nonlocal writer_transactions
        with transaction(*args, **kwargs) as connection:
            if threading.current_thread() is service._thread:
                writer_transactions += 1
                if writer_transactions == 3:  # Reservation, progress, then terminal.
                    # Deterministically model 30 seconds reaching this transaction.
                    assert clock.now == 110.0
                    clock.now = 140.0
            yield connection

    monkeypatch.setattr(service, "_transfer", transfer)
    monkeypatch.setattr(store, "_transaction", clocked_transaction)
    try:
        service.begin("minilm-en", "terminal-acquisition-charge", 0)
        final = wait(service)
        assert final["state"] == "failed" and final["failure_code"] == "model_download_limit"
        assert final["spent_seconds"] == attempts(service)[0]["spent_seconds"] == 40.0
        assert final["received_bytes"] == 4 and final["transfer_attempts"] == 1
    finally:
        asyncio.run(service.stop())


def test_failed_terminal_wait_is_charged_once_despite_repeated_lifecycle_contention(
    contended_setup, monkeypatch
):
    control = contended_setup
    service = control.service
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(install, "time", SimpleNamespace(monotonic=lambda: clock.now))
    transaction = service.store._transaction
    repeated_contention = threading.Event()
    maintenance_failures = 0

    @contextmanager
    def clocked_transaction(*args, **kwargs):
        nonlocal maintenance_failures
        try:
            with transaction(*args, **kwargs) as connection:
                yield connection
        except DatabaseBusyError:
            if threading.current_thread() is service._thread:
                # The fixture has observed the real failed BEGIN before returning
                # here. Advance the clock across each original-writer lock wait.
                clock.now = 110.0 if len(control.busy_errors) == 1 else 140.0
            elif control.writer_returned.is_set() and not service._thread.is_alive():
                maintenance_failures += 1
                clock.now += 100.0
                if maintenance_failures == 2:
                    repeated_contention.set()
            raise

    monkeypatch.setattr(service.store, "_transaction", clocked_transaction)

    async def exercise():
        await service.start()
        try:
            saved = abandon_terminal_write(control)
            assert clock.now == 140.0
            with service.store._transaction():
                await release_writer(control)
                assert await asyncio.to_thread(repeated_contention.wait, 5)
                assert attempts(service)[0] == saved  # Keep the reservation until settlement.
                assert not control.notified.is_set()
                clock.now = 1_000.0  # Maintenance delay is not another transfer charge.
            assert await asyncio.to_thread(control.notified.wait, 5)
            final = service.state("minilm-en")
            assert final["state"] == "failed" and final["can_retry"]
            failed = attempts(service)[0]
            assert final["spent_seconds"] == failed["spent_seconds"] == 40.0
            assert failed["received_bytes"] == 4 and failed["transfer_attempts"] == 1
            assert failed["cleanup_pending"] == 0 and failed["generation"] == 1
            assert maintenance_failures >= 2
            assert len(control.transfers) == 1 and control.writer_errors == []
            service.cleanup()
            assert attempts(service)[0] == failed
        finally:
            await stop_service(control)

    asyncio.run(exercise())


def test_terminal_contention_recovers_same_session_without_polling_or_another_transfer(
    contended_setup,
):
    control = contended_setup
    service = control.service

    async def exercise():
        await service.start()
        try:
            saved = abandon_terminal_write(control)
            staging = service._path(saved["staging_path"])
            assert (staging / "assets" / "asset.bin").read_bytes() == b"evil"

            # Even after SQLite is available, lifecycle maintenance may not settle
            # a terminal handoff until the exact writer thread has really exited.
            assert await asyncio.to_thread(control.live_cleanup.wait, 5)
            assert attempts(service)[0] == saved
            assert service.state("minilm-en")["state"] == "downloading"
            assert not service.state("minilm-en")["can_retry"]
            assert not control.notified.is_set()

            await release_writer(control)
            # No status request, begin(), recover() or manual cleanup drives this.
            # On 4139df1 the wait expires, then the next assertion sees downloading.
            await asyncio.to_thread(control.notified.wait, 5)
            final = service.state("minilm-en")
            assert final["state"] == "failed", final
            assert final["failure_code"] == "model_download_failed"
            assert final["generation"] == 1 and final["can_retry"]
            failed = attempts(service)[0]
            assert failed["state"] == "failed" and failed["cleanup_pending"] == 0
            assert failed["failure_code"] == final["failure_code"]
            assert failed["received_bytes"] == final["received_bytes"] == 4
            assert failed["transfer_attempts"] == final["transfer_attempts"] == 1
            assert 0 < failed["spent_seconds"] == final["spent_seconds"]
            assert all(failed[key] == saved[key] for key in ("id", "request_key", "created_at"))
            assert not staging.exists()
            assert len(control.transfers) == 1 and control.writer_errors == []

            service.begin("minilm-en", "retry-after-terminal-recovery", 1)
            assert (await asyncio.to_thread(wait, service))["state"] == "installed"
            profile = service.profile("minilm-en")
            verify_installation(installation_path(service.data_dir, profile), profile)
            retained, retried = attempts(service)
            assert retained == failed
            assert retried["generation"] == 2 and retried["state"] == "installed"
            assert retried["received_bytes"] == 4 and retried["transfer_attempts"] == 1
            assert retried["spent_seconds"] > 0 and retried["cleanup_pending"] == 0
            assert len(control.transfers) == 2
            assert control.transfers[0][1] is not control.transfers[1][1]
            assert not control.transfers[0][1].is_alive()
        finally:
            await stop_service(control)

    asyncio.run(exercise())


def test_cancel_fences_pending_terminal_accounting_before_a_new_generation(contended_setup):
    control = contended_setup
    service = control.service

    async def exercise():
        await service.start()
        try:
            saved = abandon_terminal_write(control)
            cancelled = service.cancel("minilm-en", 1)
            assert cancelled["state"] == "cancelled" and cancelled["generation"] == 2
            # Recorded file cleanup alone must not let Retry replace the writer or
            # its not-yet-settled accounting, even for a different model.
            service.cleanup()
            cancelled_attempt = attempts(service)[0]
            assert cancelled_attempt["cleanup_pending"] == 0
            assert not service.state("minilm-en")["can_retry"]
            assert not service.state("multilingual-e5")["can_retry"]
            with pytest.raises(ConflictingWriteError, match="active|finalization"):
                service.begin("minilm-en", "before-terminal-writer-exit", 2)
            assert len(control.transfers) == 1 and len(attempts(service)) == 1

            with service.store._transaction():
                await release_writer(control)
                assert await asyncio.to_thread(control.maintenance_busy.wait, 5)
                assert attempts(service)[0] == cancelled_attempt
                assert not service.state("minilm-en")["can_retry"]
                assert not service.state("multilingual-e5")["can_retry"]
                assert not control.notified.is_set()
            assert await asyncio.to_thread(control.notified.wait, 5)
            final = service.state("minilm-en")
            assert final["state"] == "cancelled" and final["generation"] == 2
            assert final["failure_code"] == "model_setup_cancelled" and final["can_retry"]
            failed = attempts(service)[0]
            assert failed["id"] == saved["id"] and failed["generation"] == 1
            assert failed["state"] == "cancelled" and failed["cleanup_pending"] == 0
            assert failed["failure_code"] == "model_setup_cancelled"
            assert failed["updated_at"] == cancelled_attempt["updated_at"]
            assert failed["received_bytes"] == 4 and failed["transfer_attempts"] == 1
            assert failed["spent_seconds"] > 0
            assert control.writer_errors == []

            service.begin("minilm-en", "after-cancelled-terminal-recovery", 2)
            assert (await asyncio.to_thread(wait, service))["state"] == "installed"
            retained, retried = attempts(service)
            assert retained == failed
            assert retried["generation"] == 3 and retried["state"] == "installed"
            assert service.state("minilm-en")["generation"] == 3
            assert len(control.transfers) == 2 and not control.transfers[0][1].is_alive()
        finally:
            await stop_service(control)

    asyncio.run(exercise())


def test_shutdown_drains_a_handoff_after_join_without_reclassifying_its_failure(contended_setup):
    control = contended_setup
    service = control.service

    async def exercise():
        await service.start()
        try:
            abandon_terminal_write(control)
            stopping = asyncio.create_task(service.stop())
            await asyncio.sleep(0)
            assert service._stop.is_set() and not stopping.done()
            with pytest.raises(ValueError, match="invalid_setup_request"):
                service.begin("multilingual-e5", "shutdown-race", 0)
            await release_writer(control)
            await asyncio.wait_for(stopping, 5)
            final = service.state("minilm-en")
            assert final["state"] == "failed"
            assert final["failure_code"] == "model_download_failed"
            assert final["generation"] == 1 and not final["can_retry"]
            failed = attempts(service)[0]
            assert failed["state"] == "failed" and failed["cleanup_pending"] == 0
            assert failed["failure_code"] == "model_download_failed"
            assert failed["received_bytes"] == 4 and failed["transfer_attempts"] == 1
            assert failed["spent_seconds"] > 0
            assert control.notified.is_set() and control.writer_errors == []
            assert len(control.transfers) == 1 and not service._thread.is_alive()
            assert service._cleanup_task.done()
        finally:
            await stop_service(control)

    asyncio.run(exercise())


def test_shutdown_surfaces_a_still_busy_terminal_drain_and_retains_it_for_retry(contended_setup):
    control = contended_setup
    service = control.service

    async def exercise():
        await service.start()
        try:
            saved = abandon_terminal_write(control)
            with service.store._transaction():
                stopping = asyncio.create_task(service.stop())
                await release_writer(control)
                with pytest.raises(DatabaseBusyError):
                    await asyncio.wait_for(stopping, 5)
                assert attempts(service)[0] == saved
                assert not service._thread.is_alive()
                assert not control.notified.is_set()
            await service.stop()
            final = service.state("minilm-en")
            assert final["state"] == "failed" and final["failure_code"] == "model_download_failed"
            failed = attempts(service)[0]
            assert failed["received_bytes"] == 4 and failed["transfer_attempts"] == 1
            assert failed["spent_seconds"] > 0 and failed["cleanup_pending"] == 0
            assert len(control.transfers) == 1 and control.writer_errors == []
        finally:
            await stop_service(control)

    asyncio.run(exercise())
