"""Authenticated PDF acquisition orchestration over durable core records."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypeVar

from starlette.requests import ClientDisconnect

from core.ingest.config import PdfIngestLimits
from core.ingest.files import ManagedFileIntegrityError, ManagedFilePathError, ManagedPdfFiles
from core.ingest.locks import FingerprintLockCoordinator
from core.persistence.errors import (
    ConflictingWriteError,
    DeletedPdfAcquisitionError,
    PersistenceError,
)
from core.persistence.models import ManagedFile, PdfAcquisition, new_id, utc_timestamp
from core.persistence.store import PersistenceStore

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
FILE_CLEANUP_PENDING_MESSAGE = (
    "The previous stored copy of this PDF is still being removed. Retry after cleanup finishes."
)


class PdfImportError(Exception):
    def __init__(self, code: str, status: int, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.status = status
        self.detail = detail


class PdfIngestService:
    def __init__(
        self,
        store: PersistenceStore,
        owner_instance_id: str,
        *,
        limits: PdfIngestLimits | None = None,
        files: ManagedPdfFiles | None = None,
        fingerprint_locks: FingerprintLockCoordinator | None = None,
        notify_jobs=None,
    ) -> None:
        self.store = store
        self.owner_instance_id = owner_instance_id
        self.limits = limits or PdfIngestLimits.from_environment()
        self.files = files or ManagedPdfFiles(store.database.config.data_dir)
        self.fingerprint_locks = fingerprint_locks or FingerprintLockCoordinator()
        self.notify_jobs = notify_jobs
        self._upload_slot = asyncio.Lock()
        self._file_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpi-pdf-file")
        self._finalization_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="corpi-pdf-finalize"
        )
        self._active_paths: set[str] = set()
        self._active_paths_lock = threading.Lock()

    async def start(self) -> None:
        """Recover complete staged inputs before the runner or trash sweep starts."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._file_executor, self.reconcile_startup)

    async def stop(self) -> None:
        # Upload handlers own and await their open writers. The application lifespan
        # waits for those requests before shutting down this executor.
        await asyncio.to_thread(self._file_executor.shutdown, True, cancel_futures=False)
        await asyncio.to_thread(self._finalization_executor.shutdown, True, cancel_futures=False)

    def create_acquisition(
        self,
        retry_key: str,
        notebook_id: str,
        intent: str,
        display_filename: str,
        target_source_id: str | None = None,
    ) -> PdfAcquisition:
        try:
            return self.store.create_pdf_acquisition(
                retry_key,
                notebook_id,
                intent,
                display_filename,
                self.owner_instance_id,
                target_source_id,
            )
        except DeletedPdfAcquisitionError as error:
            raise PdfImportError(error.code, 409, str(error)) from error
        except PersistenceError as error:
            raise PdfImportError("invalid_acquisition", 409, str(error)) from error

    async def receive_and_finalize(self, acquisition_id: str, request: Request) -> PdfAcquisition:
        if self._upload_slot.locked():
            raise PdfImportError("import_busy", 409, "Another PDF is currently being imported.")
        async with self._upload_slot:
            loop = asyncio.get_running_loop()
            accepted_at = time.monotonic()
            deadline = accepted_at + self.limits.upload_total_seconds
            acquisition = await self._before_deadline(
                loop, self.store.load_pdf_acquisition, acquisition_id, deadline=deadline
            )
            if acquisition.state == "completed" or acquisition.input_complete_at is not None:
                replay = await self._verify_replayed_upload(acquisition, request)
                if replay.state == "completed":
                    return replay
                if replay.state == "failed":
                    try:
                        replay = self.store.resume_failed_pdf_acquisition(
                            replay.id, self.owner_instance_id
                        )
                    except PersistenceError as error:
                        raise PdfImportError("retry_unavailable", 409, str(error)) from error
                result = await asyncio.get_running_loop().run_in_executor(
                    self._file_executor,
                    self._finalize,
                    replay.id,
                    True,
                    self.limits.upload_total_seconds,
                )
                if result.state == "completed":
                    await asyncio.get_running_loop().run_in_executor(
                        self._file_executor,
                        self._journal_and_remove,
                        result.id,
                        result.staging_relative_path,
                        "staging",
                        result.input_sha256,
                        result.input_size_bytes,
                    )
                return result
            acquisition = await self._before_deadline(
                loop, self._begin_upload, acquisition_id, deadline=deadline
            )
            stage_file = None
            digest = hashlib.sha256()
            signature = bytearray()
            received = 0
            try:
                content_length = request.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = -1
                    if declared_size > self.limits.max_upload_bytes:
                        raise PdfImportError(
                            "pdf_too_large", 413, "PDF exceeds the 25 MB import limit."
                        )
                with self._active_paths_lock:
                    self._active_paths.add(acquisition.staging_relative_path)
                stage_file = await self._before_deadline(
                    loop,
                    self.files.open_staging,
                    acquisition.staging_relative_path,
                    deadline=deadline,
                )
                stream = request.stream().__aiter__()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PdfImportError(
                            "upload_timeout", 408, "PDF upload exceeded its total time limit."
                        )
                    timeout = min(remaining, self.limits.upload_idle_seconds)
                    try:
                        chunk = await asyncio.wait_for(anext(stream), timeout=timeout)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as error:
                        code = (
                            "upload_timeout"
                            if time.monotonic() >= deadline
                            else "upload_idle_timeout"
                        )
                        message = (
                            "PDF upload exceeded its total time limit."
                            if code == "upload_timeout"
                            else "PDF upload stopped sending bytes for too long."
                        )
                        raise PdfImportError(code, 408, message) from error
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > self.limits.max_upload_bytes:
                        raise PdfImportError(
                            "pdf_too_large", 413, "PDF exceeds the 25 MB import limit."
                        )
                    if len(signature) < 1024:
                        signature.extend(chunk[: 1024 - len(signature)])
                    digest.update(chunk)
                    await self._before_deadline(loop, stage_file.write, chunk, deadline=deadline)
                    await self._before_deadline(
                        loop,
                        self.store.note_pdf_upload_progress,
                        acquisition.id,
                        self.owner_instance_id,
                        acquisition.finalization_generation,
                        received,
                        deadline=deadline,
                    )
                if received == 0:
                    raise PdfImportError("empty_pdf", 422, "Choose a non-empty PDF file.")
                if b"%PDF-" not in signature:
                    raise PdfImportError(
                        "invalid_pdf", 422, "The selected file does not contain a PDF header."
                    )
                await self._before_deadline(loop, stage_file.flush, deadline=deadline)
                await self._before_deadline(loop, os.fsync, stage_file.fileno(), deadline=deadline)
                await loop.run_in_executor(self._file_executor, stage_file.close)
                stage_file = None
                with self._active_paths_lock:
                    self._active_paths.discard(acquisition.staging_relative_path)
                checksum = digest.hexdigest()
                acquisition = await self._before_deadline(
                    loop,
                    self.store.record_pdf_upload,
                    acquisition.id,
                    self.owner_instance_id,
                    acquisition.finalization_generation,
                    checksum,
                    received,
                    deadline=deadline,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._fail_attempt(acquisition, "upload_timeout")
                    raise PdfImportError(
                        "upload_timeout", 408, "PDF upload exceeded its total time limit."
                    )
                result = await self._finalize_with_deadline(
                    acquisition.id, False, remaining, acquisition
                )
                if result.state == "completed":
                    await loop.run_in_executor(
                        self._file_executor,
                        self._journal_and_remove,
                        result.id,
                        result.staging_relative_path,
                        "staging",
                        result.input_sha256,
                        result.input_size_bytes,
                    )
                return result
            except PdfImportError as error:
                if stage_file is not None:
                    await loop.run_in_executor(self._file_executor, stage_file.close)
                    stage_file = None
                preserve_complete_input = (
                    acquisition.input_complete_at is not None
                    and error.code
                    in (
                        "file_cleanup_pending",
                        "import_busy",
                        "finalization_conflict",
                        "upload_timeout",
                    )
                )
                if not preserve_complete_input:
                    await loop.run_in_executor(
                        self._file_executor, self._fail_attempt, acquisition, "upload_failed"
                    )
                    await loop.run_in_executor(
                        self._file_executor,
                        self._journal_and_remove,
                        acquisition.id,
                        acquisition.staging_relative_path,
                        "staging",
                        acquisition.input_sha256,
                        acquisition.input_size_bytes,
                    )
                raise
            except (OSError, PersistenceError, ConflictingWriteError) as error:
                if stage_file is not None:
                    await loop.run_in_executor(self._file_executor, stage_file.close)
                    stage_file = None
                await loop.run_in_executor(
                    self._file_executor, self._fail_attempt, acquisition, "disk_or_store_failure"
                )
                await loop.run_in_executor(
                    self._file_executor,
                    self._journal_and_remove,
                    acquisition.id,
                    acquisition.staging_relative_path,
                    "staging",
                    acquisition.input_sha256,
                    acquisition.input_size_bytes,
                )
                logger.warning("PDF intake failed (%s)", type(error).__name__)
                raise PdfImportError(
                    "pdf_storage_failed", 500, "Corpi could not safely save this PDF."
                ) from error
            except ClientDisconnect as error:
                await loop.run_in_executor(
                    self._file_executor, self._fail_attempt, acquisition, "client_disconnected"
                )
                await loop.run_in_executor(
                    self._file_executor,
                    self._journal_and_remove,
                    acquisition.id,
                    acquisition.staging_relative_path,
                    "staging",
                    acquisition.input_sha256,
                    acquisition.input_size_bytes,
                )
                raise PdfImportError(
                    "upload_interrupted", 400, "PDF transfer ended before all bytes arrived."
                ) from error
            except asyncio.CancelledError:
                if stage_file is not None:
                    await asyncio.shield(
                        loop.run_in_executor(self._file_executor, stage_file.close)
                    )
                    stage_file = None
                await asyncio.shield(
                    loop.run_in_executor(
                        self._file_executor, self._fail_attempt, acquisition, "upload_interrupted"
                    )
                )
                await asyncio.shield(
                    loop.run_in_executor(
                        self._file_executor,
                        self._journal_and_remove,
                        acquisition.id,
                        acquisition.staging_relative_path,
                        "staging",
                        acquisition.input_sha256,
                        acquisition.input_size_bytes,
                    )
                )
                raise
            finally:
                if stage_file is not None:
                    await asyncio.shield(
                        loop.run_in_executor(self._file_executor, stage_file.close)
                    )
                with self._active_paths_lock:
                    self._active_paths.discard(acquisition.staging_relative_path)
                await asyncio.shield(
                    loop.run_in_executor(self._file_executor, self.retry_recorded_cleanup)
                )

    def retry(self, acquisition_id: str) -> PdfAcquisition:
        acquisition = self.store.load_pdf_acquisition(acquisition_id)
        if acquisition.state == "completed":
            return acquisition
        if acquisition.input_complete_at is None:
            raise PdfImportError(
                "upload_required", 409, "Select the PDF again to resume this import."
            )
        recovery = True
        if acquisition.state == "failed":
            try:
                acquisition = self.store.resume_failed_pdf_acquisition(
                    acquisition_id, self.owner_instance_id
                )
            except PersistenceError as error:
                raise PdfImportError("retry_unavailable", 409, str(error)) from error
            recovery = False
        result = self._finalize(acquisition_id, recovery, self.limits.upload_total_seconds)
        if result.state == "completed":
            self._journal_and_remove(
                result.id,
                result.staging_relative_path,
                "staging",
                result.input_sha256,
                result.input_size_bytes,
            )
        return result

    def cancel(self, acquisition_id: str) -> PdfAcquisition:
        acquisition = self.store.load_pdf_acquisition(acquisition_id)
        if acquisition.state in ("completed", "cancelled"):
            return acquisition
        result = self.store.fail_pdf_acquisition(
            acquisition.id,
            acquisition.owner_instance_id,
            acquisition.finalization_generation,
            "cancelled",
            cancelled=True,
        )
        self._journal_and_remove(
            result.id,
            result.staging_relative_path,
            "staging",
            result.input_sha256,
            result.input_size_bytes,
        )
        if result.planned_relative_path is not None:
            self._journal_and_remove(
                result.id,
                result.planned_relative_path,
                "planned_original",
                result.planned_sha256,
                result.planned_size_bytes,
            )
        return result

    def read_verified_import(self, import_id: str) -> tuple[bytes, str]:
        try:
            item = self.store.load_pdf_import(import_id)
            managed = self.store.load_import_managed_file(import_id)
            if self.store.managed_file_is_removing(managed.id):
                raise PdfImportError("file_cleanup_pending", 409, "This PDF is being removed.")
            with self.fingerprint_locks.hold(managed.sha256, managed.size_bytes, timeout=5.0):
                if self.store.managed_file_is_removing(managed.id):
                    raise PdfImportError("file_cleanup_pending", 409, "This PDF is being removed.")
                content = self.files.read_verified(
                    managed.relative_path, managed.sha256, managed.size_bytes
                )
            return content, item.display_filename
        except TimeoutError as error:
            raise PdfImportError(
                "pdf_busy", 409, "Corpi is currently finalizing or removing this PDF."
            ) from error
        except PdfImportError:
            raise
        except ManagedFileIntegrityError as error:
            raise PdfImportError("integrity_failure", 409, str(error)) from error
        except ManagedFilePathError as error:
            raise PdfImportError("integrity_failure", 409, str(error)) from error
        except PersistenceError as error:
            raise PdfImportError("pdf_unavailable", 404, "The saved PDF is unavailable.") from error

    def storage_report(self) -> dict[str, int]:
        with self.store.database.connect() as connection:
            originals = connection.execute(
                """SELECT file.relative_path,
                          CASE WHEN removal.managed_file_id IS NOT NULL
                                    AND removal.removed_at IS NULL THEN 1 ELSE 0 END AS pending
                   FROM managed_files AS file
                   LEFT JOIN managed_file_removals AS removal
                     ON removal.managed_file_id=file.id"""
            ).fetchall()
            staging_paths = {
                row[0]
                for row in connection.execute(
                    """SELECT staging_relative_path FROM pdf_acquisitions
                       WHERE state IN ('receiving','staged','finalizing','waiting_cleanup')
                          OR (state='failed' AND input_complete_at IS NOT NULL
                              AND failure_code NOT IN (
                                  'integrity_failure','integrity_conflict','source_deleted'
                              ))"""
                )
            }
            staging_paths.update(
                row[0]
                for row in connection.execute(
                    """SELECT relative_path FROM acquisition_file_cleanup
                       WHERE file_kind='staging' AND removed_at IS NULL"""
                )
            )
        original_count = 0
        original_bytes = 0
        bytes_pending_removal = 0
        for row in originals:
            try:
                actual_size = self.files.stat_size(row["relative_path"])
            except PersistenceError as error:
                logger.warning(
                    "Managed original omitted from storage report (%s)", type(error).__name__
                )
                continue
            if actual_size is None:
                continue
            original_count += 1
            original_bytes += actual_size
            if row["pending"]:
                bytes_pending_removal += actual_size
        retained_staging = 0
        for relative_path in staging_paths:
            try:
                actual_size = self.files.stat_size(relative_path)
            except PersistenceError as error:
                logger.warning(
                    "Staging file omitted from storage report (%s)", type(error).__name__
                )
                continue
            if actual_size is not None:
                retained_staging += actual_size
        return {
            "original_file_count": original_count,
            "original_bytes": original_bytes,
            "bytes_pending_removal": bytes_pending_removal,
            "retained_staging_bytes": retained_staging,
            "max_upload_bytes": self.limits.max_upload_bytes,
            "rendered_page_pixel_limit": self.limits.rendered_page_pixels,
        }

    def reconcile_startup(self) -> None:
        for acquisition in self.store.list_recoverable_pdf_acquisitions():
            try:
                result = self._finalize(acquisition.id, True, self.limits.upload_total_seconds)
                if result.state == "completed":
                    self._journal_and_remove(
                        result.id,
                        result.staging_relative_path,
                        "staging",
                        result.input_sha256,
                        result.input_size_bytes,
                    )
            except PdfImportError:
                continue
            except Exception as error:  # noqa: BLE001 - recorded work remains available for retry
                logger.warning("PDF acquisition recovery deferred (%s)", type(error).__name__)
        for acquisition in self._interrupted_uploads():
            self._fail_attempt(acquisition, "upload_interrupted")
            self._journal_and_remove(
                acquisition.id,
                acquisition.staging_relative_path,
                "staging",
                acquisition.input_sha256,
                acquisition.input_size_bytes,
            )
        self.retry_recorded_cleanup()

    def retry_recorded_cleanup(self) -> None:
        for (
            cleanup_id,
            acquisition_id,
            path,
            _file_kind,
            checksum,
            size,
        ) in self.store.list_pending_acquisition_file_cleanup():
            try:
                with self._active_paths_lock:
                    if path in self._active_paths:
                        continue
                # An active transfer owns its staging file until the request exits.
                acquisition = self.store.load_pdf_acquisition(acquisition_id)
                if acquisition.state == "receiving":
                    continue
                if checksum is not None and size is not None:
                    with self.fingerprint_locks.hold(checksum, size, timeout=0.25):
                        self.files.remove_recorded(path, checksum, size)
                        self.store.mark_acquisition_file_removed(cleanup_id)
                else:
                    self.files.remove_recorded(path, checksum, size)
                    self.store.mark_acquisition_file_removed(cleanup_id)
            except TimeoutError:
                continue
            except OSError as error:
                self.store.record_acquisition_file_cleanup_failure(cleanup_id, type(error).__name__)
            except PersistenceError as error:
                self.store.record_acquisition_file_cleanup_failure(cleanup_id, type(error).__name__)

    def maintenance(self) -> None:
        self.reconcile_startup()
        self.purge_expired_pdf_history()
        cutoff = datetime.now(UTC) - timedelta(seconds=self.limits.abandoned_staging_seconds)
        for acquisition in self._abandoned_uploads(cutoff):
            self._journal_and_remove(
                acquisition.id,
                acquisition.staging_relative_path,
                "staging",
                acquisition.input_sha256,
                acquisition.input_size_bytes,
            )

    def purge_expired_pdf_history(self) -> None:
        for expired in self.store.list_expired_pdf_import_trash(utc_timestamp()):
            try:
                managed = self.store.load_import_managed_file(expired.import_id)
                lock = self.fingerprint_locks.lock_for(managed.sha256, managed.size_bytes)
                if not lock.acquire(timeout=0.25):
                    continue
                try:
                    if expired.file_removed_at is None:
                        existing = self.store.load_managed_file_removal(managed.id)
                        if existing is None:
                            if self.store.has_live_pdf_reuse_hold(managed.id):
                                continue
                            removal = self.store.reserve_pdf_import_file_removal(
                                managed.id, managed.relative_path, expired.import_id
                            )
                            if removal is None:
                                self.store.mark_pdf_import_file_cleanup_complete(expired.import_id)
                            else:
                                existing = removal
                        if existing is not None and existing.removed_at is None:
                            try:
                                self.files.remove_recorded(managed.relative_path)
                            except (OSError, PersistenceError) as error:
                                self.store.record_managed_file_removal_failure(
                                    managed.id, type(error).__name__
                                )
                                continue
                            self.store.mark_managed_file_removed(managed.id)
                        elif existing is not None:
                            self.store.mark_pdf_import_file_cleanup_complete(expired.import_id)
                    if expired.snapshots_removed_at is None:
                        if not self._remove_migration_snapshots():
                            continue
                        self.store.mark_pdf_import_snapshots_removed(expired.import_id)
                    self.store.remove_trashed_pdf_import_records(expired.import_id)
                finally:
                    lock.release()
            except (PersistenceError, ConflictingWriteError) as error:
                logger.info("PDF history purge remains pending (%s)", type(error).__name__)

    def _remove_migration_snapshots(self) -> bool:
        snapshot_dir = self.store.database.config.snapshot_dir
        if not snapshot_dir.is_dir():
            return True
        complete = True
        for snapshot in snapshot_dir.glob("snapshot-r*.sqlite3*"):
            try:
                snapshot.unlink(missing_ok=True)
            except OSError:
                complete = False
        return complete

    def _interrupted_uploads(self) -> tuple[PdfAcquisition, ...]:
        with self.store.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pdf_acquisitions WHERE state='receiving' ORDER BY created_at,id"
            ).fetchall()
            from core.persistence.store import _pdf_acquisition

            return tuple(_pdf_acquisition(row) for row in rows)

    def _abandoned_uploads(self, cutoff: datetime) -> tuple[PdfAcquisition, ...]:
        stamp = cutoff.isoformat(timespec="microseconds").replace("+00:00", "Z")
        with self.store.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM pdf_acquisitions
                   WHERE (
                       state IN ('cancelled','completed','awaiting_upload')
                       OR (state='failed' AND (
                           input_complete_at IS NULL OR failure_code IN (
                               'integrity_failure','integrity_conflict','source_deleted',
                               'client_disconnected','upload_failed'
                           )
                       ))
                   )
                     AND last_write_at<? ORDER BY last_write_at,id""",
                (stamp,),
            ).fetchall()
            from core.persistence.store import _pdf_acquisition

            return tuple(_pdf_acquisition(row) for row in rows)

    def _begin_upload(self, acquisition_id: str) -> PdfAcquisition:
        try:
            acquisition = self.store.begin_pdf_upload(acquisition_id, self.owner_instance_id)
            return acquisition
        except PersistenceError as error:
            raise PdfImportError("invalid_acquisition", 409, str(error)) from error

    def _size_limit_error(self) -> PdfImportError:
        limit_mb = self.limits.max_upload_bytes / 1_000_000
        limit_label = (
            f"{limit_mb:g} MB" if limit_mb >= 1 else f"{self.limits.max_upload_bytes} bytes"
        )
        return PdfImportError(
            "pdf_too_large", 413, f"PDF exceeds Corpi's configured {limit_label} import limit."
        )

    async def _before_deadline(
        self,
        loop: asyncio.AbstractEventLoop,
        operation: Callable[..., _T],
        *args: object,
        deadline: float,
    ) -> _T:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PdfImportError("upload_timeout", 408, "PDF upload exceeded its total time limit.")
        future = loop.run_in_executor(self._file_executor, operation, *args)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        except TimeoutError as error:
            future.add_done_callback(_consume_background_result)
            raise PdfImportError(
                "upload_timeout", 408, "PDF upload exceeded its total time limit."
            ) from error

    async def _finalize_with_deadline(
        self,
        acquisition_id: str,
        recovery: bool,
        remaining: float,
        acquisition: PdfAcquisition,
    ) -> PdfAcquisition:
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(
            self._finalization_executor, self._finalize, acquisition_id, recovery, remaining
        )
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=max(0.001, remaining))
        except TimeoutError as error:
            current = await asyncio.to_thread(self.store.load_pdf_acquisition, acquisition.id)
            try:
                expired = await asyncio.to_thread(
                    self.store.expire_pdf_finalization,
                    current.id,
                    current.owner_instance_id,
                    current.finalization_generation,
                )
            except PersistenceError:
                expired = await asyncio.to_thread(self.store.load_pdf_acquisition, acquisition.id)
            if expired.state == "completed":
                worker.add_done_callback(_consume_background_result)
                return expired
            worker.add_done_callback(_consume_background_result)
            raise PdfImportError(
                "upload_timeout",
                408,
                "PDF finalization exceeded the total transfer time limit. "
                "The saved input can be retried.",
            ) from error

    async def _verify_replayed_upload(
        self, acquisition: PdfAcquisition, request: Request
    ) -> PdfAcquisition:
        checksum = hashlib.sha256()
        received = 0
        signature = bytearray()
        deadline = time.monotonic() + self.limits.upload_total_seconds
        stream = request.stream().__aiter__()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PdfImportError("upload_timeout", 408, "PDF replay exceeded its time limit.")
            try:
                chunk = await asyncio.wait_for(
                    anext(stream), min(remaining, self.limits.upload_idle_seconds)
                )
            except StopAsyncIteration:
                break
            except TimeoutError as error:
                raise PdfImportError(
                    "upload_idle_timeout", 408, "PDF replay stopped sending bytes for too long."
                ) from error
            if not chunk:
                continue
            received += len(chunk)
            if received > self.limits.max_upload_bytes:
                raise self._size_limit_error()
            if len(signature) < 1024:
                signature.extend(chunk[: 1024 - len(signature)])
            checksum.update(chunk)
        if b"%PDF-" not in signature:
            raise PdfImportError(
                "invalid_pdf", 422, "The selected file does not contain a PDF header."
            )
        if (
            received != acquisition.input_size_bytes
            or checksum.hexdigest() != acquisition.input_sha256
        ):
            raise PdfImportError(
                "retry_key_conflict", 409, "This retry key already identifies different PDF bytes."
            )
        return acquisition

    def _fail_attempt(self, acquisition: PdfAcquisition, code: str) -> None:
        try:
            failed = self.store.fail_pdf_acquisition(
                acquisition.id,
                acquisition.owner_instance_id,
                acquisition.finalization_generation,
                code,
            )
            if failed.planned_relative_path is not None:
                self._journal_and_remove(
                    failed.id,
                    failed.planned_relative_path,
                    "planned_original",
                    failed.planned_sha256,
                    failed.planned_size_bytes,
                )
        except PersistenceError:
            # A newer attempt or a committed outcome owns the state now.
            return

    def _journal_and_remove(
        self,
        acquisition_id: str,
        relative_path: str,
        file_kind: str,
        checksum: str | None,
        size_bytes: int | None,
    ) -> None:
        self.store.record_acquisition_file_cleanup(
            acquisition_id, relative_path, file_kind, checksum, size_bytes
        )
        with self._active_paths_lock:
            if relative_path in self._active_paths:
                return
        self.retry_recorded_cleanup()

    def _finalize(self, acquisition_id: str, recovery: bool, timeout: float) -> PdfAcquisition:
        acquisition = self.store.load_pdf_acquisition(acquisition_id)
        self._track_path(acquisition.staging_relative_path, True)
        if acquisition.planned_relative_path is not None:
            self._track_path(acquisition.planned_relative_path, True)
        try:
            return self._finalize_inner(acquisition_id, recovery, timeout)
        finally:
            try:
                current = self.store.load_pdf_acquisition(acquisition_id)
            except PersistenceError:
                current = acquisition
            self._track_path(acquisition.staging_relative_path, False)
            if current.planned_relative_path is not None:
                self._track_path(current.planned_relative_path, False)
            try:
                self.retry_recorded_cleanup()
            except Exception as error:  # noqa: BLE001 - keep the durable journal for maintenance
                logger.warning("PDF cleanup remains recorded (%s)", type(error).__name__)

    def _finalize_inner(
        self, acquisition_id: str, recovery: bool, timeout: float
    ) -> PdfAcquisition:
        acquisition = self.store.claim_pdf_finalization(
            acquisition_id, self.owner_instance_id, recovery=recovery
        )
        checksum = acquisition.input_sha256
        size_bytes = acquisition.input_size_bytes
        if checksum is None or size_bytes is None:
            raise PdfImportError(
                "upload_required", 409, "Select the PDF again to resume this import."
            )
        staging = self.files.read_verified(acquisition.staging_relative_path, checksum, size_bytes)
        remaining = max(0.0, timeout)
        registered = self.store.find_registry_managed_file(checksum, size_bytes)
        if registered is not None and self.store.managed_file_is_removing(registered.id):
            self._defer_for_file_cleanup(acquisition)
        if registered is not None and not self.store.canonical_pdf_has_eligible_owner(
            checksum, size_bytes, utc_timestamp()
        ):
            self._defer_for_file_cleanup(acquisition)
        if registered is None:
            candidates = self.store.list_legacy_pdf_candidates(
                checksum, size_bytes, utc_timestamp()
            )
            if not candidates and self.store.has_removal_marked_pdf_fingerprint(
                checksum, size_bytes
            ):
                self._defer_for_file_cleanup(acquisition)
        try:
            with self.fingerprint_locks.hold(checksum, size_bytes, remaining):
                return self._finalize_locked(acquisition, staging, registered, remaining)
        except TimeoutError as error:
            canonical = self.store.find_registry_managed_file(checksum, size_bytes)
            if canonical is not None and self.store.managed_file_is_removing(canonical.id):
                self._defer_for_file_cleanup(acquisition)
            if (
                canonical is None
                and not self.store.list_legacy_pdf_candidates(checksum, size_bytes, utc_timestamp())
                and self.store.has_removal_marked_pdf_fingerprint(checksum, size_bytes)
            ):
                self._defer_for_file_cleanup(acquisition)
            raise PdfImportError(
                "import_busy", 409, "Another operation is finalizing this PDF."
            ) from error
        except ManagedFileIntegrityError as error:
            self._fail_attempt(acquisition, "integrity_failure")
            raise PdfImportError("integrity_failure", 409, str(error)) from error
        except PdfImportError:
            raise
        except (PersistenceError, ConflictingWriteError) as error:
            self._fail_attempt(acquisition, "finalization_conflict")
            raise PdfImportError("finalization_conflict", 409, str(error)) from error

    def _finalize_locked(
        self,
        acquisition: PdfAcquisition,
        staging: bytes,
        registered: ManagedFile | None,
        remaining: float,
    ) -> PdfAcquisition:
        checksum = acquisition.input_sha256
        size_bytes = acquisition.input_size_bytes
        assert checksum is not None and size_bytes is not None
        active = self.store.find_active_notebook_pdf(acquisition.notebook_id, checksum, size_bytes)
        if active is not None:
            source, item, managed = active
            if acquisition.intent == "add":
                self._verify_reuse(acquisition, managed, staging)
                return self.store.finish_pdf_acquisition(
                    acquisition.id,
                    self.owner_instance_id,
                    acquisition.finalization_generation,
                    "duplicate",
                    existing_source_id=source.id,
                    existing_import_id=item.id,
                )
        historical = self.store.find_notebook_historical_pdf(
            acquisition.notebook_id, checksum, size_bytes, utc_timestamp()
        )
        if acquisition.intent == "add" and historical is not None:
            source, item, trashed = historical
            raise PdfImportError(
                "pdf_in_history",
                409,
                f"This PDF is already in the history for source {source.id} (import {item.id}); "
                + ("it is in trash." if trashed else "open that earlier import or use Replace."),
            )
        if acquisition.intent == "replace":
            state = self.store.load_source_pdf_state(acquisition.target_source_id or "")
            if state is not None and state.current_import_id is not None:
                current_file = self.store.load_import_managed_file(state.current_import_id)
                if current_file.sha256 == checksum and current_file.size_bytes == size_bytes:
                    self._verify_reuse(acquisition, current_file, staging)
                    return self.store.finish_pdf_acquisition(
                        acquisition.id,
                        self.owner_instance_id,
                        acquisition.finalization_generation,
                        "unchanged",
                        existing_source_id=acquisition.target_source_id,
                        existing_import_id=state.current_import_id,
                    )
        if registered is None:
            candidates = self.store.list_legacy_pdf_candidates(
                checksum, size_bytes, utc_timestamp()
            )
            if candidates:
                for candidate in candidates:
                    if self.store.managed_file_is_removing(candidate.id):
                        continue
                    try:
                        self._verify_reuse(acquisition, candidate, staging)
                        registered = candidate
                        break
                    except (ManagedFileIntegrityError, OSError, PersistenceError):
                        self.store.clear_pdf_reuse_hold(
                            acquisition.id,
                            self.owner_instance_id,
                            acquisition.finalization_generation,
                        )
                if registered is None:
                    self._fail_attempt(acquisition, "integrity_failure")
                    raise PdfImportError(
                        "integrity_failure",
                        409,
                        "Matching older stored PDFs failed integrity checks; "
                        "Corpi left them untouched.",
                    )
            elif self.store.has_removal_marked_pdf_fingerprint(checksum, size_bytes):
                self._defer_for_file_cleanup(acquisition)
        if registered is not None:
            if self.store.managed_file_is_removing(registered.id):
                self.store.mark_pdf_acquisition_waiting_cleanup(
                    acquisition.id, self.owner_instance_id, acquisition.finalization_generation
                )
                raise PdfImportError(
                    "file_cleanup_pending",
                    409,
                    FILE_CLEANUP_PENDING_MESSAGE,
                )
            self._verify_reuse(acquisition, registered, staging)
            managed_file = None
        else:
            file_id = acquisition.planned_managed_file_id or new_id()
            relative_path = acquisition.planned_relative_path or f"files/objects/{file_id}.pdf"
            acquisition = self.store.set_pdf_placement_plan(
                acquisition.id,
                self.owner_instance_id,
                acquisition.finalization_generation,
                file_id,
                relative_path,
                checksum,
                size_bytes,
            )
            self._track_path(relative_path, True)
            final_path = self.files.path_for(relative_path, create_parents=True, allow_missing=True)
            if final_path.exists():
                existing_bytes = self.files.read_verified(relative_path, checksum, size_bytes)
                if existing_bytes != staging:
                    raise PdfImportError(
                        "integrity_conflict",
                        409,
                        "Planned PDF destination contains different bytes.",
                    )
            else:
                self.files.place_staging(acquisition.staging_relative_path, relative_path)
            acquisition = self.store.mark_pdf_placement_complete(
                acquisition.id, self.owner_instance_id, acquisition.finalization_generation
            )
            self._track_path(relative_path, False)
            managed_file = ManagedFile(
                file_id, relative_path, size_bytes, checksum, utc_timestamp()
            )
        source_version_id = new_id()
        import_id = new_id()
        result = self.store.finish_pdf_acquisition(
            acquisition.id,
            self.owner_instance_id,
            acquisition.finalization_generation,
            "replaced" if acquisition.intent == "replace" else "created",
            managed_file=managed_file,
            new_source_id=new_id() if acquisition.intent == "add" else None,
            source_version_id=source_version_id,
            import_id=import_id,
            pending_replacement=acquisition.intent == "replace",
        )
        if self.notify_jobs is not None:
            try:
                self.notify_jobs()
            except RuntimeError:
                logger.warning("PDF processing job was saved but the runner wakeup was unavailable")
        return result

    def _verify_reuse(
        self, acquisition: PdfAcquisition, managed: ManagedFile, incoming: bytes
    ) -> None:
        self.store.set_pdf_reuse_hold(
            acquisition.id, self.owner_instance_id, acquisition.finalization_generation, managed.id
        )
        content = self.files.read_verified(
            managed.relative_path, managed.sha256, managed.size_bytes
        )
        if content != incoming:
            raise ManagedFileIntegrityError("matching PDF fingerprint contains different bytes")

    def _defer_for_file_cleanup(self, acquisition: PdfAcquisition) -> None:
        self.store.mark_pdf_acquisition_waiting_cleanup(
            acquisition.id, self.owner_instance_id, acquisition.finalization_generation
        )
        raise PdfImportError("file_cleanup_pending", 409, FILE_CLEANUP_PENDING_MESSAGE)

    def _track_path(self, path: str, active: bool) -> None:
        with self._active_paths_lock:
            if active:
                self._active_paths.add(path)
            else:
                self._active_paths.discard(path)


def _consume_background_result[T](future: asyncio.Future[T]) -> None:
    try:
        future.result()
    except BaseException:  # the timed-out request already reported the outcome
        return
