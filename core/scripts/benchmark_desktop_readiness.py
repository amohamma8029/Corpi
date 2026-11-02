"""Original PDFs through the native UI, with readonly disk and owned-process evidence.

Windows embedded driver only. File/select events are synthetic, not physical input.
Run after the normal embedded native build; no downloads, setup or app-data deletion.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import closing
from ctypes import wintypes as wt
from datetime import datetime
from pathlib import Path

from core.ingest.files import ManagedPdfFiles
from core.persistence.database import Database, DatabaseConfig
from core.search.handoff import SUPPORTED_PROFILE_IDS
from core.search.memory import resident_bytes
from core.search.profiles import profiles
from core.search.storage import verify_generation
from core.search.supervisor import stop_owned_process_tree

REPO = Path(__file__).resolve().parents[2]
DEFINITIONS = REPO / "core/benchmarks/search_readiness.json"
CURRENT_RECIPE = "app-preparation-1200-batch-v1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_source(source_id: str, app_data: Path) -> dict:
    database = Database(DatabaseConfig(app_data))
    with closing(database._connect(read_only=True)) as connection:
        source = dict(
            connection.execute(
                "SELECT v.id,v.content_sha256,m.relative_path,m.size_bytes FROM source_versions v "
                "JOIN managed_files m ON m.id=v.managed_file_id WHERE v.source_id=?",
                (source_id,),
            ).fetchone()
        )
        original = ManagedPdfFiles(app_data).read_verified(
            source["relative_path"], source["content_sha256"], source["size_bytes"]
        )
        source["verified_original_sha256"] = hashlib.sha256(original).hexdigest()
        jobs = []
        for row in connection.execute(
            "SELECT j.*,s.status,s.failure_code FROM jobs j JOIN job_status s ON s.job_id=j.id "
            "WHERE j.source_id=? ORDER BY j.created_at,j.rowid",
            (source_id,),
        ).fetchall():
            item = dict(row)
            item["steps"] = [
                dict(step)
                for step in connection.execute(
                    "SELECT step_key,checkpoint_json,output_extraction_version_id,"
                    "output_index_version_id "
                    "FROM job_steps WHERE job_id=? ORDER BY rowid",
                    (row["id"],),
                )
            ]
            item["attempts"] = [
                dict(attempt)
                for attempt in connection.execute(
                    "SELECT step_key,attempt_number,started_at,finished_at,failure_reason "
                    "FROM job_step_attempts WHERE job_id=? ORDER BY started_at",
                    (row["id"],),
                )
            ]
            for step in item["steps"]:
                step["checkpoint"] = json.loads(step.pop("checkpoint_json"))
            if item["attempts"]:
                item["queue_seconds"] = (
                    datetime.fromisoformat(item["attempts"][0]["started_at"])
                    - datetime.fromisoformat(item["created_at"])
                ).total_seconds()
            jobs.append(item)
        extractions = [
            dict(row)
            for row in connection.execute(
                "SELECT e.id,o.outcome FROM extraction_versions e JOIN extraction_outcomes o "
                "ON o.extraction_version_id=e.id WHERE e.source_id=?",
                (source_id,),
            )
        ]
        for extraction in extractions:
            extraction["reviews"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT state,generation FROM extraction_reviews WHERE extraction_version_id=?",
                    (extraction["id"],),
                )
            ]
            extraction["pages"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT page_number,classification,extracted_chars FROM extraction_pages "
                    "WHERE extraction_version_id=? ORDER BY page_number",
                    (extraction["id"],),
                )
            ]
            extraction["eligible_utf8_bytes"] = sum(
                len(row[0].encode("utf-8"))
                for row in connection.execute(
                    "SELECT p.text FROM passages p JOIN extraction_pages ep "
                    "ON ep.extraction_version_id=p.extraction_version_id "
                    "AND ep.source_id=p.source_id "
                    "AND ep.source_version_id=p.source_version_id "
                    "AND ep.page_number=json_extract(p.location_json,'$.page') "
                    "WHERE p.extraction_version_id=? AND ep.classification='extracted'",
                    (extraction["id"],),
                )
            )
        generations = []
        for row in connection.execute(
            "SELECT g.* FROM search_generations g JOIN index_versions i ON i.id=g.index_version_id "
            "WHERE i.source_id=?",
            (source_id,),
        ):
            generation = dict(row)
            generation["settings"] = json.loads(generation.pop("settings_json"))
            identity = generation["index_version_id"]
            generation["chunks"] = connection.execute(
                "SELECT count(*) FROM retrieval_chunks WHERE index_version_id=?",
                (identity,),
            ).fetchone()[0]
            generation["complete_batches"] = connection.execute(
                "SELECT count(*) FROM search_batches WHERE index_version_id=? AND complete=1",
                (identity,),
            ).fetchone()[0]
            generation["incomplete_batches"] = connection.execute(
                "SELECT count(*) FROM search_batches WHERE index_version_id=? AND complete=0",
                (identity,),
            ).fetchone()[0]
            for table in ("search_words", "search_vectors"):
                generation[table] = connection.execute(
                    f"SELECT count(*) FROM {table} WHERE rowid IN "
                    "(SELECT row_id FROM search_rows WHERE index_version_id=?)",
                    (identity,),
                ).fetchone()[0]
            generations.append(generation)
        readiness = [
            dict(row)
            for row in connection.execute(
                "SELECT r.*,i.index_version_id FROM source_readiness r "
                "JOIN source_readiness_indexes i ON i.source_id=r.source_id WHERE r.source_id=?",
                (source_id,),
            )
        ]
        for row in readiness:
            verify_generation(connection, row["index_version_id"])
        return {
            "source_id": source_id,
            "version": source,
            "jobs": jobs,
            "extractions": extractions,
            "generations": generations,
            "readiness": readiness,
        }


def expected_policy(choice: dict) -> dict:
    profile = choice["profile"]
    assert profile in SUPPORTED_PROFILE_IDS
    recipe = choice.get("preparation_recipe")
    assert recipe in (None, CURRENT_RECIPE)
    # Absent recipe identifies retained pre-adoption measurements, never new runs.
    assert recipe is not None or profile != "bekko-a8m-quint8-avx2"
    return {
        "batch_size": {"minilm-en": 32, "multilingual-e5": 8, "bekko-a8m-quint8-avx2": 1}[profile],
        "total_seconds": 1200 if recipe else 600,
        "budget_policy": "batch-reservation-v1" if recipe else None,
    }


def inspect_result(result: dict, fixture: dict, app_data: Path) -> list[dict]:
    expected = expected_policy(result["choice"])
    assert app_data.is_absolute()
    assert len(result["observations"]) == result["choice"]["repetitions"] > 0
    disk = []
    for ordinal, observation in enumerate(result["observations"], 1):
        assert observation.get("sourceId")
        assert observation["repetition"] == ordinal
        assert observation["fixture"] == result["choice"]["pdf"] == fixture["id"]
        assert observation["sha256"] == fixture["sha256"]
        assert observation["pages"] == fixture["pages"]
        assert observation["profile"] == result["choice"]["profile"]
        assert observation["cancellation"] == bool(result["choice"].get("cancel") and ordinal == 1)
        source = inspect_source(observation["sourceId"], app_data)
        assert source["version"]["verified_original_sha256"] == fixture["sha256"]
        if observation["readyMs"] is not None:
            assert len(source["extractions"]) == 1 and source["readiness"]
            assert all(
                row["source_version_id"] == source["version"]["id"] for row in source["readiness"]
            )
            for ready in source["readiness"]:
                publishers = [
                    job for job in source["jobs"] if job["id"] == ready["published_by_job_id"]
                ]
                assert len(publishers) == 1 and publishers[0]["status"] == "completed"
                assert {step["step_key"] for step in publishers[0]["steps"]} == {
                    "plan-search",
                    "embed-search",
                    "build-search",
                    "publish-search",
                }
            if source["extractions"][0].get("outcome") == "partial":
                assert source["extractions"][0]["reviews"]
                assert all(
                    row["state"] == "confirmed" for row in source["extractions"][0]["reviews"]
                )
            assert len(source["extractions"][0]["pages"]) == fixture["pages"]
            assert source["extractions"][0]["eligible_utf8_bytes"] <= 20_000_000
            assert len(source["generations"]) == 1
            generation = source["generations"][0]
            assert all(
                row["index_version_id"] == generation["index_version_id"]
                for row in source["readiness"]
            )
            assert generation["complete_batches"] > 0 and generation["incomplete_batches"] == 0
            assert generation["manifest_id"] == profiles()[result["choice"]["profile"]].identity
            assert 0 < generation["chunks"] <= 10000
            assert (
                generation["chunks"] == generation["search_words"] == generation["search_vectors"]
            )
            settings = generation["settings"]
            assert settings["threads"] == 2 and settings["batch_size"] == expected["batch_size"]
            assert settings["total_seconds"] == expected["total_seconds"]
            assert settings.get("budget_policy") == expected["budget_policy"]
            assert settings["max_chunks"] == 10000 and settings["max_text_bytes"] == 20_000_000
            assert settings["batch_seconds"] == 120
            assert settings["build_seconds"] == 300 and settings["memory_bytes"] == 1073741824
            assert 0 <= generation["spent_seconds"] <= expected["total_seconds"]
        if observation["cancellation"]:
            assert not source["readiness"]
            jobs = [job for job in source["jobs"] if job["kind"] == "pdf-search"]
            assert jobs and all(job["status"] == "cancelled" for job in jobs)
        else:
            assert observation["readyMs"] is not None
        disk.append(source)
    return disk


def verify_binary(binary: Path, evidence: dict) -> None:
    assert "status" in evidence or "passed" in evidence
    assert evidence.get("status", "passed") == "passed"
    assert evidence.get("passed", True) is True
    digest = evidence.get("testBinarySha256", evidence.get("binary_sha256"))
    assert digest and sha(binary) == digest


class ProcessEntry(ct.Structure):
    _fields_ = [
        ("size", wt.DWORD),
        ("usage", wt.DWORD),
        ("pid", wt.DWORD),
        ("heap", ct.c_size_t),
        ("module", wt.DWORD),
        ("threads", wt.DWORD),
        ("parent", wt.DWORD),
        ("priority", wt.LONG),
        ("flags", wt.DWORD),
        ("name", wt.WCHAR * 260),
    ]


class HostSampler:
    def __init__(self, directory: Path):
        self.directory = directory
        self.stop = threading.Event()
        self.samples: list[dict] = []
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
        for name in ("Process32FirstW", "Process32NextW"):
            method = getattr(self.kernel, name)
            method.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
            method.restype = wt.BOOL
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]
        self.kernel.CloseHandle.restype = wt.BOOL
        self.kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.kernel.OpenProcess.restype = wt.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        self.kernel.WaitForSingleObject.restype = wt.DWORD
        self.handles: dict[int, int] = {}
        self.kernel.GetSystemTimes.argtypes = [ct.POINTER(wt.FILETIME)] * 3
        self.kernel.GetSystemTimes.restype = wt.BOOL
        self.thread = threading.Thread(target=self.run)

    def tree(self, root: int) -> list[int]:
        handle = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if not handle or handle == ct.c_void_p(-1).value:
            raise OSError("process snapshot unavailable")
        try:
            row = ProcessEntry()
            row.size = ct.sizeof(row)
            parents = {}
            more = self.kernel.Process32FirstW(handle, ct.byref(row))
            if not more:
                raise OSError("process enumeration unavailable")
            while more:
                parents[int(row.pid)] = int(row.parent)
                more = self.kernel.Process32NextW(handle, ct.byref(row))
            selected = {root} if root in parents else set()
            while True:
                children = {pid for pid, parent in parents.items() if parent in selected}
                if children <= selected:
                    return sorted(selected)
                selected |= children
        finally:
            self.kernel.CloseHandle(handle)

    def run(self):
        previous = None
        while not self.stop.is_set():
            row: dict[str, object] = {"wall_ms": time.time() * 1000}
            try:
                times = [wt.FILETIME() for _ in range(3)]
                assert self.kernel.GetSystemTimes(*(ct.byref(item) for item in times))
                ticks = [
                    int(item.dwHighDateTime) * 2**32 + int(item.dwLowDateTime) for item in times
                ]
                if previous is not None:
                    elapsed = sum(ticks[1:]) - sum(previous[1:])
                    row["host_cpu_busy_percent"] = (
                        100 * (1 - (ticks[0] - previous[0]) / elapsed) if elapsed else None
                    )
                previous = ticks
                owner_path = self.directory / "core-owner.json"
                if owner_path.exists():
                    owner = json.loads(owner_path.read_text("utf-8"))
                    pids = self.tree(owner["pid"])
                    for pid in pids:
                        if pid not in self.handles:
                            handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
                            if handle:
                                self.handles[pid] = handle
                    row["core_pid"] = owner["pid"]
                    row["core_rss_bytes"] = (
                        resident_bytes(owner["pid"]) if owner["pid"] in pids else None
                    )
                    row["descendants"] = [
                        {"pid": pid, "rss_bytes": resident_bytes(pid)}
                        for pid in pids
                        if pid != owner["pid"]
                    ]
            except (OSError, ValueError, AssertionError):
                row["probe_unavailable"] = True
            self.samples.append(row)
            self.stop.wait(0.25)

    def reap_observed(self) -> dict[str, bool]:
        results = {}
        for pid, handle in self.handles.items():
            try:
                results[str(pid)] = self.kernel.WaitForSingleObject(handle, 15000) == 0
            finally:
                self.kernel.CloseHandle(handle)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--profile", choices=SUPPORTED_PROFILE_IDS, required=True)
    parser.add_argument("--pdf-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--app-data", type=Path, required=True, help="Absolute separate test library"
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--binary-evidence", type=Path, help="Passing native receipt pinning the binary"
    )
    parser.add_argument(
        "--pnpm", type=Path, help="Explicit installed pnpm executable, never provisioned"
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Readonly recovery of saved UI evidence; never reruns imports",
    )
    args = parser.parse_args()
    if not args.app_data.is_absolute():
        raise ValueError("Use an absolute separate test-library path")
    if args.inspect:
        result = json.loads((args.directory / "result.json").read_text("utf-8"))
        fixture = next(
            row
            for row in json.loads(DEFINITIONS.read_text("utf-8"))["fixtures"]
            if row["id"] == result["choice"]["pdf"]
        )
        result["disk"] = inspect_result(result, fixture, args.app_data)
        result["inspection_source_sha256"] = sha(args.directory / "result.json")
        with (args.directory / "result.inspected.json").open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({"readonly_sources_verified": len(result["disk"]), "no_import_rerun": True})
        )
        return
    if os.name != "nt" or not args.directory.is_absolute() or args.directory.exists():
        raise ValueError("Use an absent absolute result directory on the Windows benchmark host")
    if not 1 <= args.repetitions <= 3 or (args.cancel and args.repetitions < 2):
        raise ValueError("Use 1–3 repetitions; cancellation requires following fresh work")
    fixture = next(
        row
        for row in json.loads(DEFINITIONS.read_text("utf-8"))["fixtures"]
        if row["id"] == args.pdf
    )
    if sha(args.pdf_directory / fixture["filename"]) != fixture["sha256"]:
        raise ValueError("Original PDF hash mismatch")
    receipt = args.binary_evidence or REPO / "desktop/e2e-results/embedded/run.json"
    binary = args.binary or REPO / "desktop/src-tauri/target/release/corpi-desktop.exe"
    if not binary.is_absolute() or not receipt.is_absolute():
        raise ValueError("Use absolute binary and native receipt paths")
    verify_binary(binary, json.loads(receipt.read_text("utf-8")))
    pnpm = str(args.pnpm) if args.pnpm else shutil.which("pnpm.exe")
    if pnpm is None:
        raise ValueError("Supply --pnpm with the existing installed executable")
    args.directory.mkdir(parents=True)
    choice = {
        "pdf": args.pdf,
        "profile": args.profile,
        "repetitions": args.repetitions,
        "cancel": args.cancel,
        "preparation_recipe": CURRENT_RECIPE,
    }
    env = {
        **os.environ,
        "CORPI_CORE_DIR": str(REPO / "core"),
        "CORPI_E2E_PROVIDER": "embedded",
        "CORPI_E2E_RESULTS": str(args.directory),
        "CORPI_RUN_MODEL_DOWNLOAD": "0",
        "CORPI_E2E_BINARY": str(binary),
        "CORPI_E2E_DATA_DIR": str(args.app_data),
        "WEBVIEW2_USER_DATA_FOLDER": str(args.directory / "webview"),
        "CORPI_READINESS_CASE": json.dumps(choice),
        "CORPI_READINESS_PDFS": str(args.pdf_directory),
    }
    command = [pnpm, "exec", "wdio", "run", "wdio.conf.ts", "--spec", "e2e/specs/readiness.e2e.ts"]
    sampler = HostSampler(args.directory)
    sampler.thread.start()
    began = time.monotonic()
    database_size_before = (args.app_data / "corpi.sqlite3").stat().st_size
    code = None
    controller_error = None
    try:
        with (args.directory / "console.txt").open("x", encoding="utf-8") as output:
            process = subprocess.Popen(
                command, cwd=REPO / "desktop", env=env, stdout=output, stderr=subprocess.STDOUT
            )
            try:
                code = process.wait(timeout=args.repetitions * 2700 + 180)
            finally:
                stop_owned_process_tree(process)
    except (OSError, subprocess.TimeoutExpired) as error:
        controller_error = type(error).__name__ + ": " + str(error)
    finally:
        sampler.stop.set()
        sampler.thread.join(timeout=10)
        assert not sampler.thread.is_alive()
    reaped = sampler.reap_observed()
    time.sleep(1)
    owner_file = args.directory / "core-owner.json"
    orphans = (
        sampler.tree(json.loads(owner_file.read_text("utf-8"))["pid"])
        if owner_file.exists()
        else None
    )
    observed = {int(row["core_pid"]) for row in sampler.samples if "core_pid" in row}
    observed.update(child["pid"] for row in sampler.samples for child in row.get("descendants", []))
    observed_survivors = sorted(pid for pid in observed if sampler.tree(pid))
    observations_path = args.directory / "readiness.json"
    observations = (
        json.loads(observations_path.read_text("utf-8")) if observations_path.exists() else []
    )
    result = {
        "choice": choice,
        "command": command,
        "exit_code": code,
        "controller_error": controller_error,
        "elapsed_seconds": time.monotonic() - began,
        "test_binary_sha256": sha(binary),
        "binary_evidence_sha256": sha(receipt),
        "app_data": str(args.app_data),
        "observed_handle_reaping": reaped,
        "core_tree_after_exit": orphans,
        "observed_processes_after_exit": observed_survivors,
        "database_bytes_before": database_size_before,
        "database_bytes_after": (args.app_data / "corpi.sqlite3").stat().st_size,
        "observations": observations,
        "disk": [],
        "samples": sampler.samples,
        "definitions_sha256": sha(DEFINITIONS),
        "spec_sha256": sha(REPO / "desktop/e2e/specs/readiness.e2e.ts"),
        "inspector_sha256": sha(Path(__file__)),
        "input_caveat": (
            "Synthetic File/select events; original PDF bytes, actual UI/backend. "
            "No OS file-picker or cold-cache claim."
        ),
    }
    (args.directory / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    # Save UI/process evidence first: an inspector failure must not erase or
    # force repeating the real import. The same immutable source IDs can be read again.
    result["disk"] = inspect_result(result, fixture, args.app_data)
    (args.directory / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"samples", "observations", "disk"}
            },
            indent=2,
        )
    )
    raise SystemExit(
        0
        if code == 0
        and controller_error is None
        and orphans == []
        and reaped
        and all(reaped.values())
        and not observed_survivors
        else 1
    )


if __name__ == "__main__":
    main()
