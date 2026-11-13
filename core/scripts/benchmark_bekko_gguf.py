"""Pinned temporary Bekko Q8_0 CPU experiment; no application/runtime adoption."""

import argparse
import ctypes as ct
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import statistics
import struct
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from benchmark_contextual_multilingual import profiles
from benchmark_static_multilingual import load_profile
from pin_search_models import fetch

from core.search.download import download_profile
from core.search.profiles import ModelFile, verify_installation
from core.search.supervisor import stop_owned_process_tree, worker_environment

if "--worker" in sys.argv:
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="1")


class ModelParams(ct.Structure):
    _fields_ = (
        [(name, ct.c_void_p) for name in ("devices", "tensor_buft_overrides")]
        + [
            (name, ct.c_int32)
            for name in ("n_gpu_layers", "split_mode", "load_mode", "lazy_mode", "main_gpu")
        ]
        + [
            (name, ct.c_void_p)
            for name in (
                "tensor_split",
                "progress_callback",
                "progress_callback_user_data",
                "kv_overrides",
            )
        ]
        + [
            (name, ct.c_bool)
            for name in (
                "vocab_only",
                "check_tensors",
                "use_extra_bufts",
                "no_host",
                "no_alloc",
                "load_mtp",
            )
        ]
    )


class ContextParams(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_uint32)
            for name in (
                "n_ctx",
                "n_batch",
                "n_ubatch",
                "n_seq_max",
                "n_rs_seq",
                "n_outputs_max",
                "n_outputs_max_per_seq",
            )
        ]
        + [
            (name, ct.c_int32)
            for name in (
                "n_threads",
                "n_threads_batch",
                "ctx_type",
                "rope_scaling_type",
                "pooling_type",
                "attention_type",
                "flash_attn_type",
            )
        ]
        + [
            (name, ct.c_float)
            for name in (
                "rope_freq_base",
                "rope_freq_scale",
                "yarn_ext_factor",
                "yarn_attn_factor",
                "yarn_beta_fast",
                "yarn_beta_slow",
            )
        ]
        + [
            ("yarn_orig_ctx", ct.c_uint32),
            ("defrag_thold", ct.c_float),
            ("cb_eval", ct.c_void_p),
            ("cb_eval_user_data", ct.c_void_p),
            ("type_k", ct.c_int32),
            ("type_v", ct.c_int32),
            ("abort_callback", ct.c_void_p),
            ("abort_callback_data", ct.c_void_p),
        ]
        + [
            (name, ct.c_bool)
            for name in (
                "embeddings",
                "offload_kqv",
                "no_perf",
                "op_offload",
                "swa_full",
                "kv_unified",
            )
        ]
        + [("samplers", ct.c_void_p), ("n_samplers", ct.c_size_t), ("ctx_other", ct.c_void_p)]
    )


class Batch(ct.Structure):
    _fields_ = [
        ("n_tokens", ct.c_int32),
        ("token", ct.POINTER(ct.c_int32)),
        ("embd", ct.POINTER(ct.c_float)),
        ("pos", ct.POINTER(ct.c_int32)),
        ("n_seq_id", ct.POINTER(ct.c_int32)),
        ("seq_id", ct.POINTER(ct.POINTER(ct.c_int32))),
        ("logits", ct.POINTER(ct.c_int8)),
    ]


def bind(library, name, result, arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


class GGufRuntime:
    root: Path | None = None
    metadata: Path | None = None
    physical_batch = 1

    def __init__(self, path, profile):
        from benchmark_contextual_multilingual import UTF8InputTokenizer

        self.profile = profile
        verify_installation(path, profile)
        if self.root is None:
            raise ValueError("owned runtime not configured")
        root = owned(self.root)
        if not json.loads((root / "source-equivalence.json").read_text("utf-8"))[
            "source_assets_identical"
        ]:
            raise ValueError("GGUF conversion source differs from reference")
        proof = json.loads((root / "model-inspection-corrected.json").read_text("utf-8"))
        if proof["manifest"] != profile.identity or not proof["token_id_bridge_eligible"]:
            raise ValueError("token ID bridge not verified")
        runtime = root / "runtime"
        inspection = json.loads((root / "runtime-inspection.json").read_text("utf-8"))
        for item in inspection["members"]:
            data = (runtime / item["path"]).read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("inspected native runtime changed")
        if (ct.sizeof(ModelParams), ct.sizeof(ContextParams), ct.sizeof(Batch)) != (80, 160, 56):
            raise ValueError("unsupported native ABI layout")
        self.tokenizer = UTF8InputTokenizer(path, profile)
        self._dll_directory = os.add_dll_directory(str(runtime))
        self.ggml = ct.CDLL(str(runtime / "ggml.dll"))
        self.ggml_base = ct.CDLL(str(runtime / "ggml-base.dll"))
        self.dll = ct.CDLL(str(runtime / "llama.dll"))
        self._log_callback = ct.CFUNCTYPE(None, ct.c_int, ct.c_char_p, ct.c_void_p)(lambda *_: None)
        bind(self.dll, "llama_log_set", None, [ct.c_void_p, ct.c_void_p])(
            ct.cast(self._log_callback, ct.c_void_p), None
        )
        bind(self.ggml, "ggml_backend_load", ct.c_void_p, [ct.c_char_p])(
            str(runtime / "ggml-cpu-haswell.dll").encode("utf-8")
        )
        bind(self.dll, "llama_backend_init", None, [])()
        count = bind(self.ggml, "ggml_backend_dev_count", ct.c_size_t, [])()
        get_device = bind(self.ggml, "ggml_backend_dev_get", ct.c_void_p, [ct.c_size_t])
        device_type = bind(self.ggml_base, "ggml_backend_dev_type", ct.c_int, [ct.c_void_p])
        devices = [get_device(index) for index in range(count)]
        if not devices or any(device_type(device) != 0 for device in devices):
            raise ValueError("runtime did not expose CPU-only devices")
        self._devices = (ct.c_void_p * 2)(devices[0], None)
        model_params = bind(self.dll, "llama_model_default_params", ModelParams, [])()
        model_params.devices = ct.cast(self._devices, ct.c_void_p)
        model_params.n_gpu_layers = 0
        self.model = bind(
            self.dll, "llama_model_load_from_file", ct.c_void_p, [ct.c_char_p, ModelParams]
        )(str(path / profile.model_file).encode("utf-8"), model_params)
        if not self.model:
            raise ValueError("native model load failed")
        params = bind(self.dll, "llama_context_default_params", ContextParams, [])()
        params.n_ctx = params.n_batch = params.n_ubatch = 512 * self.physical_batch
        params.n_seq_max = self.physical_batch
        params.n_threads = params.n_threads_batch = 2
        params.embeddings = True
        params.pooling_type = params.attention_type = 1  # mean, non-causal, inspected enums
        params.offload_kqv = params.op_offload = False
        self.context = bind(
            self.dll, "llama_init_from_model", ct.c_void_p, [ct.c_void_p, ContextParams]
        )(self.model, params)
        if not self.context:
            raise ValueError("native context load failed")
        properties = {
            name: bind(self.dll, name, ct.c_int32, [ct.c_void_p])(self.context)
            for name in (
                "llama_n_threads",
                "llama_n_threads_batch",
                "llama_pooling_type",
                "llama_n_ctx",
            )
        }
        if properties != {
            "llama_n_threads": 2,
            "llama_n_threads_batch": 2,
            "llama_pooling_type": 1,
            "llama_n_ctx": 512 * self.physical_batch,
        }:
            raise ValueError("native context limits differ")
        if bind(self.dll, "llama_get_causal_attn", ct.c_bool, [ct.c_void_p])(self.context):
            raise ValueError("native attention is causal")
        if bind(self.dll, "llama_model_n_embd", ct.c_int32, [ct.c_void_p])(self.model) != 384:
            raise ValueError("native dimension differs")
        self.batch = bind(
            self.dll, "llama_batch_init", Batch, [ct.c_int32, ct.c_int32, ct.c_int32]
        )(512 * self.physical_batch, 0, 1)
        encoder = bind(self.dll, "llama_model_has_encoder", ct.c_bool, [ct.c_void_p])(self.model)
        self.process = bind(
            self.dll,
            "llama_encode" if encoder else "llama_decode",
            ct.c_int32,
            [ct.c_void_p, Batch],
        )
        self.get_memory = bind(self.dll, "llama_get_memory", ct.c_void_p, [ct.c_void_p])
        self.clear_memory = bind(self.dll, "llama_memory_clear", None, [ct.c_void_p, ct.c_bool])
        self.get_embedding = bind(
            self.dll, "llama_get_embeddings_seq", ct.POINTER(ct.c_float), [ct.c_void_p, ct.c_int32]
        )
        if self.metadata:
            self.metadata.write_text(
                json.dumps(
                    {
                        "backend": "llama.cpp CPU Haswell",
                        "runtime_tag": "b11429",
                        "version": bind(self.dll, "llama_version", ct.c_char_p, [])().decode(),
                        "properties": properties,
                        "model_encoder_api": bool(encoder),
                        "tokenizer": "original HF IDs; explicit whitespace display aliases only",
                        "physical_passages_per_forward": self.physical_batch,
                        "native_batch_unit": "tokens",
                        "native_batch_tokens": 512 * self.physical_batch,
                        "gpu_layers": 0,
                        "causal_attention": False,
                        "python_network_denied": True,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def embed(self, texts, *, query=False):
        from core.search.runtime import vector_bytes

        if not 1 <= len(texts) <= 32:
            raise ValueError("invalid_embedding_batch")
        output = []
        prepared_ids = []
        for text in texts:
            self.tokenizer.prepare(text, query=query)
            ids = self.tokenizer.tokenizer.encode(text).ids
            if not ids or len(ids) > 512 or min(ids) < 0 or max(ids) >= 256000:
                raise ValueError("native_input_overflow_or_unknown_id")
            prepared_ids.append(ids)
        for offset in range(0, len(prepared_ids), self.physical_batch):
            sequences = prepared_ids[offset : offset + self.physical_batch]
            memory = self.get_memory(self.context)
            if memory:
                self.clear_memory(memory, True)
            self.batch.n_tokens = sum(len(ids) for ids in sequences)
            index = 0
            for identity, ids in enumerate(sequences):
                for position, token in enumerate(ids):
                    self.batch.token[index] = token
                    self.batch.pos[index] = position
                    self.batch.n_seq_id[index] = 1
                    self.batch.seq_id[index][0] = identity
                    self.batch.logits[index] = 1
                    index += 1
            if self.process(self.context, self.batch) != 0:
                raise ValueError("native_encoding_failed")
            for identity in range(len(sequences)):
                values = self.get_embedding(self.context, identity)
                if not values:
                    raise ValueError("native_embedding_missing")
                output.append(vector_bytes([values[index] for index in range(384)], 384))
        return output


def command(root, metadata, physical_batch=1):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--directory",
        str(root),
        "--metadata",
        str(metadata),
        "--native-passages",
        str(physical_batch),
    ]


def smoke(args):
    from benchmark_contextual_multilingual import UTF8InputTokenizer
    from benchmark_inference_options import agreement
    from benchmark_shared_search import session_command

    from core.search.supervisor import EmbeddingSession

    root = owned(args.directory)
    suffix = "-" + args.tag if args.tag else ""
    target = root / f"smoke{suffix}.json"
    if target.exists():
        raise ValueError("smoke result already exists")
    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("original dense input changed")
    paragraph = raw.decode("utf-8").split("\n\f\n")[0]
    probes = [
        "Rainfall and soil moisture were measured.",
        "L'expérience mesure les précipitations.",
        "تقيس التجربة هطول الأمطار ورطوبة التربة.",
        "  Leading  and   consecutive spaces\tremain.",
    ]
    rows, reference = [], None
    deadline = time.monotonic() + 600

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("GGUF diagnostic cumulative budget exhausted")
        return min(120, seconds)

    for backend in ("ort", "gguf"):
        profile = profiles()["bekko-a8m"] if backend == "ort" else load_profile(MANIFEST)
        chosen = (
            session_command(profile, root)
            if backend == "ort"
            else command(root, root / f"gguf-runtime-metadata{suffix}.json", args.native_passages)
        )
        session = None
        try:
            with EmbeddingSession(
                args.models_directory / profile.identity,
                profile,
                command=chosen,
                load_timeout=remaining(),
            ) as session:
                vectors = session.embed([paragraph] * 8, timeout=remaining())
                times = []
                for _ in range(8):
                    began = time.perf_counter()
                    vectors = session.embed([paragraph] * 8, timeout=remaining())
                    times.append(time.perf_counter() - began)
                control = (
                    vectors
                    + session.embed(probes, timeout=remaining())
                    + session.embed(probes, query=True, timeout=remaining())
                )
                reference = reference if reference is not None else control
                counter = UTF8InputTokenizer(args.models_directory / profile.identity, profile)
                row = {
                    "backend": backend,
                    "status": "completed",
                    "manifest": profile.identity,
                    "load_seconds": session.load_seconds,
                    "request_batch_passages": 8,
                    "batch_seconds": times,
                    "passages_per_second": 8 / statistics.median(times),
                    "child_peak_bytes": session.peak_bytes,
                    "vector_agreement": agreement(reference, control),
                    "content_tokens_original_paragraph": counter.count(paragraph),
                }
            row["launcher_reaped"] = session.process.poll() is not None
        except Exception as error:
            row = {
                "backend": backend,
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure": str(error),
                "child_peak_bytes": session.peak_bytes if session else None,
                "launcher_reaped": session.process.poll() is not None if session else True,
            }
        rows.append(row)
    result = {
        "scope": "paired CPU speed/output screen; not full quality/capacity",
        "input_sha256": source["input_sha256"],
        "results": rows,
    }
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def cpu_snapshot(pid):
    """Read only owned-process time and aggregate Windows CPU/power information."""
    from ctypes import wintypes

    class FileTime(ct.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        def seconds(self):
            return ((self.high << 32) | self.low) / 10000000

    class ProcessorPower(ct.Structure):
        _fields_ = [
            (name, wintypes.ULONG)
            for name in (
                "number",
                "max_mhz",
                "current_mhz",
                "limit_mhz",
                "max_idle",
                "current_idle",
            )
        ]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    opening = bind(
        kernel, "OpenProcess", wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    )
    handle = opening(0x1000, False, pid)
    if not handle:
        raise ct.WinError(ct.get_last_error())
    create, exit_time, user, system = FileTime(), FileTime(), FileTime(), FileTime()
    try:
        function = bind(
            kernel, "GetProcessTimes", wintypes.BOOL, [wintypes.HANDLE] + [ct.POINTER(FileTime)] * 4
        )
        if not function(
            handle, ct.byref(create), ct.byref(exit_time), ct.byref(system), ct.byref(user)
        ):
            raise ct.WinError(ct.get_last_error())
    finally:
        bind(kernel, "CloseHandle", wintypes.BOOL, [wintypes.HANDLE])(handle)
    idle, total_kernel, total_user = FileTime(), FileTime(), FileTime()
    function = bind(kernel, "GetSystemTimes", wintypes.BOOL, [ct.POINTER(FileTime)] * 3)
    if not function(ct.byref(idle), ct.byref(total_kernel), ct.byref(total_user)):
        raise ct.WinError(ct.get_last_error())
    powers = (ProcessorPower * (os.cpu_count() or 1))()
    status = bind(
        ct.WinDLL("powrprof"),
        "CallNtPowerInformation",
        wintypes.ULONG,
        [ct.c_int, ct.c_void_p, wintypes.ULONG, ct.c_void_p, wintypes.ULONG],
    )(11, None, 0, ct.byref(powers), ct.sizeof(powers))
    return {
        "process_cpu_seconds": user.seconds() + system.seconds(),
        "system_cpu_seconds": total_kernel.seconds() + total_user.seconds(),
        "system_idle_seconds": idle.seconds(),
        "reported_mhz": [item.current_mhz for item in powers] if status == 0 else None,
        "reported_mhz_limits": [item.limit_mhz for item in powers] if status == 0 else None,
    }


def sustained_ort(args):
    from benchmark_shared_search import session_command

    from core.search.supervisor import EmbeddingSession

    root = owned(args.directory)
    target = root / "ort-sustained-cpu.json"
    if target.exists():
        raise ValueError("sustained profile already exists")
    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("original profile input changed")
    paragraph = raw.decode("utf-8").split("\n\f\n")[0]
    profile = profiles()["bekko-a8m"]
    rows = []
    with EmbeddingSession(
        args.models_directory / profile.identity, profile, command=session_command(profile, root)
    ) as session:
        session.embed([paragraph] * 8)
        if session.worker_pid is None:
            raise ValueError("actual interpreter PID unavailable")
        started = time.monotonic()
        while time.monotonic() - started < 180:
            before = cpu_snapshot(session.worker_pid)
            began = time.monotonic()
            session.embed([paragraph] * 8, timeout=min(120, 182 - (began - started)))
            after = cpu_snapshot(session.worker_pid)
            elapsed = time.monotonic() - began
            total = after["system_cpu_seconds"] - before["system_cpu_seconds"]
            idle = after["system_idle_seconds"] - before["system_idle_seconds"]
            rows.append(
                {
                    "elapsed_from_start": began - started,
                    "batch_seconds": elapsed,
                    "process_cpu_seconds": after["process_cpu_seconds"]
                    - before["process_cpu_seconds"],
                    "system_busy_fraction": (total - idle) / total if total else None,
                    "reported_mhz": after["reported_mhz"],
                    "reported_mhz_limits": after["reported_mhz_limits"],
                }
            )
        peak = session.peak_bytes
    windows = []
    for begin in (0, 60, 120):
        selected = [row for row in rows if begin <= row["elapsed_from_start"] < begin + 60]
        wall = sum(row["batch_seconds"] for row in selected)
        frequencies = [
            value for row in selected if row["reported_mhz"] for value in row["reported_mhz"]
        ]
        busy = [
            row["system_busy_fraction"]
            for row in selected
            if row["system_busy_fraction"] is not None
        ]
        windows.append(
            {
                "window_start_seconds": begin,
                "passages_per_second": len(selected) * 8 / wall,
                "average_process_cpu_cores": sum(row["process_cpu_seconds"] for row in selected)
                / wall,
                "average_system_busy_fraction": statistics.mean(busy) if busy else None,
                "median_reported_mhz": statistics.median(frequencies) if frequencies else None,
            }
        )
    result = {
        "scope": "180-second sustained CPU diagnosis; not capacity/readiness",
        "input_sha256": source["input_sha256"],
        "windows": windows,
        "samples": rows,
        "child_peak_bytes": peak,
        "launcher_reaped": session.process.poll() is not None,
        "frequency_scope": "OS-reported MHz; not temperature or direct hardware turbo clock",
    }
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


CPP_REVISION = "d81235049384534c167caea52b85a694f6103d14"
RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-win-cpu-x64.zip"
RUNTIME_SHA = "1283323272b04cd07905816a597a0da810918102de958f4ff6f7bbaa70ed2efe"
MODEL_REPO = "hotchpotch/bekko-embedding-v1-a8m-GGUF"
MODEL_REVISION = "4acacdea60b9393ae247a6ebf5b5707938c9755b"
MODEL_FILE = "bekko-embedding-v1-a8m-Q8_0.gguf"
MODEL_SHA = "0b0214214287e90ffb98ffb47e1071e5b193fb420d1d99b2ec8aab0ebb507fc0"
MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/bekko_gguf_model.json"


def owned(root):
    root = root.resolve(strict=True)
    marker = json.loads((root / "ownership.json").read_text("utf-8"))
    if root.name != "corpi-bekko-gguf-experiment" or marker != {
        "task": "m1-6-local-search",
        "experiment": "bekko-gguf-q8",
        "agreement_revision": 11,
        "purpose": "temporary approved CPU runtime; inspect, measure, export then remove",
    }:
        raise ValueError("unowned GGUF experiment")
    return root


def pin():
    if MANIFEST.exists():
        raise ValueError("GGUF manifest already exists")
    base = profiles()["bekko-a8m"]
    conversion_url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/conversion.json"
    conversion_bytes = fetch(conversion_url)
    conversion = json.loads(conversion_bytes)
    if conversion["published_artifacts"][MODEL_FILE] != {"size": 118639520, "sha256": MODEL_SHA}:
        raise ValueError("GGUF conversion identity changed")
    assets = [item for item in base.files if item.path != base.model_file]
    assets += [
        ModelFile(
            MODEL_FILE,
            118639520,
            MODEL_SHA,
            f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}",
        ),
        ModelFile(
            "conversion.json",
            len(conversion_bytes),
            hashlib.sha256(conversion_bytes).hexdigest(),
            conversion_url,
        ),
    ]
    profile = replace(
        base,
        id="bekko-gguf-q8",
        repository=MODEL_REPO,
        revision=MODEL_REVISION,
        pooling="mean-hf-token-id-bridge",
        input_tokens=512,
        model_file=MODEL_FILE,
        runtime="benchmark-llama-b11429-q8-hf-ids-v1",
        files=tuple(sorted(assets, key=lambda item: item.path)),
    )
    MANIFEST.write_text(json.dumps(asdict(profile), indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"manifest": profile.identity, "download_bytes": profile.download_bytes}, indent=2
        )
    )


def inspect(root):
    root = owned(root)
    folder = root / "inspection"
    if (root / "runtime-inspection.json").exists():
        raise ValueError("runtime inspection already exists")
    folder.mkdir(exist_ok=True)
    source_files = (
        "include/llama.h",
        "ggml/include/ggml-backend.h",
        "examples/embedding/embedding.cpp",
        "src/models/modern-bert.cpp",
        "LICENSE",
    )
    source_receipts = {}
    for name in source_files:
        data = fetch(f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{CPP_REVISION}/{name}")
        target = folder / name.replace("/", "-")
        if target.exists() and target.read_bytes() != data:
            raise ValueError("existing inspection source differs")
        target.write_bytes(data)
        source_receipts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    archive = fetch(RUNTIME_URL)
    if len(archive) != 19398918 or hashlib.sha256(archive).hexdigest() != RUNTIME_SHA:
        raise ValueError("pinned runtime archive integrity failure")
    zip_path = folder / "runtime.zip"
    zip_path.write_bytes(archive)
    records = []
    with zipfile.ZipFile(zip_path) as bundle:
        for item in bundle.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError("unsafe runtime member")
            if item.is_dir():
                continue
            data = bundle.read(item)
            records.append(
                {
                    "path": item.filename,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "pe_machine": struct.unpack_from(
                        "<H", data, struct.unpack_from("<I", data, 60)[0] + 4
                    )[0]
                    if data.startswith(b"MZ")
                    else None,
                }
            )
    result = {
        "runtime_tag": "b11429",
        "source_commit": CPP_REVISION,
        "archive_sha256": RUNTIME_SHA,
        "archive_bytes": len(archive),
        "members": records,
        "sources": source_receipts,
        "runtime_code_executed": False,
    }
    (root / "runtime-inspection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def prepare(root):
    root = owned(root)
    proof = json.loads((root / "runtime-inspection.json").read_text("utf-8"))
    target = root / "runtime"
    if target.exists():
        raise ValueError("runtime already prepared")
    with zipfile.ZipFile(root / "inspection/runtime.zip") as bundle:
        for item in proof["members"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"]:
                raise ValueError("unsafe inspected runtime member")
            data = bundle.read(item["path"])
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("inspected runtime member changed")
            destination = target / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    print(json.dumps({"runtime_extracted": True, "runtime_code_executed": False}))


def gguf_header(path):
    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }
    with path.open("rb") as stream:

        def number(fmt):
            size = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, stream.read(size))[0]

        def string():
            size = number("Q")
            if size > 16000000:
                raise ValueError("GGUF metadata string exceeds inspection limit")
            return stream.read(size).decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = number("I"), number("Q")
                if count > 1000000 or subtype == 9:
                    raise ValueError("unsupported GGUF metadata array")
                return [value(subtype) for _ in range(count)]
            return number(formats[kind])

        if stream.read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("unsupported GGUF format")
        tensors, entries = number("Q"), number("Q")
        if tensors > 10000 or entries > 10000:
            raise ValueError("GGUF inspection bounds exceeded")
        metadata = {}
        for _ in range(entries):
            key = string()
            metadata[key] = value(number("I"))
        tensor_info = []
        for _ in range(tensors):
            name, dimensions = string(), number("I")
            if dimensions > 4:
                raise ValueError("GGUF tensor rank unsupported")
            tensor_info.append(
                {
                    "name": name,
                    "shape": [number("Q") for _ in range(dimensions)],
                    "type": number("I"),
                    "offset": number("Q"),
                }
            )
    return metadata, tensor_info


def inspect_model(directory, output):
    from tokenizers import Tokenizer

    profile = load_profile(MANIFEST)
    installed = directory / profile.identity
    verify_installation(installed, profile)
    metadata, tensors = gguf_header(installed / MODEL_FILE)
    tokenizer = Tokenizer.from_file(str(installed / "tokenizer.json"))
    tokens = metadata["tokenizer.ggml.tokens"]
    vocab = tokenizer.get_vocab(with_added_tokens=False)
    mismatches = [
        {"id": index, "hf": tokenizer.id_to_token(index), "gguf": token}
        for index, token in enumerate(tokens)
        if tokenizer.id_to_token(index) != token
    ]
    checks = {
        key: metadata.get(key)
        for key in (
            "general.architecture",
            "general.name",
            "modern-bert.block_count",
            "modern-bert.embedding_length",
            "modern-bert.context_length",
            "modern-bert.attention.head_count",
            "modern-bert.pooling_type",
            "tokenizer.ggml.model",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token",
        )
    }
    result = {
        "manifest": profile.identity,
        "checks": checks,
        "gguf_vocabulary": len(tokens),
        "hf_model_vocabulary": len(vocab),
        "vocabulary_token_id_mismatches": mismatches,
        "tensor_types": dict(Counter(item["type"] for item in tensors)),
        "tensors": tensors,
        "token_id_bridge_eligible": len(tokens) == 256000
        and all(known_display_alias(item) for item in mismatches),
    }
    if output.exists():
        output = output.with_name("model-inspection-corrected.json")
    if output.exists():
        raise ValueError("corrected model inspection already exists")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {**result, "tensors": len(tensors), "vocabulary_token_id_mismatches": len(mismatches)},
            indent=2,
        )
    )


def known_display_alias(item):
    return (
        139 <= item["id"] <= 168
        and item["hf"] == "\u2581" * (item["id"] - 137)
        and item["gguf"] == " " * (item["id"] - 137)
    ) or (item["id"] == 226 and item["hf"] == "\t" and item["gguf"] == "<0x09>")


def export(root, output, models_directory, capacity_file):
    root = owned(root)
    if (output / "provenance.json").exists():
        raise ValueError("GGUF export already exists")
    for profile in (load_profile(MANIFEST), profiles()["bekko-a8m"]):
        verify_installation(models_directory / profile.identity, profile)
    exports = {f"raw/{path.name}": path for path in root.glob("*.json")}
    exports["raw/bekko-gguf-setup.json"] = models_directory / "bekko-gguf-setup.json"
    exports["raw/bekko-ort-full-encoding.json"] = capacity_file
    for source in (root / "inspection").iterdir():
        if source.name != "runtime.zip":
            exports[f"inspection/{source.name}"] = source
    core = Path(__file__).resolve().parents[1]
    for name in (
        "scripts/benchmark_bekko_gguf.py",
        "scripts/benchmark_shared_search.py",
        "scripts/benchmark_contextual_multilingual.py",
        "tests/search/test_gguf_controls.py",
        "benchmarks/bekko_gguf_model.json",
    ):
        exports[f"source/{name}"] = core / name
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for relative, source in exports.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("GGUF evidence export mismatch")
        receipts.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest})
    (output / "provenance.json").write_text(
        json.dumps({"files": receipts}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"exported_verified_files": len(receipts)}))


def cleanup(root, output):
    root = owned(root)
    expected = Path.home() / "AppData/Local/Temp/opencode/corpi-bekko-gguf-experiment"
    if os.name != "nt" or root != expected.resolve(strict=True):
        raise ValueError("cleanup target differs from approved owned root")
    proof = json.loads((output / "provenance.json").read_text("utf-8"))
    for item in proof["files"]:
        source = output / item["path"]
        if (
            source.stat().st_size != item["bytes"]
            or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("exported GGUF evidence changed")
    if (root / "ownership.json").read_bytes() != (output / "raw/ownership.json").read_bytes():
        raise ValueError("ownership changed after export")

    def readonly_retry(function, path, error):
        if not Path(path).resolve().is_relative_to(root):
            raise error
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        function(path)

    shutil.rmtree(root, onexc=readonly_retry)
    print(json.dumps({"exact_owned_root_removed": not root.exists(), "exports_reverified": True}))


def source_proof(root):
    root = owned(root)
    old_revision = "b24cde5de82214ada4c01f173b137c78160b13c6"
    reference = profiles()["bekko-a8m"]
    old = json.loads(
        fetch(
            f"https://huggingface.co/api/models/{reference.repository}/revision/{old_revision}?blobs=true&expand[]=sha&expand[]=siblings"
        )
    )
    new = json.loads(
        fetch(
            f"https://huggingface.co/api/models/{reference.repository}/revision/{reference.revision}?blobs=true&expand[]=sha&expand[]=siblings"
        )
    )
    old_files = {item["rfilename"]: item for item in old["siblings"]}
    new_files = {item["rfilename"]: item for item in new["siblings"]}
    names = [asset.path for asset in reference.files] + ["model.safetensors"]
    identical = all(old_files[name] == new_files[name] for name in names)
    if old["sha"] != old_revision or new["sha"] != reference.revision or not identical:
        raise ValueError("pinned GGUF source weights/config/tokenizer differ")
    (root / "source-equivalence.json").write_text(
        json.dumps(
            {
                "conversion_source_revision": old_revision,
                "reference_revision": reference.revision,
                "source_assets_identical": identical,
                "matching_assets": {name: old_files[name] for name in names},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_assets_identical": identical, "matching_assets": len(names)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-child", action="store_true")
    parser.add_argument("--inspect-model", action="store_true")
    parser.add_argument("--sustained-ort", action="store_true")
    parser.add_argument("--source-proof", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--capacity-file", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--capacity-run", type=Path)
    parser.add_argument("--tag", default="")
    parser.add_argument("--native-passages", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--models-directory", type=Path)
    parser.add_argument("worker_args", nargs="*")
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[a-z0-9-]{1,40}", args.tag):
        parser.error("tag must be a short lowercase diagnostic name")
    if args.worker:
        if args.directory is None:
            parser.error("worker needs owned runtime root")
        root = owned(args.directory)
        temporary = root / "tmp"
        temporary.mkdir(exist_ok=True)
        os.environ.update(
            TEMP=str(temporary), TMP=str(temporary), HOME=str(temporary), USERPROFILE=str(temporary)
        )

        def denied(*args, **kwargs):
            raise OSError("GGUF inference network denied")

        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        from core.search import worker as protocol

        profile = load_profile(MANIFEST)
        GGufRuntime.root, GGufRuntime.metadata = root, args.metadata
        GGufRuntime.physical_batch = args.native_passages

        class TracedRuntime(GGufRuntime):
            def __init__(self, *parameters):
                try:
                    super().__init__(*parameters)
                except Exception as error:
                    if args.metadata:
                        args.metadata.write_text(
                            json.dumps(
                                {
                                    "scope": "public pinned model loading; before any query",
                                    "failure_type": type(error).__name__,
                                    "failure": str(error),
                                    "abi_sizes": [
                                        ct.sizeof(ModelParams),
                                        ct.sizeof(ContextParams),
                                        ct.sizeof(Batch),
                                    ],
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    raise

        protocol.profiles = lambda: {profile.id: profile}
        protocol.__dict__["EmbeddingRuntime"] = TracedRuntime
        sys.argv = [sys.argv[0], *args.worker_args]
        raise SystemExit(protocol.main())
    elif args.export or args.cleanup:
        if args.directory is None or args.output_directory is None:
            parser.error("export/cleanup needs owned and output directories")
        if args.export:
            if args.models_directory is None or args.capacity_file is None:
                parser.error("export requires verified assets and original encoding result")
            export(args.directory, args.output_directory, args.models_directory, args.capacity_file)
        else:
            cleanup(args.directory, args.output_directory)
    elif args.source_proof:
        if args.directory is None:
            parser.error("source proof needs owned directory")
        source_proof(args.directory)
    elif args.pin:
        pin()
    elif args.inspect or args.prepare:
        if args.directory is None:
            parser.error("owned temporary directory required")
        (inspect if args.inspect else prepare)(args.directory)
    elif args.download or args.download_child:
        if args.models_directory is None:
            parser.error("models-directory required")
        profile = load_profile(MANIFEST)
        target = args.models_directory / profile.identity
        if args.download_child:
            print(json.dumps({"received_bytes": download_profile(target, profile)}))
        elif target.exists():
            verify_installation(target, profile)
            print("Existing GGUF model verified")
        else:
            started = time.monotonic()
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--download-child",
                    "--models-directory",
                    str(args.models_directory),
                ],
                env=worker_environment(),
            )
            try:
                if process.wait(timeout=600) != 0:
                    raise RuntimeError("GGUF setup failed; incomplete staging retained")
            finally:
                stop_owned_process_tree(process)
            result = {
                "manifest": profile.identity,
                "bytes": profile.download_bytes,
                "seconds": time.monotonic() - started,
                "launcher_reaped": process.poll() is not None,
            }
            (args.models_directory / "bekko-gguf-setup.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
    elif args.inspect_model:
        if args.models_directory is None or args.directory is None:
            parser.error("model inspection needs model and owned result directories")
        inspect_model(args.models_directory, owned(args.directory) / "model-inspection.json")
    elif args.capacity_run:
        if args.models_directory is None or args.directory is None:
            parser.error("smoke needs model and owned runtime directories")
        (sustained_ort if args.sustained_ort else smoke)(args)
    else:
        parser.error("explicit inspection/setup action required")


if __name__ == "__main__":
    main()
