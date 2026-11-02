"""Temporary official Bekko OpenVINO IR diagnostics under original resource limits."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import statistics
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import benchmark_contextual_multilingual as contextual
import benchmark_openvino_granite as backend_tools
from benchmark_inference_options import agreement
from benchmark_static_multilingual import load_profile
from pin_search_models import fetch

from core.search.download import download_profile
from core.search.profiles import ModelFile, verify_installation
from core.search.supervisor import (
    EmbeddingFailure,
    EmbeddingSession,
    stop_owned_process_tree,
    worker_environment,
)

MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/bekko_openvino_model.json"
WHEELS = {
    "openvino": (
        "2026.4.1",
        "openvino-2026.4.1-22982-cp313-cp313-win_amd64.whl",
        "37c270c99d6de23439965e97cb5106389d3c8985f3b8bb90909a6ea0270db3f2",
    ),
    "openvino-telemetry": (
        "2025.2.0",
        "openvino_telemetry-2025.2.0-py3-none-any.whl",
        "bcb667e83a44f202ecf4cfa49281715c6d7e21499daec04ff853b7f964833599",
    ),
}


def pin():
    if MANIFEST.exists():
        raise ValueError("backend manifest already exists")
    base = contextual.profiles()["bekko-a8m"]
    info = json.loads(
        fetch(
            f"https://huggingface.co/api/models/{base.repository}/revision/{base.revision}?blobs=true"
        )
    )
    files = [item for item in base.files if item.path != base.model_file]
    required = {"openvino/openvino_model.xml", "openvino/openvino_model.bin"}
    for item in info["siblings"]:
        name = item["rfilename"]
        if name not in required:
            continue
        url = f"https://huggingface.co/{base.repository}/resolve/{base.revision}/{name}"
        if "lfs" in item:
            size, digest = item["lfs"]["size"], item["lfs"]["sha256"]
        else:
            data = fetch(url)
            size, digest = len(data), hashlib.sha256(data).hexdigest()
        files.append(ModelFile(name, size, digest, url))
    if info["sha"] != base.revision or not required <= {item.path for item in files}:
        raise ValueError("incomplete pinned backend")
    profile = replace(
        base,
        id="bekko-a8m-openvino",
        model_file="openvino/openvino_model.xml",
        runtime="benchmark-openvino-2026.4.1-native-ir-f32-v1",
        files=tuple(sorted(files, key=lambda item: item.path)),
    )
    MANIFEST.write_text(json.dumps(asdict(profile), indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"manifest": profile.identity, "download_bytes": profile.download_bytes}, indent=2
        )
    )


def inspect_packages(root):
    root = backend_tools.owned_root(root)
    folder = root / "wheels"
    if folder.exists():
        raise ValueError("package inspection already exists")
    folder.mkdir()
    records = []
    for package, (version, filename, expected) in WHEELS.items():
        metadata = json.loads(fetch(f"https://pypi.org/pypi/{package}/{version}/json"))
        asset = next(item for item in metadata["urls"] if item["filename"] == filename)
        if asset["digests"]["sha256"] != expected:
            raise ValueError("inspected wheel identity changed")
        target = folder / filename
        digest = hashlib.sha256()
        with (
            urllib.request.urlopen(asset["url"], timeout=30) as response,
            target.open("wb") as stream,
        ):
            received = 0
            deadline = time.monotonic() + 600
            while block := response.read1(65536):
                received += len(block)
                if received > asset["size"] or time.monotonic() >= deadline:
                    raise ValueError("inspection transfer limit exceeded")
                digest.update(block)
                stream.write(block)
        if target.stat().st_size != asset["size"] or digest.hexdigest() != expected:
            raise ValueError("inspection wheel hash mismatch")
        with zipfile.ZipFile(target) as archive:
            wheel_metadata = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            text = archive.read(wheel_metadata).decode("utf-8")
            names = [
                name
                for name in archive.namelist()
                if name in ("openvino/__init__.py", "openvino/_ov_api.py")
                or name.startswith("openvino_telemetry/")
                and name.endswith(".py")
            ]
            sources = {}
            for name in names:
                data = archive.read(name)
                output = folder / (name.replace("/", "-") + ".inspection.txt")
                output.write_bytes(data)
                sources[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        records.append(
            {
                "package": package,
                "version": version,
                "wheel": filename,
                "sha256": expected,
                "bytes": received,
                "sources": sources,
                "metadata": [
                    line
                    for line in text.splitlines()
                    if line.startswith(("Requires-Python:", "Requires-Dist:", "License:"))
                ],
                "package_code_executed": False,
            }
        )
    (root / "package-inspection.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


def command(root, backend, metadata):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--directory",
        str(root),
        "--backend",
        backend,
        "--metadata",
        str(metadata),
    ]


def smoke(args):
    root = backend_tools.owned_root(args.directory)
    target = root / f"smoke-b{args.batch_size}.json"
    if target.exists():
        raise ValueError("backend comparison already exists")
    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("original input changed")
    paragraphs = raw.decode("utf-8").split("\n\f\n")
    if len(paragraphs) != 9950:
        raise ValueError("original capacity differs")
    probes = [
        "Rainfall and soil moisture were measured.",
        "L'expérience mesure les précipitations et l'humidité du sol.",
        "تقيس التجربة هطول الأمطار ورطوبة التربة.",
    ]
    reference = None
    rows = []
    deadline = time.monotonic() + 600

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("backend diagnostic cumulative budget exhausted")
        return min(120, seconds)

    for backend in ("ort", "openvino"):
        profile = contextual.profiles()["bekko-a8m"] if backend == "ort" else load_profile(MANIFEST)
        session = None
        started = time.monotonic()
        metadata = root / f"{backend}-b{args.batch_size}-metadata.json"
        try:
            with EmbeddingSession(
                args.models_directory / profile.identity,
                profile,
                command=command(root, backend, metadata),
                load_timeout=remaining(),
            ) as session:
                vectors = session.embed([paragraphs[0]] * args.batch_size, timeout=remaining())
                times = []
                for _ in range(8):
                    began = time.perf_counter()
                    vectors = session.embed([paragraphs[0]] * args.batch_size, timeout=remaining())
                    times.append(time.perf_counter() - began)
                control = (
                    vectors
                    + session.embed(probes, timeout=remaining())
                    + session.embed(probes, query=True, timeout=remaining())
                )
                if reference is None:
                    reference = control
                row = {
                    "backend": backend,
                    "manifest": profile.identity,
                    "status": "completed",
                    "batch_size": args.batch_size,
                    "load_seconds": session.load_seconds,
                    "batch_seconds": times,
                    "passages_per_second": args.batch_size / statistics.median(times),
                    "child_peak_bytes": session.peak_bytes,
                    "vector_agreement": agreement(reference, control),
                }
            row["launcher_reaped"] = session.process.poll() is not None
        except (EmbeddingFailure, TimeoutError) as error:
            row = {
                "backend": backend,
                "status": "failed",
                "failure": str(error),
                "child_peak_bytes": session.peak_bytes if session else None,
                "launcher_reaped": session.process.poll() is not None if session else True,
            }
        row["elapsed_seconds"] = time.monotonic() - started
        rows.append(row)
    result = {
        "scope": "paired backend screen; not full capacity/readiness/quality",
        "original_input_sha256": source["input_sha256"],
        "results": rows,
    }
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root, output, models_directory):
    root = backend_tools.owned_root(root)
    for profile in (contextual.profiles()["bekko-a8m"], load_profile(MANIFEST)):
        verify_installation(models_directory / profile.identity, profile)
    if (output / "provenance.json").exists():
        raise ValueError("backend evidence packet already exists")
    consent = root / "localappdata/Intel Corporation/openvino_telemetry"
    if consent.read_text("utf-8").strip() != "0" or list(consent.parent.iterdir()) != [consent]:
        raise ValueError("unexpected temporary consent state")
    environment = json.loads((root / "environment.json").read_text("utf-8"))
    inspection = json.loads((root / "package-inspection.json").read_text("utf-8"))
    for package, relative in (
        ("openvino", "openvino/__init__.py"),
        ("openvino-telemetry", "openvino_telemetry/__init__.py"),
    ):
        expected = next(item for item in inspection if item["package"] == package)["sources"][
            relative
        ]["sha256"]
        files = [
            Path(folder) / relative
            for folder in environment["overlay_paths"]
            if (Path(folder) / relative).exists()
        ]
        if (
            len(files) != 1
            or not files[0].resolve().is_relative_to(root)
            or file_hash(files[0]) != expected
        ):
            raise ValueError("installed backend source differs from inspected wheel")
    output.mkdir(parents=True, exist_ok=True)
    exports = {
        "raw/ownership.json": root / "ownership.json",
        "raw/package-inspection.json": root / "package-inspection.json",
        "raw/declined-consent.txt": consent,
    }
    exports["raw/bekko-openvino-setup.json"] = models_directory / "bekko-openvino-setup.json"
    rows = []
    for batch in (8, 16, 32):
        name = f"smoke-b{batch}.json"
        data = json.loads((root / name).read_text("utf-8"))
        if any(
            not item["launcher_reaped"] or item["status"] != "completed" for item in data["results"]
        ):
            raise ValueError("unresolved backend session")
        rows.append(data)
        exports[f"raw/{name}"] = root / name
        for backend in ("ort", "openvino"):
            name = f"{backend}-b{batch}-metadata.json"
            exports[f"raw/{name}"] = root / name
    for source in (root / "wheels").glob("*.inspection.txt"):
        exports[f"inspection/{source.name}"] = source
    core = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/benchmark_bekko_openvino.py",
        "scripts/benchmark_openvino_granite.py",
        "scripts/benchmark_contextual_multilingual.py",
        "tests/search/test_openvino_controls.py",
        "benchmarks/bekko_openvino_model.json",
        "benchmarks/contextual_multilingual_models.json",
    ):
        exports[f"source/{relative}"] = core / relative
    summary = {
        "scope": "paired short backend screens; no full capacity or quality pass",
        "batches": rows,
        "telemetry_declined_before_after": True,
        "package_versions": environment["versions"],
        "full_quality_and_capacity": "omitted: all measured rates below encoding-only floor",
        "best_openvino_passages_per_second": max(
            item["passages_per_second"]
            for row in rows
            for item in row["results"]
            if item["backend"] == "openvino"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    receipts = []
    for relative, source in exports.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_hash(target) != file_hash(source):
            raise ValueError("backend evidence export mismatch")
        receipts.append(
            {"path": relative, "sha256": file_hash(target), "bytes": target.stat().st_size}
        )
    (output / "provenance.json").write_text(
        json.dumps(
            {"files": receipts, "summary_sha256": file_hash(output / "summary.json")}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "exported_files": len(receipts),
                "best_openvino_passages_per_second": summary["best_openvino_passages_per_second"],
            },
            indent=2,
        )
    )


def cleanup(root, output):
    root = backend_tools.owned_root(root)
    expected = Path.home() / "AppData/Local/Temp/opencode/corpi-bekko-openvino-experiment"
    if os.name != "nt" or root != expected.resolve(strict=True):
        raise ValueError("cleanup target differs from approved temporary root")
    proof = json.loads((output / "provenance.json").read_text("utf-8"))
    if file_hash(output / "summary.json") != proof["summary_sha256"]:
        raise ValueError("exported summary changed before cleanup")
    for item in proof["files"]:
        target = output / item["path"]
        if file_hash(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise ValueError("exported evidence changed before cleanup")
    if file_hash(root / "ownership.json") != file_hash(output / "raw/ownership.json"):
        raise ValueError("temporary ownership changed")

    def readonly_retry(function, path, error):
        if not Path(path).resolve().is_relative_to(root):
            raise error
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        function(path)

    shutil.rmtree(root, onexc=readonly_retry)
    print(json.dumps({"exact_owned_root_removed": not root.exists(), "export_reverified": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--inspect-packages", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-child", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--models-directory", type=Path)
    parser.add_argument("--capacity-run", type=Path)
    parser.add_argument("--backend", choices=("ort", "openvino"), default="openvino")
    parser.add_argument("--batch-size", choices=(8, 16, 32), type=int, default=16)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("worker_args", nargs="*")
    args = parser.parse_args()
    if args.pin:
        pin()
    elif args.download or args.download_child:
        if args.models_directory is None:
            parser.error("download requires models-directory")
        profile = load_profile(MANIFEST)
        target = args.models_directory / profile.identity
        if args.download_child:
            print(json.dumps({"received_bytes": download_profile(target, profile)}))
        elif target.exists():
            verify_installation(target, profile)
            print("Existing pinned IR assets verified")
        else:
            import subprocess

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
                    raise RuntimeError("backend asset setup failed; staging retained")
            finally:
                stop_owned_process_tree(process)
            data = {
                "manifest": profile.identity,
                "bytes": profile.download_bytes,
                "elapsed_seconds": time.monotonic() - started,
                "launcher_reaped": process.poll() is not None,
            }
            (args.models_directory / "bekko-openvino-setup.json").write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
            xml = ET.parse(target / profile.model_file)
            ops = {}
            for layer in xml.findall("./layers/layer"):
                kind = layer.attrib["type"]
                ops[kind] = ops.get(kind, 0) + 1
            print(json.dumps({**data, "ir_operator_counts": ops}, indent=2))
    else:
        if args.directory is None:
            parser.error("owned directory required")
        if args.export or args.cleanup:
            if args.output_directory is None:
                parser.error("export/cleanup requires output-directory")
            if args.export:
                if args.models_directory is None:
                    parser.error("export requires models-directory for final asset verification")
                export(args.directory, args.output_directory, args.models_directory)
            else:
                cleanup(args.directory, args.output_directory)
        elif args.inspect_packages:
            inspect_packages(args.directory)
        elif args.prepare:
            backend_tools.prepare(args.directory)
        elif args.worker:
            catalog = {**contextual.profiles(), "bekko-a8m-openvino": load_profile(MANIFEST)}
            contextual.profiles = lambda: catalog
            args.profile_operators = args.float_integer_matrices = False
            backend_tools.worker(args)
        else:
            if args.models_directory is None or args.capacity_run is None:
                parser.error("comparison requires models-directory and capacity-run")
            smoke(args)


if __name__ == "__main__":
    main()
