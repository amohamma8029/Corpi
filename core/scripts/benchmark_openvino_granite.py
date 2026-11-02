"""Authorized temporary CPU-only backend comparison; never app integration.

Uses the original verified Granite graph, compiled tokenizer and trained CLS output.
Telemetry consent/config and runtime caches stay in the owned temporary root.
"""

import argparse
import json
import os
import socket
import sys
import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path


def owned_root(root: Path):
    root = root.resolve(strict=True)
    ownership = json.loads((root / "ownership.json").read_text("utf-8"))
    permitted = root.name == "corpi-granite-openvino-experiment" or (
        root.name == "corpi-bekko-openvino-experiment"
        and ownership.get("experiment") == "bekko-openvino"
    )
    if not permitted or ownership["task"] != "m1-6-local-search":
        raise ValueError("unowned backend environment")
    return root


def isolate(root: Path, backend: str):
    root = owned_root(root)
    local = root / "localappdata"
    consent = local / "Intel Corporation/openvino_telemetry"
    if consent.read_text("utf-8").strip() != "0":
        raise ValueError("telemetry consent is not declined")
    os.environ.update(
        LOCALAPPDATA=str(local),
        TEMP=str(root / "tmp"),
        TMP=str(root / "tmp"),
        HOME=str(root / "tmp"),
        USERPROFILE=str(root / "tmp"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
    )
    if backend == "openvino":
        environment = json.loads((root / "environment.json").read_text("utf-8"))
        paths = [str(Path(path).resolve()) for path in environment["overlay_paths"]]
        if not all(Path(path).is_relative_to(root) for path in paths):
            raise ValueError("backend package overlay escaped ownership")
        # uv removes bootstrap-only build paths after setup. Keep the persistent
        # owned archive site-packages, which contain the installed wheel files.
        paths = [path for path in paths if Path(path).is_dir()]
        if not paths:
            raise ValueError("backend package overlay missing")
        sys.path[:0] = paths
        for name in ("openvino", "openvino_telemetry"):
            spec = find_spec(name)
            if (
                spec is None
                or spec.origin is None
                or not Path(spec.origin).resolve().is_relative_to(root)
            ):
                raise ValueError("backend package origin outside owned root")

    def denied(*args, **kwargs):
        raise OSError("benchmark Python network connections denied")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    return consent


def worker(args):
    consent = isolate(args.directory, args.backend)
    from benchmark_contextual_multilingual import ContextualRuntime, UTF8InputTokenizer, profiles

    from core.search import worker as protocol
    from core.search.profiles import verify_installation

    class OpenVINORuntime(ContextualRuntime):
        def __init__(self, path, profile):
            verify_installation(path, profile)
            self.profile = profile
            self.tokenizer = UTF8InputTokenizer(path, profile)
            self.pad_id = json.loads((path / "config.json").read_text("utf-8"))["pad_token_id"]
            self.pooler = self.dense = None
            self.output = "last_hidden_state"
            started = time.monotonic()
            stage = "backend_import"
            try:
                ov = import_module("openvino")
                telemetry = import_module("openvino_telemetry.utils.opt_in_checker")

                if (
                    telemetry.OptInChecker().check(enable_opt_in_dialog=False)
                    != telemetry.ConsentCheckResult.DECLINED
                ):
                    raise ValueError("telemetry decline control failed")
                stage = "cpu_compile"
                core = ov.Core()
                properties = {
                    "INFERENCE_NUM_THREADS": 2,
                    "NUM_STREAMS": 1,
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_PRECISION_HINT": ov.Type.f32,
                    "CACHE_DIR": str(args.directory / "compiled-cache"),
                }
                if args.profile_operators:
                    properties["PERF_COUNT"] = True
                compile_input = str(path / profile.model_file)
                rewrites = []
                if args.float_integer_matrices:
                    parsed = core.read_model(compile_input)
                    for node in list(parsed.get_ops()):
                        if (
                            node.get_type_name() != "MatMul"
                            or node.get_output_element_type(0) != ov.Type.i32
                        ):
                            continue
                        targets = list(node.output(0).get_target_inputs())
                        casts = [
                            target.get_node()
                            for target in targets
                            if target.get_node().get_type_name() == "Convert"
                            and target.get_node().get_output_element_type(0) == ov.Type.f32
                        ]
                        if not casts:
                            continue
                        attrs = node.get_attributes()
                        left = ov.opset8.convert(node.input_value(0), ov.Type.f32)
                        right = ov.opset8.convert(node.input_value(1), ov.Type.f32)
                        replacement = ov.opset8.matmul(
                            left, right, attrs["transpose_a"], attrs["transpose_b"]
                        )
                        replacement.set_friendly_name(node.get_friendly_name() + "_centered_f32")
                        for cast in casts:
                            cast.output(0).replace(replacement.output(0))
                        rewrites.append(node.get_friendly_name())
                    if not rewrites:
                        raise ValueError("no checked integer matrix conversion paths")
                    parsed.validate_nodes_and_infer_types()
                    compile_input = parsed
                compiled = core.compile_model(
                    compile_input,
                    "CPU",
                    properties,
                )
                threads = compiled.get_property("INFERENCE_NUM_THREADS")
                streams = compiled.get_property("NUM_STREAMS")
                precision = compiled.get_property("INFERENCE_PRECISION_HINT")
                devices = compiled.get_property("EXECUTION_DEVICES")
                if (
                    threads != 2
                    or streams != 1
                    or precision != ov.Type.f32
                    or list(devices) != ["CPU"]
                ):
                    raise ValueError("CPU runtime limits not applied")
                self.inputs = {port.get_any_name() for port in compiled.inputs}
                if self.inputs != {"input_ids", "attention_mask"}:
                    raise ValueError("unsupported backend input contract")
                output = compiled.output("last_hidden_state")
                request = compiled.create_infer_request()
                calls = []

                class Facade:
                    def run(self, names, inputs):
                        if names != ["last_hidden_state"]:
                            raise ValueError("unsupported backend output contract")
                        began = time.perf_counter()
                        values = request.infer(inputs, share_inputs=False)
                        result = values[output]
                        calls.append(time.perf_counter() - began)
                        if args.profile_operators and inputs["input_ids"].shape[1] > 128:
                            entries = [
                                {
                                    "node": item.node_name,
                                    "op": item.node_type,
                                    "implementation": item.exec_type,
                                    "real_microseconds": item.real_time.total_seconds() * 1_000_000,
                                }
                                for item in request.get_profiling_info()
                            ]
                            entries.sort(key=lambda item: -item["real_microseconds"])
                            (args.directory / "openvino-dense-operators.json").write_text(
                                json.dumps(entries, indent=2) + "\n", encoding="utf-8"
                            )
                        if consent.read_text("utf-8").strip() != "0":
                            raise ValueError("telemetry consent changed")
                        return [result]

                self.model = Facade()
                metadata = {
                    "backend": "openvino",
                    "version": ov.get_version(),
                    "cpu_threads": threads,
                    "streams": streams,
                    "precision": str(precision),
                    "execution_devices": list(devices),
                    "inputs": sorted(self.inputs),
                    "output": output.get_any_name(),
                    "telemetry_declined": True,
                    "python_network_denied": True,
                    "profiling_enabled": args.profile_operators,
                    "centered_float_matrix_rewrites": rewrites,
                    "compile_and_import_seconds": time.monotonic() - started,
                }
                if args.metadata:
                    args.metadata.write_text(
                        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                    )
            except Exception as error:
                if args.metadata:
                    args.metadata.write_text(
                        json.dumps(
                            {
                                "backend": "openvino",
                                "stage": stage,
                                "error_type": type(error).__name__,
                                "error": str(error)
                                .replace(str(path), "<verified-assets>")
                                .replace(str(args.directory), "<owned-runtime>"),
                                "elapsed_seconds": time.monotonic() - started,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                raise

    if args.backend == "ort":
        import benchmark_contextual_multilingual as candidate

        candidate.METADATA = args.metadata
        factory = ContextualRuntime
    else:
        factory = OpenVINORuntime
    protocol.profiles = profiles
    protocol.__dict__["EmbeddingRuntime"] = factory
    sys.argv = [sys.argv[0], *args.worker_args]
    raise SystemExit(protocol.main())


def command(root, backend, metadata, profile_operators=False, float_integer_matrices=False):
    result = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--directory",
        str(root),
        "--metadata",
        str(metadata),
    ]
    if profile_operators:
        result.append("--profile-operators")
    if float_integer_matrices:
        result.append("--float-integer-matrices")
    return result


def prepare(root):
    """Record already-authorized installed overlays, without importing backend code."""
    from importlib.metadata import version

    root = owned_root(root)
    if (root / "localappdata/Intel Corporation/openvino_telemetry").read_text(
        "utf-8"
    ).strip() != "0":
        raise ValueError("telemetry consent is not declined")
    target = root / "environment.json"
    if target.exists():
        raise ValueError("backend environment already recorded")
    versions = {name: version(name) for name in ("openvino", "openvino-telemetry", "numpy")}
    if versions != {"openvino": "2026.4.1", "openvino-telemetry": "2025.2.0", "numpy": "2.5.3"}:
        raise ValueError("backend package versions differ from approval")
    paths = [
        str(Path(path).resolve())
        for path in sys.path
        if path and Path(path).resolve().is_relative_to(root)
    ]
    for name in ("openvino", "openvino_telemetry"):
        spec = find_spec(name)
        if (
            spec is None
            or spec.origin is None
            or not Path(spec.origin).resolve().is_relative_to(root)
        ):
            raise ValueError("backend package origin outside owned root")
    target.write_text(
        json.dumps(
            {"versions": versions, "overlay_paths": paths, "backend_imported": False}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"versions": versions, "backend_imported": False}, indent=2))


def smoke(args):
    import hashlib
    import statistics

    from benchmark_contextual_multilingual import profiles
    from benchmark_inference_options import agreement

    from core.search.profiles import verify_installation
    from core.search.supervisor import EmbeddingFailure, EmbeddingSession

    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("original dense input changed")
    paragraph = raw.decode("utf-8").split("\n\f\n")[0]
    profile = profiles()["granite-107m"]
    probes = [
        "Rainfall and soil moisture were measured.",
        "L'expérience mesure les précipitations et l'humidité du sol.",
        "تقيس التجربة هطول الأمطار ورطوبة التربة.",
    ]
    reference = None
    rows = []
    for backend in ("ort", "openvino"):
        tag = ("-profile" if args.profile_operators else "") + (
            "-float" if args.float_integer_matrices else ""
        )
        metadata = args.directory / f"{backend}-b{args.batch_size}{tag}-smoke-metadata.json"
        session = None
        try:
            with EmbeddingSession(
                args.models_directory / profile.identity,
                profile,
                command=command(
                    args.directory,
                    backend,
                    metadata,
                    args.profile_operators,
                    args.float_integer_matrices if backend == "openvino" else False,
                ),
            ) as session:
                vectors = session.embed([paragraph] * args.batch_size)
                times = []
                for _ in range(4):
                    began = time.perf_counter()
                    vectors = session.embed([paragraph] * args.batch_size)
                    times.append(time.perf_counter() - began)
                control = vectors + session.embed(probes) + session.embed(probes, query=True)
                if reference is None:
                    reference = control
                row = {
                    "backend": backend,
                    "status": "completed",
                    "batch_size": args.batch_size,
                    "load_seconds": session.load_seconds,
                    "batch_seconds": times,
                    "median_batch_seconds": statistics.median(times),
                    "documents_per_second": args.batch_size / statistics.median(times),
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
        rows.append(row)
    result = {
        "scope": "temporary backend smoke, not readiness or full quality",
        "manifest": profile.identity,
        "input_sha256": source["input_sha256"],
        "results": rows,
    }
    verify_installation(args.models_directory / profile.identity, profile)
    if (args.directory / "localappdata/Intel Corporation/openvino_telemetry").read_text(
        "utf-8"
    ).strip() != "0":
        raise ValueError("temporary telemetry consent changed")
    tag = ("-profile" if args.profile_operators else "") + (
        "-float" if args.float_integer_matrices else ""
    )
    target = args.directory / f"smoke-b{args.batch_size}{tag}-results.json"
    if target.exists():
        raise ValueError("smoke result already exists")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path)
    parser.add_argument("--capacity-run", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile-operators", action="store_true")
    parser.add_argument("--float-integer-matrices", action="store_true")
    parser.add_argument("--backend", choices=("ort", "openvino"), default="openvino")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("worker_args", nargs="*")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.directory)
    elif args.worker:
        worker(args)
    else:
        if args.models_directory is None or args.capacity_run is None:
            parser.error("comparison requires model and capacity directories")
        smoke(args)


if __name__ == "__main__":
    main()
