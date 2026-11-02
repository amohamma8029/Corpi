"""Supervised CPU inference options on the original capacity paragraph.

No database/generation settings are changed. This diagnostic records actual
thread/output settings, memory, timings and vector equivalence separately.
All calls retain the normal 1-GiB supervisor cutoff and <=2 CPU threads.
"""

import argparse
import hashlib
import json
import math
import statistics
import struct
import sys
import time
from pathlib import Path

from core.persistence.models import new_id
from core.search.profiles import profiles, verify_installation
from core.search.supervisor import EmbeddingFailure, EmbeddingSession

CHILD = """
import json, sys
from dataclasses import replace
from pathlib import Path
from core.search import worker
from core.search.runtime import EmbeddingRuntime, masked_mean_pool
threads, first, low_pool = int(sys.argv[1]), bool(int(sys.argv[2])), bool(int(sys.argv[3]))
metadata = Path(sys.argv[4])
profile_calls = int(sys.argv[5])
derived = sys.argv[6]
sys.argv = [sys.argv[0], *sys.argv[7:]]
if profile_calls:
    import onnxruntime as ort
    original_options = ort.SessionOptions
    def options():
        value = original_options()
        value.enable_profiling = True
        value.profile_file_prefix = str(metadata.with_suffix(''))
        return value
    ort.SessionOptions = options
class ConfiguredRuntime(EmbeddingRuntime):
    def __init__(self, *args, **kwargs):
        if derived:
            args = (args[0], replace(args[1], model_file=str(Path(derived).resolve())))
        super().__init__(*args, threads=threads, **kwargs)
        self.profile_calls = profile_calls
        native = self.model.model
        outputs = native.model.get_outputs()
        if first:
            native.ONNX_OUTPUT_NAMES = [outputs[0].name]
        if low_pool:
            native._pool = masked_mean_pool
        options = native.model.get_session_options()
        metadata.write_text(json.dumps({
            'intra_op_threads': options.intra_op_num_threads,
            'inter_op_threads': options.inter_op_num_threads,
            'providers': native.model.get_providers(),
            'model_outputs': [o.name for o in outputs],
            'requested_outputs': native.ONNX_OUTPUT_NAMES,
            'low_memory_mean_pool': low_pool,
            'derived_graph': bool(derived),
            'runtime_profile_identity': self.profile.identity,
        }, indent=2), encoding='utf-8')
    def embed(self, *args, **kwargs):
        result = super().embed(*args, **kwargs)
        if self.profile_calls:
            self.profile_calls -= 1
            if not self.profile_calls:
                trace = Path(self.model.model.model.end_profiling())
                operators = {}
                for event in json.loads(trace.read_text('utf-8')):
                    kernel = event.get('name', '').endswith('_kernel_time')
                    if event.get('cat') != 'Node' or not kernel:
                        continue
                    name = event.get('args', {}).get('op_name', 'unknown')
                    entry = operators.setdefault(name, {'calls': 0, 'microseconds': 0})
                    entry['calls'] += 1
                    entry['microseconds'] += event.get('dur', 0)
                data = json.loads(metadata.read_text('utf-8'))
                data['operator_profile'] = dict(sorted(operators.items(),
                    key=lambda item: -item[1]['microseconds']))
                metadata.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return result
worker.EmbeddingRuntime = ConfiguredRuntime
raise SystemExit(worker.main())
"""


def agreement(reference, actual):
    max_delta, min_cosine = 0.0, 1.0
    for left, right in zip(reference, actual, strict=True):
        a, b = struct.unpack("<384f", left), struct.unpack("<384f", right)
        max_delta = max(max_delta, max(abs(x - y) for x, y in zip(a, b, strict=True)))
        cosine = sum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))
        min_cosine = min(min_cosine, cosine)
    return {
        "bitwise_equal": reference == actual,
        "max_component_delta": max_delta,
        "minimum_cosine": min_cosine,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-run", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument(
        "--profile-graph",
        action="store_true",
        help="Collect operator timings for the baseline only; not an A/B comparison",
    )
    parser.add_argument(
        "--derived-graph",
        type=Path,
        help="Compare original against an experiment-only graph; no deployed identity is changed",
    )
    parser.add_argument("--candidate-batch-size", type=int, choices=(8, 16, 32))
    parser.add_argument("--candidate-low-memory-pool", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 8:
        parser.error("repetitions must be 2..8")
    if args.derived_graph and args.profile_graph:
        parser.error("derived-graph comparison and operator profiling are separate diagnostics")
    if (args.candidate_batch_size or args.candidate_low_memory_pool) and not args.derived_graph:
        parser.error("candidate settings require a derived graph")
    derived_hash = None
    if args.derived_graph:
        args.derived_graph = args.derived_graph.resolve(strict=True)
        derived_hash = hashlib.sha256(args.derived_graph.read_bytes()).hexdigest()
    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("original capacity input changed")
    paragraphs = raw.decode("utf-8").split("\n\f\n")
    if len(paragraphs) != source["intended_chunks"] or len(set(paragraphs)) != 1:
        raise ValueError("diagnostic expects the original repeated capacity paragraphs")
    profile = profiles()[source["profile"]]
    batch_size = source["batch_size"]
    if profile.id == "multilingual-e5":
        batch_size = 8  # Measure a configuration that stays below the existing cutoff.
    probes = [
        "Rainfall and soil moisture were measured.",
        "L'expérience mesure les précipitations et l'humidité du sol.",
        "تقيس التجربة هطول الأمطار ورطوبة التربة.",
    ]
    root = args.directory / f"inference-options-{profile.id}-{new_id()}"
    root.mkdir(parents=True)
    reference = None
    rows = []
    configurations = (
        ((2, False, False),)
        if args.profile_graph
        else ((2, False, False), (1, False, False), (2, False, True))
    )
    if args.derived_graph:
        configurations = ((2, False, False), (2, False, args.candidate_low_memory_pool))
    for number, (threads, first, low_pool) in enumerate(configurations):
        candidate = args.derived_graph if args.derived_graph and number == 1 else None
        actual_batch_size = args.candidate_batch_size if candidate else None
        actual_batch_size = actual_batch_size or batch_size
        texts = [paragraphs[0]] * actual_batch_size
        metadata = root / f"config{number}-threads{threads}-pool{low_pool}-metadata.json"
        command = [
            sys.executable,
            "-c",
            CHILD,
            str(threads),
            str(int(first)),
            str(int(low_pool)),
            str(metadata),
            str(args.repetitions + 3 if args.profile_graph else 0),
            str(candidate.resolve()) if candidate else "",
        ]
        session = None
        try:
            with EmbeddingSession(
                args.models_directory / profile.identity,
                profile,
                command=command,
                load_timeout=45,
            ) as session:
                vectors = session.embed(texts, timeout=30)  # Unmeasured warm-up.
                times = []
                for _ in range(args.repetitions):
                    began = time.perf_counter()
                    vectors = session.embed(texts, timeout=30)
                    times.append(time.perf_counter() - began)
                probes_control = session.embed(probes, timeout=30)
                probes_control += session.embed(probes, query=True, timeout=30)
                if reference is None:
                    reference = (vectors[0], probes_control)
                expected = [reference[0]] * actual_batch_size + reference[1]
                control = vectors + probes_control
            rows.append(
                {
                    "status": "passed",
                    "batch_size": actual_batch_size,
                    "threads": threads,
                    "first_output_only": first,
                    "low_memory_mean_pool": low_pool,
                    "derived_graph_sha256": derived_hash if candidate else None,
                    "actual_settings": json.loads(metadata.read_text("utf-8")),
                    "load_seconds": session.load_seconds,
                    "batch_seconds": times,
                    "median_batch_seconds": statistics.median(times),
                    "slowest_batch_seconds": max(times),
                    "child_peak_bytes": session.peak_bytes,
                    "documents_per_second": actual_batch_size / statistics.median(times),
                    "launcher_reaped": session.process.poll() is not None,
                    "vector_agreement": agreement(expected, control),
                }
            )
        except (EmbeddingFailure, TimeoutError) as error:
            rows.append(
                {
                    "status": "failed",
                    "threads": threads,
                    "batch_size": actual_batch_size,
                    "derived_graph_sha256": derived_hash if candidate else None,
                    "low_memory_mean_pool": low_pool,
                    "failure_code": error.code
                    if isinstance(error, EmbeddingFailure)
                    else "timeout",
                    "child_peak_bytes": session.peak_bytes if session else None,
                    "launcher_reaped": session.process.poll() is not None if session else True,
                }
            )
    result = {
        "scope": "supervised repeated-paragraph inference diagnostic, not readiness or quality",
        "profile": profile.id,
        "upstream_manifest": profile.identity,
        "batch_size": batch_size,
        "input_sha256": source["input_sha256"],
        "repetitions": args.repetitions,
        "configurations": rows,
    }
    verify_installation(args.models_directory / profile.identity, profile)
    if args.derived_graph:
        if hashlib.sha256(args.derived_graph.read_bytes()).hexdigest() != derived_hash:
            raise ValueError("derived graph changed during comparison")
    (root / "options-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
