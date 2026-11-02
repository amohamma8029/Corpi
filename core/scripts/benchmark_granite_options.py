"""Supervised Granite CPU options on the unchanged original dense paragraph.

No generation, asset or app setting is modified. Profiling is separate from A/B
timing. All calls retain the usual child RSS cutoff and <=2 compute threads.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from benchmark_contextual_multilingual import profiles
from benchmark_inference_options import agreement

from core.persistence.models import new_id
from core.search.profiles import verify_installation
from core.search.supervisor import EmbeddingFailure, EmbeddingSession

CONFIGURATIONS = {
    "baseline": {"batch": 16, "threads": 2, "arena": False},
    "batch8": {"batch": 8, "threads": 2, "arena": False},
    "batch32": {"batch": 32, "threads": 2, "arena": False},
    "arena": {"batch": 16, "threads": 2, "arena": True},
    "one-thread": {"batch": 16, "threads": 1, "arena": False},
    "dynamic-block": {
        "batch": 16,
        "threads": 2,
        "arena": False,
        "entries": {"session.dynamic_block_base": "4"},
    },
    "affinity": {"batch": 16, "threads": 2, "arena": False},
}

CHILD = """
import json, os, socket, sys, time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, sys.argv[1])
import benchmark_contextual_multilingual as candidate
from core.search import worker
config = json.loads(Path(sys.argv[2]).read_text('utf-8'))
metadata = Path(config['metadata'])
sys.argv = [sys.argv[0], *sys.argv[3:]]
candidate.METADATA = metadata
socket.socket.connect = candidate.denied
socket.socket.connect_ex = candidate.denied
original_session = candidate.ort.InferenceSession
affinity_control = None
def session(path, providers, sess_options):
    global affinity_control
    sess_options.intra_op_num_threads = config['threads']
    sess_options.inter_op_num_threads = config['threads']
    sess_options.enable_cpu_mem_arena = config['arena']
    for key, value in config.get('entries', {}).items():
        sess_options.add_session_config_entry(key, value)
    before = None
    if config.get('affinity_cpus'):
        import benchmark_cpu_affinity as affinity
        caller_cpu, worker_cpu = config['affinity_cpus']
        caller = affinity.pin_calling_thread(caller_cpu)
        before = {row['thread_id'] for row in affinity.thread_affinities()}
        sess_options.add_session_config_entry(
            'session.intra_op_thread_affinities', str(worker_cpu + 1))
        affinity_control = {'caller': caller, 'worker_cpu': worker_cpu,
                            'ort_one_based_entry': str(worker_cpu + 1)}
    if config['profile_calls']:
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = str(metadata.with_suffix(''))
        sess_options.optimized_model_filepath = str(metadata.with_suffix('.optimized.onnx'))
    model = original_session(path, providers=providers, sess_options=sess_options)
    if before is not None:
        observed = affinity.thread_affinities()
        created = [row for row in observed if row['thread_id'] not in before]
        matched = [row for row in created if row['group'] == 0 and row['mask'] == 1 << worker_cpu]
        if len(created) != 1 or len(matched) != 1:
            raise ValueError('expected exactly one separately pinned ORT compute worker')
        affinity_control.update(created_threads=created, actual_threads=observed, verified=True)
    return model
candidate.ort.InferenceSession = session
class MeasuredRuntime(candidate.ContextualRuntime):
    def __init__(self, *args):
        if config.get('derived_graph'):
            from dataclasses import replace
            args = (args[0], replace(args[1], model_file=config['derived_graph']))
        super().__init__(*args)
        self.calls = []
        self.native_calls = []
        self.profile_calls = config['profile_calls']
        original_run = self.model.run
        def run(*args, **kwargs):
            started = time.perf_counter()
            result = original_run(*args, **kwargs)
            self.native_calls.append(time.perf_counter() - started)
            return result
        self.model.run = run
        data = json.loads(metadata.read_text('utf-8'))
        options = self.model.get_session_options()
        data.update(cpu_memory_arena=options.enable_cpu_mem_arena,
                    memory_pattern=options.enable_mem_pattern,
                    graph_optimization=str(options.graph_optimization_level),
                     session_entries=config.get('entries', {}),
                     execution_mode=str(options.execution_mode),
                     affinity_control=affinity_control)
        metadata.write_text(json.dumps(data, indent=2), encoding='utf-8')
    def embed(self, *args, **kwargs):
        started = time.perf_counter()
        result = super().embed(*args, **kwargs)
        self.calls.append(time.perf_counter() - started)
        data = json.loads(metadata.read_text('utf-8'))
        data.update(runtime_embed_seconds=self.calls, native_run_seconds=self.native_calls)
        if self.profile_calls:
            self.profile_calls -= 1
            if not self.profile_calls:
                trace = Path(self.model.end_profiling())
                operators = {}
                for event in json.loads(trace.read_text('utf-8')):
                    if event.get('cat') != 'Node':
                        continue
                    if not event.get('name', '').endswith('_kernel_time'):
                        continue
                    name = event.get('args', {}).get('op_name', 'unknown')
                    entry = operators.setdefault(name, {'calls': 0, 'microseconds': 0})
                    entry['calls'] += 1
                    entry['microseconds'] += event.get('dur', 0)
                data['operators'] = dict(sorted(operators.items(),
                    key=lambda item: -item[1]['microseconds']))
        metadata.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return result
worker.profiles = candidate.profiles
worker.EmbeddingRuntime = MeasuredRuntime
raise SystemExit(worker.main())
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-run", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--operator-profile", action="store_true")
    parser.add_argument("--configurations", nargs="+", choices=list(CONFIGURATIONS))
    parser.add_argument("--derived-graph", type=Path)
    parser.add_argument(
        "--affinity-trial", action="store_true", help="Windows physical-core ABBA diagnostic"
    )
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 8:
        parser.error("repetitions must be 2..8")
    selected = args.configurations or [key for key in CONFIGURATIONS if key != "affinity"]
    cpu_topology = None
    if args.affinity_trial:
        from benchmark_cpu_affinity import topology

        if args.operator_profile or args.derived_graph or args.configurations:
            parser.error("affinity-trial is separate from other configuration/profiling selection")
        cpu_topology = topology()
        selected = ["baseline", "affinity", "affinity", "baseline"]
    elif "affinity" in selected:
        parser.error("select the complete paired test with affinity-trial")
    if args.operator_profile:
        selected = ["baseline"]
    if selected[0] != "baseline":
        parser.error("baseline must be first for vector equivalence")
    source = json.loads((args.capacity_run / "capacity-results.json").read_text("utf-8"))
    raw = (args.capacity_run / "input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["input_sha256"]:
        raise ValueError("original dense capacity input changed")
    paragraphs = raw.decode("utf-8").split("\n\f\n")
    if len(paragraphs) != 9950 or len(set(paragraphs)) != 1:
        raise ValueError("expected original repeated 9,950-passage control")
    profile = profiles()["granite-107m"]
    derived_hash = None
    if args.derived_graph:
        args.derived_graph = args.derived_graph.resolve(strict=True)
        derived_hash = hashlib.sha256(args.derived_graph.read_bytes()).hexdigest()
        if args.operator_profile:
            parser.error("derived comparison is separate from operator profiling")
        selected = ["baseline", "derived"]
    root = args.directory / f"options-{new_id()}"
    root.mkdir()
    probes = [
        "Rainfall and soil moisture were measured.",
        "L'expérience mesure les précipitations et l'humidité du sol.",
        "تقيس التجربة هطول الأمطار ورطوبة التربة.",
    ]
    reference = None
    rows = []
    diagnostic_started = time.monotonic()

    def remaining_timeout():
        remaining = 600 - (time.monotonic() - diagnostic_started)
        if remaining <= 0:
            raise TimeoutError("diagnostic total budget exceeded")
        return min(120, remaining)

    for trial, label in enumerate(selected, start=1):
        config = {
            **CONFIGURATIONS["baseline" if label == "derived" else label],
            "metadata": str(root / f"{trial:02d}-{label}-metadata.json"),
            "profile_calls": args.repetitions + 3 if args.operator_profile else 0,
        }
        if label == "affinity":
            assert cpu_topology is not None
            config["affinity_cpus"] = cpu_topology["selected_cpus"]
        if label == "derived":
            config["derived_graph"] = str(args.derived_graph)
        config_file = root / f"{trial:02d}-{label}-config.json"
        config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            "-c",
            CHILD,
            str(Path(__file__).resolve().parent),
            str(config_file),
        ]
        session = None
        try:
            with EmbeddingSession(
                args.models_directory / profile.identity,
                profile,
                command=command,
                load_timeout=remaining_timeout(),
            ) as session:
                texts = [paragraphs[0]] * config["batch"]
                vectors = session.embed(texts, timeout=remaining_timeout())  # unmeasured warm-up
                times = []
                for _ in range(args.repetitions):
                    started = time.perf_counter()
                    vectors = session.embed(texts, timeout=remaining_timeout())
                    times.append(time.perf_counter() - started)
                probe_vectors = session.embed(probes, timeout=remaining_timeout()) + session.embed(
                    probes, query=True, timeout=remaining_timeout()
                )
                if reference is None:
                    reference = (vectors[0], probe_vectors)
                expected = [reference[0]] * config["batch"] + reference[1]
                row = {
                    "configuration": label,
                    "trial": trial,
                    "status": "passed",
                    "batch_size": config["batch"],
                    "threads": config["threads"],
                    "cpu_arena": config["arena"],
                    "derived_graph_sha256": derived_hash if label == "derived" else None,
                    "load_seconds": session.load_seconds,
                    "batch_seconds": times,
                    "median_batch_seconds": statistics.median(times),
                    "documents_per_second": config["batch"] / statistics.median(times),
                    "child_peak_bytes": session.peak_bytes,
                    "vector_agreement": agreement(expected, vectors + probe_vectors),
                    "actual_settings": json.loads(Path(config["metadata"]).read_text("utf-8")),
                }
            row["launcher_reaped"] = session.process.poll() is not None
        except (EmbeddingFailure, TimeoutError) as error:
            row = {
                "configuration": label,
                "trial": trial,
                "status": "failed",
                "batch_size": config["batch"],
                "threads": config["threads"],
                "cpu_arena": config["arena"],
                "failure_code": error.code if isinstance(error, EmbeddingFailure) else "timeout",
                "child_peak_bytes": session.peak_bytes if session else None,
                "launcher_reaped": session.process.poll() is not None if session else True,
            }
        rows.append(row)
    result = {
        "scope": "bounded dense inference options; not readiness or full retrieval quality",
        "profile": profile.id,
        "manifest": profile.identity,
        "input_sha256": source["input_sha256"],
        "profiling_only": args.operator_profile,
        "repetitions": args.repetitions,
        "cpu_topology": cpu_topology,
        "elapsed_seconds": time.monotonic() - diagnostic_started,
        "total_budget_seconds": 600,
        "configurations": rows,
    }
    verify_installation(args.models_directory / profile.identity, profile)
    if (
        args.derived_graph
        and hashlib.sha256(args.derived_graph.read_bytes()).hexdigest() != derived_hash
    ):
        raise ValueError("derived graph changed during inference")
    (root / "options-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
