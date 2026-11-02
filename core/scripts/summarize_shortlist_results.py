"""Validate/export the three-model comparison without running inference."""

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

from benchmark_shared_search import candidates

from core.search.profiles import verify_installation

NAMES = ("potion-multilingual-static", "amgix-static-retrieval", "bekko-a8m")
GROUPS = {"english": 13, "cross_language": 10, "non_english": 10}
NATIVE = {
    NAMES[0]: "potion-native-chunks-v1",
    NAMES[1]: "amgix-native-chunks-v1",
    NAMES[2]: "bekko-native-chunks-v1",
}


def read(path):
    return json.loads(path.read_text("utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(data):
    if data["status"] != "completed" or len(data["quality_cases"]) != 33:
        raise ValueError("incomplete quality result")
    result = {}
    for group, denominator in GROUPS.items():
        cases = [item for item in data["quality_cases"] if item["question_group"] == group]
        if len(cases) != denominator:
            raise ValueError("question denominator changed")
        result[group] = {
            "questions": denominator,
            **{
                method: sum(item["scores"][method]["span_covered"] for item in cases)
                for method in ("word", "vector", "hybrid")
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-directory", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--full-capacity-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_directory / "provenance.json").exists():
        raise ValueError("evidence packet already exists")
    shared = args.runs_directory / "shared-multilingual-v3"
    corpus_file = shared / "shared-corpus.json"
    corpus_hash = digest(corpus_file)
    definition = Path(__file__).resolve().parents[1] / "benchmarks/search_questions.json"
    definition_hash = digest(definition)
    exports = {
        "shared/shared-corpus.json": corpus_file,
        "definitions/search_questions.json": definition,
    }
    rows = []
    reference = None
    baseline = {}
    for name in NAMES:
        profile = candidates()[name]
        verify_installation(args.models_directory / profile.identity, profile)
        path = shared / f"{name}-quality.json"
        data = read(path)
        if (data["corpus_sha256"], data["definition_sha256"], data["manifest"]) != (
            corpus_hash,
            definition_hash,
            profile.identity,
        ):
            raise ValueError("shared model/input identity changed")
        words = [
            (item["id"], item["language"], item["source"], item["top_chunks"]["word"])
            for item in data["quality_cases"]
        ]
        if reference is None:
            reference = words
            baseline = {(item["id"], item["language"]): item for item in data["quality_cases"]}
        if words != reference:
            raise ValueError("native word rankings differ on shared inputs")
        changes = {"gained": [], "lost": []}
        for item in data["quality_cases"]:
            before = baseline[(item["id"], item["language"])]["scores"]["vector"]["span_covered"]
            after = item["scores"]["vector"]["span_covered"]
            if before != after:
                changes["gained" if after else "lost"].append(
                    {key: item[key] for key in ("id", "language", "question_group")}
                )
        native_dir = args.runs_directory / NATIVE[name]
        native_file = native_dir / f"{name}-quality.json"
        native = read(native_file)
        native_corpus = native_dir / "shared-corpus.json"
        if (
            native["corpus_sha256"] != digest(native_corpus)
            or native["definition_sha256"] != definition_hash
            or native["manifest"] != profile.identity
        ):
            raise ValueError("native input identity changed")
        rows.append(
            {
                "profile": name,
                "manifest": profile.identity,
                "shared": counts(data),
                "native": counts(native),
                "changes_from_potion_shared_vector": changes,
                "shared_embedding_and_storage_seconds": sum(
                    source["embedding_and_storage_seconds"] for source in data["sources"].values()
                ),
                "shared_child_peak_bytes": data["child_peak_bytes"],
                "shared_load_seconds": data["load_seconds"],
                "native_chunks": {key: value["chunks"] for key, value in native["sources"].items()},
                "languages": {
                    language: {
                        group: {
                            "questions": sum(
                                item["language"] == language and item["question_group"] == group
                                for item in data["quality_cases"]
                            ),
                            "vector": sum(
                                item["scores"]["vector"]["span_covered"]
                                for item in data["quality_cases"]
                                if item["language"] == language and item["question_group"] == group
                            ),
                        }
                        for group in ("cross_language", "non_english")
                    }
                    for language in ("french", "arabic")
                },
            }
        )
        exports[f"shared/{name}-quality.json"] = path
        exports[f"native/{name}-quality.json"] = native_file
        exports[f"native/{name}-corpus.json"] = native_corpus
    capacity = read(args.full_capacity_run / "capacity-results.json")
    if not (
        capacity["readiness_reached"]
        and capacity["status"] == "completed"
        and capacity["manifest"] == candidates()[NAMES[1]].identity
        and capacity["planned_chunks"] == capacity["committed_chunks"] == 9950
        and capacity["input_sha256"] == capacity["original_input_sha256"]
        and capacity["input_sha256"] == digest(args.full_capacity_run / "input.txt")
        and capacity["preparation_wall_seconds"] < 600
        and capacity["following_small_ready"]
        and all(
            item["launcher_reaped"] and item["peak_bytes"] <= 1073741824
            for item in capacity["observed_sessions"]
        )
    ):
        raise ValueError("full preparation evidence does not meet stated boundary")
    for filename in (
        "shortlist-inspection.json",
        "amgix-static-retrieval-capacity-encoding.json",
        "bekko-a8m-speed-screen.json",
        "bekko-a8m-speed-screen-batch32.json",
        "bekko-a8m-runtime-metadata.json",
        "potion-capacity-encoding.json",
        "potion-capacity-varied-encoding.json",
    ):
        exports[f"raw/{filename}"] = shared / filename
    exports["raw/amgix-full-preparation.json"] = args.full_capacity_run / "capacity-results.json"
    for name in NAMES[1:]:
        exports[f"raw/{name}-setup.json"] = args.models_directory / f"{name}-setup.json"
    core = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/benchmark_shared_search.py",
        "scripts/benchmark_static_multilingual.py",
        "scripts/benchmark_contextual_multilingual.py",
        "scripts/benchmark_search_capacity.py",
        "scripts/inspect_shortlist_models.py",
        "scripts/summarize_shortlist_results.py",
        "tests/search/test_static_candidate.py",
        "tests/search/test_contextual_candidate.py",
        "benchmarks/static_retrieval_model.json",
        "benchmarks/contextual_multilingual_models.json",
        "benchmarks/static_multilingual_model.json",
    ):
        exports[f"source/{relative}"] = core / relative
    summary = {
        "scope": "three-model diagnostics; single runs; Potion results reused",
        "corpus_sha256": corpus_hash,
        "definition_sha256": definition_hash,
        "identical_word_rankings": True,
        "profiles": rows,
        "original_input_sha256": capacity["input_sha256"],
        "full_preparation_seconds": capacity["preparation_wall_seconds"],
        "full_preparation_child_peak_bytes": max(
            item["peak_bytes"] for item in capacity["observed_sessions"]
        ),
        "full_preparation_parent_peak_bytes": capacity["parent_observed_peak_bytes"],
        "bekko_full_capacity": "not run: screens below encoding-only floor of 16.58 passages/s",
        "hardware_platform": platform.platform(),
        "processor": platform.processor(),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    receipts = []
    for relative, source in exports.items():
        target = args.output_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != digest(source):
            raise ValueError("evidence export integrity failure")
        receipts.append(
            {"path": relative, "sha256": digest(target), "bytes": target.stat().st_size}
        )
    (args.output_directory / "provenance.json").write_text(
        json.dumps(
            {"files": receipts, "summary_sha256": digest(args.output_directory / "summary.json")},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
