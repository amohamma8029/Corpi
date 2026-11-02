"""Diagnose saved multilingual misses without inference or changing frozen labels."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from analyze_potion_misses import read_json, reconstruct, span_detail
from benchmark_contextual_multilingual import UTF8InputTokenizer
from benchmark_pdf_search import DEFINITIONS, coverage
from benchmark_shared_search import candidates, hit, static_profile
from benchmark_static_multilingual import StaticTokenizer

from core.search.profiles import verify_installation

PROFILES = ("amgix-static-retrieval", "potion-multilingual-static", "bekko-a8m")
NATIVE_FOLDERS = ("amgix-native-chunks-v1", "potion-native-chunks-v1", "bekko-native-chunks-v1")


def cases(data):
    rows = data["quality_cases"]
    result = {(row["id"], row["language"]): row for row in rows}
    if data["status"] != "completed" or len(rows) != len(result) or len(result) != 33:
        raise ValueError("expected 33 distinct completed cases")
    return result


def ranked_hits(document, case, manifest, method):
    identities = case["candidate_top_chunks"][method]
    if len(identities) > 50 or len(identities) != len(set(identities)):
        raise ValueError("invalid retained candidate ranking")
    if identities[:10] != case["top_chunks"][method]:
        raise ValueError("candidate prefix differs from original top ten")
    lookup = {int(chunk["id"]): chunk for chunk in document["chunks"]}
    return [hit(lookup[int(identity)], 0, manifest) for identity in identities]


def first_complete(ranking, evidence, report):
    return next(
        (
            limit
            for limit in range(1, len(ranking) + 1)
            if coverage(ranking[:limit], evidence, report)[1]
        ),
        None,
    )


def details(document, case, manifest):
    rankings, report, lookup = reconstruct(document, case, manifest)
    evidence = document["expected_evidence"][case["id"]]
    candidates = ranked_hits(document, case, manifest, "vector")
    groups = []
    for alternatives in evidence:
        choices = [span_detail(document, rankings["vector"], span) for span in alternatives]
        best_index = max(range(len(choices)), key=lambda index: choices[index]["covered_fraction"])
        best = choices[best_index]
        best["first_group_complete_prefix"] = first_complete(candidates, [alternatives], report)
        best["all_supporting_candidate_hits"] = span_detail(
            document, candidates, alternatives[best_index]
        )["supporting_hits"]
        groups.append(best)
    intersections = [group["covered_codepoints"] for group in groups]
    classification = (
        "complete"
        if case["scores"]["vector"]["span_covered"]
        else "partial_required_span"
        if all(intersections)
        else "missing_required_group"
        if any(intersections)
        else "same_passage_wrong_fragment"
        if case["scores"]["vector"]["passage_hit"]
        else "no_required_evidence_in_top10"
    )
    # Require at least one retained hit for each group, even if no one hit contains
    # the whole span. Distinguish low-ranked fragments from absent candidate evidence.
    union_complete = coverage(candidates, evidence, report)[1]
    return {
        "scores": case["scores"],
        "classification": classification,
        "first_complete_candidate_prefix": first_complete(candidates, evidence, report),
        "retained_candidates": len(candidates),
        "top50_complete": union_complete,
        "required_groups": groups,
        "top10_vector_text": [
            {"rank": rank, "chunk_id": identity, "text": lookup[int(identity)]["text"]}
            for rank, identity in enumerate(case["top_chunks"]["vector"], 1)
        ],
    }


def analyze(runs, potion_recheck, models_directory=None, trial_directory=None):
    shared = runs / "shared-multilingual-v3"
    corpus_file = shared / "shared-corpus.json"
    corpus = read_json(corpus_file)
    corpus_hash = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    definition_hash = hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest()
    definitions = read_json(DEFINITIONS)
    questions = {item["id"]: item for item in definitions["questions"]}
    receipts = {}

    def retain(path):
        receipts[f"{path.parent.name}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_json(path)

    retain(corpus_file)
    catalogs = {}
    native_catalogs = {}
    for name, folder in zip(PROFILES, NATIVE_FOLDERS, strict=True):
        data = retain(shared / f"{name}-quality.json")
        if (data["corpus_sha256"], data["definition_sha256"]) != (corpus_hash, definition_hash):
            raise ValueError("shared input/definition changed")
        native_corpus_file = runs / folder / "shared-corpus.json"
        native_corpus = retain(native_corpus_file)
        native = retain(runs / folder / f"{name}-quality.json")
        if native["manifest"] != data["manifest"] or native["definition_sha256"] != definition_hash:
            raise ValueError("native model/definition changed")
        if native["corpus_sha256"] != hashlib.sha256(native_corpus_file.read_bytes()).hexdigest():
            raise ValueError("native inputs changed")
        catalogs[name] = (data, cases(data))
        native_catalogs[name] = (native, cases(native), native_corpus)
    baseline, baseline_cases = catalogs[PROFILES[1]]
    recheck = retain(potion_recheck)
    if (recheck["corpus_sha256"], recheck["definition_sha256"], recheck["manifest"]) != (
        corpus_hash,
        definition_hash,
        baseline["manifest"],
    ):
        raise ValueError("Potion recheck model/input changed")
    recheck_cases = cases(recheck)
    for key, original in baseline_cases.items():
        fresh = recheck_cases[key]
        if fresh["scores"] != original["scores"] or fresh["top_chunks"] != original["top_chunks"]:
            raise ValueError("Potion recheck changed original scores/rankings")
    catalogs[PROFILES[1]] = (baseline, recheck_cases)
    # Reconstruct every saved score, not just the failures selected for inspection.
    for key, case in catalogs[PROFILES[0]][1].items():
        reference_words = case["top_chunks"]["word"]
        for name in PROFILES:
            data, rows = catalogs[name]
            selected = rows[key]
            document = corpus["documents"][selected["source"]]
            reconstruct(document, selected, data["manifest"])
            ranked_hits(document, selected, data["manifest"], "vector")
            if selected["top_chunks"]["word"] != reference_words:
                raise ValueError("word control changed on identical inputs")
            native, native_rows, native_corpus = native_catalogs[name]
            reconstruct(
                native_corpus["documents"][selected["source"]], native_rows[key], native["manifest"]
            )
    misses = []
    for key, case in catalogs[PROFILES[0]][1].items():
        if case["scores"]["vector"]["span_covered"] or case["question_group"] == "english":
            continue
        question = questions[case["id"]]
        entry = {key: case[key] for key in ("id", "language", "source", "question_group")}
        entry["query"] = (
            question[case["language"]]
            if case["question_group"] == "cross_language"
            else question["query"]
        )
        entry["models"] = {}
        for name in PROFILES:
            data, rows = catalogs[name]
            native, native_rows, native_corpus = native_catalogs[name]
            native_case = native_rows[key]
            native_document = native_corpus["documents"][case["source"]]
            native_rankings, _, _ = reconstruct(native_document, native_case, native["manifest"])
            native_groups = [
                max(
                    (
                        span_detail(native_document, native_rankings["vector"], span)
                        for span in group
                    ),
                    key=lambda item: item["covered_fraction"],
                )
                for group in native_document["expected_evidence"][case["id"]]
            ]
            entry["models"][name] = {
                **details(corpus["documents"][case["source"]], rows[key], data["manifest"]),
                "native_scores": native_case["scores"],
                "native_required_groups": native_groups,
                "native_chunks": len(native_document["chunks"]),
            }
        misses.append(entry)
    if len(misses) != 7:
        raise ValueError("expected five cross-language and two French shared-input misses")
    compact = [
        {
            "id": row["id"],
            "language": row["language"],
            "group": row["question_group"],
            "classification": row["models"][PROFILES[0]]["classification"],
            "first_complete_prefix": {
                name: row["models"][name]["first_complete_candidate_prefix"] for name in PROFILES
            },
            "native_complete": {
                name: row["models"][name]["native_scores"]["vector"]["span_covered"]
                for name in PROFILES
            },
            "amgix_missing_slices": [
                group["missing_slices"] for group in row["models"][PROFILES[0]]["required_groups"]
            ],
        }
        for row in misses
    ]
    result = {
        "scope": "saved rankings only; no inference, reranking or changed labels",
        "definition_sha256": definition_hash,
        "corpus_sha256": corpus_hash,
        "input_receipts": receipts,
        "all_three_models_scores_reconstructed": True,
        "summary": compact,
        "misses": misses,
    }
    if models_directory is not None:
        tokenizers = {}
        catalog = candidates()
        for name, manifest in corpus["candidate_manifests"].items():
            profile = catalog[name]
            if profile.identity != manifest:
                raise ValueError("original chunk-counting model identity changed")
            installed = models_directory / manifest
            verify_installation(installed, profile)
            cls = StaticTokenizer if static_profile(profile) else UTF8InputTokenizer
            tokenizers[name] = cls(installed, profile)
        fits = []
        for entry in misses:
            if entry["source"] != "udhr-french":
                continue
            document = corpus["documents"][entry["source"]]
            groups = entry["models"][PROFILES[0]]["required_groups"]
            ids = {group["passage_id"] for group in groups}
            whole = "\n\n".join(p["text"] for p in document["passages"] if p["id"] in ids)
            counts = {name: tokenizer.count(whole) for name, tokenizer in tokenizers.items()}
            fits.append(
                {
                    "id": entry["id"],
                    "whole_saved_passages_text": whole,
                    "content_tokens": counts,
                    "maximum_content_tokens": max(counts.values()),
                    "fits_original_64_token_budget": max(counts.values()) <= 64,
                }
            )
        result["french_whole_statement_fit"] = fits
    if trial_directory is not None:
        trial_file = trial_directory / "shared-corpus.json"
        trial_corpus = retain(trial_file)
        trial_hash = hashlib.sha256(trial_file.read_bytes()).hexdigest()
        if trial_corpus.get("original_shared_corpus_sha256") != corpus_hash:
            raise ValueError("packing trial does not derive from original shared inputs")
        if trial_corpus["candidate_manifests"] != corpus["candidate_manifests"]:
            raise ValueError("packing trial changed declared counting tokenizers")
        for source, document in corpus["documents"].items():
            changed = trial_corpus["documents"][source]
            if any(
                changed[key] != document[key]
                for key in ("pdf_sha256", "saved_passages_sha256", "passages", "expected_evidence")
            ):
                raise ValueError("packing trial changed source text or frozen evidence")
        summaries = []
        word_control = None
        for name in PROFILES:
            trial_data = retain(trial_directory / f"{name}-quality.json")
            original, original_rows = catalogs[name]
            if (
                trial_data["manifest"],
                trial_data["corpus_sha256"],
                trial_data["definition_sha256"],
            ) != (original["manifest"], trial_hash, definition_hash):
                raise ValueError("packing trial model/inputs changed")
            rows = cases(trial_data)
            current_words = [(key, row["top_chunks"]["word"]) for key, row in rows.items()]
            if word_control is None:
                word_control = current_words
            if current_words != word_control:
                raise ValueError("packing trial word rankings differ between models")
            changes = []
            for key, row in rows.items():
                reconstruct(trial_corpus["documents"][row["source"]], row, trial_data["manifest"])
                before = original_rows[key]["scores"]["vector"]["span_covered"]
                after = row["scores"]["vector"]["span_covered"]
                if before != after:
                    changes.append(
                        {
                            "id": row["id"],
                            "language": row["language"],
                            "group": row["question_group"],
                            "change": "gained" if after else "lost",
                        }
                    )
            summaries.append(
                {
                    "profile": name,
                    "changes": changes,
                    "vector_coverage": {
                        group: {
                            "original": sum(
                                row["scores"]["vector"]["span_covered"]
                                for row in original_rows.values()
                                if row["question_group"] == group
                            ),
                            "trial": sum(
                                row["scores"]["vector"]["span_covered"]
                                for row in rows.values()
                                if row["question_group"] == group
                            ),
                        }
                        for group in ("english", "cross_language", "non_english")
                    },
                }
            )
        result["packing_trial"] = {
            "scope": "new chunks; original frozen benchmark unchanged",
            "corpus_sha256": trial_hash,
            "profiles": summaries,
            "original_chunks": sum(len(doc["chunks"]) for doc in corpus["documents"].values()),
            "trial_chunks": sum(len(doc["chunks"]) for doc in trial_corpus["documents"].values()),
            "source_passages_labels_and_declared_tokenizers_unchanged": True,
            "identical_trial_word_rankings": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-directory", type=Path, required=True)
    parser.add_argument("--potion-rankings-recheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models-directory", type=Path, help="Optional token-count inspection only"
    )
    parser.add_argument("--trial-directory", type=Path)
    parser.add_argument("--export-evidence", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be absent with an existing parent")
    result = analyze(
        args.runs_directory,
        args.potion_rankings_recheck,
        args.models_directory,
        args.trial_directory,
    )
    if args.export_evidence:
        if args.trial_directory is None:
            parser.error("export-evidence requires trial-directory")
        folder = args.output.parent / "evidence"
        if folder.exists():
            parser.error("evidence export already exists")
        core = Path(__file__).resolve().parents[1]
        sources = {
            f"source/{relative}": core / relative
            for relative in (
                "scripts/analyze_amgix_misses.py",
                "scripts/analyze_potion_misses.py",
                "scripts/benchmark_shared_search.py",
                "scripts/benchmark_pdf_search.py",
                "src/core/search/chunking.py",
                "tests/search/test_chunking.py",
                "benchmarks/search_questions.json",
            )
        }
        for filename in (
            "shared-corpus.json",
            "shared-summary.json",
            *(f"{profile}-quality.json" for profile in PROFILES),
        ):
            sources[f"trial/{filename}"] = args.trial_directory / filename
        sources["original/potion-top50-recheck.json"] = args.potion_rankings_recheck
        for profile in PROFILES:
            sources[f"original/{profile}-quality.json"] = (
                args.runs_directory / "shared-multilingual-v3" / f"{profile}-quality.json"
            )
        receipts = []
        for relative, source in sources.items():
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("analysis evidence export mismatch")
            receipts.append({"path": relative, "sha256": digest, "bytes": target.stat().st_size})
        result["export_receipts"] = receipts
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))
    if "french_whole_statement_fit" in result:
        print(json.dumps(result["french_whole_statement_fit"], indent=2, ensure_ascii=True))
    if "packing_trial" in result:
        print(json.dumps(result["packing_trial"], indent=2))


if __name__ == "__main__":
    main()
