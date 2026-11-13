"""Recheck saved English rankings against frozen citation spans; no inference.

Only top-ten ranks were retained. Missing items' deeper rank is explicitly unknown.
Native-chunk comparisons are separate observations, never same-input replacements.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark_pdf_search import DEFINITIONS, coverage
from benchmark_shared_search import hit

PROFILE = "potion-multilingual-static"


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def english_cases(data):
    rows = [case for case in data["quality_cases"] if case["question_group"] == "english"]
    if len(rows) != 13 or len({case["id"] for case in rows}) != 13:
        raise ValueError("expected thirteen distinct English questions")
    return {case["id"]: case for case in rows}


def reconstruct(document, case, manifest):
    lookup = {int(chunk["id"]): chunk for chunk in document["chunks"]}
    report = SimpleNamespace(passages=[SimpleNamespace(**p) for p in document["passages"]])
    rankings = {
        method: [hit(lookup[int(identity)], 0, manifest) for identity in identities]
        for method, identities in case["top_chunks"].items()
    }
    for method, ranking in rankings.items():
        if len(ranking) != len({item.chunk_id for item in ranking}) or len(ranking) > 10:
            raise ValueError("saved ranking has duplicate or excessive hits")
        measured = dict(
            zip(
                ("passage_hit", "span_covered"),
                coverage(ranking, document["expected_evidence"][case["id"]], report),
                strict=True,
            )
        )
        if measured != case["scores"][method]:
            raise ValueError("reconstructed evidence scores disagree with retained results")
    return rankings, report, lookup


def span_detail(document, ranking, span):
    passages = {p["id"]: p for p in document["passages"]}
    text = passages[span["passage_id"]]["text"]
    start, end = span["start"], span["end"]
    segments, supporting = [], []
    for rank, item in enumerate(ranking, start=1):
        for link in item.mappings:
            if link.passage_id != span["passage_id"]:
                continue
            left = max(start, link.start_offset or 0)
            right = min(end, len(text) if link.end_offset is None else link.end_offset)
            if left < right:
                segments.append((left, right))
                supporting.append(
                    {"rank": rank, "chunk_id": item.chunk_id, "text": text[left:right]}
                )
    cursor, missing = start, []
    for left, right in sorted(segments):
        if left > cursor:
            missing.append({"start": cursor, "end": left, "text": text[cursor:left]})
        cursor = max(cursor, right)
    if cursor < end:
        missing.append({"start": cursor, "end": end, "text": text[cursor:end]})
    covered = end - start - sum(part["end"] - part["start"] for part in missing)
    return {
        "passage_id": span["passage_id"],
        "page": span["page"],
        "quote": text[start:end],
        "expected_codepoints": end - start,
        "covered_codepoints": covered,
        "covered_fraction": covered / (end - start),
        "supporting_hits": supporting,
        "missing_slices": missing,
    }


def analyze(shared, native, recheck_file=None):
    definitions = read_json(DEFINITIONS)
    question_hash = hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest()
    corpus_path = shared / "shared-corpus.json"
    corpus = read_json(corpus_path)
    potion = read_json(shared / f"{PROFILE}-quality.json")
    e5 = read_json(shared / "multilingual-e5-quality.json")
    native_corpus = read_json(native / "shared-corpus.json")
    native_potion = read_json(native / f"{PROFILE}-quality.json")
    receipts = {}
    for directory, inputs in (
        (shared, ("shared-corpus.json", f"{PROFILE}-quality.json", "multilingual-e5-quality.json")),
        (native, ("shared-corpus.json", f"{PROFILE}-quality.json")),
    ):
        for name in inputs:
            receipts[f"{directory.name}/{name}"] = hashlib.sha256(
                (directory / name).read_bytes()
            ).hexdigest()
    shared_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    native_hash = hashlib.sha256((native / "shared-corpus.json").read_bytes()).hexdigest()
    for data, expected_hash in (
        (potion, shared_hash),
        (e5, shared_hash),
        (native_potion, native_hash),
    ):
        if data["definition_sha256"] != question_hash or data["corpus_sha256"] != expected_hash:
            raise ValueError("saved results do not match unchanged frozen input")
    if (
        corpus["definition_sha256"] != question_hash
        or native_corpus["definition_sha256"] != question_hash
    ):
        raise ValueError("frozen labels changed")
    if potion["manifest"] != native_potion["manifest"]:
        raise ValueError("native comparison used a different Potion artifact")
    cases, native_cases, e5_cases = map(english_cases, (potion, native_potion, e5))
    recheck_cases = None
    if recheck_file is not None:
        recheck = read_json(recheck_file)
        if (
            recheck["status"] != "completed"
            or recheck["manifest"] != potion["manifest"]
            or recheck["definition_sha256"] != question_hash
            or recheck["corpus_sha256"] != shared_hash
        ):
            raise ValueError("rankings recheck differs from original model/input")
        original_all = {(row["id"], row["language"]): row for row in potion["quality_cases"]}
        if len(recheck["quality_cases"]) != len(original_all):
            raise ValueError("rankings recheck has a different question count")
        for row in recheck["quality_cases"]:
            old = original_all[(row["id"], row["language"])]
            if row["scores"] != old["scores"] or row["top_chunks"] != old["top_chunks"]:
                raise ValueError("recheck top-ten scores/rankings changed")
            for method, identities in row["candidate_top_chunks"].items():
                if len(identities) > 50 or len(identities) != len(set(identities)):
                    raise ValueError("invalid retained candidate ranking")
                if identities[:10] != old["top_chunks"][method]:
                    raise ValueError("candidate prefix differs from original top ten")
        recheck_cases = english_cases(recheck)
        receipts["rankings-recheck.json"] = hashlib.sha256(recheck_file.read_bytes()).hexdigest()
    questions = {q["id"]: q for q in definitions["questions"]}
    misses, comparisons = [], []
    for identity, case in cases.items():
        document = corpus["documents"][case["source"]]
        rankings, report, lookup = reconstruct(document, case, potion["manifest"])
        native_case = native_cases[identity]
        reconstruct(
            native_corpus["documents"][case["source"]], native_case, native_potion["manifest"]
        )
        reconstruct(document, e5_cases[identity], e5["manifest"])
        if case["top_chunks"]["word"] != e5_cases[identity]["top_chunks"]["word"]:
            raise ValueError("word-search control differs on identical chunks")
        word_ids, vector_ids = map(set, (case["top_chunks"]["word"], case["top_chunks"]["vector"]))
        combined = list(
            {item.chunk_id: item for item in rankings["word"] + rankings["vector"]}.values()
        )
        entry = {
            "id": identity,
            "source": case["source"],
            "query": questions[identity]["query"],
            "shared_scores": case["scores"],
            "e5_shared_vector_complete": e5_cases[identity]["scores"]["vector"]["span_covered"],
            "native_scores": native_case["scores"],
            "word_vector_shared_chunk_ids": sorted(word_ids & vector_ids),
            "word_vector_top10_jaccard": len(word_ids & vector_ids) / len(word_ids | vector_ids),
            "word_vector_top10_union_complete": coverage(
                combined, document["expected_evidence"][identity], report
            )[1],
            "union_scope": "up to twenty diagnostic hits, not top-ten fusion",
        }
        if recheck_cases is not None:
            first_complete = {}
            for method, identities in recheck_cases[identity]["candidate_top_chunks"].items():
                candidate_hits = [
                    hit(lookup[int(key)], 0, potion["manifest"]) for key in identities
                ]
                first_complete[method] = next(
                    (
                        limit
                        for limit in range(1, len(candidate_hits) + 1)
                        if coverage(
                            candidate_hits[:limit], document["expected_evidence"][identity], report
                        )[1]
                    ),
                    None,
                )
            entry["first_complete_candidate_prefix"] = first_complete
        comparisons.append(entry)
        if case["scores"]["vector"]["span_covered"]:
            continue
        groups = []
        for group in document["expected_evidence"][identity]:
            alternatives = [span_detail(document, rankings["vector"], span) for span in group]
            groups.append(max(alternatives, key=lambda item: item["covered_fraction"]))
        intersections = [group["covered_codepoints"] for group in groups]
        classification = (
            "partial_required_span"
            if all(intersections)
            else "missing_required_group"
            if any(intersections)
            else "same_passage_wrong_fragment"
            if case["scores"]["vector"]["passage_hit"]
            else "no_required_evidence_in_top10"
        )
        misses.append(
            {
                **entry,
                "classification": classification,
                "required_groups_best_alternative": groups,
                "top10_vector_text": [
                    {"rank": rank, "chunk_id": chunk_id, "text": lookup[int(chunk_id)]["text"]}
                    for rank, chunk_id in enumerate(case["top_chunks"]["vector"], start=1)
                ],
                "rank_beyond_top10": "fresh top-fifty recheck; original top-ten results identical"
                if recheck_cases is not None
                else (
                    "unknown; original runner saved only top-ten IDs "
                    "and overwrote earlier-source vectors"
                ),
            }
        )
    if len(misses) != 7:
        raise ValueError("expected exactly seven shared-input Potion English misses")
    summary = {
        "english_questions": 13,
        "shared_complete": {
            method: sum(case["scores"][method]["span_covered"] for case in cases.values())
            for method in ("word", "vector", "hybrid")
        },
        "native_complete": {
            method: sum(case["scores"][method]["span_covered"] for case in native_cases.values())
            for method in ("word", "vector", "hybrid")
        },
        "miss_classifications": {
            key: sum(case["classification"] == key for case in misses)
            for key in sorted({case["classification"] for case in misses})
        },
        "shared_vector_misses_recovered_by_word": [
            case["id"] for case in misses if case["shared_scores"]["word"]["span_covered"]
        ],
        "shared_vector_misses_recovered_by_fixed_hybrid": [
            case["id"] for case in misses if case["shared_scores"]["hybrid"]["span_covered"]
        ],
        "shared_vector_misses_recovered_by_native_vector": [
            case["id"] for case in misses if case["native_scores"]["vector"]["span_covered"]
        ],
        "shared_vector_completes_word_misses": [
            case["id"]
            for case in comparisons
            if case["shared_scores"]["vector"]["span_covered"]
            and not case["shared_scores"]["word"]["span_covered"]
        ],
    }
    return {
        "scope": "saved-rankings/evidence analysis; no model inference or new retrieval rankings",
        "definition_sha256": question_hash,
        "corpus_sha256": shared_hash,
        "manifest": potion["manifest"],
        "input_receipts": receipts,
        "scores_reconstructed_and_verified": True,
        "summary": summary,
        "misses": misses,
        "all_english_comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-directory", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rankings-recheck-file", type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be absent with an existing parent")
    result = analyze(args.shared_directory, args.native_directory, args.rankings_recheck_file)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
