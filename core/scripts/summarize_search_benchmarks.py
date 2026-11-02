"""Summarize explicit public benchmark results without exporting paths or source text."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*/result.json")):
        raw = path.read_bytes()
        result = json.loads(raw)
        row = {
            key: result.get(key)
            for key in (
                "pdf",
                "pdf_sha256",
                "profile",
                "manifest",
                "status",
                "failure",
                "run",
                "overlap_override",
                "batch_size",
                "threads",
                "pages",
                "chunks",
                "text_bytes",
                "core_import_to_ready_seconds",
                "search_preparation_seconds",
                "budget_charged_seconds",
                "model_reloaded",
                "parent_observed_peak_bytes",
                "platform",
                "processor",
                "python",
                "sqlite",
                "fastembed",
                "onnxruntime",
                "pipeline_version",
                "parser_name",
                "parser_version",
                "normalization_version",
                "fixture_definition_sha256",
                "effective_settings",
            )
        }
        row["result_sha256"] = hashlib.sha256(raw).hexdigest()
        row["child_peak_bytes"] = result.get("embedding", {}).get("child_peak_bytes")
        row["usable_measurement"] = (
            result["status"] == "ready" and (row["child_peak_bytes"] or 0) > 20_000_000
        )
        totals = {}
        for case in result.get("quality_cases", []):
            group = case.get("question_group") or (
                "english" if case["language"] == "english" else "cross_language"
            )
            for method, scores in case["scores"].items():
                keys = [group + ":" + method]
                if group != "english":
                    keys.append(group + ":" + case["language"] + ":" + method)
                for key in keys:
                    metric = totals.setdefault(
                        key, {"questions": 0, "passage_hits": 0, "span_covered": 0}
                    )
                    metric["questions"] += 1
                    metric["passage_hits"] += int(scores["passage_hit"])
                    metric["span_covered"] += int(scores["span_covered"])
        row["quality"] = totals
        rows.append(row)
    text = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.compact:
        for row in rows:
            if row["usable_measurement"]:
                print(
                    row["pdf"],
                    row["profile"],
                    "overlap",
                    row["overlap_override"],
                    "seconds",
                    round(row["core_import_to_ready_seconds"], 2),
                    "chunks",
                    row["chunks"],
                    "hybrid",
                    row["quality"].get("english:hybrid"),
                    "cross",
                    row["quality"].get("cross_language:hybrid"),
                    "non-English",
                    row["quality"].get("non_english:hybrid"),
                )
    else:
        print(text)


if __name__ == "__main__":
    main()
