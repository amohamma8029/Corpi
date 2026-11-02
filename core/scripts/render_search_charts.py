"""Render scoped PR previews from retained benchmark results, without new downloads.

Pillow is already installed with FastEmbed. Generated images, captions and
provenance belong in ignored evidence storage, not public architecture docs.
"""

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAMES = {
    "minilm-en": "English MiniLM",
    "multilingual-e5": "Multilingual E5",
    "bge-small-en": "BGE English",
    "multilingual-minilm": "Multilingual MiniLM",
}
COLORS = {
    "minilm-en": "#147d92",
    "multilingual-e5": "#7551b8",
    "bge-small-en": "#41628f",
    "multilingual-minilm": "#a06438",
}


@dataclass
class Bar:
    profile: str
    value: float
    detail: str
    label: str | None = None


def render(path: Path, title: str, subtitle: str, groups, maximum, tick, axis, notes):
    height = 290 + sum(50 + 64 * len(bars) for _, bars in groups) + 34 * len(notes)
    image = Image.new("RGB", (1600, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    heading = ImageFont.load_default(size=42)
    body = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=22)
    ink, muted = "#172b43", "#546477"
    draw.rectangle((0, 0, 1600, 12), fill="#147d92")
    draw.text((48, 42), title, font=heading, fill=ink)
    draw.text((48, 104), subtitle, font=small, fill=muted)
    left, right, top = 355, 1135, 205
    bottom = height - 75 - 34 * len(notes)
    for value in range(0, maximum + 1, tick):
        x = left + (right - left) * value / maximum
        draw.line((x, top - 12, x, bottom), fill="#e1e7ed", width=2)
        draw.text((x, top - 40), str(value), font=small, fill=muted, anchor="mm")
    y = top
    for name, bars in groups:
        draw.rectangle((40, y, 1560, y + 44), fill="#f2f5f8")
        draw.text((52, y + 10), name, font=body, fill=ink)
        y += 50
        for bar in bars:
            if not math.isfinite(bar.value) or not 0 <= bar.value <= maximum:
                raise ValueError("invalid chart value")
            center = y + 26
            draw.text(
                (52, center), bar.label or NAMES[bar.profile], font=body, fill=ink, anchor="lm"
            )
            end = left + (right - left) * bar.value / maximum
            if bar.value:
                draw.rounded_rectangle(
                    (left, center - 16, end, center + 16), radius=5, fill=COLORS[bar.profile]
                )
            draw.text((end + 12, center), bar.detail, font=small, fill=ink, anchor="lm")
            y += 64
    draw.text((left, bottom + 24), axis, font=small, fill=muted)
    y = bottom + 74
    for note in notes:
        if draw.textlength(note, font=small) > 1504:
            raise ValueError("chart note exceeds image width")
        draw.text((48, y), note, font=small, fill=muted)
        y += 34
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--platform-checks", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-baseline", type=Path)
    parser.add_argument("--filter-after", type=Path)
    args = parser.parse_args()
    if bool(args.filter_baseline) != bool(args.filter_after):
        parser.error("supply both filter baseline and after results")
    raw_results = args.results.read_bytes()
    raw_platform = args.platform_checks.read_bytes()
    results = json.loads(raw_results)
    platform = json.loads(raw_platform)
    ready = [r for r in results if r["status"] == "ready" and r["usable_measurement"]]
    selected = []
    timing_groups = []
    current_profiles = ("minilm-en", "multilingual-e5", "bge-small-en")
    for pdf, label in (
        ("bert", "BERT / 16 pages"),
        ("http-semantics", "HTTP Semantics / 194 pages"),
    ):
        bars = []
        for profile in current_profiles:
            rows = [
                r
                for r in ready
                if r["pdf"] == pdf
                and r["profile"] == profile
                and r["pipeline_version"] == "supervised-search-v2"
                and r["overlap_override"] == (8 if profile == "minilm-en" else None)
            ]
            if len(rows) != (2 if profile == "minilm-en" else 1):
                raise ValueError(f"ambiguous current timing group: {pdf}/{profile}")
            selected.extend(rows)
            times = [r["core_import_to_ready_seconds"] for r in rows]
            median = statistics.median(times)
            detail = f"{median:.1f} s | n={len(rows)}"
            if len(rows) > 1:
                detail += f", max {max(times):.1f} s"
            bars.append(Bar(profile, median, detail))
        timing_groups.append((label, bars))

    quality_groups = []
    historical = []
    for language, expected, label in (
        ("english", 13, "English questions / English sources (13 questions)"),
        ("cross_language", 10, "French + Arabic queries / English sources (10 questions)"),
    ):
        bars = []
        for profile in NAMES:
            rows = [
                r
                for r in ready
                if r["profile"] == profile
                and r["pipeline_version"] is None
                and r["overlap_override"] is None
            ]
            if {r["pdf"] for r in rows} != {"attention", "bert", "http-semantics"} or len(
                rows
            ) != 3:
                raise ValueError(f"ambiguous historical quality group: {profile}")
            scores = [
                r["quality"][f"{language}:hybrid"]
                for r in rows
                if f"{language}:hybrid" in r["quality"]
            ]
            total = sum(s["questions"] for s in scores)
            covered = sum(s["span_covered"] for s in scores)
            if total != expected or not 0 <= covered <= total:
                raise ValueError("quality denominator or coverage mismatch")
            bars.append(Bar(profile, 100 * covered / total, f"{covered}/{total}"))
            if language == "english":
                historical.extend(rows)
        quality_groups.append((label, bars))

    setup_bars = []
    for profile in ("minilm-en", "multilingual-e5"):
        setup = platform["managed_setup"][profile]
        if setup["transfer_attempts"] != 1:
            raise ValueError("setup chart expects the recorded single-attempt observations")
        seconds = setup["setup_seconds"]
        setup_bars.append(
            Bar(profile, seconds, f"{seconds:.1f} s | {setup['received_bytes'] / 1e6:.1f} MB")
        )

    memory_bars = []
    for profile in current_profiles:
        peaks = [r["child_peak_bytes"] / 1024**2 for r in selected if r["profile"] == profile]
        memory_bars.append(Bar(profile, max(peaks), f"{max(peaks):.0f} MiB"))

    args.output.mkdir(parents=True, exist_ok=True)
    render(
        args.output / "01-pdf-readiness-preview.png",
        "PDF import to searchable readiness",
        "PREVIEW | Windows core API benchmark snapshot",
        timing_groups,
        400,
        100,
        "Elapsed seconds (lower is faster)",
        [
            "Installed models; fresh imports; model reloads; two threads. Core API, not UI timing.",
            "MiniLM: overlap 8, batch 32. E5: batch 8. BGE: batch 32. Profile-specific chunks.",
            "Earlier snapshot; 194 pages, not the required 200-page fixture. Acceptance pending.",
        ],
    )
    render(
        args.output / "02-evidence-coverage-preview.png",
        "Does search find the expected evidence?",
        "PREVIEW | Historical four-candidate comparison",
        quality_groups,
        100,
        20,
        "Questions with all required evidence in the top 10 (%)",
        [
            "Benchmark-only hybrid: equal-weight RRF, k=60; up to 50 hits per method, top 10.",
            "Fixed English-source PDFs only; no French/Arabic source PDF evaluation in this set.",
            "Model-specific chunks; E5 batch 8, others 32. Evidence coverage, not answer accuracy.",
        ],
    )
    render(
        args.output / "03-model-setup-preview.png",
        "First-use model setup",
        "PREVIEW | Fresh authenticated downloads on Windows; one observation per model",
        [("Download + verification + installation", setup_bars)],
        360,
        60,
        "Setup seconds (network-dependent)",
        [
            "One transfer attempt each; files checked against pinned sizes and SHA-256 hashes.",
            "Separate from PDF preparation. Installed models support offline local inference.",
        ],
    )
    render(
        args.output / "04-inference-memory-preview.png",
        "Observed inference-child memory",
        "PREVIEW | Windows; maximum sampled peak across the BERT and 194-page runs above",
        [("Inference child only", memory_bars)],
        1024,
        256,
        "Sampled resident memory (MiB)",
        [
            "Interpreter RSS sampled every 250 ms. Parent/UI and between-sample spikes excluded.",
            "Two threads; MiniLM/BGE batch 32, E5 batch 8. Near-10k capacity remains unverified.",
        ],
    )

    assets = []
    for name in ("search-ready.png", "model-switch-ready.png"):
        raw = (args.screenshots / name).read_bytes()
        with Image.open(args.screenshots / name) as image:
            image.verify()
        (args.output / name).write_bytes(raw)
        assets.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest()})
    data = {
        "status": "preview; final acceptance pending",
        "results_sha256": hashlib.sha256(raw_results).hexdigest(),
        "platform_checks_sha256": hashlib.sha256(raw_platform).hexdigest(),
        "current_timing_records": selected,
        "historical_quality_records": historical,
        "platform_checks": platform,
        "screenshots": assets,
    }
    if args.filter_baseline:
        assert args.filter_after is not None
        before_raw = args.filter_baseline.read_bytes()
        after_raw = args.filter_after.read_bytes()
        before, after = json.loads(before_raw), json.loads(after_raw)
        if any(
            r["stored_chunks"] != 50000
            or r["eligible_chunks"] != 5000
            or not r["word_and_vector_exact_controls_passed"]
            for r in (before, after)
        ):
            raise ValueError("filter chart expects the verified 50k/5k corpus controls")
        bars = [
            Bar(
                "minilm-en",
                r["timings"]["word"]["median"],
                f"{r['timings']['word']['median']:.3f} s | n={r['repetitions']}",
                label,
            )
            for label, r in (("Before", before), ("After", after))
        ]
        render(
            args.output / "05-word-search-fix-preview.png",
            "Word-search query-plan fix",
            "PREVIEW | Windows; same saved corpus and exact expected results",
            [("50,000 stored chunks / 5,000 eligible / top 50", bars)],
            14,
            2,
            "Median elapsed seconds (lower is faster)",
            [
                "Common-word workload with stronger ineligible decoys; five requests per version.",
                "Filtering and canonical hits matched exact eligible controls before and after.",
                "Controlled vectors/storage benchmark; not model inference or PDF preparation.",
            ],
        )
        data["filter_comparison"] = {
            "baseline": before,
            "after": after,
            "baseline_sha256": hashlib.sha256(before_raw).hexdigest(),
            "after_sha256": hashlib.sha256(after_raw).hexdigest(),
        }
    (args.output / "chart-data.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "captions.md").write_text(
        "# PR image previews\n\n"
        "These images report the saved observations, not full M1.6 acceptance. Upload the PNGs "
        "with the PR; ignored local paths are not public evidence. "
        "Regenerate after affected benchmarks change.\n\n"
        "- `01-pdf-readiness-preview.png`: earlier Windows core API timings for BERT and RFC 9110. "
        "MiniLM shows median/slowest of two runs; E5/BGE have one observation each. "
        "Installed models, "
        "fresh indexes, model reloads. Different model chunking/settings; 194 pages, not 200.\n"
        "- `02-evidence-coverage-preview.png`: historical four-candidate "
        "expected-evidence coverage "
        "in benchmark hybrid top 10. Denominators are 13 English questions and ten French/Arabic "
        "queries against English sources. No answer-accuracy or non-English-source support claim.\n"
        "- `03-model-setup-preview.png`: actual fresh download, verification and installation. "
        "One observation per model; times depend on the network "
        "and are separate from PDF readiness.\n"
        "- `04-inference-memory-preview.png`: maximum sampled inference-child RSS across the "
        "displayed current-pipeline runs. Excludes parent/UI memory and between-sample spikes.\n"
        "- `search-ready.png`: real native Windows app showing completed search preparation "
        "and the retained original for the controlled two-page fixture. "
        "The alternative shows a failed earlier setup attempt before explicit Retry.\n"
        "- `model-switch-ready.png`: same native flow after multilingual activation; "
        "the original remains available. Embedded selection uses a synthetic DOM event.\n\n"
        "When filter inputs are supplied, `05-word-search-fix-preview.png` compares the "
        "same controlled 50k corpus before/after the SQL query-plan fix. "
        "Five requests per version; exact eligible result controls pass in both. "
        "It does not measure model inference or PDF preparation.\n\n"
        "Hardware: ThinkPad T490, Intel i7-10510U, about 16 GiB RAM, Windows 11. "
        "`chart-data.json` retains selected raw-result hashes, settings/metadata, input checksums "
        "and unchanged screenshot checksums. Historical summaries have missing definition/settings "
        "fields; consult the retained benchmark report "
        "before interpreting them as final measurements.\n",
        encoding="utf-8",
    )
    for image in sorted(args.output.glob("*.png")):
        with Image.open(image) as opened:
            opened.verify()
        print(image.name)


if __name__ == "__main__":
    main()
