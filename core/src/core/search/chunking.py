"""Deterministic saved-passage chunks with reconstructable code-point slices."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core.persistence.models import ExtractionReport, Passage
from core.search.profiles import ModelProfile

CHUNKER_VERSION = "passage-token-v2"


@dataclass(frozen=True)
class PassageSlice:
    passage: Passage
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.passage.text[self.start : self.end]


@dataclass(frozen=True)
class Chunk:
    text: str
    mappings: tuple[PassageSlice, ...]


def eligible_passages(report: ExtractionReport) -> tuple[Passage, ...]:
    if report.outcome is None or (
        report.outcome.outcome != "complete"
        and (report.review is None or report.review.state != "confirmed")
    ):
        raise ValueError("extraction_not_confirmed")
    pages = {page.page_number for page in report.pages if page.classification == "extracted"}
    return tuple(
        passage
        for passage in sorted(report.passages, key=lambda p: p.ordinal)
        if isinstance(passage.location, dict) and passage.location.get("page") in pages
    )


def _append(parts: list[PassageSlice], part: PassageSlice) -> list[PassageSlice]:
    if parts and parts[-1].passage.id == part.passage.id and parts[-1].end == part.start:
        return [*parts[:-1], PassageSlice(part.passage, parts[-1].start, part.end)]
    return [*parts, part]


def _text(parts: Sequence[PassageSlice], separator: str) -> str:
    return separator.join(part.text for part in parts)


def make_chunks(
    passages: Sequence[Passage],
    profile: ModelProfile,
    count: Callable[[str], int],
    validate_input: Callable[[str], str],
    *,
    max_text_bytes: int = 20_000_000,
    max_chunks: int = 10_000,
    preserve_fitting_passages: bool = False,
) -> tuple[Chunk, ...]:
    """Never alter passages or silently truncate. Page boundaries are conservative.

    A binary token-fit search finds a valid advancing slice, not an assumed
    globally maximal BPE prefix. Every chosen slice is checked again before use.
    """
    if sum(len(p.text.encode("utf-8")) for p in passages) > max_text_bytes:
        raise ValueError("search_text_limit")
    chunks: list[Chunk] = []
    pending: list[PassageSlice] = []
    previous_page = None
    has_new_text = False

    def fits(parts: list[PassageSlice]) -> bool:
        text = _text(parts, profile.separator)
        if count(text) > profile.chunk_tokens:
            return False
        try:
            validate_input(text)
        except ValueError:
            return False
        return True

    def flush(*, overlap: bool) -> None:
        nonlocal pending, has_new_text
        if not pending or not has_new_text:
            pending = []
            return
        text = _text(pending, profile.separator)
        validate_input(text)
        chunks.append(Chunk(text, tuple(pending)))
        if len(chunks) > max_chunks:
            raise ValueError("search_chunk_limit")
        tail = list(pending) if overlap else []
        while len(tail) > 1 and count(_text(tail[1:], profile.separator)) >= profile.overlap_tokens:
            tail = tail[1:]
        if tail and count(_text(tail, profile.separator)) > profile.overlap_tokens:
            first = tail[0]
            low, high = first.start, first.end
            while low < high:
                middle = (low + high) // 2
                trial = [PassageSlice(first.passage, middle, first.end), *tail[1:]]
                if count(_text(trial, profile.separator)) <= profile.overlap_tokens:
                    high = middle
                else:
                    low = middle + 1
            tail = (
                tail[1:]
                if low == first.end
                else [PassageSlice(first.passage, low, first.end), *tail[1:]]
            )
        pending = tail
        has_new_text = False

    for passage in passages:
        page = passage.location.get("page") if isinstance(passage.location, dict) else None
        if page != previous_page:
            flush(overlap=False)
        previous_page = page
        offset = 0
        while offset < len(passage.text):
            # Bound token-counting work even for a multi-megabyte paragraph.
            # This is only a search window: every chosen input is token-checked,
            # and subsequent windows retain all remaining source code points.
            window_end = min(len(passage.text), offset + max(1024, profile.chunk_tokens * 16))
            whole = PassageSlice(passage, offset, window_end)
            if fits(_append(pending, whole)):
                pending = _append(pending, whole)
                has_new_text = True
                offset = window_end
                if offset == len(passage.text):
                    break
                flush(overlap=True)
                continue
            if (
                preserve_fitting_passages
                and pending
                and offset == 0
                and window_end == len(passage.text)
                and fits([whole])
            ):
                # Benchmark-only opt-in: keep a fitting saved passage whole instead
                # of splitting it merely to fill the preceding chunk. Deployed
                # callers retain the default policy and its generation identity.
                flush(overlap=True)
                continue
            low, high = offset, window_end
            while low < high:
                middle = (low + high + 1) // 2
                if fits(_append(pending, PassageSlice(passage, offset, middle))):
                    low = middle
                else:
                    high = middle - 1
            if low == offset:
                if pending:
                    # Flush a chunk only once; overlap alone must not loop forever.
                    before = _text(pending, profile.separator)
                    flush(overlap=True)
                    if _text(pending, profile.separator) == before:
                        pending = []
                    if not fits(_append(pending, PassageSlice(passage, offset, offset + 1))):
                        pending = []
                    continue
                raise ValueError("model_cannot_encode_character")
            # Prefer a nearby whitespace boundary, retaining every original code point.
            candidate = passage.text.rfind(" ", offset, low)
            if candidate > offset and candidate >= low - 64:
                low = candidate + 1
            pending = _append(pending, PassageSlice(passage, offset, low))
            has_new_text = True
            offset = low
            flush(overlap=True)
    flush(overlap=False)
    return tuple(chunks)
