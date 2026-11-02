"""Recover horizontal RTL lines emitted in left-to-right glyph order.

Coordinates distinguish visual glyph order from already-logical RTL streams.
Keep each glyph's complete Unicode mapping intact: a glyph may map to a
ligature or several code points. This is layout recovery, not translation.
"""

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class GlyphText:
    text: str
    x: float | None


def _rtl(text: str) -> bool:
    return any(unicodedata.bidirectional(c) in ("R", "AL") for c in text)


def logical_line_text(units: list[GlyphText]) -> str:
    original = "".join(unit.text for unit in units)
    anchors = [unit.x for unit in units if unit.x is not None and _rtl(unit.text)]
    if len(anchors) < 2:
        return original
    steps = [b - a for a, b in zip(anchors[:-1], anchors[1:], strict=True) if abs(b - a) > 0.1]
    if not steps or sum(step > 0 for step in steps) < 0.9 * len(steps):
        return original  # Already logical, overlapping or ambiguous glyph placement.
    # A dominant LTR line with an embedded RTL label keeps its paragraph order.
    left = sum(any(unicodedata.bidirectional(c) == "L" for c in unit.text) for unit in units)
    if left >= len(anchors):
        return original

    ending = ""
    while units and "\n" in units[-1].text:
        ending = units[-1].text + ending
        units = units[:-1]
    pieces = []
    offset = 0
    while offset < len(units):
        unit = units[offset]
        if _rtl(unit.text) or any(unicodedata.bidirectional(c) == "NSM" for c in unit.text):
            pieces.append(unit.text)
            offset += 1
            continue
        end = offset + 1
        while (
            end < len(units)
            and not _rtl(units[end].text)
            and not any(unicodedata.bidirectional(c) == "NSM" for c in units[end].text)
        ):
            end += 1
        run = "".join(u.text for u in units[offset:end])
        # Preserve Latin/numeric runs and their internal punctuation, but keep
        # boundary spaces separate so they move with the reversed paragraph.
        leading = run[: len(run) - len(run.lstrip())]
        trailing = run[len(run.rstrip()) :]
        middle = run.strip()
        if not middle:
            pieces.append(run)
            offset = end
            continue
        pieces.extend(part for part in (leading, middle, trailing) if part)
        offset = end
    return "".join(reversed(pieces)) + ending
