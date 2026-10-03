"""Faithfully normalized text v1 (M1.5 decision 2).

Meaning is preserved: original list numbering and markers stay as extracted, ambiguous
line-break hyphens stay hyphenated, and lines join conservatively. The saved PDF remains the
exact original — this normalization only shapes passage text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_NORMALIZATION_FORM = "NFC"
_WHITESPACE = re.compile(r"\s+")
_WORD_HYPHEN = re.compile(r"\w-$")


def normalize_cell(raw: str) -> str:
    """Normalize one cell's whitespace before cells are joined."""
    return _WHITESPACE.sub(" ", unicodedata.normalize(_NORMALIZATION_FORM, raw)).strip()


def normalize_table_row(cells: Sequence[str]) -> str:
    """Normalize each cell first, then join with tab separators so they survive."""
    return "\t".join(normalize_cell(cell) for cell in cells)


def normalize_text(raw: str) -> str:
    """Normalize one block's text: NFC, collapsed whitespace, conservative line joins."""
    text = unicodedata.normalize(_NORMALIZATION_FORM, raw)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    joined = ""
    for line in lines:
        if not line:
            continue
        if not joined:
            joined = line
        elif _WORD_HYPHEN.search(joined):
            # Ambiguous line-break hyphen: keep it — "re-/creation" never becomes "recreation".
            joined = joined + line
        else:
            joined = joined + " " + line
    return joined
